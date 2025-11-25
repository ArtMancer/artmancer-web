"""
Benchmark API endpoint for batch evaluation of diffusion image editing models.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from typing import List, Optional
from fastapi.responses import FileResponse

from ...models.schemas import BenchmarkRequest, BenchmarkResponse, BenchmarkResult, BenchmarkSummary
from ...services.benchmark_service import BenchmarkSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.post("/validate")
async def validate_benchmark_folder(
    request: Request,
    upload_type: str = Form("zip"),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Validate benchmark dataset structure without running benchmark.
    
    Supports two upload types:
    1. ZIP file: Upload a ZIP file containing: input/, mask/, groundtruth/ folders
    2. Folder: Upload folder structure directly (multiple files with relative paths)
    
    Returns validation status and image count.
    """
    import tempfile
    import shutil
    
    temp_file = None
    temp_dir = None
    try:
        # Log incoming request for debugging
        # Check if file was actually uploaded (not just None from default)
        has_file = file is not None and hasattr(file, 'filename') and file.filename is not None and file.filename != ""
        has_files = files is not None and len(files) > 0
        
        logger.info(f"📥 Validation request: upload_type={upload_type}, has_file={has_file}, has_files={has_files}")
        logger.info(f"📥 File details: file={file}, files={files}")
        
        if upload_type == "folder" and has_files and files is not None:
            # Handle folder upload
            logger.info(f"📁 Validating folder upload: {len(files)} files")
            
            # Fast validation: Check folder structure from filenames without saving all files
            # Count files in each folder from relative paths
            folder_counts = {"input": 0, "mask": 0, "groundtruth": 0}
            image_extensions = {'.jpg', '.jpeg', '.png'}
            
            for uploaded_file in files:
                if not uploaded_file.filename:
                    continue
                
                # Parse relative path: "folder_name/input/00001.jpg" or "input/00001.jpg"
                parts = uploaded_file.filename.replace('\\', '/').split('/')
                
                # Find which folder this file belongs to
                for folder_name in ["input", "mask", "groundtruth"]:
                    if folder_name in parts:
                        # Check if it's an image file
                        file_ext = Path(uploaded_file.filename).suffix.lower()
                        if file_ext in image_extensions:
                            folder_counts[folder_name] += 1
                        break
            
            logger.info(f"📊 Quick folder structure check: {folder_counts}")
            
            # Quick validation: Check if all 3 folders exist and have files
            if folder_counts["input"] == 0 or folder_counts["mask"] == 0 or folder_counts["groundtruth"] == 0:
                return {
                    "success": False,
                    "message": f"❌ Missing or empty folders. Found: input={folder_counts['input']}, mask={folder_counts['mask']}, groundtruth={folder_counts['groundtruth']}",
                    "image_count": 0,
                    "details": folder_counts,
                }
            
            # Check if counts match
            if len(set(folder_counts.values())) > 1:
                return {
                    "success": False,
                    "message": f"❌ Image count mismatch:\n  - input: {folder_counts['input']}\n  - mask: {folder_counts['mask']}\n  - groundtruth: {folder_counts['groundtruth']}",
                    "image_count": 0,
                    "details": folder_counts,
                }
            
            # Quick validation passed - return success
            # Full validation with file saving will happen during actual benchmark run
            return {
                "success": True,
                "message": f"✅ Detected {folder_counts['input']} valid image sets",
                "image_count": folder_counts["input"],
                "details": folder_counts,
            }
            
        elif upload_type == "zip" and has_file and file is not None:
            # Handle ZIP file upload
            logger.info(f"📦 Validating ZIP file: {file.filename}")
            
            # Save uploaded file to temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            
            # Extract ZIP to temporary directory
            temp_dir = tempfile.mkdtemp(prefix="benchmark_zip_")
            import zipfile
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Validate extracted folder structure (optimized - only check structure and counts)
            validation = BenchmarkSystem.validate_benchmark_folder(temp_dir)
            return validation
        else:
            # Better error message with details
            has_file = file is not None and file.filename is not None
            has_files = files is not None and len(files) > 0
            
            error_detail = f"Invalid request configuration. upload_type='{upload_type}', "
            error_detail += f"has_file={has_file}, "
            error_detail += f"has_files={has_files}"
            
            if upload_type == "zip" and not has_file:
                error_detail = "ZIP file upload requires a file. Please upload a ZIP file."
            elif upload_type == "folder" and not has_files:
                error_detail = "Folder upload requires files. Please select a folder to upload."
            else:
                error_detail = f"Invalid upload configuration. Expected upload_type='zip' with file, or upload_type='folder' with files. Got: upload_type='{upload_type}', has_file={has_file}, has_files={has_files}"
            
            logger.error(f"❌ Validation error: {error_detail}")
            raise HTTPException(status_code=400, detail=error_detail)
            
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file format")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
    finally:
        # Cleanup temporary files
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(
    upload_type: str = Form("zip"),
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    task_type: str = Form("object-removal"),
    prompt: str = Form("remove the object"),
    sample_count: int = Form(0),
    num_inference_steps: int = Form(40),
    guidance_scale: float = Form(1.0),
    true_cfg_scale: float = Form(4.0),
    negative_prompt: str = Form(None),
    seed: int = Form(None),
    input_quality: str = Form("high"),
):
    """
    Run benchmark evaluation on a dataset.
    
    Upload a ZIP file containing: input/, mask/, groundtruth/ folders.
    
    The prompt parameter will be used for ALL images in the benchmark.
    
    Sample selection:
    - sample_count = 0: Process all images
    - sample_count > 0: Process randomly selected samples
    
    Task implementation status:
    - object-removal: ✅ Implemented (Qwen Image Edit 2509)
    - white-balance: 🚧 TODO - Coming soon with Pix2Pix
    - object-insert: 🚧 TODO - Coming soon with Qwen Image Edit 2509 insertion mode
    """
    import tempfile
    import shutil
    import zipfile
    
    # Validate prompt (required)
    if not prompt or prompt.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Prompt is required for benchmark. This prompt will be used for all images."
        )
    
    # Check if task is implemented
    if task_type == "white-balance":
        raise HTTPException(
            status_code=501,
            detail="White balancing task is not yet implemented. Coming soon with Pix2Pix model integration."
        )
    
    if task_type == "object-insert":
        raise HTTPException(
            status_code=501,
            detail="Object insertion task is not yet implemented. Coming soon with Qwen Image Edit 2509 insertion mode."
        )
    
    # Only Object Removal is implemented
    if task_type != "object-removal":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task type: {task_type}. Only 'object-removal' is currently supported."
        )
    
    start_time = time.time()
    temp_file = None
    temp_dir = None
    
    try:
        if upload_type == "folder" and files:
            # Handle folder upload
            logger.info(f"📁 Running benchmark with folder upload: {len(files)} files")
            
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_folder_"))
            
            # Reconstruct folder structure from uploaded files
            for uploaded_file in files:
                relative_path = uploaded_file.filename
                if not relative_path:
                    continue
                
                file_path = temp_dir / relative_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file.file, f)
            
            logger.info(f"📁 Reconstructed folder structure in: {temp_dir}")
            
            # Initialize benchmark system with reconstructed folder
            benchmark = BenchmarkSystem(temp_dir)
            
        elif upload_type == "zip" and file:
            # Handle ZIP file upload
            logger.info(f"📦 Running benchmark with ZIP file: {file.filename}")
            
            # Save uploaded file to temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            
            # Extract ZIP to temporary directory
            temp_dir = tempfile.mkdtemp(prefix="benchmark_zip_")
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Initialize benchmark system with extracted folder
            benchmark = BenchmarkSystem(Path(temp_dir))
        else:
            raise HTTPException(
                status_code=400,
                detail="Please provide either a ZIP file (upload_type=zip) or folder files (upload_type=folder)"
            )
        
        # Load and validate dataset with sample selection
        validation_result = benchmark.load_and_validate(sample_count=sample_count)
        logger.info(f"✅ Dataset validated: {validation_result}")
        
        # Prepare generation parameters
        generation_kwargs = {}
        if num_inference_steps:
            generation_kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale:
            generation_kwargs["guidance_scale"] = guidance_scale
        if true_cfg_scale:
            generation_kwargs["true_cfg_scale"] = true_cfg_scale
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            generation_kwargs["seed"] = seed
        if input_quality:
            generation_kwargs["input_quality"] = input_quality
        
        # Generate images
        # NOTE: The same prompt will be used for ALL images in the benchmark
        logger.info(f"📝 Using prompt for all images: '{prompt}'")
        generated_results = benchmark.generate_images(
            task_type=task_type,
            prompt=prompt,  # Same prompt for all images
            **generation_kwargs
        )
        
        # Calculate metrics
        results_with_metrics = benchmark.calculate_metrics(generated_results)
        
        # Export results
        exported_files = benchmark.export_results(formats=["csv", "json", "latex"])
        
        # Convert to response format
        benchmark_results = []
        for result in results_with_metrics:
            metrics = result.get("metrics", {})
            benchmark_results.append(
                BenchmarkResult(
                    image_id=result["filename"].replace(".jpg", "").replace(".jpeg", ""),
                    filename=result["filename"],
                    success=result.get("success", False),
                    psnr=metrics.get("psnr"),
                    ssim=metrics.get("ssim"),
                    lpips=metrics.get("lpips"),
                    clip_score=metrics.get("clip_score"),
                    delta_e=metrics.get("de00"),
                    generation_time=result.get("generation_time", 0.0),
                    error=result.get("error"),
                )
            )
        
        # Calculate summary
        successful = sum(1 for r in benchmark_results if r.success)
        failed = len(benchmark_results) - successful
        
        # Calculate statistics for each metric
        def calc_stats(values):
            if not values:
                return None
            import numpy as np
            values = [v for v in values if v is not None]
            if not values:
                return None
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
        
        summary = BenchmarkSummary(
            total_images=len(benchmark_results),
            successful=successful,
            failed=failed,
            psnr=calc_stats([r.psnr for r in benchmark_results if r.psnr is not None]),
            ssim=calc_stats([r.ssim for r in benchmark_results if r.ssim is not None]),
            lpips=calc_stats([r.lpips for r in benchmark_results if r.lpips is not None]),
            clip_score=calc_stats([r.clip_score for r in benchmark_results if r.clip_score is not None]),
            delta_e=calc_stats([r.delta_e for r in benchmark_results if r.delta_e is not None]),
            generation_time=calc_stats([r.generation_time for r in benchmark_results]),
        )
        
        total_time = time.time() - start_time
        
        # Cleanup temporary files
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Cleanup benchmark system
        benchmark.cleanup()
        
        return BenchmarkResponse(
            success=True,
            results=benchmark_results,
            summary=summary,
            total_time=total_time,
            exported_files=exported_files,
        )
        
    except Exception as e:
        logger.exception(f"❌ Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.post("/upload")
async def upload_benchmark_dataset(file: UploadFile = File(...)):
    """
    Upload benchmark dataset (ZIP file).
    
    Returns the path where the dataset was saved.
    """
    # Save uploaded file to temporary location
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_upload_"))
    filename = file.filename or "benchmark_dataset.zip"
    file_path = temp_dir / filename
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"✅ Uploaded benchmark dataset: {file_path}")
        
        return {
            "success": True,
            "file_path": str(file_path),
            "filename": file.filename,
        }
    except Exception as e:
        logger.exception(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

