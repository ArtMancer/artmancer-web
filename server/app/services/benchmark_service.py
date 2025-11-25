"""
Benchmark System for Diffusion Image Editing.

Handles loading, validation, generation, and evaluation of benchmark datasets.
Supports folder, ZIP, and Parquet input formats.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException
from PIL import Image

from ..core.config import settings
from ..models.schemas import GenerationRequest
from ..services.generation_service import GenerationService
from ..services.image_processing import image_to_base64
from ..utils.metrics_utils import compute_all_metrics

logger = logging.getLogger(__name__)


class BenchmarkSystem:
    """Benchmark system for evaluating diffusion image editing models."""

    def __init__(self, input_path: str | Path):
        """
        Initialize benchmark system.
        
        Args:
            input_path: Path to folder, ZIP file, or Parquet file
        """
        self.input_path = Path(input_path)
        self.temp_dir: Optional[Path] = None
        self.data_dir: Optional[Path] = None
        self.input_images: List[Path] = []
        self.mask_images: List[Path] = []
        self.groundtruth_images: List[Path] = []
        
        # Initialize generation service
        self._generation_service = GenerationService()
        
        # Results storage
        self.results: List[Dict] = []
        
    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract ZIP file to temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        logger.info(f"📦 Extracting ZIP to {self.temp_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        return self.temp_dir
    
    def _load_parquet(self, parquet_path: Path) -> Tuple[List[bytes], List[bytes], List[bytes]]:
        """Load data from Parquet file."""
        logger.info(f"📊 Loading Parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Expected columns: input, mask, groundtruth (all as bytes or base64)
        inputs = []
        masks = []
        groundtruths = []
        
        for _, row in df.iterrows():
            # Handle bytes or base64 strings
            if 'input' in row:
                inputs.append(row['input'])
            if 'mask' in row:
                masks.append(row['mask'])
            if 'groundtruth' in row:
                groundtruths.append(row['groundtruth'])
        
        return inputs, masks, groundtruths
    
    @staticmethod
    def _find_benchmark_root(root_path: Path) -> Optional[Path]:
        """
        Find the folder containing input/, mask/, groundtruth/ subfolders.
        Handles cases where ZIP contains a wrapper folder.
        
        Args:
            root_path: Root path to search from
        
        Returns:
            Path to folder containing input/, mask/, groundtruth/ or None if not found
        """
        # First check if root_path itself contains the required folders
        if all((root_path / folder).exists() and (root_path / folder).is_dir() 
               for folder in ["input", "mask", "groundtruth"]):
            return root_path
        
        # Search in subdirectories (max depth 2 to avoid deep recursion)
        for item in root_path.iterdir():
            if item.is_dir():
                # Check if this subdirectory contains the required folders
                if all((item / folder).exists() and (item / folder).is_dir() 
                       for folder in ["input", "mask", "groundtruth"]):
                    logger.info(f"📁 Found benchmark folder structure in: {item.name}")
                    return item
        
        return None
    
    @staticmethod
    def validate_benchmark_folder(root_path: str | Path) -> Dict:
        """
        Validate benchmark folder structure and return status.
        
        Automatically detects folder structure even if ZIP contains a wrapper folder.
        Supports both:
        - Direct structure: input/, mask/, groundtruth/
        - Wrapped structure: wrapper_folder/input/, wrapper_folder/mask/, wrapper_folder/groundtruth/
        
        Args:
            root_path: Path to root folder (may contain wrapper folder)
        
        Returns:
            Dictionary with:
                - success: bool
                - message: str (error or success message)
                - image_count: int
                - details: dict with counts per folder
        """
        root_path = Path(root_path)
        
        if not root_path.exists():
            return {
                "success": False,
                "message": f"❌ Folder not found: {root_path}",
                "image_count": 0,
                "details": {},
            }
        
        # Find the actual benchmark folder (handles wrapper folders)
        benchmark_root = BenchmarkSystem._find_benchmark_root(root_path)
        if benchmark_root is None:
            return {
                "success": False,
                "message": f"❌ Could not find benchmark folder structure. Expected: input/, mask/, groundtruth/ folders (may be inside a wrapper folder)",
                "image_count": 0,
                "details": {},
            }
        
        input_dir = benchmark_root / "input"
        mask_dir = benchmark_root / "mask"
        groundtruth_dir = benchmark_root / "groundtruth"
        
        required_folders = {
            "input": input_dir,
            "mask": mask_dir,
            "groundtruth": groundtruth_dir,
        }
        
        # Check folder existence
        missing_folders = [name for name, path in required_folders.items() if not path.exists()]
        if missing_folders:
            return {
                "success": False,
                "message": f"❌ Missing required folder(s): {', '.join(missing_folders)}. Expected structure: input/, mask/, groundtruth/",
                "image_count": 0,
                "details": {},
            }
        
        # Fast validation: Only count files, don't load all file names
        # Use parallel counting for better performance
        from concurrent.futures import ThreadPoolExecutor
        
        def count_images_in_folder(folder_path: Path) -> int:
            """Count image files in a folder quickly."""
            if not folder_path.exists():
                return 0
            count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                count += len(list(folder_path.glob(ext)))
            return count
        
        # Count files in parallel for speed
        with ThreadPoolExecutor(max_workers=3) as executor:
            input_count_future = executor.submit(count_images_in_folder, input_dir)
            mask_count_future = executor.submit(count_images_in_folder, mask_dir)
            gt_count_future = executor.submit(count_images_in_folder, groundtruth_dir)
            
            counts = {
                "input": input_count_future.result(),
                "mask": mask_count_future.result(),
                "groundtruth": gt_count_future.result(),
            }
        
        # Check image count consistency
        if len(set(counts.values())) > 1:
            return {
                "success": False,
                "message": f"❌ Image count mismatch:\n  - input: {counts['input']}\n  - mask: {counts['mask']}\n  - groundtruth: {counts['groundtruth']}",
                "image_count": 0,
                "details": counts,
            }
        
        if counts["input"] == 0:
            return {
                "success": False,
                "message": "❌ No valid images found in folder. Supported formats: .jpg, .jpeg, .png",
                "image_count": 0,
                "details": counts,
            }
        
        # Skip detailed naming validation for speed - only check counts match
        # Detailed naming check can be done later if needed
        logger.info(f"✅ Quick validation passed: {counts['input']} images in each folder")
        
        return {
            "success": True,
            "message": f"✅ Detected {counts['input']} valid image sets",
            "image_count": counts["input"],
            "details": counts,
        }
    
    def _load_folder(self, folder_path: Path, sample_indices: Optional[List[int]] = None) -> None:
        """Load images from folder structure."""
        # Find the actual benchmark folder (handles wrapper folders)
        benchmark_root = self._find_benchmark_root(folder_path)
        if benchmark_root is None:
            validation = self.validate_benchmark_folder(folder_path)
            raise HTTPException(status_code=400, detail=validation["message"])
        
        input_dir = benchmark_root / "input"
        mask_dir = benchmark_root / "mask"
        groundtruth_dir = benchmark_root / "groundtruth"
        
        # Validate structure
        validation = self.validate_benchmark_folder(folder_path)
        if not validation["success"]:
            raise HTTPException(status_code=400, detail=validation["message"])
        
        # Update data_dir to point to the actual benchmark folder
        self.data_dir = benchmark_root
        
        # Get all image files
        input_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")) + sorted(input_dir.glob("*.png"))
        mask_files = sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg")) + sorted(mask_dir.glob("*.png"))
        gt_files = sorted(groundtruth_dir.glob("*.jpg")) + sorted(groundtruth_dir.glob("*.jpeg")) + sorted(groundtruth_dir.glob("*.png"))
        
        # Apply sample selection if provided
        if sample_indices is not None:
            input_files = [input_files[i] for i in sample_indices if i < len(input_files)]
            mask_files = [mask_files[i] for i in sample_indices if i < len(mask_files)]
            gt_files = [gt_files[i] for i in sample_indices if i < len(gt_files)]
            logger.info(f"📊 Selected {len(input_files)} samples from {validation['image_count']} total images")
        
        self.input_images = input_files
        self.mask_images = mask_files
        self.groundtruth_images = gt_files
        # data_dir already set to benchmark_root above
        
        logger.info(f"✅ Loaded {len(self.input_images)} image pairs from {benchmark_root}")
    
    def load_and_validate(self, sample_count: int = 0) -> Dict[str, int]:
        """
        Load and validate benchmark dataset.
        
        Args:
            sample_count: Number of samples to process (0 = all)
        
        Returns:
            Dictionary with validation results
        """
        logger.info(f"📂 Loading benchmark dataset from: {self.input_path}")
        
        try:
            # Determine sample indices
            sample_indices = None
            if sample_count > 0:
                # First validate to get total count
                if self.input_path.suffix == ".zip":
                    temp_dir = self._extract_zip(self.input_path)
                    validation = self.validate_benchmark_folder(temp_dir)
                else:
                    validation = self.validate_benchmark_folder(self.input_path)
                
                if not validation["success"]:
                    raise HTTPException(status_code=400, detail=validation["message"])
                
                total_count = validation["image_count"]
                
                # Handle sample selection
                if sample_count > total_count:
                    logger.warning(f"⚠️ Requested {sample_count} samples but only {total_count} available. Using all.")
                    sample_count = total_count
                
                if sample_count > 0:
                    import random
                    # Use fixed seed for reproducibility
                    random.seed(42)
                    sample_indices = sorted(random.sample(range(total_count), min(sample_count, total_count)))
                    logger.info(f"📊 Selected {len(sample_indices)} samples: {sample_indices[:5]}..." if len(sample_indices) > 5 else f"📊 Selected {len(sample_indices)} samples: {sample_indices}")
            
            if self.input_path.suffix == ".zip":
                # Extract ZIP
                extracted_dir = self._extract_zip(self.input_path)
                self._load_folder(extracted_dir, sample_indices)
            elif self.input_path.suffix == ".parquet":
                # Handle Parquet (in-memory, no folder structure)
                inputs, masks, groundtruths = self._load_parquet(self.input_path)
                # Convert to PIL Images and store
                # Note: For Parquet, we'll handle differently in generate_images
                raise NotImplementedError("Parquet format support coming soon")
            else:
                # Assume folder
                self._load_folder(self.input_path, sample_indices)
            
            return {
                "total_pairs": len(self.input_images),
                "input_count": len(self.input_images),
                "mask_count": len(self.mask_images),
                "groundtruth_count": len(self.groundtruth_images),
            }
        except Exception as e:
            logger.exception(f"❌ Failed to load dataset: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")
    
    def generate_images(
        self,
        task_type: str = "object-removal",
        prompt: str = "remove the object",
        **generation_kwargs
    ) -> List[Dict]:
        """
        Generate edited images for all input pairs.
        
        Args:
            task_type: Task type ("object-removal", "object-insert", "white-balance")
            prompt: Prompt for generation (SAME prompt used for ALL images)
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of generation results
        
        Task Implementation Status:
        - object-removal: ✅ Implemented (Qwen Image Edit 2509)
        - white-balance: 🚧 TODO - Coming soon with Pix2Pix
        - object-insert: 🚧 TODO - Coming soon with Qwen Image Edit 2509 insertion mode
        """
        logger.info(f"🎨 Generating images for {len(self.input_images)} pairs using prompt: '{prompt}'")
        
        generated_results = []
        
        for i, (input_path, mask_path) in enumerate(zip(self.input_images, self.mask_images)):
            try:
                # Load images
                input_img = Image.open(input_path).convert("RGB")
                mask_img = Image.open(mask_path).convert("L")  # Grayscale mask
                
                # Convert to base64
                input_b64 = image_to_base64(input_img)
                mask_b64 = image_to_base64(mask_img)
                
                # Create generation request
                request = GenerationRequest(
                    prompt=prompt,
                    input_image=input_b64,
                    mask_image=mask_b64,
                    task_type=task_type,
                    **generation_kwargs
                )
                
                # Generate
                result = self._generation_service.generate(request)
                
                if result.get("success"):
                    generated_results.append({
                        "index": i,
                        "filename": input_path.name,
                        "generated_image": result["image"],
                        "generation_time": result.get("generation_time", 0.0),
                        "success": True,
                    })
                    logger.info(f"✅ Generated {i+1}/{len(self.input_images)}: {input_path.name}")
                else:
                    generated_results.append({
                        "index": i,
                        "filename": input_path.name,
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                    })
                    logger.error(f"❌ Failed to generate {input_path.name}")
                    
            except Exception as e:
                logger.exception(f"❌ Error generating image {i+1}: {e}")
                generated_results.append({
                    "index": i,
                    "filename": input_path.name if i < len(self.input_images) else "unknown",
                    "success": False,
                    "error": str(e),
                })
        
        return generated_results
    
    def calculate_metrics(self, generated_results: List[Dict]) -> List[Dict]:
        """
        Calculate metrics for all generated images.
        
        Args:
            generated_results: List of generation results with 'generated_image' (base64)
        
        Returns:
            List of results with metrics
        """
        logger.info(f"📊 Calculating metrics for {len(generated_results)} images...")
        
        results = []
        
        for gen_result, gt_path in zip(generated_results, self.groundtruth_images):
            if not gen_result.get("success"):
                results.append({
                    **gen_result,
                    "metrics": None,
                })
                continue
            
            try:
                # Load groundtruth
                gt_img = Image.open(gt_path).convert("RGB")
                
                # Decode generated image
                from ..services.image_processing import base64_to_image
                gen_img = base64_to_image(gen_result["generated_image"])
                
                # Calculate metrics
                metrics = compute_all_metrics(gen_img, gt_img)
                
                results.append({
                    **gen_result,
                    "metrics": metrics,
                })
                
                logger.info(
                    f"📊 Metrics for {gen_result['filename']}: "
                    f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}, "
                    f"LPIPS={metrics['lpips']:.4f}, ΔE00={metrics['de00']:.3f}"
                )
                
            except Exception as e:
                logger.exception(f"❌ Error calculating metrics for {gen_result['filename']}: {e}")
                results.append({
                    **gen_result,
                    "metrics": None,
                    "metrics_error": str(e),
                })
        
        self.results = results
        return results
    
    def export_results(
        self,
        output_dir: Optional[Path] = None,
        formats: List[str] = ["csv", "json", "latex"]
    ) -> Dict[str, str]:
        """
        Export benchmark results.
        
        Args:
            output_dir: Output directory (default: temp directory)
            formats: List of formats to export ("csv", "json", "latex")
        
        Returns:
            Dictionary with paths to exported files
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="benchmark_results_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Prepare data for export
        export_data = []
        for result in self.results:
            if result.get("metrics"):
                export_data.append({
                    "image_id": result["filename"].replace(".jpg", "").replace(".jpeg", ""),
                    "psnr": result["metrics"]["psnr"],
                    "ssim": result["metrics"]["ssim"],
                    "lpips": result["metrics"]["lpips"],
                    "clip_score": result["metrics"].get("clip_score"),
                    "delta_e": result["metrics"]["de00"],
                    "generation_time": result.get("generation_time", 0.0),
                    "success": result.get("success", False),
                })
            else:
                export_data.append({
                    "image_id": result["filename"].replace(".jpg", "").replace(".jpeg", ""),
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "clip_score": None,
                    "delta_e": None,
                    "generation_time": result.get("generation_time", 0.0),
                    "success": result.get("success", False),
                    "error": result.get("error", "Unknown error"),
                })
        
        df = pd.DataFrame(export_data)
        
        # Export CSV
        if "csv" in formats:
            csv_path = output_dir / "benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            exported_files["csv"] = str(csv_path)
            logger.info(f"✅ Exported CSV to {csv_path}")
        
        # Export JSON
        if "json" in formats:
            json_path = output_dir / "benchmark_results.json"
            json_data = {
                "summary": self._calculate_summary(df),
                "per_image_results": export_data,
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            exported_files["json"] = str(json_path)
            logger.info(f"✅ Exported JSON to {json_path}")
        
        # Export LaTeX table
        if "latex" in formats:
            latex_path = output_dir / "benchmark_results.tex"
            latex_table = self._generate_latex_table(df)
            with open(latex_path, "w") as f:
                f.write(latex_table)
            exported_files["latex"] = str(latex_path)
            logger.info(f"✅ Exported LaTeX table to {latex_path}")
        
        return exported_files
    
    def _calculate_summary(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics."""
        numeric_cols = ["psnr", "ssim", "lpips", "clip_score", "delta_e", "generation_time"]
        summary = {}
        
        for col in numeric_cols:
            if col in df.columns:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    summary[col] = {
                        "mean": float(valid_data.mean()),
                        "std": float(valid_data.std()),
                        "min": float(valid_data.min()),
                        "max": float(valid_data.max()),
                        "median": float(valid_data.median()),
                        "count": int(len(valid_data)),
                    }
        
        summary["total_images"] = len(df)
        summary["successful"] = int(df["success"].sum())
        summary["failed"] = int((~df["success"]).sum())
        
        return summary
    
    def _generate_latex_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX-formatted table for research paper."""
        summary = self._calculate_summary(df)
        
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Benchmark Results Summary}\n"
        latex += "\\label{tab:benchmark_results}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\toprule\n"
        latex += "Metric & Mean $\\pm$ Std & Min & Max \\\\\n"
        latex += "\\midrule\n"
        
        metric_labels = {
            "psnr": "PSNR (dB)",
            "ssim": "SSIM",
            "lpips": "LPIPS",
            "clip_score": "CLIP-S",
            "delta_e": "$\\Delta$E00",
        }
        
        for metric in ["psnr", "ssim", "lpips", "clip_score", "delta_e"]:
            if metric in summary:
                stats = summary[metric]
                label = metric_labels.get(metric, metric.upper())
                latex += f"{label} & {stats['mean']:.3f} $\\pm$ {stats['std']:.3f} & {stats['min']:.3f} & {stats['max']:.3f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"🧹 Cleaned up temporary directory: {self.temp_dir}")

