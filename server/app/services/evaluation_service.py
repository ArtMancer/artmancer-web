from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import HTTPException
from PIL import Image

from ..models.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationMetrics,
)
from ..services.image_processing import base64_to_image, image_to_base64
from ..services.visualization import create_visualization_service
from ..core.config import settings
from ..utils.metrics_utils import compute_all_metrics
from ..services.generation_service import GenerationService
from ..models.schemas import GenerationRequest

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating image quality and similarity metrics."""

    def __init__(self) -> None:
        # Initialize visualization service for logging
        self._visualization = create_visualization_service(
            output_dir=settings.visualization_dir if settings.visualization_dir else None,
            enabled=settings.enable_visualization,
        )
        # Initialize generation service for generating images
        self._generation_service = GenerationService()

    def _calculate_metrics(
        self, original: Image.Image, target: Image.Image
    ) -> EvaluationMetrics:
        """
        Calculate evaluation metrics between original and target images.
        
        Args:
            original: Original image (PIL Image)
            target: Target/reference image (PIL Image)
        
        Returns:
            EvaluationMetrics with calculated scores (PSNR, SSIM, LPIPS, ΔE00)
        """
        try:
            # Compute all metrics
            metrics_dict = compute_all_metrics(original, target)
            
            metrics = EvaluationMetrics(
                psnr=metrics_dict["psnr"],
                ssim=metrics_dict["ssim"],
                lpips=metrics_dict["lpips"],
                de00=metrics_dict["de00"],
                evaluation_time=0.0,  # Will be set by caller
            )
            
            logger.info(
                f"📊 Metrics calculated: PSNR={metrics.psnr:.2f}dB, "
                f"SSIM={metrics.ssim:.4f}, LPIPS={metrics.lpips:.4f}, ΔE00={metrics.de00:.3f}"
            )
            return metrics
        except Exception as e:
            logger.exception(f"❌ Error calculating metrics: {e}")
            # Return empty metrics on error
            return EvaluationMetrics(evaluation_time=0.0)

    def evaluate_single(
        self,
        original_image: str,
        target_image: str,
        conditional_images: Optional[list[str]] = None,
        input_image: Optional[str] = None,
        prompt: Optional[str] = None,
        reference_image: Optional[str] = None,
        task_type: str = "object-removal",
        filename: Optional[str] = None,
        request_id: Optional[str] = None,
        save_visualization: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a single image pair.
        
        Args:
            original_image: Base64 encoded original image
            target_image: Base64 encoded target image
            conditional_images: Optional list of base64 encoded conditional images
            input_image: Optional base64 encoded input image
            filename: Optional filename for this evaluation pair
            request_id: Optional request ID for logging
        
        Returns:
            EvaluationResult with metrics
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not original_image or not original_image.strip():
                raise ValueError("original_image is required and cannot be empty")
            if not target_image or not target_image.strip():
                raise ValueError("target_image is required and cannot be empty")
            
            original = base64_to_image(original_image)
            # IMPORTANT: target image is ONLY used for metrics calculation, NOT passed to the model
            # The model should generate based on original + conditional images + prompt only
            target = base64_to_image(target_image)
            
            # Ensure images have same size for comparison (only for metrics, not for generation)
            if original.size != target.size:
                logger.info(f"🔄 Resizing target from {target.size} to {original.size} (for metrics only)")
                target = target.resize(original.size, Image.Resampling.LANCZOS)
            
            # Load conditional images if provided
            conditional_imgs = None
            if conditional_images:
                conditional_imgs = [base64_to_image(img) for img in conditional_images]
                logger.info(f"📸 Loaded {len(conditional_imgs)} conditional images")
            
            # Load input image if provided
            input_img = None
            if input_image:
                input_img = base64_to_image(input_image)
                logger.info("📸 Loaded input image")
            
            # Load reference image if provided (for object-insert task)
            ref_img = None
            if reference_image:
                ref_img = base64_to_image(reference_image)
                logger.info("📸 Loaded reference image for object-insert task")
            
            # Generate image from original + conditional images + prompt
            logger.info("🎨 Generating image for evaluation...")
            generated_img = None
            generation_time = 0.0
            gen_task_type = None  # Will be set during generation
            model_file_used = None  # Will be set during generation
            actual_model_used = None  # Will be set from generation result
            
            try:
                # Map evaluation task_type to generation task_type
                # evaluation task_type: "white-balance" | "object-insert" | "object-removal"
                # generation task_type: "insertion" | "removal"
                gen_task_type = "removal"  # default
                if task_type == "object-insert":
                    gen_task_type = "insertion"
                elif task_type == "object-removal":
                    gen_task_type = "removal"
                elif task_type == "white-balance":
                    # White balance: use dedicated model if configured, otherwise use removal model
                    from ..core.config import settings
                    if settings.model_file_white_balance:
                        gen_task_type = "white-balance"
                        logger.info("🎨 White-balance task: using dedicated white balance model")
                    else:
                        gen_task_type = "removal"
                        logger.info("⚠️ White-balance task: using removal model (no dedicated model configured)")
                
                logger.info(f"🎯 Evaluation task_type: {task_type} -> Generation task_type: {gen_task_type}")
                
                # Determine model file used based on generation task type
                from ..core.config import settings
                if gen_task_type == "insertion":
                    model_file_used = settings.model_file_insertion
                elif gen_task_type == "white-balance" and settings.model_file_white_balance:
                    model_file_used = settings.model_file_white_balance
                else:  # removal or white-balance fallback
                    model_file_used = settings.model_file_removal
                
                logger.info(f"📦 Model file used: {model_file_used}")
                
                # Prepare generation request
                # For white-balance, mask is not required
                mask_img = None
                if gen_task_type != "white-balance":
                    # Extract mask from conditional images (first one should be mask)
                    if conditional_imgs and len(conditional_imgs) > 0:
                        # First conditional image should be the mask
                        mask_img = conditional_imgs[0]
                        logger.info("📸 Using first conditional image as mask")
                    else:
                        # If no conditional images, we can't generate - skip generation
                        logger.warning("⚠️ No conditional images provided, skipping generation")
                        raise ValueError("Conditional images (mask) required for generation")
                else:
                    # White balance doesn't need mask
                    logger.info("🎨 White-balance task: mask not required")
                
                # Prepare generation request
                # For object-insert, ensure reference_image is provided
                gen_reference_image = None
                if gen_task_type == "insertion":
                    if ref_img:
                        gen_reference_image = image_to_base64(ref_img)
                    else:
                        logger.warning("⚠️ Object-insert task but no reference image provided, falling back to removal")
                        gen_task_type = "removal"
                
                # CRITICAL: Generation request uses ONLY original image, NOT target image
                # Target image is only used later for metrics calculation
                # For white-balance, mask_image is not required (can be empty string or None)
                gen_request = GenerationRequest(
                    prompt=prompt or "",  # Empty prompt is allowed for white-balance
                    input_image=image_to_base64(original),  # Only original, NOT target
                    mask_image=image_to_base64(mask_img) if mask_img else "",  # Empty for white-balance
                    reference_image=gen_reference_image,  # Only for object-insert task
                    task_type=gen_task_type,  # Explicitly set task_type for white-balance
                )
                
                # Generate image with explicit task type
                gen_start = time.time()
                # We need to ensure the correct pipeline is used
                # The generation service will use task_type from request if provided
                if gen_task_type == "insertion" and not gen_reference_image:
                    logger.warning("⚠️ Insertion task but no reference image, using removal model")
                    gen_request.task_type = "removal"  # Override task_type
                elif gen_task_type == "removal" and gen_reference_image:
                    logger.warning("⚠️ Removal task but reference image provided, ignoring reference")
                    gen_request.reference_image = None
                elif gen_task_type == "white-balance":
                    # Ensure task_type is set for white-balance
                    gen_request.task_type = "white-balance"
                    logger.info("✅ White-balance task_type set in generation request")
                
                # Generate image - model does NOT see target image
                gen_result = self._generation_service.generate(gen_request)
                generation_time = time.time() - gen_start
                
                # Extract generated image and model info from result
                if isinstance(gen_result, dict) and "image" in gen_result:
                    generated_img = base64_to_image(gen_result["image"])
                    # Get actual model used from generation result
                    actual_model_used = gen_result.get("model_used", f"qwen_local_{gen_task_type}")
                else:
                    raise ValueError("Invalid generation result format")
                
                logger.info(f"✅ Image generated in {generation_time:.2f}s (model did NOT see target image)")
                logger.info(f"📦 Actual model used: {actual_model_used}")
                
                # Resize generated to match target if needed (only for metrics comparison)
                if generated_img.size != target.size:
                    logger.info(f"🔄 Resizing generated from {generated_img.size} to {target.size} (for metrics only)")
                    generated_img = generated_img.resize(target.size, Image.Resampling.LANCZOS)
                
            except Exception as e:
                logger.exception(f"❌ Error generating image: {e}")
                # If generation fails, fall back to comparing original with target
                logger.warning("⚠️ Falling back to original vs target comparison")
                generated_img = original
            
            # Calculate metrics: compare GENERATED image with TARGET image
            # Note: Target image is ONLY used here for metrics, model never saw it
            metrics = self._calculate_metrics(generated_img, target)
            metrics.evaluation_time = time.time() - start_time
            
            # Save visualization for logging/reporting (only if enabled)
            if request_id and save_visualization:
                metadata = {
                    "filename": filename,
                    "prompt": prompt,
                    "evaluation_task_type": task_type,  # Task selected in evaluation UI
                    "generation_task_type": gen_task_type if gen_task_type else None,  # Task used for generation
                    "model_file_used": str(model_file_used) if model_file_used else None,  # Model file path
                    "actual_model_used": actual_model_used if actual_model_used else None,  # Model name from generation
                    "has_reference_image": reference_image is not None,
                    "metrics": {
                        "psnr": metrics.psnr,
                        "ssim": metrics.ssim,
                        "lpips": metrics.lpips,
                        "de00": metrics.de00,
                        "evaluation_time": metrics.evaluation_time,
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "has_conditional_images": conditional_imgs is not None,
                    "has_input_image": input_img is not None,
                    # Keep old task_type for backward compatibility
                    "task_type": task_type,
                }
                self._visualization.save_evaluation_visualization(
                    request_id=request_id,
                    original=original,
                    target=target,
                    generated=generated_img,
                    conditional_images=conditional_imgs,
                    input_image=input_img,
                    reference_image=ref_img,
                    metadata=metadata,
                    filename=filename,
                    save_images=save_visualization,
                )
            
            return EvaluationResult(
                filename=filename,
                metrics=metrics,
                success=True,
                evaluation_task_type=task_type,
                generation_task_type=gen_task_type if gen_task_type else None,
                model_file_used=str(model_file_used) if model_file_used else None,
                actual_model_used=actual_model_used if actual_model_used else None,
            )
        except Exception as e:
            logger.exception("❌ Error evaluating image pair")
            return EvaluationResult(
                filename=filename,
                metrics=EvaluationMetrics(evaluation_time=time.time() - start_time),
                success=False,
                error=str(e),
                evaluation_task_type=task_type,
                generation_task_type=None,
                model_file_used=None,
                actual_model_used=None,
            )

    def evaluate_batch(
        self,
        image_pairs: List[Dict[str, str]],
        conditional_images: Optional[list[str]] = None,
        input_image: Optional[str] = None,
        task_type: str = "object-removal",
        reference_image: Optional[str] = None,
        request_id: Optional[str] = None,
        save_visualization: bool = False,  # Default False for batch to save storage
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple image pairs in batch.
        
        Args:
            image_pairs: List of dictionaries with 'original_image' and 'target_image' keys
            conditional_images: Optional list of base64 encoded conditional images (shared across pairs)
            input_image: Optional base64 encoded input image (shared across pairs)
            task_type: Evaluation task type (white-balance, object-insert, object-removal)
            reference_image: Optional reference image for object-insert task
            request_id: Optional request ID for logging
        
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for idx, pair in enumerate(image_pairs):
            logger.info(f"📊 Evaluating pair {idx + 1}/{len(image_pairs)}")
            
            # Use pair-specific request_id if available, otherwise use base request_id
            pair_request_id = f"{request_id}_pair_{idx + 1}" if request_id else None
            
            # Use pair-specific reference_image if provided, otherwise use shared reference_image
            pair_reference_image = pair.get("reference_image") or reference_image
            
            result = self.evaluate_single(
                pair["original_image"],
                pair["target_image"],
                conditional_images=conditional_images,
                input_image=input_image,
                prompt=pair.get("prompt"),
                reference_image=pair_reference_image,
                task_type=task_type,
                filename=pair.get("filename"),
                request_id=pair_request_id,
                save_visualization=save_visualization,  # Passed from evaluate_batch parameter
            )
            results.append(result)
        
        return results

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """
        Main evaluation method that handles both single and batch evaluation.
        
        Args:
            request: EvaluationRequest with image data
        
        Returns:
            EvaluationResponse with all results
        """
        start_time = time.time()
        results: List[EvaluationResult] = []
        
        # Generate unique request ID for logging
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        try:
            # Log task type
            task_type = request.task_type or "object-removal"
            logger.info(f"📊 Starting evaluation with task_type: {task_type}")
            
            # Validate reference image for object-insert task
            if task_type == "object-insert" and not request.reference_image:
                logger.warning("⚠️ Reference image missing for object-insert task")
            
            # Get save_visualization setting (default True for single, False for batch)
            save_vis = request.save_visualization if request.save_visualization is not None else True
            
            # Single image evaluation
            if request.original_image and request.target_image:
                logger.info("📊 Starting single image evaluation")
                result = self.evaluate_single(
                    request.original_image,
                    request.target_image,
                    conditional_images=request.conditional_images,
                    input_image=request.input_image,
                    prompt=request.prompt,
                    reference_image=request.reference_image,
                    task_type=task_type,
                    filename=None,
                    request_id=request_id,
                    save_visualization=save_vis,
                )
                results.append(result)
            
            # Batch evaluation
            elif request.image_pairs:
                logger.info(f"📊 Starting batch evaluation for {len(request.image_pairs)} pairs")
                # For batch, default to False to save storage unless explicitly requested
                batch_save_vis = request.save_visualization if request.save_visualization is not None else False
                if batch_save_vis:
                    logger.info("⚠️ Visualization enabled for batch - this may use significant disk space")
                else:
                    logger.info("💾 Visualization disabled for batch to save storage")
                
                pair_dicts = [
                    {
                        "original_image": pair.original_image,
                        "target_image": pair.target_image,
                        "prompt": pair.prompt,
                        "filename": pair.filename,
                    }
                    for pair in request.image_pairs
                ]
                results = self.evaluate_batch(
                    pair_dicts,
                    conditional_images=request.conditional_images,
                    input_image=request.input_image,
                    reference_image=request.reference_image,
                    task_type=task_type,
                    request_id=request_id,
                    save_visualization=batch_save_vis,
                )
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either (original_image and target_image) or image_pairs must be provided",
                )
            
            # Calculate statistics
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            total_time = time.time() - start_time
            
            logger.info(f"✅ Evaluation completed: {successful} successful, {failed} failed in {total_time:.2f}s")
            # Only log visualization path if visualization was saved
            save_vis = request.save_visualization if request.save_visualization is not None else (True if request.original_image and request.target_image else False)
            if save_vis:
                logger.info(f"📁 Evaluation logs saved to: visualizations/{request_id}")
            else:
                logger.info("💾 Visualization skipped to save storage")
            
            return EvaluationResponse(
                success=True,
                results=results,
                total_pairs=len(results),
                successful_evaluations=successful,
                failed_evaluations=failed,
                total_evaluation_time=round(total_time, 2),
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("❌ Unexpected error during evaluation")
            raise HTTPException(status_code=500, detail=str(e)) from e

