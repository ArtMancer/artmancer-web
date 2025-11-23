from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for saving visualization images for debugging and monitoring."""

    def __init__(self, output_dir: Path | str, enabled: bool = True):
        """
        Initialize visualization service.

        Args:
            output_dir: Directory to save visualization images
            enabled: Whether visualization is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Visualization output directory: {self.output_dir}")

    def save_generation_visualization(
        self,
        request_id: str,
        original: Image.Image,
        mask: Optional[Image.Image],  # Optional for white-balance task
        conditional_images: list[Image.Image],
        output: Image.Image,
        reference_image: Optional[Image.Image] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Save all images from a generation request for visualization.

        Args:
            request_id: Unique identifier for this request (timestamp or UUID)
            original: Original input image
            mask: Mask image
            conditional_images: List of conditional images [mask_cond, background, obj, ...]
            output: Generated output image
            reference_image: Optional reference image for object insertion
            metadata: Optional metadata to save as JSON
        """
        if not self.enabled:
            return

        try:
            # Create subdirectory for this request
            request_dir = self.output_dir / request_id
            request_dir.mkdir(parents=True, exist_ok=True)

            # Save input images
            original.save(request_dir / "01_original.png", "PNG")
            # Save original mask for reference (before processing) - skip if None (white-balance)
            if mask is not None:
                mask.save(request_dir / "02_mask_original.png", "PNG")

            # Save conditional images
            if len(conditional_images) >= 1:
                # For insertion task: conditional_images only has [reference] - 1 image
                # For removal task: conditional_images has [mask, background, object, mae] - 4 images
                if reference_image:
                    # This is insertion task: [reference] - 1 image only
                    conditional_images[0].save(request_dir / "06_conditional_reference.png", "PNG")
                    logger.info("📸 Conditional reference image saved (insertion task, only reference in conditionals)")
                else:
                    # This is removal task: [mask, background, object, mae] - 4 images
                    if len(conditional_images) >= 1:
                        conditional_images[0].save(request_dir / "03_conditional_mask.png", "PNG")
                    if len(conditional_images) >= 2:
                        conditional_images[1].save(request_dir / "04_conditional_background.png", "PNG")
                    if len(conditional_images) >= 3:
                        conditional_images[2].save(request_dir / "05_conditional_object.png", "PNG")
                    if len(conditional_images) >= 4:
                        conditional_images[3].save(request_dir / "06_conditional_mae.png", "PNG")
                        logger.info("📸 MAE conditional image saved (removal task)")

                # Save additional conditional images if any (beyond the 4th)
                for i, cond_img in enumerate(conditional_images[4:], start=7):
                    cond_img.save(request_dir / f"{i:02d}_conditional_extra_{i-6}.png", "PNG")

            # Save reference image if provided (for object insertion)
            if reference_image:
                reference_image.save(request_dir / "07_reference_image.png", "PNG")
                logger.info("📸 Reference image saved")

            # Save output image
            output.save(request_dir / "99_output.png", "PNG")

            # Save prompt to text file for easy reading
            if metadata and "prompt" in metadata:
                prompt_file = request_dir / "00_prompt.txt"
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {metadata['prompt']}\n")
                    if metadata.get("negative_prompt"):
                        f.write(f"\nNegative Prompt: {metadata['negative_prompt']}\n")
                logger.info("📝 Prompt saved to text file")

            # Save metadata if provided
            if metadata:
                import json

                metadata_file = request_dir / "metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Visualization saved to: {request_dir}")
        except Exception as exc:
            logger.warning(f"⚠️ Failed to save visualization: {exc}", exc_info=True)

    def save_evaluation_visualization(
        self,
        request_id: str,
        original: Image.Image,
        target: Image.Image,
        generated: Optional[Image.Image] = None,
        conditional_images: Optional[list[Image.Image]] = None,
        input_image: Optional[Image.Image] = None,
        reference_image: Optional[Image.Image] = None,
        metadata: Optional[dict] = None,
        filename: Optional[str] = None,
        save_images: bool = True,  # Set False to only save metadata
    ) -> None:
        """
        Save all images from an evaluation request for logging and reporting.
        
        Args:
            request_id: Unique identifier for this evaluation request
            original: Original input image
            target: Target/reference image for comparison
            conditional_images: Optional list of conditional images (mask, background, etc.)
            input_image: Optional input image for conditional evaluation
            metadata: Optional metadata including metrics and evaluation info
            filename: Optional original filename for this evaluation pair
        """
        if not self.enabled:
            return

        try:
            # Create subdirectory for this evaluation request
            request_dir = self.output_dir / request_id
            request_dir.mkdir(parents=True, exist_ok=True)
            
            # If filename provided, create subdirectory for this pair
            if filename:
                pair_dir = request_dir / filename.replace("/", "_").replace("\\", "_")
                pair_dir.mkdir(parents=True, exist_ok=True)
                request_dir = pair_dir

            # Save images only if save_images is True (to save storage for batch)
            if save_images:
                # Save main evaluation images
                original.save(request_dir / "01_original.png", "PNG")
                target.save(request_dir / "02_target.png", "PNG")
                if generated:
                    generated.save(request_dir / "99_generated.png", "PNG")
                    logger.info("📸 Saved generated image")
                logger.info("📸 Saved original and target images")

                # Save input image if provided
                if input_image:
                    input_image.save(request_dir / "00_input.png", "PNG")
                    logger.info("📸 Saved input image")

                # Save reference image if provided (for object-insert task)
                if reference_image:
                    reference_image.save(request_dir / "06_reference_image.png", "PNG")
                    logger.info("📸 Saved reference image")

                # Save conditional images if provided
                if conditional_images:
                    for idx, cond_img in enumerate(conditional_images, start=1):
                        cond_img.save(request_dir / f"03_conditional_{idx:02d}.png", "PNG")
                    logger.info(f"📸 Saved {len(conditional_images)} conditional images")
            else:
                logger.info("💾 Skipping image save to conserve storage (metadata only)")

            # Save metadata with metrics
            if metadata:
                import json
                metadata_file = request_dir / "metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info("📝 Saved evaluation metadata")

            # Save evaluation summary to text file
            summary_file = request_dir / "00_evaluation_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"Evaluation ID: {request_id}\n")
                if filename:
                    f.write(f"Filename: {filename}\n")
                if metadata:
                    if metadata.get("task_type"):
                        f.write(f"Task Type: {metadata['task_type']}\n")
                    if metadata.get("prompt"):
                        f.write(f"Prompt: {metadata['prompt']}\n")
                    if metadata.get("has_reference_image"):
                        f.write(f"Has Reference Image: {metadata['has_reference_image']}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                
                if metadata:
                    f.write("Metrics:\n")
                    if metadata.get("metrics"):
                        metrics = metadata["metrics"]
                        f.write(f"  PSNR: {metrics.get('psnr', 'N/A')} dB (higher is better, >30 good, >40 excellent)\n")
                        f.write(f"  SSIM: {metrics.get('ssim', 'N/A')} (higher is better, >0.90 good, >0.95 excellent)\n")
                        f.write(f"  LPIPS: {metrics.get('lpips', 'N/A')} (lower is better, <0.10 good, <0.05 excellent)\n")
                        f.write(f"  ΔE00: {metrics.get('de00', 'N/A')} (lower is better, <1.0 imperceptible, 1-2 barely noticeable)\n")
                        f.write(f"  Evaluation Time: {metrics.get('evaluation_time', 'N/A')}s\n")
                
                f.write("\nImages Saved:\n")
                f.write("  - 00_input.png (if provided)\n")
                f.write("  - 01_original.png\n")
                f.write("  - 02_target.png\n")
                if conditional_images:
                    f.write(f"  - 03_conditional_XX.png ({len(conditional_images)} files)\n")
                f.write("  - metadata.json\n")
            
            logger.info(f"💾 Evaluation visualization saved to: {request_dir}")
        except Exception as exc:
            logger.warning(f"⚠️ Failed to save evaluation visualization: {exc}", exc_info=True)


def create_visualization_service(
    output_dir: Optional[str] = None,
    enabled: bool = True,
) -> VisualizationService:
    """
    Create a visualization service instance.

    Args:
        output_dir: Directory path for visualization output. Defaults to 'visualizations' in server root.
        enabled: Whether visualization is enabled

    Returns:
        VisualizationService instance
    """
    from ..core.config import BASE_DIR

    if output_dir is None:
        output_dir = BASE_DIR / "visualizations"
    else:
        output_dir = Path(output_dir)

    return VisualizationService(output_dir=output_dir, enabled=enabled)

