#!/usr/bin/env python3
"""
Command-line interface for running ArtMancer benchmarks.

Usage:
    python -m app.cli.benchmark_cli --input <path> --prompt <text> [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.benchmark_service import BenchmarkSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_summary(results: list, output_dir: Path):
    """Print benchmark summary."""
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    
    # Filter successful results
    successful = [r for r in results if r.get("success") and r.get("metrics")]
    failed = [r for r in results if not r.get("success") or not r.get("metrics")]
    
    print(f"\nTotal Images: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # Calculate aggregate metrics
        psnr_values = [r["metrics"]["psnr"] for r in successful]
        ssim_values = [r["metrics"]["ssim"] for r in successful]
        lpips_values = [r["metrics"]["lpips"] for r in successful]
        de00_values = [r["metrics"]["de00"] for r in successful]
        
        import statistics
        
        print("\n" + "-"*70)
        print("  METRICS SUMMARY")
        print("-"*70)
        print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
        print("-"*70)
        
        metrics_data = [
            ("PSNR (dB)", psnr_values),
            ("SSIM", ssim_values),
            ("LPIPS", lpips_values),
            ("ΔE00", de00_values),
        ]
        
        for name, values in metrics_data:
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)
            median = statistics.median(values)
            
            print(f"{name:<15} {mean:<12.4f} {std:<12.4f} {min_val:<12.4f} {max_val:<12.4f} {median:<12.4f}")
        
        # Generation time
        gen_times = [r.get("generation_time", 0.0) for r in successful]
        if gen_times:
            total_time = sum(gen_times)
            avg_time = statistics.mean(gen_times)
            print(f"\nGeneration Time:")
            print(f"  Total: {total_time:.2f}s")
            print(f"  Average per image: {avg_time:.2f}s")
    
    if failed:
        print(f"\n⚠️  {len(failed)} images failed:")
        for r in failed[:5]:  # Show first 5 failures
            error = r.get("error") or r.get("metrics_error") or "Unknown error"
            print(f"  - {r.get('filename', 'unknown')}: {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print("\n" + "="*70)
    print(f"Results exported to: {output_dir}")
    print("="*70 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run ArtMancer benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on ZIP file
  python -m app.cli.benchmark_cli --input benchmark_data.zip --prompt "remove the object"
  
  # Run with custom parameters
  python -m app.cli.benchmark_cli --input ./data/ --prompt "remove object" \\
      --samples 10 --steps 50 --quality high --output ./results
  
  # Run with all options
  python -m app.cli.benchmark_cli --input data.zip --prompt "remove" \\
      --task object-removal --samples 0 --steps 40 \\
      --guidance-scale 1.0 --cfg-scale 4.0 \\
      --negative-prompt "blurry, low quality" --seed 42 \\
      --quality high --output ./benchmark_results
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to benchmark folder or ZIP file (must contain input/, mask/, groundtruth/)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Prompt for all images (required)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="object-removal",
        choices=["object-removal", "object-insert", "white-balance"],
        help="Task type (default: object-removal)"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=0,
        help="Number of samples to process (0 = all, default: 0)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps (default: 40)"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale (default: 1.0)"
    )
    
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale (default: 4.0)"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt (optional)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["super_low", "low", "medium", "high", "original"],
        help="Input quality preset (default: high)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./benchmark_results",
        help="Output directory for results (default: ./benchmark_results)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Validate task
    if args.task != "object-removal":
        logger.error(f"❌ Task '{args.task}' is not yet implemented. Only 'object-removal' is supported.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize benchmark system
        logger.info(f"📂 Initializing benchmark system with: {input_path}")
        
        # Initialize benchmark system
        benchmark = BenchmarkSystem(input_path)
        
        # Validate dataset first
        logger.info("🔍 Validating dataset structure...")
        
        # For ZIP files, we need to extract first to validate
        if input_path.suffix == ".zip":
            import tempfile
            import zipfile
            temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_validate_"))
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            validation = benchmark.validate_benchmark_folder(temp_dir)
            # Cleanup temp dir after validation
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            validation = benchmark.validate_benchmark_folder(input_path)
        
        if not validation["success"]:
            logger.error(f"❌ Validation failed: {validation['message']}")
            sys.exit(1)
        
        logger.info(f"✅ Validation passed: {validation['image_count']} image sets detected")
        
        # Load and validate dataset
        logger.info(f"📥 Loading dataset (sample_count={args.samples})...")
        
        # Handle sample_count validation
        sample_count = args.samples
        if sample_count > 0 and sample_count > validation["image_count"]:
            logger.warning(f"⚠️  Requested {sample_count} samples but only {validation['image_count']} available. Using all.")
            sample_count = validation["image_count"]
        
        try:
            load_result = benchmark.load_and_validate(sample_count=sample_count)
        except Exception as e:
            # Handle HTTPException from benchmark service
            error_msg = str(e)
            # Check if it's an HTTPException (has detail attribute)
            if hasattr(e, 'detail'):
                error_msg = getattr(e, 'detail', str(e))
            elif "detail" in str(e):
                # Try to extract from string representation
                import re
                match = re.search(r"detail[=:]\s*['\"]([^'\"]+)['\"]", str(e))
                if match:
                    error_msg = match.group(1)
            logger.error(f"❌ Failed to load dataset: {error_msg}")
            sys.exit(1)
        logger.info(f"✅ Loaded {load_result['total_pairs']} image pairs")
        
        # Prepare generation parameters
        generation_kwargs = {
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "true_cfg_scale": args.cfg_scale,
            "input_quality": args.quality,
        }
        
        if args.negative_prompt:
            generation_kwargs["negative_prompt"] = args.negative_prompt
        
        if args.seed is not None:
            generation_kwargs["seed"] = args.seed
        
        # Generate images
        logger.info(f"🎨 Generating images with prompt: '{args.prompt}'...")
        generated_results = benchmark.generate_images(
            task_type=args.task,
            prompt=args.prompt,
            **generation_kwargs
        )
        
        successful_generations = sum(1 for r in generated_results if r.get("success"))
        logger.info(f"✅ Generated {successful_generations}/{len(generated_results)} images")
        
        # Calculate metrics
        logger.info("📊 Calculating metrics...")
        results = benchmark.calculate_metrics(generated_results)
        
        successful_metrics = sum(1 for r in results if r.get("metrics"))
        logger.info(f"✅ Calculated metrics for {successful_metrics}/{len(results)} images")
        
        # Export results
        logger.info(f"💾 Exporting results to {output_dir}...")
        exported_files = benchmark.export_results(
            output_dir=output_dir,
            formats=["csv", "json", "latex"]
        )
        
        logger.info("✅ Results exported:")
        for format_type, file_path in exported_files.items():
            logger.info(f"  - {format_type.upper()}: {file_path}")
        
        # Print summary
        print_summary(results, output_dir)
        
        # Save summary to file
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Benchmark Summary\n")
            f.write(f"{'='*70}\n")
            f.write(f"Input: {input_path}\n")
            f.write(f"Task: {args.task}\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Total Images: {len(results)}\n")
            f.write(f"Successful: {sum(1 for r in results if r.get('success') and r.get('metrics'))}\n")
            f.write(f"Failed: {sum(1 for r in results if not r.get('success') or not r.get('metrics'))}\n")
            f.write(f"\nResults exported to: {output_dir}\n")
        
        logger.info(f"✅ Summary saved to: {summary_file}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

