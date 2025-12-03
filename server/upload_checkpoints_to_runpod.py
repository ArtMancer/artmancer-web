#!/usr/bin/env python3
"""
Script to upload checkpoint safetensors files to RunPod Network Volume via S3-compatible API.

IMPORTANT: This requires a separate S3 API key (NOT your regular RunPod API key).
1. Go to RunPod Console → Settings → S3 API Keys → Create an S3 API key
2. Save the access key (user_***) and secret (rps_***)
3. The access key is your User ID (found in the key description)

RunPod provides an S3-compatible API for network volumes in these datacenters:
- EUR-IS-1, EU-RO-1, EU-CZ-1, US-KS-2, US-CA-2

When attached to a Serverless endpoint, network volumes are mounted at /runpod-volume.
Files uploaded to s3://bucket/checkpoints/ will be accessible at /runpod-volume/checkpoints/

References:
- https://docs.runpod.io/storage/s3-api
- https://docs.runpod.io/storage/network-volumes

Usage:
    python upload_checkpoints_to_runpod.py [BUCKET_NAME] [USER_ID] [S3_SECRET]
    
    Or set environment variables in .env file:
    RUNPOD_S3_USER_ID=user_XXXXX
    RUNPOD_S3_SECRET=rps_XXXXX
    RUNPOD_S3_BUCKET=wjda09uwzl
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except ImportError:
    print("❌ Error: boto3 library is not installed")
    print("   Install it with: pip install boto3")
    print("   Or: uv pip install boto3")
    sys.exit(1)

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    # Fallback if tqdm is not available - create a dummy context manager
    class tqdm:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
        
        def update(self, n):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

# Load environment variables from .env file if exists
def load_env_file(env_path: Path) -> None:
    """Load environment variables from .env file."""
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)


def upload_file_to_runpod_s3(
    file_path: Path,
    bucket_name: str,
    s3_key: str,
    user_id: str,
    s3_secret: str,
    region: str = "eu-ro-1",
    endpoint_url: str = "https://s3api-eu-ro-1.runpod.io",
) -> bool:
    """
    Upload a file to RunPod S3 using boto3 with proper AWS Signature v4 authentication.
    
    Args:
        file_path: Local file path to upload
        bucket_name: S3 bucket name (Network Volume ID)
        s3_key: S3 object key (path in bucket)
        user_id: RunPod User ID (from S3 API key description, e.g., user_XXXXX)
        s3_secret: RunPod S3 API key secret (e.g., rps_XXXXX)
        region: AWS region (datacenter)
        endpoint_url: S3 endpoint URL
    
    Returns:
        True if upload successful, False otherwise
    """
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    try:
        # Configure boto3 S3 client for RunPod S3-compatible API
        # Note: RunPod S3 may require specific authentication format
        # Try different configurations if this doesn't work
        s3_config = Config(
            signature_version='s3v4',
            s3={
                'addressing_style': 'path',
                'payload_signing_enabled': False,  # Disable payload signing for compatibility
            },
            # Disable SSL verification if needed (not recommended for production)
            # retries={'max_attempts': 3}
        )
        
        # Use RunPod S3 API credentials
        # Access Key ID = User ID (e.g., user_XXXXX)
        # Secret Access Key = S3 API key secret (e.g., rps_XXXXX)
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=user_id,
            aws_secret_access_key=s3_secret,
            region_name=region,
            config=s3_config,
        )
        
        # Test connection first with a simple operation
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                print(f"\n   ⚠️  Bucket '{bucket_name}' not found.")
                print("   Make sure the bucket name matches your network volume ID exactly.")
            elif error_code == '403':
                print("\n   ⚠️  Access denied. Check your API key and bucket permissions.")
            # Continue anyway - bucket might exist but head_bucket failed
        
        # Upload file with progress tracking
        print(f"   Uploading {file_size_mb:.1f}MB...", end="", flush=True)
        
        # Use boto3's upload_file which handles multipart upload automatically
        s3_client.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            Callback=lambda bytes_transferred: print(".", end="", flush=True)
        )
        
        print(" ✅")
        return True
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"\n   ❌ AWS Error ({error_code}): {error_message}")
        
        if error_code == 'SignatureDoesNotMatch':
            print("\n   💡 Troubleshooting:")
            print("   1. Verify you're using S3 API key (NOT regular RunPod API key)")
            print("   2. Check User ID matches the one in S3 API key description")
            print("   3. Verify bucket name matches your Network Volume ID exactly")
            print("   4. Ensure network volume is in a datacenter with S3 API support")
            print("   5. Check RunPod docs: https://docs.runpod.io/storage/s3-api#setup-and-authentication")
        
        return False
    except Exception as e:
        print(f"\n   ❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main function to upload checkpoint files."""
    # Load .env file if exists
    env_path = Path(__file__).parent / ".env"
    load_env_file(env_path)
    
    # Configuration
    # RunPod S3-compatible API endpoints by datacenter:
    # EUR-IS-1: https://s3api-eur-is-1.runpod.io
    # EU-RO-1:  https://s3api-eu-ro-1.runpod.io (default)
    # EU-CZ-1:  https://s3api-eu-cz-1.runpod.io
    # US-KS-2:  https://s3api-us-ks-2.runpod.io
    # US-CA-2:  https://s3api-us-ca-2.runpod.io
    region = os.getenv("RUNPOD_S3_REGION", "eu-ro-1")
    endpoint_url = os.getenv("RUNPOD_S3_ENDPOINT", f"https://s3api-{region}.runpod.io")
    checkpoints_dir = Path("./checkpoints")
    
    # Parse arguments - S3 credentials từ environment variable hoặc command line
    bucket_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RUNPOD_S3_BUCKET", "wjda09uwzl")
    user_id = os.getenv("RUNPOD_S3_USER_ID") or (sys.argv[2] if len(sys.argv) > 2 else None)
    s3_secret = os.getenv("RUNPOD_S3_SECRET") or (sys.argv[3] if len(sys.argv) > 3 else None)
    
    # Files to upload
    files = [
        "insertion_cp.safetensors",
        "removal_cp.safetensors",
        "wb_cp.safetensors",
    ]
    
    print("🚀 Uploading checkpoints to RunPod S3...")
    print(f"   Bucket: s3://{bucket_name}/")
    print(f"   Region: {region}")
    print(f"   Endpoint: {endpoint_url}")
    print("")
    print("ℹ️  Important notes:")
    print("   - Bucket name must be your Network Volume ID (found in RunPod Console → Storage)")
    print("   - S3 API is only available for: EUR-IS-1, EU-RO-1, EU-CZ-1, US-KS-2, US-CA-2")
    print("   - If authentication fails, try uploading via RunPod Console web interface")
    print("")
    
    # Check if S3 credentials are provided
    if not user_id or not s3_secret:
        print("❌ Error: RunPod S3 API credentials are required")
        print("")
        print("   IMPORTANT: You need a separate S3 API key (NOT your regular RunPod API key)")
        print("")
        print("   Step 1: Create S3 API key in RunPod Console:")
        print("     1. Go to Settings → S3 API Keys → Create an S3 API key")
        print("     2. Save the access key (user_XXXXX) and secret (rps_XXXXX)")
        print("     3. The access key is your User ID (found in key description)")
        print("")
        print("   Step 2: Configure credentials:")
        print("")
        print("   Option 1: Create .env file (recommended):")
        print("     echo 'RUNPOD_S3_USER_ID=user_XXXXX' >> server/.env")
        print("     echo 'RUNPOD_S3_SECRET=rps_XXXXX' >> server/.env")
        print("     echo 'RUNPOD_S3_BUCKET=wjda09uwzl' >> server/.env  # Network Volume ID")
        print("")
        print("   Option 2: Set environment variables:")
        print("     export RUNPOD_S3_USER_ID='user_XXXXX'")
        print("     export RUNPOD_S3_SECRET='rps_XXXXX'")
        print("")
        print("   Option 3: Pass as command line arguments:")
        print("     python upload_checkpoints_to_runpod.py BUCKET_ID user_XXXXX rps_XXXXX")
        print("")
        print("   Reference: https://docs.runpod.io/storage/s3-api#setup-and-authentication")
        sys.exit(1)
    
    # Check if checkpoints directory exists
    if not checkpoints_dir.exists():
        print(f"❌ Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    
    # Check if files exist
    missing_files = []
    for file in files:
        file_path = checkpoints_dir / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Warning: Some files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("")
        response = input("Continue with available files? (y/n) ")
        if response.lower() != "y":
            sys.exit(1)
    
    # Upload each file
    success_count = 0
    failed_files = []
    
    for file in files:
        file_path = checkpoints_dir / file
        
        if not file_path.exists():
            print(f"⏭️  Skipping {file} (not found)")
            continue
        
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"📤 Uploading {file} ({file_size_mb:.1f}MB)...")
        
        # Upload to checkpoints/ directory in network volume
        # Files will be accessible at /runpod-volume/checkpoints/ on Serverless workers
        s3_key = f"checkpoints/{file}"
        
        if upload_file_to_runpod_s3(
            file_path=file_path,
            bucket_name=bucket_name,
            s3_key=s3_key,
            user_id=user_id,
            s3_secret=s3_secret,
            region=region,
            endpoint_url=endpoint_url,
        ):
            print(f"   ✅ {file} uploaded successfully")
            success_count += 1
        else:
            print(f"   ❌ Failed to upload {file}")
            failed_files.append(file)
        print("")
    
    # Summary
    print("=" * 60)
    print("📊 Upload Summary:")
    print(f"   Success: {success_count}/{len(files)}")
    if failed_files:
        print("   Failed:")
        for file in failed_files:
            print(f"     - {file}")
        sys.exit(1)
    else:
        print("   ✅ All files uploaded successfully!")
        print("")
        print("📋 Next steps:")
        print("   1. Attach network volume to your Serverless endpoint in RunPod console")
        print("   2. Files will be accessible at /runpod-volume/checkpoints/ on workers")
        print("")
        print("📋 Verify upload with:")
        print(f"   aws s3 ls --region {region} --endpoint-url {endpoint_url} s3://{bucket_name}/checkpoints/")
        print("")
        print("   Make sure AWS CLI is configured with S3 API credentials:")
        print("   aws configure")
        print("   # Use User ID as Access Key ID")
        print("   # Use S3 API secret as Secret Access Key")


if __name__ == "__main__":
    main()

