"""
Job cleanup service for managing expired job states in Modal Dict.

Provides utilities for cleaning up old job states based on expiration rules:
- Completed jobs: Delete after 1 hour
- Error jobs: Delete after 1 hour
- Queued/Processing jobs: Delete after 24 hours
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import modal


def cleanup_expired_jobs(job_state_dictio: "modal.Dict") -> int:
    """
    Cleanup expired job states from Modal Dict.
    
    Args:
        job_state_dictio: Modal Dict instance containing job states
    
    Returns:
        Number of deleted jobs
    
    Note:
        Expiration rules:
        - Completed jobs: Delete after 1 hour
        - Error jobs: Delete after 1 hour
        - Queued/Processing jobs: Delete after 24 hours
    """
    tracking_key = "__job_tracking_list__"
    current_time = time.time()
    deleted_count = 0
    
    # Get tracking list (list of job IDs)
    try:
        job_ids = job_state_dictio.get(tracking_key, [])
    except Exception:
        job_ids = []
    
    # Cleanup expired jobs
    valid_job_ids: list[str] = []
    for job_id in job_ids:
        try:
            job_state = job_state_dictio.get(job_id)
            if job_state is None:
                # Job already deleted, skip
                continue
            
            created_at = job_state.get("created_at", current_time)
            status = job_state.get("status", "unknown")
            age_seconds = current_time - created_at
            
            # Determine if job should be deleted based on expiration rules
            should_delete = _should_delete_job(status, age_seconds)
            
            if should_delete:
                try:
                    del job_state_dictio[job_id]
                    deleted_count += 1
                    print(
                        f"🧹 [JobManager] Deleted expired job {job_id} "
                        f"(status={status}, age={age_seconds:.0f}s)"
                    )
                except Exception:
                    pass  # Job might have been deleted already
            else:
                valid_job_ids.append(job_id)
        except Exception as e:
            # Skip invalid job IDs
            print(f"⚠️ [JobManager] Error checking job {job_id}: {e}")
            continue
    
    # Update tracking list with valid job IDs
    if deleted_count > 0:
        job_state_dictio[tracking_key] = valid_job_ids
        print(
            f"✅ [JobManager] Cleanup completed: deleted {deleted_count} expired jobs"
        )
    
    return deleted_count


def _should_delete_job(status: str, age_seconds: float) -> bool:
    """
    Determine if a job should be deleted based on status and age.
    
    Args:
        status: Job status string
        age_seconds: Age of job in seconds
    
    Returns:
        True if job should be deleted, False otherwise
    
    Note:
        Expiration rules:
        - Completed jobs: Delete after 1 hour (3600 seconds)
        - Error jobs: Delete after 1 hour (3600 seconds)
        - Queued/Processing jobs: Delete after 24 hours (86400 seconds)
    """
    # Completed or error jobs: delete after 1 hour
    if status in ("done", "error"):
        return age_seconds > 3600
    
    # Queued or processing jobs: delete after 24 hours
    if status in ("queued", "processing", "initializing_h100"):
        return age_seconds > 86400
    
    # Unknown status: don't delete (conservative)
    return False
