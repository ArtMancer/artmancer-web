"""
Job cleanup service for managing expired job states in Modal Dict.
"""
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
    valid_job_ids = []
    for job_id in job_ids:
        try:
            job_state = job_state_dictio.get(job_id)
            if job_state is None:
                # Job already deleted, skip
                continue
            
            created_at = job_state.get("created_at", current_time)
            status = job_state.get("status", "unknown")
            age_seconds = current_time - created_at
            
            # Expiration rules:
            # - Completed jobs: Delete after 1 hour
            # - Error jobs: Delete after 1 hour
            # - Queued/Processing jobs: Delete after 24 hours
            should_delete = False
            if status == "done" and age_seconds > 3600:  # 1 hour
                should_delete = True
            elif status == "error" and age_seconds > 3600:  # 1 hour
                should_delete = True
            elif status in ["queued", "processing", "initializing_a100"] and age_seconds > 86400:  # 24 hours
                should_delete = True
            
            if should_delete:
                try:
                    del job_state_dictio[job_id]
                    deleted_count += 1
                    print(f"🧹 [JobManager] Deleted expired job {job_id} (status={status}, age={age_seconds:.0f}s)")
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
        print(f"✅ [JobManager] Cleanup completed: deleted {deleted_count} expired jobs")
    
    return deleted_count

