"""
Scheduler module for periodically recomputing PCA for image embeddings.
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from .pca import compute_and_save_pca, should_recompute_pca
from datetime import datetime

logger = logging.getLogger(__name__)

# Create a scheduler
scheduler = BackgroundScheduler()

def check_and_update_pca():
    """Check if PCA needs to be recomputed and do so if necessary."""
    try:
        logger.info(f"PCA check at {datetime.now().isoformat()}")
        if should_recompute_pca():  # Use new interval-based logic
            logger.info("Recomputing PCA")
            pca_model = compute_and_save_pca()
            if pca_model:
                logger.info(f"PCA recomputed successfully. Model ID: {pca_model['id']}, embeddings: {pca_model['num_embeddings']}")
            else:
                logger.warning("Failed to recompute PCA")
        else:
            logger.info("No need to recompute PCA")
    except Exception as e:
        logger.error(f"Error in PCA update job: {e}")

def start_scheduler():
    """Start the background scheduler for PCA updates."""
    if not scheduler.running:
        # Add job to check and update PCA every hour
        scheduler.add_job(
            check_and_update_pca,
            trigger=IntervalTrigger(hours=1),  # Run every hour
            id='pca_update',
            name='Update PCA model',
            replace_existing=True
        )
        
        # Start the scheduler
        scheduler.start()
        logger.info("PCA update scheduler started")

def stop_scheduler():
    """Stop the background scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("PCA update scheduler stopped")