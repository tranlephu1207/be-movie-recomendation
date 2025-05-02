import schedule
import time
import subprocess
import logging
from datetime import datetime
import os
from google.cloud import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_evaluation():
    """
    Run the model evaluation and deployment process.
    """
    try:
        logger.info("Starting model evaluation process")
        
        # Run the evaluation script
        result = subprocess.run(
            ['python', 'scripts/evaluate_and_deploy.py'],
            capture_output=True,
            text=True
        )
        
        # Get bucket name from environment variable
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable must be set")
        
        # Initialize storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Create log entry with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_content = f"=== Evaluation Run: {timestamp} ===\n"
        
        if result.returncode == 0:
            logger.info("Model evaluation completed successfully")
            log_content += "STATUS: SUCCESS\n"
            log_content += result.stdout
        else:
            logger.error("Model evaluation failed")
            log_content += "STATUS: FAILED\n"
            log_content += result.stderr
        
        # Upload log to GCS
        log_blob = bucket.blob(f"logs/evaluation_{timestamp}.log")
        log_blob.upload_from_string(log_content)
        logger.info(f"Evaluation log saved to gs://{bucket_name}/logs/evaluation_{timestamp}.log")
            
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        
        # Try to log the error to GCS if possible
        try:
            if 'bucket' in locals() and bucket:
                error_blob = bucket.blob(f"logs/evaluation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                error_blob.upload_from_string(f"ERROR: {str(e)}")
        except Exception:
            pass

def main():
    """
    Main function to schedule and run evaluations.
    """
    # Schedule daily evaluation at 2 AM
    schedule.every().day.at("02:00").do(run_evaluation)
    
    logger.info("Scheduler started. Will run evaluations daily at 2 AM.")
    
    # Run initial evaluation
    run_evaluation()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 