# Movie Recommendation System

A comprehensive movie recommendation system with automatic model evaluation, deployment, and monitoring capabilities.

## Prerequisites

- Python 3.8+
- Google Cloud Storage account (for model storage and logging)
- Required Python packages (install using `pip install -r requirements.txt`)

## Environment Setup

1. Set up your environment variables:
```bash
# Option 1: Using .env file (recommended)
# Make sure your .env file exists in the project root with the following variables:
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

# Option 2: Using environment variables directly
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
export GCS_BUCKET_NAME="your-bucket-name"
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Python path:
```bash
# Add the project root directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

4. Create required directories:
```bash
# Create static directory for serving static files
mkdir -p static

# Create templates directory if it doesn't exist
mkdir -p templates
```

## Starting the System

The system consists of two main components that need to be running:

### 1. Monitoring Dashboard

Start the monitoring dashboard:
```bash
# Make sure you're in the project root directory
cd /path/to/be-movie-recomendation

# Verify .env file exists and contains required variables
cat .env

python scripts/monitoring_dashboard.py
```
The dashboard will be available at `http://localhost:8001`

### 2. Model Evaluation Scheduler

Start the evaluation scheduler:
```bash
# Make sure you're in the project root directory
cd /path/to/be-movie-recomendation
python scripts/schedule_evaluation.py
```
This will:
- Run an initial model evaluation
- Schedule daily evaluations at 2 AM
- Automatically deploy the best performing model

## System Components

### Monitoring Dashboard
- Access at `http://localhost:8001`
- View overall metrics and experiment-specific details
- Monitor hit rates and interaction types over time
- Compare performance of different models

### Metrics Structure
The system tracks the following metrics:

1. **Overall Metrics**:
   - Total recommendations
   - Total interactions
   - Hit rate (interactions / recommendations)

2. **Experiment-specific Metrics**:
   - Hit rate per experiment
   - Interaction types distribution
   - Time-based metrics

3. **Accessing Metrics**:
```python
# Get overall summary
summary = metrics.get_summary()
# Returns: {
#   "total_recommendations": int,
#   "total_interactions": int,
#   "hit_rate": float
# }

# Get experiment summary
experiment_summary = metrics.get_experiment_summary("experiment_id")
# Returns: {
#   "total_recommendations": int,
#   "total_interactions": int,
#   "hit_rate": float,
#   "interaction_types": dict
# }

# Get filtered interactions
interactions = metrics.get_interactions(
    user_id=123,
    movie_id=456,
    experiment_id="experiment_id",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)
```

### Model Evaluation
- Runs daily at 2 AM
- Evaluates models based on hit rate metrics
- Automatically deploys the best performing model
- Logs evaluation results to Google Cloud Storage

### Tracking User Interactions

To track user interactions and monitor hit rates, you need to implement the following in your application:

1. **Track Recommendations**:
```python
# When serving recommendations to users
recommendations = await get_recommendations(
    query="action movies",
    experiment_id="new_algorithm_v1"  # Unique ID for your experiment
)
```

2. **Track User Interactions**:
```python
# Track when a user clicks on a recommendation
await track_interaction(
    user_id=123,
    movie_id=456,
    recommendation_id=recommendations[0]["recommendation_id"],
    interaction_type="click",  # Can be: click, view, like, dislike
    experiment_id="new_algorithm_v1"
)

# Track when a user views a recommendation
await track_interaction(
    user_id=123,
    movie_id=456,
    recommendation_id=recommendations[0]["recommendation_id"],
    interaction_type="view",
    experiment_id="new_algorithm_v1"
)
```

3. **Monitor Hit Rates**:
- Access the monitoring dashboard at `http://localhost:8001`
- View overall hit rates for each experiment
- Analyze interaction patterns
- Compare performance across different models

4. **Interaction Types**:
- `click`: User clicked on a recommendation
- `view`: User viewed a recommendation
- `like`: User liked a recommendation
- `dislike`: User disliked a recommendation

5. **Best Practices**:
- Use consistent `experiment_id` for each A/B test
- Track all relevant user interactions
- Monitor hit rates regularly through the dashboard
- Compare performance across different time periods

## Logging and Storage

- Evaluation logs are stored in Google Cloud Storage under the `logs/` directory
- Model versions are tracked in `model_version.json`
- Interaction metrics are stored in the metrics database
- User interaction data is stored and used for hit rate calculations

## Troubleshooting

If you encounter issues:
1. Check the logs in Google Cloud Storage
2. Verify your environment variables:
   - Check that `.env` file exists and contains all required variables
   - Verify the values in `.env` are correct
   - Make sure the `.env` file is in the project root directory
3. Verify Google Cloud Storage bucket:
   - Ensure `GCS_BUCKET_NAME` in `.env` is set to a valid bucket name
   - Verify the bucket exists and is accessible
   - Check your Google Cloud project has access to the bucket
4. Check that both the dashboard and scheduler are running
5. Verify that user interactions are being tracked correctly
6. Check the experiment IDs match between recommendations and interactions
7. If you see "ModuleNotFoundError: No module named 'models'":
   - Make sure you're in the project root directory
   - Verify that PYTHONPATH is set correctly
   - Try running the scripts from the project root directory
8. If you see "Directory 'static' does not exist":
   - Create the required directories using the commands in the Environment Setup section
   - Make sure you're in the project root directory when running the commands
9. If you see "Bucket name must be provided":
   - Check that `GCS_BUCKET_NAME` is set in your `.env` file
   - Verify the bucket name is correct and accessible
   - Try running `gsutil ls gs://$GCS_BUCKET_NAME` to verify bucket access
10. If you see "AttributeError: 'RecommendationMetrics' object has no attribute 'metrics'":
    - Use the correct methods to access metrics: `get_summary()`, `get_experiment_summary()`, or `get_interactions()`
    - Check the Metrics Structure section for the correct way to access metrics data

## Support

For additional support or questions, please contact the development team.
