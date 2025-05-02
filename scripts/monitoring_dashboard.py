from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
from models.metrics import RecommendationMetrics
import uvicorn
import os
from google.cloud import storage
from dotenv import load_dotenv

from utils.gcloud import init_credentials

# Load environment variables from .env file
load_dotenv()
# Initialize FastAPI app
app = FastAPI(title="Recommendation System Monitoring Dashboard")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize metrics
metrics = RecommendationMetrics()

# Initialize Google Cloud Storage client
storage_client = storage.Client()

def create_hit_rate_plot(experiment_id: str, days: int = 30) -> str:
    """
    Create a plot of hit rate over time.
    
    Args:
        experiment_id: Experiment identifier
        days: Number of days to plot
        
    Returns:
        HTML string containing the plot
    """
    # Get interactions for the time period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    interactions = metrics.get_interactions(
        experiment_id=experiment_id,
        start_time=start_time,
        end_time=end_time
    )
    
    if not interactions:
        return "No data available for the selected period"
    
    # Convert to DataFrame
    df = pd.DataFrame(interactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by day and calculate hit rate
    daily_stats = df.groupby(df['timestamp'].dt.date).agg({
        'recommendation_id': 'nunique',
        'interaction_type': 'count'
    }).reset_index()
    
    daily_stats['hit_rate'] = daily_stats['interaction_type'] / daily_stats['recommendation_id']
    
    # Create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add hit rate line
    fig.add_trace(
        go.Scatter(
            x=daily_stats['timestamp'],
            y=daily_stats['hit_rate'],
            name="Hit Rate",
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add interactions bar
    fig.add_trace(
        go.Bar(
            x=daily_stats['timestamp'],
            y=daily_stats['interaction_type'],
            name="Interactions",
            opacity=0.3
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"Hit Rate and Interactions for {experiment_id}",
        xaxis_title="Date",
        yaxis_title="Hit Rate",
        yaxis2_title="Number of Interactions",
        template="plotly_white"
    )
    
    return fig.to_html(full_html=False)

def create_interaction_type_pie(experiment_id: str, days: int = 30) -> str:
    """
    Create a pie chart of interaction types.
    
    Args:
        experiment_id: Experiment identifier
        days: Number of days to include
        
    Returns:
        HTML string containing the plot
    """
    # Get interactions for the time period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    interactions = metrics.get_interactions(
        experiment_id=experiment_id,
        start_time=start_time,
        end_time=end_time
    )
    
    if not interactions:
        return "No data available for the selected period"
    
    # Count interaction types
    interaction_counts = {}
    for interaction in interactions:
        itype = interaction['interaction_type']
        interaction_counts[itype] = interaction_counts.get(itype, 0) + 1
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(interaction_counts.keys()),
        values=list(interaction_counts.values()),
        hole=.3
    )])
    
    fig.update_layout(
        title=f"Interaction Types for {experiment_id}",
        template="plotly_white"
    )
    
    return fig.to_html(full_html=False)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Main dashboard page.
    """
    # Get available experiments
    df = metrics._read_interactions()
    experiments = df["experiment_id"].unique().tolist()
    experiments = [exp for exp in experiments if exp]  # Remove empty strings
    
    # Create plots for each experiment
    plots = {}
    for exp_id in experiments:
        plots[exp_id] = {
            "hit_rate": create_hit_rate_plot(exp_id),
            "interaction_types": create_interaction_type_pie(exp_id)
        }
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "experiments": experiments,
            "plots": plots
        }
    )

@app.get("/experiment/{experiment_id}", response_class=HTMLResponse)
async def experiment_dashboard(request: Request, experiment_id: str):
    """
    Detailed dashboard for a specific experiment.
    """
    # Get experiment metrics
    experiment_metrics = metrics.get_experiment_summary(experiment_id)
    if experiment_metrics is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Create plots
    hit_rate_plot = create_hit_rate_plot(experiment_id)
    interaction_plot = create_interaction_type_pie(experiment_id)
    
    return templates.TemplateResponse(
        "experiment.html",
        {
            "request": request,
            "experiment_id": experiment_id,
            "metrics": experiment_metrics["summary"],
            "hit_rate_plot": hit_rate_plot,
            "interaction_plot": interaction_plot
        }
    )

if __name__ == "__main__":
    # Initialize GCS client
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set")
    
    init_credentials(credentials_path)
    # Get bucket name from environment variable
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable must be set")
    
    # Create necessary directories if they don't exist in the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Check if static directory exists, create if not
    static_blob = bucket.blob("static/")
    if not static_blob.exists():
        static_blob.upload_from_string("")
    
    # Check if templates directory exists, create if not
    templates_blob = bucket.blob("templates/")
    if not templates_blob.exists():
        templates_blob.upload_from_string("")
    
    # Run the dashboard
    uvicorn.run(app, host="0.0.0.0", port=8001)