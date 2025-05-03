from kfp import dsl, compiler
from kfp.dsl import component, Input, Dataset
from google.cloud import aiplatform
import os
from datetime import datetime

PROJECT_ID = "hallowed-hold-454809-q7"
LOCATION = "asia-southeast1"
BUCKET_URI = "gs://namtua-movie-data"
PIPELINE_ROOT_PATH = f"{BUCKET_URI}/pipelines"
PIPELINE_NAME = "movie_pipeline.yaml"
MODEL_TAG=os.environ.get("CI_COMMIT_TAG",os.environ.get("CI_COMMIT_SHA",datetime.now().strftime("%Y-%m-%d-%H%M%S")))

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage", "pandas==1.5.3","numpy==1.24.4", "surprise==0.1", "joblib==1.3.2"],
)
def svd_train_and_push_model(
    bucket: str,
    svd_model_upload_path: str,
    ratings_small_path: str,
):
    from surprise import Dataset, Reader, SVD
    import joblib
    import pandas as pd
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket)

    def download_csv(blob_path, output_path):
        blob = bucket.blob(blob_path)
        blob.download_to_filename(output_path)

    def upload_model(gcs_path, model_path):
        # Extract bucket and blob path
        blob = bucket.blob(gcs_path)

        # Upload the file
        blob.upload_from_filename(model_path)
        print(f"Uploaded {model_path} to {gcs_path}")

    # Download each file
    download_csv(ratings_small_path, "ratings_small.csv")

    ratings_small_df = pd.read_csv("ratings_small.csv")
    ratings_small_df = ratings_small_df.drop(columns=["timestamp"])

    new_user_ratings = pd.DataFrame(
        [
            {"userId": 9999, "movieId": 1172, "rating": 5.0},
            {"userId": 9999, "movieId": 1061, "rating": 4.5},
            {"userId": 9999, "movieId": 1029, "rating": 4.0},
        ]
    )

    ratings = pd.concat([ratings_small_df, new_user_ratings], ignore_index=True)

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    trainset = data.build_full_trainset()

    # Huấn luyện SVD model
    svd_model = SVD()
    svd_model.fit(trainset)

    joblib.dump(svd_model, "svd_model.pkl")
    upload_model(svd_model_upload_path, "svd_model.pkl")


@dsl.pipeline(
    name="movie-recommendation-svd-pipeline",
    pipeline_root=PIPELINE_ROOT_PATH,
)
def movie_pipeline():
    svd_train_and_push_model_task = svd_train_and_push_model(
        bucket="namtua-movie-data",
        ratings_small_path="data/recommendation_data/ratings_small.csv",
        svd_model_upload_path=f"models/svd_{MODEL_TAG}.pkl",
    )


compiler.Compiler().compile(pipeline_func=movie_pipeline, package_path=PIPELINE_NAME)
print("compile sucess")


aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
job = aiplatform.PipelineJob(
    display_name="movie_pipeline",
    template_path=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT_PATH,
)
job.submit()
