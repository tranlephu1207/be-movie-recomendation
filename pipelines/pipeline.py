from kfp import dsl, compiler
from kfp.dsl import component, Input, Dataset, Output, Model
from google.cloud import aiplatform
import os
from datetime import datetime


PROJECT_ID = "hallowed-hold-454809-q7"
LOCATION = "asia-southeast1"
BUCKET_URI = "gs://namtua-movie-data"
PIPELINE_ROOT_PATH = f"{BUCKET_URI}/pipelines"
PIPELINE_NAME = "movie_pipeline.yaml"
BUCKET = "namtua-movie-data"
MODEL_TAG = os.environ.get(
    "CI_COMMIT_TAG",
    os.environ.get("CI_COMMIT_SHA", datetime.now().strftime("%Y-%m-%d-%H%M%S")),
)


@component(
    base_image="python:3.10", packages_to_install=["pandas", "google-cloud-storage"]
)
def preprocess_data(
    bucket: str,
    ratings_small_path: str,
    ratings_cleaned: Output[Dataset],
):
    import pandas as pd
    import os
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(ratings_small_path)
    blob.download_to_filename("ratings_small.csv")

    df = pd.read_csv("ratings_small.csv")
    df = df.drop(columns=["timestamp"], errors="ignore")

    new_user_ratings = pd.DataFrame(
        [
            {"userId": 9999, "movieId": 1172, "rating": 5.0},
            {"userId": 9999, "movieId": 1061, "rating": 4.5},
            {"userId": 9999, "movieId": 1029, "rating": 4.0},
        ]
    )

    df = pd.concat([df, new_user_ratings], ignore_index=True)
    df.to_csv(ratings_cleaned.path + ".csv", index=False)


@component(
    base_image="python:3.10",
    packages_to_install=[
        "joblib==1.3.2",
        "pandas==1.5.3",
        "numpy==1.24.4",
        "surprise==0.1",
        "google-cloud-storage",
    ],
)
def train_and_upload_model(
    ratings_cleaned: Input[Dataset], bucket: str, svd_model_upload_path: str
):
    from surprise import Dataset, Reader, SVD
    import pandas as pd
    import joblib
    import os
    from google.cloud import storage

    df = pd.read_csv(ratings_cleaned.path + ".csv")

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd_model = SVD()
    svd_model.fit(trainset)
    joblib.dump(svd_model, "svd_model.pkl")

    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(svd_model_upload_path)
    blob.upload_from_filename("svd_model.pkl")
    print(f"Uploaded svd_model.pkl to gs://{bucket.name}/{svd_model_upload_path}")


## Can implement evaluation to check model metrics


@dsl.pipeline(
    name="movie-recommendation-svd-pipeline",
    pipeline_root=PIPELINE_ROOT_PATH,
)
def movie_pipeline():

    preprocess_task = preprocess_data(
        bucket=BUCKET,
        ratings_small_path="data/recommendation_data/ratings_small.csv",
    )

    train_task = train_and_upload_model(
        ratings_cleaned=preprocess_task.outputs["ratings_cleaned"],
        bucket=BUCKET,
        svd_model_upload_path=f"models/svd_{MODEL_TAG}.pkl",
    )


compiler.Compiler().compile(pipeline_func=movie_pipeline, package_path=PIPELINE_NAME)
print("compile success")

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
job = aiplatform.PipelineJob(
    display_name="movie_pipeline",
    template_path=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT_PATH,
)
job.submit()
