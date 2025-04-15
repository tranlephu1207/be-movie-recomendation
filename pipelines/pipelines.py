from kfp import dsl, compiler
from kfp.dsl import component
from google.cloud import aiplatform

PROJECT_ID = "hallowed-hold-454809-q7"
LOCATION = "asia-southeast1"
BUCKET_URI = "gs://namtua-movie-data"
PIPELINE_ROOT_PATH = f"{BUCKET_URI}/pipelines"
PIPELINE_NAME = "movie_pipeline.yaml"


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage", "pandas", "numpy"],
)
def preprocess_data(
    bucket: str,
    credits_path: str,
    keywords_path: str,
    links_small_path: str,
    metadata_path: str,
    ratings_small_path: str,
):
    import pandas as pd
    import numpy as np

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket)

    def download_csv(blob_path, output_path):
        blob = bucket.blob(blob_path)
        blob.download_to_filename(output_path)

    # Download each file
    download_csv(credits_path, "credits.csv")
    download_csv(keywords_path, "keywords.csv")
    download_csv(links_small_path, "links_small.csv")
    download_csv(metadata_path, "movies_metadata.csv")
    download_csv(ratings_small_path, "ratings_small.csv")

    credits_df = pd.read_csv("credits.csv")
    keywords_df = pd.read_csv("keywords.csv")
    links_small_df = pd.read_csv("links_small.csv")
    metadata_df = pd.read_csv("movies_metadata.csv")
    ratings_small_df = pd.read_csv("ratings_small.csv")
    links_small_df = links_small_df[links_small_df["tmdbId"].notnull()][
        "tmdbId"
    ].astype("int")

    metadata_df["id"] = pd.to_numeric(
        metadata_df["id"], errors="coerce", downcast="integer"
    )
    metadata_df = metadata_df.dropna(subset=["id"]).astype({"id": "int"})

    # chuyển cột sang định dạng số (interger), nếu dòng đó không chuyển được sang dạng số, chuyển thành NaN

    # chỉ lấy tập con của metadata
    metadata_df = metadata_df[metadata_df["id"].isin(links_small_df)]

    keywords_df["id"] = keywords_df["id"].astype("int")
    credits_df["id"] = credits_df["id"].astype("int")
    metadata_df["id"] = metadata_df["id"].astype("int")

    metadata_df = metadata_df.merge(credits_df, on="id")
    metadata_df = metadata_df.merge(keywords_df, on="id")

    metadata_small = metadata_df[
        [
            "genres",
            "id",
            "overview",
            "popularity",
            "production_companies",
            "spoken_languages",
            "title",
            "vote_average",
            "vote_count",
            "cast",
            "crew",
            "keywords",
        ]
    ]

    numeric_cols = metadata_small.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = metadata_small.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Biến số:", numeric_cols)
    print("Biến phân loại:", categorical_cols)
    metadata_small["popularity"] = metadata_small["popularity"].astype("float")

    features = [
        "crew",
        "cast",
        "keywords",
        "genres",
        "production_companies",
        "spoken_languages",
    ]
    for feature in features:
        metadata_small[feature] = metadata_small[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i["job"] == "Director":
                return i["name"]
        return np.nan

    def get_list(
        x,
    ):  # return list feature chứa tên diễn viên, key words của bộ phim, đội ngũ làm phim, thể loại phim,...
        if isinstance(x, list):  # Kiểm tra xem x có phải là một danh sách không
            names = [i["name"] for i in x]
            if len(names) > 5:
                names = names[:5]
            return names

        return []

    metadata_small["director"] = metadata_small["crew"].apply(get_director)

    features = [
        "cast",
        "keywords",
        "genres",
        "production_companies",
        "spoken_languages",
    ]
    for feature in features:
        metadata_small[feature] = metadata_small[feature].apply(get_list)

    metadata_small = metadata_small.drop(columns="crew")
    metadata_small["rating"] = (
        metadata_small["vote_average"] * metadata_small["popularity"]
    ) / metadata_small["popularity"].mean()
    metadata_small["rating_class"] = pd.qcut(
        metadata_small["rating"], q=[0, 0.3, 0.6, 1.0], labels=["low", "medium", "high"]
    )

    ratings_small_df = ratings_small_df.drop(columns=["timestamp"])


@dsl.pipeline(
    name="movie-recommendation-svd-pipeline",
    pipeline_root=PIPELINE_ROOT_PATH,
)
def movie_pipeline():
    preprocess_task = preprocess_data(
        bucket="namtua-movie-data",
        metadata_path="data/recommendation_data/movies_metadata.csv",
        links_small_path="data/recommendation_data/links_small.csv",
        credits_path="data/recommendation_data/credits.csv",
        keywords_path="data/recommendation_data/keywords.csv",
        ratings_small_path="data/recommendation_data/ratings_small.csv",
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
