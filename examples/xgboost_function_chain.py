"""End-to-end XGBoost rerank demo for Milvus FunctionChain.

The demo performs the complete workflow:

1. Train an XGBoost regression model and save it in UBJ format.
2. Upload the UBJ file to the MinIO bucket used by Milvus.
3. Register the object as a Milvus FileResource.
4. Create a collection and insert vector plus scalar feature data.
5. Rewrite the search score with ``fn.xgboost`` at the L0 rerank stage.
6. Predict the returned rows locally and assert that every Milvus score is correct.

Requirements:
    pip install pymilvus numpy xgboost minio

The Milvus server must be built with XGBoost FunctionChain support enabled.

The MinIO bucket must be the same bucket configured in Milvus. FileResource
paths are relative to the bucket root; unlike normal Milvus data paths, they do
not include ``minio.rootPath``.

Example:
    python examples/xgboost_function_chain.py \
        --milvus-uri http://localhost:19530 \
        --minio-endpoint localhost:9000 \
        --minio-bucket a-bucket
"""

import argparse
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xgboost as xgb
from minio import Minio

from pymilvus import DataType, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.function_chain import col, fn

DIM = 8
NUM_FEATURES = 3
NUM_ROWS = 20
FEATURE_FIELDS = ("price", "ctr", "freshness")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--token", default=None, help="Milvus token, for example root:Milvus")
    parser.add_argument("--minio-endpoint", default="localhost:9000")
    parser.add_argument("--minio-access-key", default="minioadmin")
    parser.add_argument("--minio-secret-key", default="minioadmin")
    parser.add_argument("--minio-bucket", default="a-bucket")
    parser.add_argument("--minio-secure", action="store_true")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the collection, FileResource, and MinIO object after validation",
    )
    return parser.parse_args()


def train_and_save_model(model_path: Path):
    """Train a supported gbtree/reg:squarederror model and save it as UBJ."""
    rng = np.random.default_rng(19530)
    features = rng.uniform(0.0, 1.0, size=(512, NUM_FEATURES)).astype(np.float32)
    labels = (
        1.8 * features[:, 0]
        + 3.2 * features[:, 1]
        + 1.4 * features[:, 2]
        + 0.8 * features[:, 0] * features[:, 1]
    ).astype(np.float32)

    model = xgb.train(
        {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "max_depth": 3,
            "eta": 0.15,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "seed": 19530,
            "nthread": 1,
        },
        xgb.DMatrix(features, label=labels),
        num_boost_round=30,
    )
    model.save_model(model_path)
    return model


def upload_model(minio_client, bucket: str, object_path: str, model_path: Path):
    if not minio_client.bucket_exists(bucket):
        raise RuntimeError(
            f"MinIO bucket {bucket!r} does not exist. Use the bucket configured in Milvus."
        )
    minio_client.fput_object(
        bucket,
        object_path,
        str(model_path),
        content_type="application/octet-stream",
    )


def create_collection(client: MilvusClient, collection_name: str):
    schema = client.create_schema(enable_dynamic_field=False, auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    for field in FEATURE_FIELDS:
        schema.add_field(field, DataType.FLOAT)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )


def generate_rows():
    rng = np.random.default_rng(2026)
    embeddings = rng.normal(size=(NUM_ROWS, DIM)).astype(np.float32)
    features = rng.uniform(0.0, 1.0, size=(NUM_ROWS, NUM_FEATURES)).astype(np.float32)
    return [
        {
            "id": row_id,
            "embedding": embeddings[row_id].tolist(),
            **{
                field: float(features[row_id, feature_index])
                for feature_index, field in enumerate(FEATURE_FIELDS)
            },
        }
        for row_id in range(NUM_ROWS)
    ]


def search_with_xgboost(client, collection_name: str, resource_name: str):
    chain = FunctionChain(FunctionChainStage.L0_RERANK, name="xgboost_rerank").map(
        "$score",
        fn.xgboost(
            *(col(field) for field in FEATURE_FIELDS),
            model_resource=resource_name,
            output="default",
        ),
    )

    # The zero query makes every original IP score zero, so the final ordering
    # is determined only by the XGBoost model output.
    return client.search(
        collection_name=collection_name,
        data=[[0.0] * DIM],
        anns_field="embedding",
        search_params={"metric_type": "IP"},
        limit=NUM_ROWS,
        output_fields=list(FEATURE_FIELDS),
        function_chains=chain,
    )[0]


def validate_scores(model, hits):
    returned_ids = {hit["id"] for hit in hits}
    expected_ids = set(range(NUM_ROWS))
    if returned_ids != expected_ids:
        message = f"Returned ids differ from inserted ids: {sorted(returned_ids)}"
        raise AssertionError(message)

    returned_features = np.asarray(
        [[hit["entity"][field] for field in FEATURE_FIELDS] for hit in hits],
        dtype=np.float32,
    )
    expected_scores = model.predict(xgb.DMatrix(returned_features))
    actual_scores = np.asarray([hit["distance"] for hit in hits], dtype=np.float32)

    score_matches = np.isclose(actual_scores, expected_scores, rtol=1e-5, atol=1e-5)
    max_abs_error = float(np.max(np.abs(actual_scores - expected_scores)))
    if not np.all(score_matches):
        mismatch_indexes = np.flatnonzero(~score_matches).tolist()
        message = (
            f"Score validation failed: mismatch indexes={mismatch_indexes}, "
            f"max_abs_error={max_abs_error:.8g}"
        )
        raise AssertionError(message)

    if np.any(actual_scores[:-1] < actual_scores[1:] - 1e-5):
        message = f"XGBoost scores are not sorted descending: {actual_scores}"
        raise AssertionError(message)

    print(" id | milvus_score | local_xgboost_score")
    print("----+--------------+--------------------")
    for hit, actual, expected in zip(hits, actual_scores, expected_scores):
        print(f"{hit['id']:3d} | {actual:12.7f} | {expected:18.7f}")
    print(f"\nMaximum absolute score error: {max_abs_error:.8g}")
    return True


def remove_file_resource_with_retry(client, resource_name: str):
    # Dropping a collection and releasing its FileResource reference are
    # asynchronous on the server, so cleanup may need a short retry.
    last_error = None
    for _ in range(10):
        try:
            client.remove_file_resource(resource_name)
            return
        except Exception as exc:  # noqa: BLE001 - preserve the server error for the final retry
            last_error = exc
            time.sleep(1)
    raise last_error


def main():
    args = parse_args()
    suffix = uuid.uuid4().hex[:8]
    collection_name = f"xgboost_e2e_{suffix}"
    resource_name = f"xgboost_model_{suffix}"
    object_path = f"xgboost_demo/{resource_name}/model.ubj"

    client_kwargs = {"uri": args.milvus_uri}
    if args.token:
        client_kwargs["token"] = args.token
    client = MilvusClient(**client_kwargs)
    minio_client = Minio(
        args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=args.minio_secure,
    )

    model_uploaded = False
    resource_registered = False
    collection_created = False

    with TemporaryDirectory(prefix="milvus_xgboost_") as temp_dir:
        model_path = Path(temp_dir) / "model.ubj"
        try:
            print("1. Training XGBoost model and writing UBJ file")
            model = train_and_save_model(model_path)

            print(f"2. Uploading {model_path.name} to s3://{args.minio_bucket}/{object_path}")
            upload_model(minio_client, args.minio_bucket, object_path, model_path)
            model_uploaded = True

            print(f"3. Registering Milvus FileResource {resource_name!r}")
            client.add_file_resource(name=resource_name, path=object_path)
            resource_registered = True

            print(f"4. Creating collection {collection_name!r} and inserting {NUM_ROWS} rows")
            create_collection(client, collection_name)
            collection_created = True
            client.insert(collection_name=collection_name, data=generate_rows())
            client.load_collection(collection_name)

            print("5. Searching with XGBoost L0 rerank")
            hits = search_with_xgboost(client, collection_name, resource_name)
            if len(hits) != NUM_ROWS:
                raise AssertionError(f"Expected {NUM_ROWS} hits, got {len(hits)}")

            print("6. Comparing Milvus scores with local XGBoost predictions")
            score_is_expected = validate_scores(model, hits)
            if not score_is_expected:
                message = "SCORE VALIDATION: FAILED"
                raise AssertionError(message)
            print("\nSCORE VALIDATION: PASSED")
        finally:
            if args.keep_artifacts:
                print(
                    f"\nArtifacts kept: collection={collection_name}, "
                    f"resource={resource_name}, object={object_path}"
                )
            else:
                if collection_created:
                    client.drop_collection(collection_name)
                if resource_registered:
                    remove_file_resource_with_retry(client, resource_name)
                if model_uploaded:
                    minio_client.remove_object(args.minio_bucket, object_path)


if __name__ == "__main__":
    main()
