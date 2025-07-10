import os

from pymilvus.stage.stage_operation import StageOperation

# You need to upload the local file path or folder path for import & migration & spark.
LOCAL_DIR_OR_FILE_PATH = "/Users/zilliz/Desktop/1.parquet"

# The value of the URL is fixed.
# For overseas regions, it is: https://api.cloud.zilliz.com
# For regions in China, it is: https://api.cloud.zilliz.com.cn
CLOUD_ENDPOINT = "https://api.cloud.zilliz.com"
API_KEY = "_api_key_for_cluster_org_"

# This is currently a private preview feature. If you need to use it, please submit a request and contact us.
# Before using this feature, you need to create a stage using the stage API.
STAGE_NAME = "_stage_name_for_project_"
PATH = "_path_for_stage"

def main():
    stage_operation = StageOperation(
        cloud_endpoint=CLOUD_ENDPOINT,
        api_key=API_KEY,
        stage_name=STAGE_NAME,
        path=PATH
    )
    result = stage_operation.upload_file_to_stage(LOCAL_DIR_OR_FILE_PATH)
    print(f"\nuploadFileToStage results: {result}")

if __name__ == "__main__":
    main()