from pymilvus.bulk_writer.stage_file_manager import StageFileManager

if __name__ == "__main__":
    stage_file_manager = StageFileManager(
        cloud_endpoint='https://api.cloud.zilliz.com',
        api_key='_api_key_for_cluster_org_',
        stage_name='_stage_name_for_project_',
    )
    result = stage_file_manager.upload_file_to_stage("/Users/zilliz/data/", "data/")
    print(f"\nuploadFileToStage results: {result}")
