from pymilvus.bulk_writer.stage_manager import StageManager

PROJECT_ID = "_id_for_project_"
REGION_ID = "_id_for_region_"
STAGE_NAME = "_stage_name_for_project_"

if __name__ == "__main__":
    stage_manager = StageManager(
        cloud_endpoint="https://api.cloud.zilliz.com",
        api_key="_api_key_for_cluster_org_",
    )

    stage_manager.create_stage(PROJECT_ID, REGION_ID, STAGE_NAME)
    print(f"\nStage {STAGE_NAME} created")

    stage_list = stage_manager.list_stages(PROJECT_ID, 1, 10)
    print(f"\nlistStages results: ", stage_list.json()['data'])

    stage_manager.delete_stage(STAGE_NAME)
    print(f"\nStage {STAGE_NAME} deleted")
