from pymilvus.bulk_writer.volume_manager import VolumeManager

PROJECT_ID = "_id_for_project_"
REGION_ID = "_id_for_region_"
VOLUME_NAME = "_volume_name_for_project_"

if __name__ == "__main__":
    volume_manager = VolumeManager(
        cloud_endpoint="https://api.cloud.zilliz.com",
        api_key="_api_key_for_cluster_org_",
    )

    volume_manager.create_volume(PROJECT_ID, REGION_ID, VOLUME_NAME)
    print(f"\nVolume {VOLUME_NAME} created")

    volume_list = volume_manager.list_volumes(PROJECT_ID, 1, 10)
    print(f"\nlistVolumes results: ", volume_list.json()['data'])

    volume_manager.delete_volume(VOLUME_NAME)
    print(f"\nVolume {VOLUME_NAME} deleted")
