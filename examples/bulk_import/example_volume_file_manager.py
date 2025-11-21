from pymilvus.bulk_writer.constants import ConnectType
from pymilvus.bulk_writer.volume_file_manager import VolumeFileManager

if __name__ == "__main__":
    volume_file_manager = VolumeFileManager(
        cloud_endpoint='https://api.cloud.zilliz.com',
        api_key='_api_key_for_cluster_org_',
        volume_name='_volume_name_for_project_',
        connect_type=ConnectType.AUTO,
    )
    result = volume_file_manager.upload_file_to_volume("/Users/zilliz/data/", "data/")
    print(f"\nuploadFileToVolume results: {result}")
