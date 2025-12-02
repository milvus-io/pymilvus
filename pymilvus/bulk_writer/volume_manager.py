import logging

from pymilvus.bulk_writer.volume_restful import create_volume, delete_volume, list_volumes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolumeManager:
    def __init__(self, cloud_endpoint: str, api_key: str):
        """
        Args:
            cloud_endpoint (str): The fixed cloud endpoint URL.
                - For international regions: https://api.cloud.zilliz.com
                - For regions in China: https://api.cloud.zilliz.com.cn
            api_key (str): The API key associated with your organization or cluster.
        """
        self.cloud_endpoint = cloud_endpoint
        self.api_key = api_key

    def create_volume(self, project_id: str, region_id: str, volume_name: str):
        """
        Create a volume under the specified project and regionId.
        """
        create_volume(self.cloud_endpoint, self.api_key, project_id, region_id, volume_name)

    def delete_volume(self, volume_name: str):
        """
        Delete a volume.
        """
        delete_volume(self.cloud_endpoint, self.api_key, volume_name)

    def list_volumes(self, project_id: str, current_page: int = 1, page_size: int = 10):
        """
        Paginated query of the volume list under a specified projectId.
        """
        return list_volumes(self.cloud_endpoint, self.api_key, project_id, current_page, page_size)
