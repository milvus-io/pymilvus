import logging
from typing import Optional

from pymilvus.bulk_writer.volume_restful import (
    create_volume,
    delete_volume,
    describe_volume,
    list_volumes,
)

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

    def create_volume(
        self,
        project_id: str,
        region_id: str,
        volume_name: str,
        volume_type: Optional[str] = None,
        storage_integration_id: Optional[str] = None,
        path: Optional[str] = None,
    ):
        """
        Create a volume under the specified project and regionId.

        Args:
            project_id (str): id of the project
            region_id (str): id of the region
            volume_name (str): name of the volume
            volume_type (str): volume type: MANAGED (default) or EXTERNAL
            storage_integration_id (str): Storage Integration ID, required for EXTERNAL type
            path (str): storage path, required for EXTERNAL type
        """
        return create_volume(
            self.cloud_endpoint,
            self.api_key,
            project_id,
            region_id,
            volume_name,
            volume_type,
            storage_integration_id,
            path,
        )

    def delete_volume(self, volume_name: str):
        """
        Delete a volume.
        """
        delete_volume(self.cloud_endpoint, self.api_key, volume_name)

    def describe_volume(self, volume_name: str):
        """
        Get detailed information about a specific volume.
        """
        return describe_volume(self.cloud_endpoint, self.api_key, volume_name)

    def list_volumes(
        self,
        project_id: str,
        current_page: int = 1,
        page_size: int = 10,
        volume_type: Optional[str] = None,
    ):
        """
        Paginated query of the volume list under a specified projectId.

        Args:
            project_id (str): id of the project
            current_page (int): the current page
            page_size (int): the size of each page
            volume_type (str): filter by volume type: MANAGED or EXTERNAL
        """
        return list_volumes(
            self.cloud_endpoint, self.api_key, project_id, current_page, page_size, volume_type
        )
