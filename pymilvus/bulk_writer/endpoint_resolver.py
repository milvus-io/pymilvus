import http
import logging

from pymilvus.bulk_writer.constants import ConnectType

logger = logging.getLogger("EndpointResolver")
logging.basicConfig(level=logging.INFO)


class EndpointResolver:
    @staticmethod
    def resolve_endpoint(
        default_endpoint: str, cloud: str, region: str, connect_type: ConnectType
    ) -> str:
        logger.info(
            "Start resolving endpoint, cloud=%s, region=%s, connectType=%s",
            cloud,
            region,
            connect_type,
        )
        if cloud == "ali":
            default_endpoint = EndpointResolver._resolve_oss_endpoint(region, connect_type)
        logger.info("Resolved endpoint: %s, reachable check passed", default_endpoint)
        return default_endpoint

    @staticmethod
    def _resolve_oss_endpoint(region: str, connect_type: ConnectType) -> str:
        internal_endpoint = f"oss-{region}-internal.aliyuncs.com"
        public_endpoint = f"oss-{region}.aliyuncs.com"

        if connect_type == ConnectType.INTERNAL:
            logger.info("Forced INTERNAL endpoint selected: %s", internal_endpoint)
            EndpointResolver._check_endpoint_reachable(internal_endpoint, True)
            return internal_endpoint
        if connect_type == ConnectType.PUBLIC:
            logger.info("Forced PUBLIC endpoint selected: %s", public_endpoint)
            EndpointResolver._check_endpoint_reachable(public_endpoint, True)
            return public_endpoint
        if EndpointResolver._check_endpoint_reachable(internal_endpoint, False):
            logger.info("AUTO mode: internal endpoint reachable, using %s", internal_endpoint)
            return internal_endpoint
        logger.warning(
            "AUTO mode: internal endpoint not reachable, fallback to public endpoint %s",
            public_endpoint,
        )
        EndpointResolver._check_endpoint_reachable(public_endpoint, True)
        return public_endpoint

    @staticmethod
    def _check_endpoint_reachable(endpoint: str, raise_error: bool) -> bool:
        try:
            conn = http.client.HTTPSConnection(endpoint, timeout=5)
            conn.request("HEAD", "/")
            resp = conn.getresponse()
            code = resp.status
            logger.debug("Checked endpoint %s, response code=%s", endpoint, code)
        except Exception as e:
            if raise_error:
                logger.exception("Endpoint %s not reachable, throwing exception", endpoint)
                raise RuntimeError(str(e)) from e
            logger.warning("Endpoint %s not reachable, will fallback if needed", endpoint)
            return False
        else:
            return 200 <= code < 400
