import logging

from app.domain.external.health_checker import HealthChecker
from app.domain.models.health_status import HealthStatus
from app.infrastructure.storage.minio import MinioStore
from core.config import get_settings

logger = logging.getLogger(__name__)


class MinioHealthChecker(HealthChecker):
    """MinIO health checker"""

    def __init__(self, minio_store: MinioStore, bucket_name: str | None = None) -> None:
        self._minio_store = minio_store
        settings = get_settings()
        self._bucket_name = bucket_name or settings.minio_bucket_name

    async def check(self) -> HealthStatus:
        """Check MinIO connectivity and bucket availability."""
        try:
            result = await self._minio_store.ping(bucket_name=self._bucket_name)
            if not result.get("ok", False):
                return HealthStatus(
                    service="minio",
                    status="error",
                    details=f"bucket_not_exists: {self._bucket_name}",
                )
            return HealthStatus(service="minio", status="ok")
        except Exception as e:
            logger.error(f"MinIO health check failed: {str(e)}")
            return HealthStatus(service="minio", status="error", details=str(e))
