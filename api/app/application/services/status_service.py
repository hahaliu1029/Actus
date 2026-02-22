import asyncio
import logging
from typing import List

from app.domain.external.health_checker import HealthChecker
from app.domain.models.health_status import HealthStatus

logger = logging.getLogger(__name__)


class StatusService:
    """Aggregates health checks for system services."""

    def __init__(self, checkers: List[HealthChecker]) -> None:
        """Create the service with the health checkers."""
        self._checkers = checkers

    async def check_all(self) -> List[HealthStatus]:
        """Run all health checks and return their statuses."""
        results = await asyncio.gather(
            *(checker.check() for checker in self._checkers),
            return_exceptions=True,
        )

        statuses: List[HealthStatus] = []
        for checker, result in zip(self._checkers, results):
            if isinstance(result, Exception):
                service = getattr(checker, "service_name", checker.__class__.__name__)
                logger.error(f"{service} health check failed: {str(result)}")
                statuses.append(
                    HealthStatus(service=str(service), status="error", details=str(result))
                )
            else:
                statuses.append(result)

        # Include FastAPI itself as always-ok.
        statuses.append(HealthStatus(service="fastapi", status="ok"))
        return statuses
