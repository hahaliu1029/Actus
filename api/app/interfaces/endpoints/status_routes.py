import logging
import os
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from app.application.services.status_service import StatusService
from app.domain.models.health_status import HealthStatus
from app.infrastructure.storage.minio import get_minio
from app.interfaces.dependencies import AdminUser, CurrentUser
from app.interfaces.schemas import Response
from app.interfaces.service_dependencies import get_status_service
from core.config import get_settings
from fastapi import APIRouter, Depends, File, UploadFile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/status", tags=["状态模块"])


@router.get(
    "/",
    response_model=Response[List[HealthStatus]],
    summary="系统健康检查",
    description="检查系统的postgres, redis,fastapi等服务的健康状态",
)
async def get_status(
    current_user: CurrentUser,
    status_service: StatusService = Depends(get_status_service),
) -> Response:
    """系统健康检查，检查postgres/redis/fastapi/minio等服务"""
    statues = await status_service.check_all()

    if any(item.status == "error" for item in statues):
        return Response.fail(503, "系统存在服务异常", statues)

    return Response.success(msg="系统健康检查成功", data=statues)


@router.get(
    "/minio",
    response_model=Response,
    summary="MinIO 健康检查",
    description="检查 MinIO 连接；smoke=true 时会执行 put/get/remove 读写自检。",
)
async def get_minio_status(
    current_user: CurrentUser,
    smoke: bool = False,
    bucket: str | None = None,
) -> Response:
    settings = get_settings()
    bucket_name = bucket or settings.minio_bucket_name
    store = get_minio()

    try:
        data = (
            await store.smoke_test(bucket_name=bucket_name)
            if smoke
            else await store.ping(bucket_name=bucket_name)
        )
        return Response.success(data)
    except Exception as e:
        logger.exception("MinIO health check failed")
        return Response.fail(
            code=500, msg="MinIO health check failed", data={"error": str(e)}
        )


@router.post(
    "/minio/upload",
    response_model=Response,
    summary="MinIO 上传文件测试",
    description="使用 multipart/form-data 上传文件到 MinIO（仅限管理员）",
)
async def upload_minio_file(
    admin_user: AdminUser,
    file: UploadFile = File(...),
    bucket: str | None = None,
    object_name: str | None = None,
    prefix: str = "uploads",
    presign: bool = True,
    expiry_seconds: int = 3600,
) -> Response:
    settings = get_settings()
    bucket_name = bucket or settings.minio_bucket_name
    store = get_minio()

    normalized_object = (object_name or "").strip().lstrip("/").replace("\\", "/")
    if not normalized_object:
        filename = os.path.basename(file.filename or "upload.bin")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix_clean = prefix.strip().strip("/").replace("\\", "/")
        normalized_object = (
            f"{prefix_clean}/{timestamp}-{uuid4().hex}-{filename}"
            if prefix_clean
            else f"{timestamp}-{uuid4().hex}-{filename}"
        )

    try:
        if not await store.bucket_exists(bucket_name):
            return Response.fail(
                code=400,
                msg="bucket_not_exists",
                data={"bucket": bucket_name, "object": normalized_object},
            )

        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)

        data = await store.upload_fileobj(
            bucket_name=bucket_name,
            object_name=normalized_object,
            data=file.file,
            length=size,
            content_type=file.content_type or "application/octet-stream",
        )
        data.update(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": size,
            }
        )
        if presign:
            data["presigned_get_url"] = await store.presigned_get_url(
                bucket_name=bucket_name,
                object_name=normalized_object,
                expiry_seconds=expiry_seconds,
            )
        return Response.success(data)
    except Exception as e:
        logger.exception("MinIO upload failed")
        return Response.fail(
            code=500, msg="MinIO upload failed", data={"error": str(e)}
        )
