from models.dbschemas import CeleryTaskExecution
from sqlalchemy import select, update, delete
from datetime import datetime, timedelta, timezone
import hashlib
import json
import uuid


class IdempotencyManager:

    RUNNING_STATES = ("PENDING", "STARTED")
    GRACE_SECONDS = 60

    def __init__(self, db_client, db_engine):
        self.db_client = db_client
        self.db_engine = db_engine

    @staticmethod
    def create_args_hash(task_name: str, task_args: dict) -> str:
        payload = json.dumps({"task_name": task_name, "task_args": task_args},
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get_latest_execution(self, task_name: str, task_args_hash: str):
        async with self.db_client() as session:
            result = await session.execute(
                select(CeleryTaskExecution)
                .where(CeleryTaskExecution.task_name == task_name,
                       CeleryTaskExecution.task_args_hash == task_args_hash)
                .order_by(CeleryTaskExecution.created_at.desc(), CeleryTaskExecution.execution_id.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def create_task_record(self, task_name: str, task_args: dict, task_args_hash: str,
                                 celery_task_id: str):
        record = CeleryTaskExecution(
            task_name=task_name,
            task_args_hash=task_args_hash,
            celery_task_id=uuid.UUID(celery_task_id) if celery_task_id else None,
            status="STARTED",
            task_args=task_args,
            started_at=datetime.now(timezone.utc),
        )
        async with self.db_client() as session:
            async with session.begin():
                session.add(record)
            await session.refresh(record)
        return record

    async def mark_started(self, execution_id: int):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(
                    update(CeleryTaskExecution)
                    .where(CeleryTaskExecution.execution_id == execution_id)
                    .values(status="STARTED", started_at=datetime.now(timezone.utc),
                            completed_at=None, result=None)
                )

    async def update_task_status(self, execution_id: int, status: str, result: dict = None):
        values = {"status": status, "result": result}
        if status in ("SUCCESS", "FAILURE"):
            values["completed_at"] = datetime.now(timezone.utc)
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(
                    update(CeleryTaskExecution)
                    .where(CeleryTaskExecution.execution_id == execution_id)
                    .values(**values)
                )

    def is_still_running(self, record, task_time_limit: int) -> bool:
        if record.status not in self.RUNNING_STATES:
            return False
        started_at = record.started_at or record.created_at
        if started_at is None:
            return False
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - started_at
        return elapsed < timedelta(seconds=task_time_limit + self.GRACE_SECONDS)

    async def should_execute_task(self, task_name: str, task_args: dict, celery_task_id: str,
                                  task_time_limit: int = 600):
        task_args_hash = self.create_args_hash(task_name, task_args)
        existing = await self.get_latest_execution(task_name, task_args_hash)

        if existing and celery_task_id and str(existing.celery_task_id) == str(celery_task_id):
            await self.mark_started(existing.execution_id)
            return True, existing

        if existing and self.is_still_running(existing, task_time_limit):
            return False, existing

        record = await self.create_task_record(task_name, task_args, task_args_hash, celery_task_id)
        return True, record

    async def cleanup_old_tasks(self, time_retention: int = 86400) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=time_retention)
        async with self.db_client() as session:
            async with session.begin():
                result = await session.execute(
                    delete(CeleryTaskExecution)
                    .where(CeleryTaskExecution.created_at < cutoff)
                )
            return result.rowcount
