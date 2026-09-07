"""Bounded in-process research job manager for the HTTP control plane."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import BoundedSemaphore, Lock
from uuid import uuid4


class JobQueueFullError(RuntimeError):
    """Raised when the configured research queue has no free capacity."""


@dataclass
class ResearchJob:
    job_id: str
    pdf_path: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None


class ResearchJobManager:
    """Bound both running workers and queued jobs to prevent local memory exhaustion."""

    def __init__(self, runner, max_workers: int = 1, max_queue_size: int = 8, max_retained_jobs: int = 256):
        workers = max(1, max_workers)
        queue = max(0, max_queue_size)
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="research-job")
        self._capacity = BoundedSemaphore(workers + queue)
        self._max_retained_jobs = max(16, max_retained_jobs)
        self._jobs: dict[str, ResearchJob] = {}
        self._futures: dict[str, Future] = {}
        self._lock = Lock()

    def submit(self, pdf_path: str) -> ResearchJob:
        if not self._capacity.acquire(blocking=False):
            raise JobQueueFullError("Research job queue is full; retry later.")
        job = ResearchJob(job_id=uuid4().hex, pdf_path=pdf_path)
        try:
            with self._lock:
                self._trim_finished_locked()
                self._jobs[job.job_id] = job
                self._futures[job.job_id] = self._executor.submit(self._run, job.job_id, pdf_path)
            return self.get(job.job_id)
        except Exception:
            self._capacity.release()
            raise

    def get(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, pdf_path: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                self._capacity.release()
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = self._runner(pdf_path)
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.finished_at = datetime.now(timezone.utc).isoformat()
            return
        finally:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job.status == "running":
                    job.status = "completed"
                    job.result = result
                    job.finished_at = datetime.now(timezone.utc).isoformat()
            self._capacity.release()

    def _trim_finished_locked(self) -> None:
        finished = [job for job in self._jobs.values() if job.status in {"completed", "failed"}]
        excess = len(self._jobs) - self._max_retained_jobs + 1
        if excess <= 0:
            return
        for job in sorted(finished, key=lambda item: item.finished_at or item.created_at)[:excess]:
            self._jobs.pop(job.job_id, None)
            self._futures.pop(job.job_id, None)


def build_default_job_manager(runner):
    workers = int(os.getenv("RESEARCH_MAX_CONCURRENT_JOBS", "1"))
    queue = int(os.getenv("RESEARCH_MAX_QUEUED_JOBS", "8"))
    retained = int(os.getenv("RESEARCH_MAX_RETAINED_JOBS", "256"))
    return ResearchJobManager(runner, max_workers=workers, max_queue_size=queue, max_retained_jobs=retained)


__all__ = ["JobQueueFullError", "ResearchJob", "ResearchJobManager", "build_default_job_manager"]
