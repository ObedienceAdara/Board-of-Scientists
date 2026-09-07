"""Bounded in-process research job manager for the HTTP control plane."""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4


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
    """Bound concurrency so HTTP callers cannot create unlimited workers."""

    def __init__(self, runner, max_workers: int = 1, max_retained_jobs: int = 256):
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="research-job")
        self._max_retained_jobs = max(16, max_retained_jobs)
        self._jobs: dict[str, ResearchJob] = {}
        self._futures: dict[str, Future] = {}
        self._lock = Lock()

    def submit(self, pdf_path: str) -> ResearchJob:
        job = ResearchJob(job_id=uuid4().hex, pdf_path=pdf_path)
        with self._lock:
            self._jobs[job.job_id] = job
            self._trim_finished_locked()
            future = self._executor.submit(self._run, job.job_id, pdf_path)
            self._futures[job.job_id] = future
        return self.get(job.job_id)

    def get(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, pdf_path: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = self._runner(pdf_path)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = datetime.now(timezone.utc).isoformat()
            return
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.result = result
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _trim_finished_locked(self) -> None:
        finished = [
            job for job in self._jobs.values()
            if job.status in {"completed", "failed"}
        ]
        excess = len(self._jobs) - self._max_retained_jobs
        if excess <= 0:
            return
        for job in sorted(finished, key=lambda item: item.finished_at or item.created_at)[:excess]:
            self._jobs.pop(job.job_id, None)
            self._futures.pop(job.job_id, None)


def build_default_job_manager(runner):
    workers = int(os.getenv("RESEARCH_MAX_CONCURRENT_JOBS", "1"))
    retained = int(os.getenv("RESEARCH_MAX_RETAINED_JOBS", "256"))
    return ResearchJobManager(runner, max_workers=workers, max_retained_jobs=retained)


__all__ = ["ResearchJob", "ResearchJobManager", "build_default_job_manager"]
