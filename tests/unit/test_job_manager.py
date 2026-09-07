import time

from board_of_scientists.graph.job_manager import ResearchJobManager


def test_job_manager_runs_and_retains_result():
    manager = ResearchJobManager(lambda path: {"path": path, "ok": True}, max_workers=1)
    try:
        job = manager.submit("paper.pdf")
        deadline = time.time() + 2
        while time.time() < deadline:
            current = manager.get(job.job_id)
            if current and current.status == "completed":
                assert current.result == {"path": "paper.pdf", "ok": True}
                assert current.error is None
                return
            time.sleep(0.01)
        raise AssertionError("job did not complete")
    finally:
        manager.shutdown()


def test_job_manager_captures_runner_failure():
    def fail(_path):
        raise ValueError("boom")

    manager = ResearchJobManager(fail, max_workers=1)
    try:
        job = manager.submit("paper.pdf")
        deadline = time.time() + 2
        while time.time() < deadline:
            current = manager.get(job.job_id)
            if current and current.status == "failed":
                assert current.result is None
                assert "ValueError: boom" == current.error
                return
            time.sleep(0.01)
        raise AssertionError("job did not fail")
    finally:
        manager.shutdown()
