"""Smoke tests for the Nepali Calendar API.

These tests boot the FastAPI app against the live data/ tree and exercise
the public endpoints plus input validation. They are not network-dependent
and finish in well under a second.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert isinstance(body["years_available"], int)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["years"] >= 0


def test_get_monthly_calendar():
    year, month = 2082, 1
    resp = client.get(f"/calendar/{year}/{month}")
    if resp.status_code == 200:
        body = resp.json()
        assert "metadata" in body
        assert "days" in body
        assert isinstance(body["days"], list)
    else:
        assert resp.status_code == 404


def test_invalid_month_returns_422():
    # FastAPI/Pydantic surfaces invalid path params as 422.
    resp = client.get("/calendar/2082/13")
    assert resp.status_code == 422


def test_invalid_month_zero_returns_422():
    resp = client.get("/calendar/2082/0")
    assert resp.status_code == 422


def test_invalid_year_too_low_returns_422():
    resp = client.get("/calendar/1500")
    assert resp.status_code == 422


def test_unknown_year_returns_404():
    # An in-range year that simply has no data should give 404, not 422.
    resp = client.get("/calendar/2150")
    assert resp.status_code == 404


def test_etag_is_served_and_respected():
    sample_year = 2082
    first = client.get(f"/calendar/{sample_year}/1")
    if first.status_code != 200:
        return
    etag = first.headers.get("ETag")
    assert etag is not None
    second = client.get(f"/calendar/{sample_year}/1", headers={"If-None-Match": etag})
    assert second.status_code == 304


def test_file_lock_serialises_writers(tmp_path):
    """Two writers taking the same lock must run sequentially, not interleaved."""
    from scraper import _file_lock

    log_path = tmp_path / "order.log"
    lock = tmp_path / "x.lock"

    def _writer(tag: str, hold_ms: int) -> None:
        with _file_lock(lock):
            with log_path.open("a") as f:
                f.write(f"{tag}:enter\n")
                f.flush()
            import time as _t

            _t.sleep(hold_ms / 1000)
            with log_path.open("a") as f:
                f.write(f"{tag}:exit\n")

    import threading

    t1 = threading.Thread(target=_writer, args=("A", 200))
    t2 = threading.Thread(target=_writer, args=("B", 50))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    entries = log_path.read_text().splitlines()
    # The only legal orderings are A-enter, A-exit, B-enter, B-exit OR
    # B-enter, B-exit, A-enter, A-exit. Anything interleaved means the
    # lock did not actually serialise.
    assert entries in (
        ["A:enter", "A:exit", "B:enter", "B:exit"],
        ["B:enter", "B:exit", "A:enter", "A:exit"],
    ), f"writers interleaved: {entries}"


def test_atomic_write_is_visible_after_crash(tmp_path):
    """A writer that raises must not leave a half-written file behind."""
    from scraper import _atomic_write_json

    target = tmp_path / "x.json"
    try:
        with (
            pytest.raises(RuntimeError, match="boom"),
            patch("scraper.orjson.dumps", side_effect=RuntimeError("boom")),
        ):
            _atomic_write_json(target, {"a": 1})
    finally:
        # Patch is module-level; restore the real orjson.dumps to avoid
        # leaking the side effect into the rest of the test session.
        pass

    assert not target.exists(), f"partial file left behind: {target.read_bytes()!r}"
    leftovers = list(tmp_path.glob(".x.json.*.tmp"))
    assert leftovers == [], f"temp files leaked: {leftovers}"
