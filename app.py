import hashlib
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("nepali-calendar-api")

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
MIN_YEAR = 1900
MAX_YEAR = 2200
CACHE_CONTROL = "public, max-age=86400, immutable"

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _load_data()
    yield


app = FastAPI(
    title="Nepali Calendar API",
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url=None,
    lifespan=_lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=512)


_CALENDAR: dict[int, dict] = {}
_MONTHLY: dict[tuple[int, int], dict] = {}
_ETAG_DATA: bytes = b""


def _load_data() -> None:
    """Eagerly load every JSON file into memory once at startup.

    The dataset is static and small (~9MB), so an in-memory dict keeps the
    API off the disk on every request, which is essential for tiny servers
    where cold reads hurt p99 latency and steal CPU from the event loop.
    """
    global _ETAG_DATA
    if not DATA_DIR.is_dir():
        log.warning("DATA_DIR %s does not exist; API will return 404s", DATA_DIR)
        return

    monthly: dict[tuple[int, int], dict] = {}
    yearly: dict[int, dict] = {}

    for year_dir in DATA_DIR.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for month_file in year_dir.glob("*.json"):
            if not month_file.stem.isdigit():
                continue
            month = int(month_file.stem)
            if not 1 <= month <= 12:
                continue
            try:
                raw = month_file.read_bytes()
                data = orjson.loads(raw)
            except (orjson.JSONDecodeError, OSError) as exc:
                log.error("Skipping corrupt file %s: %s", month_file, exc)
                continue
            monthly[(year, month)] = data

    # Build yearly aggregates from monthly data so the /calendar/{year}
    # endpoint works even when only the directory layout has been scraped.
    by_year: dict[int, dict[int, dict]] = {}
    for (year, month), payload in monthly.items():
        by_year.setdefault(year, {})[month] = payload

    for year, months in by_year.items():
        if len(months) == 12:
            yearly[year] = {str(m): months[m] for m in range(1, 13)}

    # Also pick up pre-aggregated yearly files (e.g. data/2082.json) if they
    # are newer or missing, so the scraper's --single output is honoured.
    for year_file in DATA_DIR.glob("*.json"):
        if not year_file.stem.isdigit():
            continue
        year = int(year_file.stem)
        if not MIN_YEAR <= year <= MAX_YEAR:
            continue
        try:
            raw = year_file.read_bytes()
            data = orjson.loads(raw)
        except (orjson.JSONDecodeError, OSError) as exc:
            log.error("Skipping corrupt file %s: %s", year_file, exc)
            continue
        # Only adopt the file if it covers a full year.
        if isinstance(data, dict) and all(str(m) in data for m in range(1, 13)):
            yearly[year] = data

    _MONTHLY.update(monthly)
    _CALENDAR.update(yearly)
    _ETAG_DATA = hashlib.sha1(
        repr(sorted(_CALENDAR.keys())).encode()
    ).hexdigest()[:16].encode()
    log.info(
        "Loaded %d years and %d months into memory",
        len(_CALENDAR),
        len(_MONTHLY),
    )


@app.get("/", summary="Root Endpoint")
async def root() -> dict:
    return {
        "message": "Welcome to the Nepali Calendar API. Use /docs for interactive API documentation.",
        "years_available": len(_CALENDAR),
    }


@app.get("/health", summary="Health Check")
async def health() -> dict:
    return {"status": "ok", "years": len(_CALENDAR), "months": len(_MONTHLY)}


def _validate_year(year: int) -> None:
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise HTTPException(
            status_code=400,
            detail=f"Year must be between {MIN_YEAR} and {MAX_YEAR}",
        )


def _maybe_not_modified(request: Request, etag: str) -> Response | None:
    if request.headers.get("if-none-match") == etag:
        return Response(
            status_code=304, headers={"ETag": etag, "Cache-Control": CACHE_CONTROL}
        )
    return None


@app.get(
    "/calendar/{year}",
    summary="Get Yearly Calendar",
    responses={404: {"description": "Year not found"}},
)
async def get_year_calendar(
    request: Request,
    year: int = PathParam(..., ge=MIN_YEAR, le=MAX_YEAR, description="BS year"),
) -> Response:
    _validate_year(year)
    payload = _CALENDAR.get(year)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Calendar data for year {year} not found. Please run the scraper for this year.",
        )
    etag = f'W/"{year}-{_ETAG_DATA.decode()}"'
    not_modified = _maybe_not_modified(request, etag)
    if not_modified is not None:
        return not_modified
    body = orjson.dumps(payload)
    return Response(
        content=body,
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": CACHE_CONTROL},
    )


@app.get(
    "/calendar/{year}/{month}",
    summary="Get Monthly Calendar",
    responses={404: {"description": "Year/Month not found"}},
)
async def get_month_calendar(
    request: Request,
    year: int = PathParam(..., ge=MIN_YEAR, le=MAX_YEAR, description="BS year"),
    month: int = PathParam(..., ge=1, le=12, description="BS month (1-12)"),
) -> Response:
    _validate_year(year)
    payload = _MONTHLY.get((year, month))
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Calendar data for {year}/{month} not found. Please run the scraper for this year.",
        )
    etag = f'W/"{year}-{month}-{_ETAG_DATA.decode()}"'
    not_modified = _maybe_not_modified(request, etag)
    if not_modified is not None:
        return not_modified
    body = orjson.dumps(payload)
    return Response(
        content=body,
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": CACHE_CONTROL},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        log_level=os.environ.get("LOG_LEVEL", "info"),
        access_log=False,
    )
