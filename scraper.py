import argparse
import asyncio
import fcntl
import logging
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path

import aiohttp
import orjson
from bs4 import BeautifulSoup

BASE_URL = "https://nepalicalendar.rat32.com/index_nep.php"
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()

MIN_YEAR = 1900
MAX_YEAR = 2200
MAX_MONTH = 12

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)
MAX_RETRIES = 3
RETRY_BASE_MS = 3000
BETWEEN_MONTH_MS = 2000
BETWEEN_YEAR_MS = 5000
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

log = logging.getLogger("scraper")


def _try_load_existing(file_path: Path) -> dict | None:
    """Read file atomically, returning None if it does not exist.

    Using a single open() with FileNotFoundError closes the classic
    TOCTOU window where the file could disappear between exists() and open().
    """
    try:
        return orjson.loads(file_path.read_bytes())
    except FileNotFoundError:
        return None
    except (orjson.JSONDecodeError, OSError) as exc:
        log.warning("Corrupt file %s (%s); re-scraping", file_path, exc)
        return None


def _atomic_write_json(file_path: Path, payload: dict) -> None:
    """Write JSON atomically: temp file in the same directory, then rename.

    os.replace is atomic on POSIX and Windows, so readers (the API) will
    never see a half-written file even if the process dies mid-write.
    Writing the temp file in the same directory guarantees rename is
    on the same filesystem and therefore atomic.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        dir=file_path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        os.replace(tmp_name, file_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an exclusive OS-level advisory lock on `lock_path`.

    Used to serialise the per-year aggregate write so that two scraper
    processes cannot read a half-updated set of month files and bake that
    inconsistency into data/<year>.json. The lock is released on context
    exit (including exceptions) and the lock file is left on disk as a
    tiny sentinel so future runs can find it without races of their own.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _extract_text(soup: BeautifulSoup, selector: str) -> str:
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else ""


def _extract_list(soup: BeautifulSoup, selector: str, remove_tags: list[str] | None = None) -> list[str]:
    el = soup.select_one(selector)
    if el is None:
        return []
    for tag in remove_tags or []:
        child = el.select_one(tag)
        if child:
            child.decompose()
    return [line.strip() for line in el.get_text().split("\n") if line.strip()]


def _parse_cell(cell_html: str, day_index: int) -> dict:
    """Parse one day cell from the calendar grid."""
    cell = BeautifulSoup(cell_html, "html.parser")
    n = _extract_text(cell, "#nday")
    e = _extract_text(cell, "#eday")
    t = _extract_text(cell, "#dashi")
    f = _extract_text(cell, "#fest")
    color = ""
    font_tag = cell.select_one("#nday font")
    if font_tag and font_tag.has_attr("color"):
        color = (font_tag["color"] or "").lower()
    return {
        "d": (day_index % 7) + 1,
        "n": n,
        "e": e,
        "t": t,
        "f": f,
        "h": color == "red",
    }


async def fetch_month(
    session: aiohttp.ClientSession,
    year: int,
    month: int,
    *,
    save_dir: bool = True,
) -> dict | None:
    """Fetch a single month's calendar with bounded retries."""
    if not (MIN_YEAR <= year <= MAX_YEAR) or not 1 <= month <= MAX_MONTH:
        log.error("Refusing to fetch out-of-range %d/%02d", year, month)
        return None

    file_path = DATA_DIR / str(year) / f"{month}.json"
    month_log = f"[{year}/{month:02d}]"

    existing = _try_load_existing(file_path)
    if existing is not None:
        log.info("%s Skipped: file already exists", month_log)
        return existing

    payload = {
        "selYear": year,
        "selMonth": month,
        "viewCalander": "View Calander",
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        log.info("%s Attempt %d/%d", month_log, attempt, MAX_RETRIES)
        try:
            async with session.post(BASE_URL, data=payload) as resp:
                if resp.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                    )
                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            metadata = {
                "en": _extract_text(soup, "#entarikYr"),
                "np": _extract_text(soup, "#yren"),
            }
            log.info("%s Metadata: %r (%s)", month_log, metadata["np"], metadata["en"])

            holi_fest = _extract_list(soup, "#holi", ["b", "a"])
            marriage = _extract_list(soup, "#bibah", ["b"])
            bratabandha = _extract_list(soup, "#bratabandha", ["b"])

            days = [
                _parse_cell(str(cell), i)
                for i, cell in enumerate(soup.select(".cells"))
            ]

            data = {
                "metadata": metadata,
                "days": days,
                "holiFest": holi_fest,
                "marriage": marriage,
                "bratabandha": bratabandha,
            }

            if save_dir:
                _atomic_write_json(file_path, data)
                log.info("%s Saved %s (%d days)", month_log, file_path, len(days))

            return data
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            last_error = err
            log.warning("%s Failed: %s", month_log, err)
            if attempt < MAX_RETRIES:
                wait_ms = RETRY_BASE_MS * attempt
                log.info("%s Retrying in %.1fs", month_log, wait_ms / 1000)
                await asyncio.sleep(wait_ms / 1000)

    log.error("%s Giving up after %d attempts: %s", month_log, MAX_RETRIES, last_error)
    return None


@asynccontextmanager
async def _session():
    connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        timeout=REQUEST_TIMEOUT,
        connector=connector,
        headers={"User-Agent": USER_AGENT},
    ) as session:
        yield session


async def scrape_year(
    session: aiohttp.ClientSession,
    year: int,
    *,
    save_single: bool = True,
    save_dir: bool = True,
) -> None:
    if not (MIN_YEAR <= year <= MAX_YEAR):
        log.error("Refusing to scrape out-of-range year %d", year)
        return

    log.info("Starting scrape for year %d", year)
    year_data: dict[str, dict] = {}
    success = 0

    for month in range(1, MAX_MONTH + 1):
        data = await fetch_month(session, year, month, save_dir=save_dir)
        if data is not None:
            year_data[str(month)] = data
            success += 1
        else:
            log.error("Failed to gather data for %d/%02d", year, month)
        if month < MAX_MONTH:
            await asyncio.sleep(BETWEEN_MONTH_MS / 1000)

    if success == MAX_MONTH and save_single:
        year_file = DATA_DIR / f"{year}.json"
        # Serialise with any other scraper touching the same year so we
        # never read a partial month set and write a mixed aggregate.
        with _file_lock(DATA_DIR / f"{year}.lock"):
            _atomic_write_json(year_file, year_data)
        log.info("Saved aggregated file: %s", year_file)

    log.info("Year %d done: success=%d failed=%d", year, success, MAX_MONTH - success)


async def scrape_years(
    start_year: int,
    end_year: int,
    *,
    save_single: bool = True,
    save_dir: bool = True,
) -> None:
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with _session() as session:
        for year in range(start_year, end_year + 1):
            await scrape_year(session, year, save_single=save_single, save_dir=save_dir)
            if year < end_year:
                log.info("Waiting %.1fs before next year", BETWEEN_YEAR_MS / 1000)
                await asyncio.sleep(BETWEEN_YEAR_MS / 1000)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nepali Calendar Scraper")
    parser.add_argument(
        "years",
        nargs="+",
        type=int,
        help="One year (e.g. 2082) or a range (e.g. 2070 2085)",
    )
    parser.add_argument(
        "--single",
        nargs="?",
        const="json",
        help="Also produce a single aggregated year JSON (data/<year>.json)",
    )
    parser.add_argument(
        "--dir",
        nargs="?",
        const="format",
        help="Also produce per-month files in data/<year>/<month>.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if len(args.years) == 1:
        start_year = end_year = args.years[0]
    elif len(args.years) == 2:
        start_year, end_year = args.years
    else:
        log.error("Provide either one year or a start/end pair")
        return 2

    if not (MIN_YEAR <= start_year <= MAX_YEAR) or not (MIN_YEAR <= end_year <= MAX_YEAR):
        log.error("Years must be between %d and %d", MIN_YEAR, MAX_YEAR)
        return 2

    save_single = bool(args.single)
    save_dir = bool(args.dir)
    if not save_single and not save_dir:
        # Default behaviour matches the original script: produce both.
        save_single = True
        save_dir = True

    asyncio.run(
        scrape_years(
            start_year,
            end_year,
            save_single=save_single,
            save_dir=save_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
