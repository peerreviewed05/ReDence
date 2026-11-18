import base64
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _ensure_redence_package() -> None:
    """Allow `import redence.*` when this file is loaded from agent/."""
    parent = str(_PACKAGE_ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

LAST_SCREENSHOT_B64 = None
SESSION = {
    "playwright": None,
    "browser": None,
    "context": None,
    "page": None,
}


def _cleanup_browser_session() -> None:
    """Close Playwright resources and stop the driver. Safe to call multiple times."""
    for key in ("page", "context", "browser"):
        obj = SESSION.get(key)
        if obj is not None:
            try:
                obj.close()
            except Exception:
                pass
        SESSION[key] = None
    pw = SESSION.get("playwright")
    if pw is not None:
        try:
            pw.stop()
        except Exception:
            pass
    SESSION["playwright"] = None


def _load_dotenv_fallback(dotenv_path: Path) -> None:
    """
    Minimal .env loader (no external dependencies).
    - Reads lines like KEY=VALUE
    - Ignores blank lines and comments starting with '#'
    - Does not override existing environment variables
    """
    try:
        if not dotenv_path.exists():
            return
        raw = dotenv_path.read_text(encoding="utf-8", errors="ignore")
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if key in os.environ:
                continue
            # Trim optional surrounding quotes.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            os.environ[key] = val
    except Exception:
        pass


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


_load_dotenv_fallback(Path(__file__).resolve().parent / ".env")
_load_dotenv_fallback(_PACKAGE_ROOT / ".env")

# Optional: if python-dotenv exists, load it too (but do not override existing values).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

# Import after .env so OLLAMA_HOST applies (module-level ollama.chat() binds host at import).
from ollama import Client

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from playwright.sync_api import sync_playwright

_OLLAMA_CLIENT: Client | None = None


def _ollama_host() -> str | None:
    host = (os.environ.get("OLLAMA_HOST") or os.environ.get("REDENCE_OLLAMA_HOST") or "").strip()
    return host or None


def _ollama_client() -> Client:
    global _OLLAMA_CLIENT
    host = _ollama_host()
    if _OLLAMA_CLIENT is None:
        _OLLAMA_CLIENT = Client(host=host) if host else Client()
    return _OLLAMA_CLIENT


# Consolidated environment configuration (defaults are intentionally True).
REDENCE_LOG_CONSOLE_ENABLED = _env_bool("REDENCE_LOG_CONSOLE", True)


def _resolve_headless() -> bool:
    """Headless by default; honors USE_HEADED or PLAYWRIGHT_HEADLESS from redence/.env."""
    if os.environ.get("USE_HEADED") is not None:
        return not _env_bool("USE_HEADED", False)
    return _env_bool("PLAYWRIGHT_HEADLESS", True)


HEADLESS = _resolve_headless()
USE_HEADED = not HEADLESS
# JPEG capture, vision.changed, and Ollama image delivery (one switch).
if os.environ.get("REDENCE_VISION_ENABLED") is not None:
    REDENCE_VISION_ENABLED = _env_bool("REDENCE_VISION_ENABLED", True)
else:
    REDENCE_VISION_ENABLED = _env_bool("REDENCE_SCREENSHOT_VISION", True)
REDENCE_ARCHIVE_SCREENSHOT_ENABLED = _env_bool("REDENCE_ARCHIVE_SCREENSHOT_ENABLED", True)
REDENCE_CAPTCHA_MODE = os.environ.get("REDENCE_CAPTCHA_MODE", "auto").strip().lower() or "auto"
if REDENCE_CAPTCHA_MODE not in ("auto", "llm"):
    REDENCE_CAPTCHA_MODE = "auto"
REDENCE_CAPTCHA_DEBUG = _env_bool("REDENCE_CAPTCHA_DEBUG", True)

MODEL = os.environ.get("REDENCE_INVESTIGATOR_MODEL", "investigator")

# Structured per-run logger (JSON lines)
def _setup_run_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"redence.run.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("%(message)s")
        fh = logging.FileHandler(run_dir / "runs.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if REDENCE_LOG_CONSOLE_ENABLED:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            logger.addHandler(sh)

    return logger


def _cleanup_run_logger(logger: logging.Logger | None) -> None:
    """Close and detach handlers so file descriptors are not leaked across runs."""
    if logger is None:
        return
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


def _truncate(s, limit=2000):
    try:
        s = str(s)
    except Exception:
        return "<unstringifiable>"
    return s if len(s) <= limit else s[:limit] + f"... <truncated {len(s) - limit} chars>"


def _log_event(logger: logging.Logger, event: str, **fields):
    payload = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Never break the agent due to logging issues
        pass


def _message_get(message, key: str, default=None):
    """Read a field from an Ollama message (dict or object)."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _log_model_thinking(
    logger: logging.Logger,
    output_lines: list,
    message,
    *,
    step: int,
) -> None:
    """Persist Ollama `thinking` alongside existing content/tool logging."""
    thinking = _message_get(message, "thinking") or ""
    if not isinstance(thinking, str):
        thinking = str(thinking)
    thinking = thinking.strip()
    if not thinking:
        return
    output_lines.append("\n=== Model Thinking ===")
    output_lines.append(thinking)
    _log_event(
        logger,
        "model_thinking",
        step=step,
        thinking=thinking,
        thinking_len=len(thinking),
    )


# Realistic Chrome User-Agent and headers to avoid bot detection
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


def extract_url_from_prompt(prompt):
    """
    Extract a URL or domain from the prompt when user asks to check/visit/go to a website.
    Returns normalized URL (with https://) or None if no website detected.
    """
    prompt_lower = prompt.lower()
    # Trigger phrases that indicate user wants to visit a specific site
    triggers = ("check ", "visit ", "go to ", "open ", "look at ", "goto ", "go to")
    if not any(prompt_lower.startswith(t) or f" {t}" in prompt_lower for t in triggers):
        # Also match bare domains like "amazon.com" or "what's on amazon.com"
        pass  # We'll still try to find URLs/domains below

    # Full URL: https://example.com or http://example.com/path
    url_match = re.search(
        r"https?://[^\s\)\"\']+",
        prompt,
        re.IGNORECASE,
    )
    if url_match:
        url = url_match.group(0).rstrip(".,;:!?)")
        return url if "://" in url else f"https://{url}"

    # Domain pattern: amazon.com, www.example.com, sub.domain.co.uk
    domain_match = re.search(
        r"(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}(?:/[^\s\)\"\']*)?",
        prompt,
    )
    if domain_match:
        domain = domain_match.group(0).rstrip(".,;:!?)")
        return f"https://{domain}" if not domain.startswith("http") else domain

    return None

# -------------------------
# CAPTCHA SUPPORT (Playwright / selenium_captcha_processing)
# Same bootstrap and API as demo_playwright_recaptcha.py
# -------------------------
def _local_captcha_package_root() -> Path | None:
    """Playwright-based sources next to this agent script."""
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir / "mycaptcha"):
        if (
            (candidate / "__init__.py").is_file()
            and (candidate / "bypassing.py").is_file()
            and (candidate / "detectors" / "detect_re_captcha_playwright.py").is_file()
        ):
            return candidate
    return None


def _purge_captcha_package_modules() -> None:
    package_name = "selenium_captcha_processing"
    for key in list(sys.modules):
        if key == package_name or key.startswith(package_name + "."):
            del sys.modules[key]


def _loaded_captcha_package_is_local_playwright() -> bool:
    """Local factory only registers a reCaptcha Playwright detector."""
    try:
        from selenium_captcha_processing.captcha import Captcha
        from selenium_captcha_processing.factory import make_detector_type

        return make_detector_type(Captcha.GeeTest) is None
    except Exception:
        return False


def _load_local_captcha_package(repo_root: Path) -> None:
    package_name = "selenium_captcha_processing"
    spec = importlib.util.spec_from_file_location(
        package_name,
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load local selenium_captcha_processing package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


def _bootstrap_selenium_captcha_processing() -> None:
    """
    Load Playwright captcha sources from this repo when present.

    Must run before `import selenium_captcha_processing`: the pip package is
    Selenium-based (`page.find_element`) and breaks with Playwright Page objects.
    """
    package_name = "selenium_captcha_processing"
    repo_root = _local_captcha_package_root()

    if package_name in sys.modules:
        if repo_root is not None and not _loaded_captcha_package_is_local_playwright():
            _purge_captcha_package_modules()
        else:
            return

    if repo_root is not None:
        _load_local_captcha_package(repo_root)
        try:
            from selenium_captcha_processing.detectors.detect_re_captcha_playwright import (  # noqa: F401
                DetectReCaptchaPlaywright,
            )
        except ModuleNotFoundError as e:
            raise ImportError(
                "Local captcha package loaded but a Playwright module is missing. "
                "Copy the full folder (detectors/, solvers/, utils/, factory.py, etc.). "
                "Also run: pip uninstall selenium-captcha-processing -y "
                "(pip package conflicts with this fork)."
            ) from e
        return

    try:
        import selenium_captcha_processing  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "No local Playwright captcha package found. "
            f"Expected __init__.py and detectors/detect_re_captcha_playwright.py under "
            f"{Path(__file__).resolve().parent}. "
            "Do not pip install selenium-captcha-processing; use the local fork in this repo."
        ) from e


_bootstrap_selenium_captcha_processing()
from selenium_captcha_processing import BypassCaptcha, Config
from selenium_captcha_processing.captcha import Captcha
from selenium_captcha_processing.factory import (
    make_default_utils,
    make_detector_type,
    make_solver_type,
)

_CAPTCHA_LOGGER = logging.getLogger("redence.captcha")


def _ensure_captcha_logger() -> None:
    if _CAPTCHA_LOGGER.handlers:
        return
    _CAPTCHA_LOGGER.setLevel(logging.DEBUG)
    _CAPTCHA_LOGGER.propagate = False
    if REDENCE_LOG_CONSOLE_ENABLED:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        _CAPTCHA_LOGGER.addHandler(sh)


def _log_captcha(run_logger, event: str, **fields) -> None:
    """Log captcha events to run log (JSON) and redence.captcha console logger."""
    _ensure_captcha_logger()
    if run_logger is not None:
        _log_event(run_logger, event, **fields)
    try:
        payload = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        _CAPTCHA_LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def _captcha_config() -> Config:
    try:
        wait_after_solving = int(os.environ.get("REDENCE_CAPTCHA_WAIT", "2"))
    except ValueError:
        wait_after_solving = 2
    return Config(
        wait_after_solving=wait_after_solving,
        default_element_waiting=1.0,
        raise_exceptions_in_detection=False,
        raise_exceptions_in_solving=False,
    )


def _captcha_detection_scores(page, config: Config) -> list:
    """Run each registered detector and return raw scores (before classify threshold)."""
    utils = make_default_utils(config)
    scores = []
    for captcha in Captcha:
        detector_type = make_detector_type(captcha)
        if detector_type is None:
            scores.append({
                "captcha": captcha.value,
                "has_detector": False,
            })
            continue

        entry = {
            "captcha": captcha.value,
            "has_detector": True,
            "has_solver": make_solver_type(captcha) is not None,
        }
        try:
            detector = detector_type(page, utils, config)
            prob = float(detector.detected())
            entry["score"] = prob
            entry["above_threshold"] = prob >= config.detection_threshold
            entry["detection_threshold"] = config.detection_threshold
        except Exception as ex:
            entry["score"] = None
            entry["error"] = str(ex)
        scores.append(entry)
    return scores


def _captcha_notes_payload(notes) -> list:
    return [
        {"captcha": note.captcha.value, "prob": note.prob}
        for note in (notes or [])
    ]


def handle_captcha(page, logger=None, context: str = "unknown") -> bool:
    """Detect and solve captchas via BypassCaptcha (library detectors, not HTML heuristics)."""
    page_url = getattr(page, "url", None)
    try:
        import selenium_captcha_processing.factory as _captcha_factory_mod

        package_info = {
            "factory_module": getattr(_captcha_factory_mod, "__file__", None),
            "is_local_playwright": _loaded_captcha_package_is_local_playwright(),
        }
    except Exception as ex:
        package_info = {"error": str(ex)}

    _log_captcha(
        logger,
        "captcha_bypass_start",
        context=context,
        page_url=page_url,
        mode=REDENCE_CAPTCHA_MODE,
        package=package_info,
    )

    if REDENCE_CAPTCHA_MODE != "auto":
        _log_captcha(
            logger,
            "captcha_bypass_skipped",
            context=context,
            page_url=page_url,
            reason="REDENCE_CAPTCHA_MODE is not auto",
        )
        return True

    config = _captcha_config()
    _log_captcha(
        logger,
        "captcha_config",
        context=context,
        page_url=page_url,
        wait_after_solving=config.wait_after_solving,
        default_element_waiting=config.default_element_waiting,
        detection_threshold=config.detection_threshold,
        raise_exceptions_in_detection=config.raise_exceptions_in_detection,
        raise_exceptions_in_solving=config.raise_exceptions_in_solving,
    )

    if REDENCE_CAPTCHA_DEBUG:
        lib_logger = logging.getLogger("selenium_captcha_processing")
        lib_logger.setLevel(logging.DEBUG)
        if not lib_logger.handlers:
            lib_logger.addHandler(logging.StreamHandler(sys.stdout))

    t0 = time.perf_counter()
    pre_scores = _captcha_detection_scores(page, config)
    _log_captcha(
        logger,
        "captcha_detection_scores",
        context=context,
        page_url=page_url,
        phase="pre_classify",
        scores=pre_scores,
    )

    bypass = BypassCaptcha(page, config=config)
    notes = bypass.classification.classify()
    _log_captcha(
        logger,
        "captcha_classify_result",
        context=context,
        page_url=page_url,
        notes_count=len(notes),
        notes=_captcha_notes_payload(notes),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )

    if not notes:
        _log_captcha(
            logger,
            "captcha_bypass_done",
            context=context,
            page_url=page_url,
            solved=True,
            reason="no_captcha_above_threshold",
        )
        return True

    _log_captcha(
        logger,
        "captcha_solve_start",
        context=context,
        page_url=page_url,
        notes=_captcha_notes_payload(notes),
    )

    t_solve = time.perf_counter()
    try:
        solved = bypass.solving.solve(notes)
    except Exception as ex:
        _log_captcha(
            logger,
            "captcha_solve_error",
            context=context,
            page_url=page_url,
            error=str(ex),
            duration_ms=int((time.perf_counter() - t_solve) * 1000),
        )
        solved = False
    else:
        _log_captcha(
            logger,
            "captcha_solve_result",
            context=context,
            page_url=page_url,
            solved=bool(solved),
            duration_ms=int((time.perf_counter() - t_solve) * 1000),
        )

    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception as ex:
        _log_captcha(
            logger,
            "captcha_wait_load_state",
            context=context,
            page_url=page_url,
            ok=False,
            error=str(ex),
        )

    post_scores = _captcha_detection_scores(page, config)
    _log_captcha(
        logger,
        "captcha_detection_scores",
        context=context,
        page_url=getattr(page, "url", page_url),
        phase="post_solve",
        scores=post_scores,
    )

    _log_captcha(
        logger,
        "captcha_bypass_done",
        context=context,
        page_url=getattr(page, "url", page_url),
        solved=bool(solved),
        reason="solve_attempt_finished",
        total_duration_ms=int((time.perf_counter() - t0) * 1000),
    )
    return solved


# BU searches via search bar?
def search_web(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r["title"],
                "href": r["href"],
                "body": r["body"]
            })
    return results


# Navigation timeout (ms) - increased for slow sites
NAV_TIMEOUT = 45000
# Retry with lighter wait if initial load times out
NAV_RETRY_WAIT = "commit"
# Set USE_HEADED=1 or PLAYWRIGHT_HEADLESS=false for a visible browser (needs $DISPLAY).


def _launch_browser(playwright):
    """Launch browser, preferring real Chrome over Chromium to reduce bot detection."""
    headless = HEADLESS
    if not headless and not os.environ.get("DISPLAY"):
        print("USE_HEADED set but no $DISPLAY; falling back to headless")
        headless = True
    try:
        return playwright.chromium.launch(
            channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None,
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
            ],
        )
    except Exception:
        return playwright.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
            ],
        )


STEALTH_INIT_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    window.chrome = { runtime: {}, loadTimes: function() {}, csi: function() {} };
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'], configurable: true });
"""


def _goto_with_retry(page, url):
    """Navigate to URL with retry: try domcontentloaded first, fallback to commit."""
    try:
        return page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    except Exception:
        try:
            response = page.goto(url, wait_until=NAV_RETRY_WAIT, timeout=NAV_TIMEOUT)
            page.wait_for_timeout(2000)  # allow some content to load
            return response
        except Exception:
            raise


def _url_requires_http_status_check(url: str) -> bool:
    if _env_bool("REDENCE_SKIP_HTTP_STATUS_CHECK", False):
        return False
    return url.lower().startswith(("http://", "https://"))


def _probe_http_status(page, url: str) -> int | None:
    """Navigate to url and return HTTP status, or None if unavailable."""
    try:
        response = _goto_with_retry(page, url)
        return response.status if response else None
    except Exception:
        return None


def _write_http_status_file(screens_dir: Path, status: int | None) -> Path:
    path = screens_dir / "http_status.txt"
    path.write_text(f"{status if status is not None else 'unknown'}\n", encoding="utf-8")
    return path


def _save_screenshot(page, screens_dir, step_ref, label, selector=None):
    """Save screenshot to screens_dir if provided. If selector given, highlight that element with red outline.

    Returns the saved path as a string, or None if skipped/failed.
    """
    if not screens_dir or not step_ref:
        return None
    step_ref[0] += 1
    path = Path(screens_dir) / f"{step_ref[0]:03d}_{label}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if selector:
            locator = page.locator(selector).first
            try:
                locator.evaluate("""el => {
                    el.dataset._hlOutline = el.style.outline || '';
                    el.dataset._hlShadow = el.style.boxShadow || '';
                    el.style.outline = '4px solid #ff0000';
                    el.style.boxShadow = '0 0 0 4px rgba(255,0,0,0.5)';
                }""")
                page.wait_for_timeout(100)
                page.screenshot(path=str(path))
            finally:
                locator.evaluate("""el => {
                    el.style.outline = el.dataset._hlOutline || '';
                    el.style.boxShadow = el.dataset._hlShadow || '';
                }""")
        else:
            page.screenshot(path=str(path))
        return str(path)
    except Exception as e:
        logging.exception(f"Failed to save screenshot {path}: {e}")
        return None

# -------------------------
# 🧠 SET OF MARKS (SoM)
# -------------------------
def extract_marks(page, max_elements=40):
    elements = page.query_selector_all(
        "a, button, input, textarea, select, [role='button'], [role='link']"
    )

    marks = []

    for el in elements:
        if len(marks) >= max_elements:
            break

        try:
            if not el.is_visible():
                continue

            tag = el.evaluate("el => el.tagName.toLowerCase()")

            role = el.get_attribute("role")
            aria_label = el.get_attribute("aria-label")
            input_type = el.get_attribute("type")
            placeholder = el.get_attribute("placeholder")
            value = el.get_attribute("value")

            text = (
                el.inner_text().strip()
                or placeholder
                or value
                or aria_label
                or ""
            )

            if not text and tag not in ["input", "button"]:
                continue

            enabled = True
            try:
                enabled = not el.is_disabled()
            except Exception:
                pass

            # -------------------------
            # Spatial position
            # -------------------------
            try:
                box = el.bounding_box()
                position = {
                    "x": int(box["x"]) if box else None,
                    "y": int(box["y"]) if box else None,
                }
            except Exception:
                position = {"x": None, "y": None}

            # -------------------------
            # DOM container + section
            # -------------------------
            def get_ancestor_selector(node):
                return page.evaluate(
                    """el => {
                        function getSelector(el) {
                            if (!el) return null;
                            if (el.id) return el.tagName.toLowerCase() + "#" + el.id;
                            let path = [];
                            while (el && el.nodeType === 1) {
                                let selector = el.tagName.toLowerCase();
                                let sibling = el;
                                let nth = 1;
                                while (sibling = sibling.previousElementSibling) {
                                    if (sibling.tagName === el.tagName) nth++;
                                }
                                selector += `:nth-of-type(${nth})`;
                                path.unshift(selector);
                                el = el.parentElement;
                            }
                            return path.join(" > ");
                        }

                        let current = el;
                        while (current) {
                            const tag = current.tagName.toLowerCase();

                            if (["header", "nav", "main", "footer", "aside"].includes(tag)) {
                                return tag;
                            }

                            current = current.parentElement;
                        }

                        return "unknown";
                    }""",
                    el,
                )

            section = get_ancestor_selector(el)

            # container = nearest non-trivial parent (div with class/id if possible)
            container = page.evaluate(
                """el => {
                    function simplify(el) {
                        if (!el) return null;
                        if (el.id) return el.tagName.toLowerCase() + "#" + el.id;
                        if (el.className) {
                            const cls = el.className.toString().split(" ")[0];
                            if (cls) return el.tagName.toLowerCase() + "." + cls;
                        }
                        return el.tagName.toLowerCase();
                    }

                    let current = el.parentElement;
                    while (current) {
                        const tag = current.tagName.toLowerCase();

                        if (tag === "div" || tag === "section" || tag === "form") {
                            return simplify(current);
                        }

                        current = current.parentElement;
                    }

                    return null;
                }""",
                el,
            )

            selector = page.evaluate(
                """el => {
                    if (!el) return null;
                    if (el.id) return el.tagName.toLowerCase() + "#" + el.id;
                    let path = [];
                    while (el && el.nodeType === 1) {
                        let selector = el.tagName.toLowerCase();
                        let sibling = el;
                        let nth = 1;
                        while (sibling = sibling.previousElementSibling) {
                            if (sibling.tagName === el.tagName) nth++;
                        }
                        selector += `:nth-of-type(${nth})`;
                        path.unshift(selector);
                        el = el.parentElement;
                    }
                    return path.join(" > ");
                }""",
                el,
            )

            marks.append({
                "id": len(marks) + 1,

                # Semantic
                "text": text[:100],
                "tag": tag,
                "role": role,
                "aria_label": aria_label,
                "type": input_type,
                "placeholder": placeholder,

                # State
                "visible": True,
                "enabled": enabled,

                # DOM context
                "section": section,
                "container": container,

                # Spatial
                "position": position,

                # Fallback
                "selector": selector,
            })

        except Exception:
            continue

    return marks


def fetch_page(url, screens_dir=None, step_ref=None, logger=None):
    page = SESSION.get("page")
    if not page:
        return {"error": True, "message": "fetch_page_error : No active browser page session"}

    try: 
        if url != page.url: 
            _goto_with_retry(page, url)
            # page.wait_for_timeout(3000)   
        # Wait for  content
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                page.wait_for_timeout(3000)

        # -------------------------
        # CAPTCHA BYPASS (library classify + solve)
        # -------------------------
        handle_captcha(page, logger=logger, context="fetch_page")

        try:
            page.wait_for_function(
                """() => {
                    const imgs = Array.from(document.images);
                    if (!imgs.length) return true;
                    return imgs.every(i => i.complete);
                }""",
                timeout=10000,
            )
        except Exception:
            page.wait_for_timeout(1500)

        _save_screenshot(page, screens_dir, step_ref, "fetch")

        content = page.content()

        soup = BeautifulSoup(content, "html.parser")
        for s in soup(["script", "style"]):
            s.extract()

        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]

    except Exception as e:
        _save_screenshot(page, screens_dir, step_ref, "error")
        # Capture richer diagnostics
        err_type = type(e).__name__
        err_msg = str(e)

        # Optional: include current URL if navigation partially succeeded
        current_url = None
        try:
            current_url = page.url
        except Exception:
            pass

        error_payload = {
            "error": True,
            "type": err_type,
            "message": err_msg,
            "requested_url": url,
            "current_url": current_url,
        }

        return error_payload

def interact_page(url, steps, screens_dir=None, step_ref=None, logger=None):
    """
    Interact with a web page using Set-of-Marks (SoM).

    STRICT MODE:
    - EXACTLY ONE action per call.
    """

    # -------------------------
    # Enforce SINGLE ACTION
    # -------------------------
    if not isinstance(steps, dict):
        return "Error: interact_page now requires a single action dict, not a list"
    step = steps

    try:
        page = SESSION["page"]
        if not url:
            url = page.url

        # -------------------------
        # Navigate if needed (preserve in-session URL changes, e.g. search -> results)
        # -------------------------
        if url and page.url != url:
            if not page.url.startswith(url):
                _goto_with_retry(page, url)
                page.wait_for_timeout(3000)

        _save_screenshot(page, screens_dir, step_ref, "interact_initial")

        marks = extract_marks(page)

        results = []

        # -------------------------
        # Resolve step
        # -------------------------
        action = step.get("action")
        mark_id = step.get("id")
        selector = step.get("selector")

        if mark_id:
            match = next((m for m in marks if m["id"] == mark_id), None)
            if not match:
                return f"Invalid id: {mark_id}"
            selector = match["selector"]

        if action != "press" and not selector:
            return "Error: missing selector or id"

        # -------------------------
        # PRE-ACTION VALIDATION
        # -------------------------
        if action != "press":
            locator = page.locator(selector)

            if locator.count() == 0:
                return f"Validation failed: Element not found ({selector})"

            first = locator.first
            try:
                first.scroll_into_view_if_needed(timeout=2000)
            except:
                logging.exception(f"Validation failed: Element not visible ({selector})")

            #if not first.is_enabled():
            #    return f"Validation failed: Element not interactable ({selector})"

        # -------------------------
        # Execute ONE action
        # -------------------------
        try:
            if action == "click":
                first.scroll_into_view_if_needed(timeout=5000)

                _save_screenshot(
                    page,
                    screens_dir,
                    step_ref,
                    f"click_{_safe_filename(str(mark_id or selector)[:30])}",
                    selector=selector,
                )

                first.click(timeout=5000, force=True)
                page.wait_for_timeout(500)

                results.append(f"Clicked: {mark_id or selector}")

            elif action == "fill":
                value = step.get("value", "")

                _save_screenshot(
                    page,
                    screens_dir,
                    step_ref,
                    f"fill_{_safe_filename(str(mark_id or selector)[:30])}",
                    selector=selector,
                )

                first.fill(value, timeout=5000)
                page.wait_for_timeout(300)

                results.append(f"Filled {mark_id or selector}")

            elif action == "press":
                key = step.get("key", "Enter")

                if selector:
                    locator = page.locator(selector)

                    if locator.count() > 0:
                        first = locator.first

                        if not first.is_visible():
                            return f"Validation failed: Element not visible ({selector})"

                        if not first.is_enabled():
                            return f"Validation failed: Element not interactable ({selector})"

                        first.focus(timeout=5000)

                        _save_screenshot(
                            page,
                            screens_dir,
                            step_ref,
                            f"press_{key}",
                            selector=selector,
                        )

                page.keyboard.press(key)
                page.wait_for_timeout(500)

                results.append(
                    f"Pressed {key}" + (f" on {mark_id or selector}" if selector else "")
                )

            else:
                return f"Unknown action: {action}"

        except Exception as ex:
            _save_screenshot(page, screens_dir, step_ref, "error")
            return f"Error on {action}: {ex}"

        # -------------------------
        # Final state (ALWAYS)
        # -------------------------
        page.wait_for_timeout(1000)

        # -------------------------
        # CAPTCHA BYPASS (library classify + solve)
        # -------------------------
        handle_captcha(page, logger=logger, context="interact_page")

        content = page.content()


        interact_final_path = _save_screenshot(page, screens_dir, step_ref, "interact_final")
        
        # add screenshot check to detect if worked
        screenshot = take_screenshot(
            label="interact_final_vision_check",
            screens_dir=screens_dir,
            step_ref=step_ref,
            logger=logger,
        )

        vision_changed = screenshot.get("vision_changed", False)
        
        # -------------------------
        # Clean text
        # -------------------------
        soup = BeautifulSoup(content, "html.parser")
        for s in soup(["script", "style"]):
            s.extract()

        text = soup.get_text(separator=" ", strip=True)[:3000]
        # -------------------------
        # Extract new SoM for agent
        # -------------------------
        marks = extract_marks(page)

        marks_text = "\n".join( [
                f"[{m['id']}] {m['text']} "
                f"(tag={m['tag']}, role={m['role']}, type={m['type']}, "
                f"section={m['section']}, container={m['container']}, "
                f"pos={m['position']})"
                for m in marks
            ]    )   # option insetad of marks to reduce tokens
        
        '''Tried richer: [f"[{m['id']}] {m['text']}" for m in marks] but seems to confuse/break so see why'''
        return json.dumps({
                "type": "page_state",
                "url": page.url,
                "title": page.title(),
                "text": text,
                "marks": marks,
                "action_result": {
                "success": True,
                "error": None
                 },
                "interact_final_path": interact_final_path,
                "vision_check_path": screenshot.get("path"),
                "vision": {
                      "changed": vision_changed,
                 },
               })
    except Exception as e:
        logging.exception(f"Failed to interact: {e}")
        return f"Failed to interact: {e}"

def _safe_filename(s):
    """Make string safe for use in filenames."""
    return re.sub(r"[^\w\-.]", "_", s)[:30]

def take_screenshot(label="snapshot", screens_dir=None, step_ref=None, logger=None):
    """
    Capture a screenshot of the current page in the ACTIVE browser session.

    - Does not navigate; preserves cookies, login, and post-interaction DOM state
    - Saves full-page PNG for user/debugging
    - Optionally returns base64 JPEG for vision models
    """

    if not screens_dir:
        screens_dir = SCREENS_DIR

    page = SESSION.get("page")
    if not page:
        return {"error": "No active browser session", "label": label}

    try:
        vision_changed = False
        # -------------------------
        # File naming
        # -------------------------
        if step_ref is not None:
            step_ref[0] += 1
            filename = f"{step_ref[0]:03d}_{_safe_filename(label)}.png"
        else:
            filename = f"{_safe_filename(label)}.png"

        path = Path(screens_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            page.wait_for_function(
                """() => {
                    const imgs = Array.from(document.images);
                    if (!imgs.length) return true;
                    return imgs.every(i => i.complete);
                }""",
                timeout=10000,
            )
        except Exception:
            page.wait_for_timeout(1500)

        # -------------------------
        # Save PNG (user-facing)
        # -------------------------
        page.screenshot(path=str(path), full_page=True)

        # -------------------------
        # Optional vision JPEG (model-facing)
        # -------------------------
        vision_b64 = None
        vision_enabled = REDENCE_VISION_ENABLED
        vision_jpeg_error = None

        if vision_enabled:
            try:
                jpg_bytes = page.screenshot(type="jpeg", quality=45, full_page=False)
                vision_b64 = base64.b64encode(jpg_bytes).decode("ascii")
                # TRYING VISION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                global LAST_SCREENSHOT_B64

                vision_changed = False

                if LAST_SCREENSHOT_B64:
                    # for convenience compare first N bytes
                    vision_changed = vision_b64[:500] != LAST_SCREENSHOT_B64[:500]

                LAST_SCREENSHOT_B64 = vision_b64
            
            except Exception as e:
                vision_b64 = None
                vision_jpeg_error = str(e) or "jpeg_screenshot_failed"
                if logger:
                    _log_event(logger, "screenshot_vision_jpeg_error", error=vision_jpeg_error)

        # -------------------------
        # Extract text preview
        # -------------------------
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style"]):
            s.extract()

        text = soup.get_text(separator=" ", strip=True)[:1500]

        result = {
            "path": str(path),
            "text_preview": text,
            "vision_b64_jpeg": vision_b64,
            "vision_enabled": vision_enabled,
            "vision_b64_len": len(vision_b64) if isinstance(vision_b64, str) else 0,
            "vision_jpeg_error": vision_jpeg_error,
            "vision_changed": vision_changed,
            "url": page.url,  # important: actual current URL
            "label": label,
        }

        if logger:
            _log_event(
                logger,
                "take_screenshot_done",
                url=page.url,
                label=label,
                path=str(path),
                text_preview=_truncate(text, 500),
                vision_enabled=vision_enabled,
                vision_b64_len=result["vision_b64_len"],
                vision_jpeg_error=vision_jpeg_error,
            )

        return result

    except Exception as e:
        current_url = None
        try:
            current_url = SESSION.get("page") and SESSION["page"].url
        except Exception:
            pass
        if logger:
            _log_event(logger, "take_screenshot_error", url=current_url, label=label, error=str(e))

        return {"error": str(e), "url": current_url, "label": label}


def _tool_result_from_capture(capture: dict, vision_delivered: bool = False) -> dict:
    """Lean JSON for model-facing tool messages (no embedded JPEG)."""
    if capture.get("error"):
        return capture
    return {
        "path": capture.get("path"),
        "text_preview": capture.get("text_preview"),
        "vision_changed": capture.get("vision_changed"),
        "vision_delivered": vision_delivered,
        "url": capture.get("url"),
        "label": capture.get("label"),
        "vision_jpeg_error": capture.get("vision_jpeg_error") if vision_delivered else None,
    }


def _attach_archive_screenshot_to_user_message(
    messages: list,
    *,
    url: str,
    archive_path: str | Path | None,
    logger: logging.Logger | None = None,
) -> bool:
    """Add oldest archive crawl PNG to the first user message (before prefetch)."""
    if not REDENCE_VISION_ENABLED or not REDENCE_ARCHIVE_SCREENSHOT_ENABLED:
        return False
    if not archive_path:
        return False

    try:
        _ensure_redence_package()
        from redence.archive_screenshot import load_archive_vision_b64
    except ImportError:
        return False

    vision_b64, png_path = load_archive_vision_b64(archive_path)
    if not vision_b64 or png_path is None:
        if logger:
            _log_event(
                logger,
                "archive_screenshot_missing",
                archive_path=str(archive_path),
                url=url,
            )
        return False

    if not messages or messages[0].get("role") != "user":
        return False

    user = dict(messages[0])
    prior = (user.get("content") or "").rstrip()
    user["content"] = (
        f"{prior}\n"
        """Archived landing-page screenshot from the pipeline crawl is attached.
It was captured earlier and may differ from the live page after prefetch.
Use it for an initial visual check of the landing page.
Use model_take_screenshot subsequently when you need the current browser view
(e.g. after form interaction or redirects)."""
    )
    user["images"] = [vision_b64]
    messages[0] = user

    if logger:
        _log_event(
            logger,
            "archive_screenshot_attached",
            url=url,
            archive_path=str(archive_path),
            png_path=str(png_path),
            vision_b64_len=len(vision_b64),
        )
    return True


def _append_model_vision_message(messages: list, capture: dict) -> bool:
    """Attach a multimodal user turn for Ollama after model_take_screenshot."""
    if not REDENCE_VISION_ENABLED:
        return False
    if capture.get("error"):
        return False

    vision_b64 = capture.get("vision_b64_jpeg")
    if not vision_b64:
        return False

    messages.append({
        "role": "user",
        "content": (
            f"Screenshot of the current browser view ({capture.get('url', 'unknown')}). "
            "Use this image together with prior marks and page text to inspect the page "
            "and understand it and/or decide the next step."
        ),
        "images": [vision_b64],
    })
    return True


def ask_model(messages):
    # Normalize tool_calls so arguments are dicts (Ollama expects dict, not JSON string)
    normalized_messages = [_normalize_tool_call_message(m) for m in messages]
    response = _ollama_client().chat(
        model=MODEL,
        messages=normalized_messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the internet for informatiom or cross-checking claims/information on website, "
                        "or brand-impersonation evidence. Do NOT use for WHOIS, registrar, DNS, "
                        "or SSL — pre-collected domain intelligence is already in the prompt. "
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_page",
                    "description": "Fetch the contents of a web page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"}
                        },
                        "required": ["url"]
                    },
                },
            },
            {
  "type": "function",
  "function": {
    "name": "interact_page",
    "description": (
      "Interact with a web page using Set-of-Marks (SoM). "
      "IMPORTANT RULES:\n"
      "- You MUST perform EXACTLY ONE action per call.\n"
      "- Do NOT provide multiple steps.\n"
      "- Each call represents a single interaction with the page.\n"
      "- After each action, you will receive updated page state.\n"
      #"- Do NOT use XPath or JQuery style selectors.\n\n"

      "The page will return a numbered list of interactive elements (marks). "
      "Each element has an 'id'. Use this 'id' for actions instead of raw CSS selectors whenever possible.\n\n"

      "Preferred usage:\n"
      "- Use 'id' from the SoM list to click or fill elements\n"
      "- ALWAYS prefer 'id'\n"
      "- Use 'press' with a key like 'Enter' to submit forms\n\n"

      "Fallback:\n"
      "- ONLY use 'selector' if NO matching id exists\n"
      
      "Examples:\nTask: Search for laptop in search bar\n"
      "Click a button: {\"action\": \"click\", \"id\": 3}\n"
      "Fill input: {\"action\": \"fill\", \"id\": 5, \"value\": \"laptop\"}\n"
      "Submit search: {\"action\": \"press\", \"id\": 5, \"key\": \"Enter\"}"
    ),
    "parameters": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string",
          "description": "URL of the page to interact with"
        },
        "action": {
          "type": "string",
          "enum": ["click", "fill", "press"],
          "description": "Type of interaction (exactly one per call)"
        },
        "id": {
          "type": "integer",
          "description": "SoM element id (preferred over selector)"
        },
        "selector": {
          "type": "string",
          "description": "CSS selector (fallback if id is unavailable)"
        },
        "value": {
          "type": "string",
          "description": "Text to input (required for 'fill')"
        },
        "key": {
          "type": "string",
          "description": "Keyboard key to press (e.g., 'Enter')"
        }
      },
      "required": ["url", "action"]
    }
  }
},
            {
                "type": "function",
                "function": {
                    "name": "model_take_screenshot",
                    "description": "Capture a screenshot of the current browser view for visual inspection. "
                    "Does not navigate; use fetch_page first if you need to load a different URL. "
                    "You will receive the screenshot image on the next turn. "
                    "Use when marks/text are not enough (modals, CAPTCHA UI, layout, errors). "
                    "Use when you need to visually inspect the current page. "
                    "Returns the screenshot path, the actual current URL, and a short text preview.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short label for the screenshot filename",
                            },
                        },
                        "required": [],
                    },
                },
            },
           
        ],
    )

    return response


def _normalize_tool_call_message(msg):
    """Ensure tool_calls have arguments as dict (Ollama expects dict, not JSON string)."""
    if "tool_calls" not in msg:
        return msg
    normalized = dict(msg)
    normalized["tool_calls"] = []
    for tc in msg["tool_calls"]:
        tc_copy = dict(tc)
        func = tc_copy.get("function", {})
        if isinstance(func, dict):
            func_copy = dict(func)
            args = func_copy.get("arguments")
            if isinstance(args, str):
                func_copy["arguments"] = json.loads(args) if args else {}
            normalized["tool_calls"].append({**tc_copy, "function": func_copy})
        else:
            normalized["tool_calls"].append(tc_copy)
    return normalized


def _resolve_screens_dir() -> Path:
    override = os.environ.get("REDENCE_SCREENS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "screens" / "p2"


SCREENS_DIR = _resolve_screens_dir()


class AgentRunError(Exception):
    """Raised when the agent cannot start or complete a run."""


def run_agent(
    url: str,
    *,
    prompt: str | None = None,
    archive_path: str | Path | None = None,
) -> Path:
    """
    Run the agent for a single URL from the phish pipeline (or manual --url).

    ``archive_path`` is the crawl run root from the DB (playwright/<slug>/); when set,
    the oldest PNG under its screenshots/ folder is attached to the initial user turn.

    Returns the run output directory under screens/. Raises AgentRunError on failure.
    """
    url = (url or "").strip()
    if not url:
        raise AgentRunError("empty url")

    if prompt is None:
        _ensure_redence_package()
        from redence.agent_config import build_agent_prompt

        prompt = build_agent_prompt(url)

    messages = [{"role": "user", "content": prompt}]

    # Create screens folder for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    screens_dir = SCREENS_DIR / f"run_{run_id}"
    screens_dir.mkdir(parents=True, exist_ok=True)
    step_ref = [0]
    logger = _setup_run_logger(screens_dir)

    archive_vision_attached = _attach_archive_screenshot_to_user_message(
        messages,
        url=url,
        archive_path=archive_path,
        logger=logger,
    )

    _log_event(
        logger,
        "run_start",
        model=MODEL,
        ollama_host=_ollama_host() or "http://127.0.0.1:11434 (default)",
        prompt=_truncate(prompt, 4000),
        screens_dir=str(screens_dir),
        archive_path=str(archive_path) if archive_path else None,
        archive_vision_attached=archive_vision_attached,
    )

    # -------------------------
    # 🧠 Initialize persistent browser session ONCE
    # -------------------------
    from playwright.sync_api import sync_playwright

    timer_stop = threading.Event()
    _cleanup_browser_session()

    try:
        try:
            p = sync_playwright().start()
            SESSION["playwright"] = p
            browser = _launch_browser(p)
            SESSION["browser"] = browser
        except Exception as e:
            logging.exception(f"Failed to start Playwright: {e}")
            raise AgentRunError(f"Failed to start Playwright: {e}") from e

        try:
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=BROWSER_HEADERS["User-Agent"],
                extra_http_headers={
                    "Accept-Language": BROWSER_HEADERS["Accept-Language"],
                },
                locale="en-US",
                java_script_enabled=True,
                ignore_https_errors=True,
            )

            context.set_default_navigation_timeout(NAV_TIMEOUT)
            context.add_init_script(STEALTH_INIT_SCRIPT)

            page = context.new_page()
            SESSION["context"] = context
            SESSION["page"] = page
        except Exception as e:
            logging.exception("Failed to initialize browser context/page: %s", e)
            raise AgentRunError(f"Failed to initialize browser context/page: {e}") from e

        # -------------------------
        # Timer
        # -------------------------
        def _timer_thread():
            start = time.time()
            while not timer_stop.wait(1):
                elapsed = int(time.time() - start)
                mins, secs = divmod(elapsed, 60)
                print(f"\rTime elapsed: {mins}:{secs:02d}  ", end="", flush=True)

        timer = threading.Thread(target=_timer_thread, daemon=True)
        timer.start()

        output_lines = [f"=== User Query ===\n{prompt}\n"]

        # -------------------------
        # HTTP status gate (before any model call)
        # -------------------------
        if url and _url_requires_http_status_check(url):
            http_status = _probe_http_status(page, url)
            _log_event(logger, "http_status_check", url=url, status=http_status)
            if http_status != 200:
                status_path = _write_http_status_file(screens_dir, http_status)
                output_lines.append(
                    f"\n=== Skipped model run ===\n"
                    f"HTTP status {http_status if http_status is not None else 'unknown'} "
                    f"(expected 200). Wrote {status_path.name}.\n"
                )
                (screens_dir / "output.txt").write_text(
                    "\n".join(output_lines),
                    encoding="utf-8",
                )
                _log_event(
                    logger,
                    "run_end",
                    skipped_model=True,
                    reason="http_status_not_200",
                    http_status=http_status,
                    http_status_file=str(status_path),
                )
                return screens_dir

        # -------------------------
        # Prefetch URL (explicit from pipeline, not parsed from prompt)
        # -------------------------
        _log_event(logger, "prefetch_url", url=url)
        if url:

            t0 = time.perf_counter()
            page_content = fetch_page(
                url,
                screens_dir=screens_dir,
                step_ref=step_ref,
                logger=logger,
            )

            _log_event(
                logger,
                "prefetch_fetched",
                url=url,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                content_preview=_truncate(page_content, 800),
            )
            try:
                page_marks = extract_marks(page)
            except Exception as e:
                _log_event(logger, "prefetch_marks_error", url=url, error=str(e))
                page_marks = []

            archive_note = ""
            if archive_vision_attached:
                archive_note = (
                    "- An archived crawl-time landing screenshot is already attached to the user message; "
                    "the live content below may differ.\n"
                )

            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"""
The user wants to check {url}.

The page has been fetched:

{page_content}

The marks for the current page are provided:
{page_marks}


IMPORTANT:
- Use the existing context
- Do NOT call fetch_page again unless absolutely necessary
- Prefer interact_page with ONE action per call
{archive_note}- Use model_take_screenshot when you need to visually inspect the **current** live page (after interactions), not to duplicate the archived landing view

""",
                },
            )

        # -------------------------
        # Main agent loop
        # -------------------------
        while True:
            step_start = time.perf_counter()
            step_idx = len([m for m in messages if m.get("role") == "tool"]) + 1

            _log_event(logger, "step_start", step=step_idx)
            response = ask_model(messages)
            message = response["message"]
            _log_model_thinking(logger, output_lines, message, step=step_idx)

            # -------------------------
            # FINAL ANSWER
            # -------------------------
            if "tool_calls" not in message:
                timer_stop.set()
                print("\nFinal Answer:\n")
                print(message["content"])

                output_lines.append(f"\n=== Final Answer ===\n{message['content']}")

                (screens_dir / "output.txt").write_text(
                    "\n".join(output_lines),
                    encoding="utf-8",
                )

                thinking = (_message_get(message, "thinking") or "")
                if not isinstance(thinking, str):
                    thinking = str(thinking)
                _log_event(
                    logger,
                    "run_end",
                    final_answer_preview=_truncate(message.get("content", ""), 2000),
                    thinking_len=len(thinking.strip()),
                    thinking_preview=_truncate(thinking.strip(), 2000) if thinking.strip() else None,
                )
                break

            # -------------------------
            # Log reasoning
            # -------------------------
            if message.get("content"):
                output_lines.append("\n=== Agent Thinking ===")
                output_lines.append(message["content"])

            # Normalize assistant message
            messages.append(_normalize_tool_call_message(message))

            # -------------------------
            # Execute tools
            # -------------------------
            for tool in message["tool_calls"]:
                name = tool["function"]["name"]
                raw_args = tool["function"]["arguments"]
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception as e:
                    result = {"error": True, "message": f"Invalid JSON arguments: {e}", "raw_args": raw_args}
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result)
                    })
                    continue

                _log_event(logger, "tool_start", tool=name, arguments=args)

                t0 = time.perf_counter()
                capture_for_vision = None

                try:
                    # -------------------------
                    # WEB SEARCH
                    # -------------------------
                    if name == "web_search":
                        try:
                            result = search_web(args["query"])
                        except Exception as e:
                            result = {"error": True, "message": str(e)}

                    # -------------------------
                    # FETCH PAGE
                    # -------------------------
                    elif name == "fetch_page":
                        result = fetch_page(
                            args["url"],
                            screens_dir=screens_dir,
                            step_ref=step_ref,
                            logger=logger,
                        )

                    # -------------------------
                    # INTERACT PAGE (UPDATED)
                    # -------------------------
                    elif name == "interact_page":
                        result = interact_page(
                            args.get("url"),
                            args,
                            screens_dir=screens_dir,
                            step_ref=step_ref,
                            logger=logger,
                        )

                    # -------------------------
                    # MODEL SCREENSHOT (vision delivered to model)
                    # -------------------------
                    elif name == "model_take_screenshot":
                        capture_for_vision = take_screenshot(
                            label=args.get("label", "snapshot"),
                            screens_dir=screens_dir,
                            step_ref=step_ref,
                            logger=logger,
                        )
                        will_deliver_vision = (
                            REDENCE_VISION_ENABLED
                            and not capture_for_vision.get("error")
                            and bool(capture_for_vision.get("vision_b64_jpeg"))
                        )
                        result = _tool_result_from_capture(
                            capture_for_vision,
                            vision_delivered=will_deliver_vision,
                        )

                    else:
                        result = f"Unknown tool: {name}"

                    dur = int((time.perf_counter() - t0) * 1000)

                    tool_end_fields = {"tool": name, "duration_ms": dur}
                    if name == "web_search":
                        if isinstance(result, dict) and result.get("error"):
                            tool_end_fields["error"] = result.get("message")
                        elif isinstance(result, list):
                            tool_end_fields["result_count"] = len(result)
                            tool_end_fields["results"] = [
                                {
                                    "title": r.get("title"),
                                    "href": r.get("href"),
                                    "body": _truncate(r.get("body"), 500),
                                }
                                for r in result
                            ]
                    elif name == "fetch_page":
                        if isinstance(result, dict) and result.get("error"):
                            tool_end_fields["error"] = result.get("message")
                            if result.get("current_url"):
                                tool_end_fields["url"] = result.get("current_url")
                        else:
                            preview = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                            tool_end_fields["result_preview"] = _truncate(preview, 800)
                    elif name == "model_take_screenshot":
                        src = result if isinstance(result, dict) else {}
                        tool_end_fields["screenshot_path"] = src.get("path") or (
                            capture_for_vision or {}
                        ).get("path")
                        tool_end_fields["url"] = src.get("url") or (capture_for_vision or {}).get("url")
                        tool_end_fields["label"] = src.get("label") or (capture_for_vision or {}).get("label")
                        tool_end_fields["vision_changed"] = src.get("vision_changed")
                        tool_end_fields["vision_delivered"] = src.get("vision_delivered")
                        tool_end_fields["vision_b64_len"] = (capture_for_vision or {}).get("vision_b64_len")
                        if src.get("error") or (capture_for_vision or {}).get("error"):
                            tool_end_fields["error"] = src.get("error") or (capture_for_vision or {}).get("error")
                    elif name == "interact_page":
                        parsed = None
                        if isinstance(result, dict):
                            parsed = result
                        elif isinstance(result, str):
                            try:
                                parsed = json.loads(result)
                            except Exception:
                                parsed = None
                        if isinstance(parsed, dict) and parsed.get("type") == "page_state":
                            tool_end_fields["url"] = parsed.get("url")
                            tool_end_fields["interact_final_path"] = parsed.get("interact_final_path")
                            tool_end_fields["vision_check_path"] = parsed.get("vision_check_path")
                            tool_end_fields["page_state_changed"] = (
                                (parsed.get("vision") or {}).get("changed")
                            )
                            tool_end_fields["result_preview"] = _truncate(parsed.get("text"), 800)
                        elif isinstance(result, str):
                            tool_end_fields["result_preview"] = _truncate(result, 800)
                            if result.startswith("Error") or result.startswith("Failed") or result.startswith("Validation"):
                                tool_end_fields["error"] = result

                    _log_event(logger, "tool_end", **tool_end_fields)

                except Exception as ex:
                    dur = int((time.perf_counter() - t0) * 1000)

                    _log_event(
                        logger,
                        "tool_error",
                        tool=name,
                        duration_ms=dur,
                        error=str(ex),
                    )

                    result = f"Error: {ex}"

                # -------------------------
                # Feed result back to model
                # -------------------------
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False)
                        if not isinstance(result, str)
                        else result,
                    }
                )

                if name == "model_take_screenshot" and capture_for_vision:
                    vision_delivered = _append_model_vision_message(messages, capture_for_vision)
                    _log_event(
                        logger,
                        "model_vision_attached",
                        url=capture_for_vision.get("url"),
                        vision_delivered=vision_delivered,
                        vision_b64_len=len(capture_for_vision.get("vision_b64_jpeg") or ""),
                    )

            _log_event(
                logger,
                "step_end",
                duration_ms=int((time.perf_counter() - step_start) * 1000),
            )

        return screens_dir
    finally:
        timer_stop.set()
        _cleanup_browser_session()
        _cleanup_run_logger(logger)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Marketing analyst web agent")
    parser.add_argument(
        "--url",
        help="Run once for this URL (uses pipeline prompt template)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="REPL: free-form prompts (extracts URL from text when possible)",
    )
    args = parser.parse_args()

    if args.interactive:
        while True:
            try:
                q = input("\nAsk: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            target_url = extract_url_from_prompt(q)
            if target_url:
                run_agent(target_url, prompt=q)
            else:
                print("No URL found in prompt; use --url or include a link/domain.")
        return

    if args.url:
        run_dir = run_agent(args.url)
        print(f"\nRun output: {run_dir}")
        return

    parser.print_help()
    print("\nFor DB-driven batch processing, use: python -m redence.run_agent_pipeline")


if __name__ == "__main__":
    _main()
