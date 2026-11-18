# Collect domain registrar / WHOIS / DNS / SSL metadata for a URL.

from __future__ import annotations

import socket
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = PACKAGE_DIR / "dataset"

INTEL_FILENAME = "whois_intel.txt"
WHOIS_TIMEOUT_SECONDS = 20
SSL_TIMEOUT_SECONDS = 15


def _normalize_netloc(url: str) -> str:
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    try:
        host = urlparse(u).netloc.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _domain_from_url(url: str) -> str:
    host = _normalize_netloc(url)
    if host:
        return host
    parsed = urlparse(url.strip())
    return (parsed.netloc or parsed.path or "unknown").lower()


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _section(title: str, body: str) -> str:
    line = "=" * 72
    text = body.rstrip()
    if not text:
        text = "(no data)"
    return f"{line}\n{title}\n{line}\n{text}\n\n"


def _collect_dns(domain: str) -> str:
    lines: list[str] = []
    for family, label in ((socket.AF_INET, "A"), (socket.AF_INET6, "AAAA")):
        try:
            results = socket.getaddrinfo(
                domain,
                None,
                family=family,
                type=socket.SOCK_STREAM,
            )
            addrs = sorted({item[4][0] for item in results})
            lines.append(f"{label} records: {', '.join(addrs) if addrs else '(none)'}")
        except socket.gaierror as exc:
            lines.append(f"{label} records: lookup failed ({exc})")
        except OSError as exc:
            lines.append(f"{label} records: error ({exc})")
    return "\n".join(lines)


def _collect_whois(domain: str) -> str:
    try:
        import whois
    except ImportError:
        return "python-whois is not installed (pip install python-whois)."

    try:
        socket.setdefaulttimeout(WHOIS_TIMEOUT_SECONDS)
        record = whois.whois(domain)
    except Exception as exc:
        return f"WHOIS lookup failed: {exc}"
    finally:
        socket.setdefaulttimeout(None)

    if record is None:
        return "WHOIS lookup returned no data."

    lines: list[str] = []
    for key in sorted(record.keys()):
        value = _format_value(record.get(key))
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "WHOIS lookup returned an empty record."


def _collect_ssl(domain: str) -> str:
    lines: list[str] = []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection((domain, 443), timeout=SSL_TIMEOUT_SECONDS) as raw:
            with ctx.wrap_socket(raw, server_hostname=domain) as tls:
                lines.append(f"TLS version: {tls.version()}")
                cipher = tls.cipher()
                if cipher:
                    lines.append(f"Cipher: {cipher[0]} ({cipher[1]}, {cipher[2]})")

                der = tls.getpeercert(binary_form=True)
                if not der:
                    lines.append("No certificate returned.")
                    return "\n".join(lines)

                pem = ssl.DER_cert_to_PEM_cert(der)
                lines.append("")
                lines.append("Certificate (PEM):")
                lines.append(pem.rstrip())
    except Exception as exc:
        lines.append(f"SSL/TLS probe failed: {exc}")

    return "\n".join(lines)


def intel_path_for_url(url: str, out_root: str | Path | None = None) -> Path:
    domain = _domain_from_url(url)
    root = Path(out_root) if out_root is not None else DEFAULT_OUT_ROOT
    return root / domain / INTEL_FILENAME


def read_intel_text(url: str, out_root: str | Path | None = None) -> str | None:
    """Return whois_intel.txt contents for the URL's domain, or None if missing."""
    path = intel_path_for_url(url, out_root)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def build_intel_report(url: str) -> str:
    """Return a plain-text report for the URL's domain."""
    url = url.strip()
    domain = _domain_from_url(url)
    parts = [
        _section("SUMMARY", f"URL: {url}\nDomain: {domain}\nCollected: {_utc_now_str()}"),
        _section("DNS", _collect_dns(domain)),
        _section("WHOIS / REGISTRAR", _collect_whois(domain)),
        _section("SSL / TLS", _collect_ssl(domain)),
    ]
    return "".join(parts)


def write_intel_file(url: str, out_root: str | Path) -> str:
    """
    Write whois_intel.txt under the domain archive folder.
    Returns the absolute path written.
    """
    domain = _domain_from_url(url)
    base_dir = Path(out_root) / domain
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / INTEL_FILENAME
    report = build_intel_report(url)
    out_path.write_text(report, encoding="utf-8")
    return str(out_path.resolve())








