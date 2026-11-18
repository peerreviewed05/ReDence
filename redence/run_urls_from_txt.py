"""Run the ReDence controller for a URL list with a per-site budget and resume log."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redence.controller import main as run_controller
from redence.config import load_env


def load_urls(path):
    urls = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        line = line.split(',', 1)[0].strip()
        url = line if line.startswith(('http://', 'https://')) else 'https://' + line
        if url not in urls:
            urls.append(url)
    return urls


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('urls_file', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/redence'))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--throttle', type=float, default=2)
    p.add_argument('--no-resume', action='store_true')
    args, controller_args = p.parse_known_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / 'progress.jsonl'
    finished = set()
    if progress.exists() and not args.no_resume:
        for line in progress.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get('status') == 'ok':
                    finished.add(row['url'])
            except (ValueError, KeyError):
                pass
    urls = load_urls(args.urls_file)[max(0, args.start-1):]
    if args.limit > 0:
        urls = urls[:args.limit]
    failures = 0
    for url in urls:
        if url in finished:
            continue
        folder = args.output_dir / hashlib.sha256(url.encode()).hexdigest()[:16] / str(time.time_ns())
        print(f'Investigating {url}', flush=True)
        code = run_controller(['--url', url, '--output-dir', str(folder), *controller_args])
        failures += bool(code)
        with progress.open('a') as stream:
            stream.write(json.dumps({'url': url, 'status': 'ok' if code == 0 else 'incomplete',
                                     'report': str(folder / 'report.json')}) + '\n')
        if args.throttle > 0:
            time.sleep(args.throttle)
    return 2 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
