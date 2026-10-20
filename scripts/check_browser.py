"""Check actual WebGL 2 shaders in headless Chrome using only Python's stdlib."""
import argparse
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chrome', default=shutil.which('google-chrome') or shutil.which('chromium') or
                        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    parser.add_argument('--software', action='store_true', help='Use SwiftShader instead of the real GPU')
    args = parser.parse_args()
    if not Path(args.chrome).exists():
        parser.error('Chrome not found; supply --chrome /path/to/chrome')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = tempfile.mkdtemp(prefix='qclassify-browser-')
    process = None
    try:
        command = [args.chrome, '--headless=new', '--no-first-run', '--no-default-browser-check',
                   f'--user-data-dir={profile}', '--dump-dom', '--timeout=20000', '--virtual-time-budget=15000']
        if args.software:
            command += ['--enable-unsafe-swiftshader', '--use-angle=swiftshader']
        command += [f'http://127.0.0.1:{server.server_port}/web/test/browser.html']
        # Some Chrome helpers keep inherited pipes open after dump-dom. Use
        # files and stop the process group once the browser result is written.
        stdout_path, stderr_path = Path(profile) / 'stdout.html', Path(profile) / 'stderr.log'
        with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            deadline = time.monotonic() + 35
            body = ''
            while time.monotonic() < deadline:
                body = stdout_path.read_text()
                if re.search(r'<pre id="result">(?:PASS|FAIL) ', body) or process.poll() is not None:
                    break
                time.sleep(0.1)
        match = re.search(r'<pre id="result">PASS (.*?)</pre>', body, re.DOTALL)
        if not match:
            raise RuntimeError(f'Browser validation failed:\n{body[-6000:]}\n{stderr_path.read_text()[-2000:]}')
        report = json.loads(html.unescape(match.group(1)))
        (ROOT / 'artifacts/browser-validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        server.shutdown()
        server.server_close()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == '__main__':
    main()
