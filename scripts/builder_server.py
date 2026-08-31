"""
builder_server.py — local helper that lets the Experience Builder's Export
button actually run scripts/export_experience.py.

    py -3.12 scripts/builder_server.py            # listen on 127.0.0.1:8798
    py -3.12 scripts/builder_server.py --port N

Leave it running in a terminal while authoring. The builder page (opened via
file:// as usual, or via http://127.0.0.1:8798/) detects it with GET /ping;
its Export dialog then shows a "Run export" button which:

  1. POSTs the current project JSON to /export,
  2. which is written to BHR_Experience.bhrx.json (the single source of
     truth stays in sync with what you exported),
  3. then runs export_experience.py on it, streaming the exporter's output
     back into the dialog as it happens.

Stdlib only — no pip installs. Binds to localhost only. One export runs at a
time (a second request gets 409).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER_DIR = ROOT / "tools" / "experience_builder"
PROJECT_PATH = ROOT / "BHR_Experience.bhrx.json"
EXPORT_SCRIPT = ROOT / "scripts" / "export_experience.py"
DEFAULT_PORT = 8798

_export_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    """Serves the builder statically; /ping and /export are the API."""

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # Chrome Private Network Access preflight (page -> localhost)
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/ping":
            body = json.dumps({"ok": True, "project": PROJECT_PATH.name}).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/export":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        if not _export_lock.acquire(blocking=False):
            self.send_response(409)
            self._cors()
            self.end_headers()
            self.wfile.write(b"an export is already running\n")
            return
        try:
            self._run_export()
        finally:
            _export_lock.release()

    def _run_export(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            project = payload["project"]
        except Exception as exc:
            self.send_response(400)
            self._cors()
            self.end_headers()
            self.wfile.write(f"bad request: {exc}\n".encode())
            return

        # The posted project becomes the canonical .bhrx — what you export
        # is exactly what is on disk afterwards.
        PROJECT_PATH.write_text(
            json.dumps(project, indent=2, ensure_ascii=False),
            encoding="utf-8")

        cmd = [sys.executable, str(EXPORT_SCRIPT), str(PROJECT_PATH)]
        if payload.get("no_frames"):
            cmd.append("--no-frames")

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # no Content-Length -> chunked/close-delimited streaming
        self.end_headers()

        def line(txt: str):
            try:
                self.wfile.write((txt.rstrip("\n") + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass   # client went away; let the export finish regardless

        line(f"[server] saved {PROJECT_PATH.name}")
        line(f"[server] running: {' '.join(cmd[1:])}")
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    bufsize=1)
            for out in proc.stdout:
                line(out)
            code = proc.wait()
        except Exception as exc:
            line(f"[server] FAILED to run the exporter: {exc}")
            code = 1
        line(f"[exit {code}]")

    def log_message(self, fmt, *args):     # quieter: API + errors only
        if "/ping" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    handler = partial(Handler, directory=str(BUILDER_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"[builder-server] serving {BUILDER_DIR.name}/ + export API on "
          f"http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    print(f"[builder-server] the Export dialog in the builder page will now "
          f"offer 'Run export'")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[builder-server] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
