"""Tiny localhost receiver: saves POSTed bodies to krillion-slm/data/<path>.

Run: python probe/receive.py   (listens on 127.0.0.1:8765)
From the page: fetch('http://127.0.0.1:8765/reveal/2026-11-17.json', {method:'POST', mode:'no-cors', body})
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


class H(BaseHTTPRequestHandler):
    def do_OPTIONS(self):  # CORS + Private/Local Network Access preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_POST(self):
        rel = Path(self.path.lstrip("/"))
        if ".." in rel.parts or rel.is_absolute():
            self.send_response(400); self.end_headers(); return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        out = DATA / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        print(f"saved {out} ({len(body):,} bytes)", flush=True)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


HTTPServer(("127.0.0.1", 8765), H).serve_forever()
