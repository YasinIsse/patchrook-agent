"""Setup refuses unsupported decision servers before installation or model pulls."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from threading import Thread

import pytest


@pytest.mark.parametrize('version', ['0.34.9', 'unrecognized', None])
def test_setup_rejects_unsupported_server_before_installing(tmp_path, version):
    class VersionServer(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == '/api/version'
            content = json.dumps({'version': version}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    script = tmp_path / 'checkout/scripts/setup.sh'
    script.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parents[1] / 'scripts/setup.sh', script)
    commands = tmp_path / 'bin'
    commands.mkdir()
    calls = tmp_path / 'ollama-calls'
    for name in ('git', 'rg'):
        tool = commands / name
        tool.write_text('#!/bin/sh\nexit 0\n')
        tool.chmod(0o755)
    ollama = commands / 'ollama'
    ollama.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> '
                      + shlex.quote(str(calls)) + '\n[ "$1" = list ]\n')
    ollama.chmod(0o755)
    server = ThreadingHTTPServer(('127.0.0.1', 0), VersionServer)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                   OLLAMA_HOST=f'http://127.0.0.1:{server.server_port}')
        result = subprocess.run(['bash', str(script), '--python', sys.executable],
                                env=env, capture_output=True, text=True, timeout=15)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

    assert result.returncode == 1
    assert 'Tev1 requires Ollama 0.35 or newer' in result.stderr
    assert calls.read_text().splitlines() == ['list']
    assert not (script.parent.parent / '.venv').exists()
