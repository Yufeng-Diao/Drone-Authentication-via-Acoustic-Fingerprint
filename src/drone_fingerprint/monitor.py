"""Read-only loopback monitoring of paper sweeps, independent of the trainer."""

from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit

STATUSES = ('pending', 'running', 'completed', 'failed', 'interrupted')
ASSETS = {'/': ('monitor.html', 'text/html'), '/monitor.css': ('monitor.css', 'text/css'),
          '/monitor.js': ('monitor.js', 'text/javascript')}


def read_json(path, limit=32 * 1024 * 1024):
    with path.open('rb') as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f'Monitor JSON exceeds {limit} bytes: {path.name}')
    return json.loads(content)


def inside(root, relative):
    """Ledger paths may name only files inside the explicitly selected root."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Ledger path escapes the selected monitoring directory')
    return path


def log_tail(path, limit=32768):
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - limit))
            content = stream.read(limit)
        if size > limit and b'\n' in content:
            content = content.partition(b'\n')[2]
        return {'text': content.decode('utf-8', errors='replace'), 'truncated': size > limit, 'available': True}
    except FileNotFoundError:
        return {'text': 'Waiting for the attempt log at the configured log root.', 'truncated': False, 'available': False}


def lock_owner(output):
    """Observe the existing Linux flock without acquiring it or disturbing resume."""
    try:
        stat = (output / '.lock').stat()
        target = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
        for line in Path('/proc/locks').read_text().splitlines():
            parts = line.split()
            if len(parts) != 8 or parts[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
                continue
            major, minor, inode = parts[5].split(':')
            if (int(major, 16), int(minor, 16), int(inode)) == target:
                return int(parts[4])
        return None
    except (FileNotFoundError, PermissionError):
        return None


def process_metrics(pid):
    if pid is None:
        return None
    try:
        root = Path('/proc') / str(pid)
        # The process name in stat may contain spaces or parentheses.
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
        metrics = {'pid': pid, 'cpu_seconds': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
                   'rss_bytes': int(status['VmRSS'].split()[0]) * 1024}
        try:
            io = dict(line.split(':', 1) for line in (root / 'io').read_text().splitlines())
            metrics.update({key: int(io[key]) for key in ('read_bytes', 'write_bytes')})
        except PermissionError:
            pass
        return metrics
    except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError):
        return None


class Monitor:
    """Cache immutable plans and changed ledgers; never open WAVs, arrays or models."""

    def __init__(self, output: Path, logs: Path):
        self.output, self.logs = output.resolve(), logs.resolve()
        self.plan_signature = self.state_signature = None
        self.plan, self.state, self.rows, self.counts, self.coverage = None, None, [], {}, []
        self.lock = threading.Lock()

    def refresh(self):
        try:
            plan_path = inside(self.output, 'plan.json')
            state_path = inside(self.output, 'sweep.json')
            plan_stat, state_stat = plan_path.stat(), state_path.stat()
        except FileNotFoundError:
            self.state = None
            self.rows = []
            return
        plan_signature = (plan_stat.st_ino, plan_stat.st_mtime_ns, plan_stat.st_size)
        state_signature = (state_stat.st_ino, state_stat.st_mtime_ns, state_stat.st_size)
        if self.plan_signature == plan_signature and self.state_signature == state_signature and self.state is not None:
            return
        plan = self.plan if self.plan_signature == plan_signature else read_json(plan_path)
        state = read_json(state_path)
        if plan['schema_version'] != 1 or state['schema_version'] != 1:
            raise ValueError('Unsupported sweep schema')
        if set(state['tasks']) != {task['id'] for task in plan['tasks']} or len(state['tasks']) != len(plan['tasks']):
            raise ValueError('Plan and ledger task identities disagree')
        if state['status'] not in STATUSES:
            raise ValueError('Unknown sweep status')
        rows, grid = [], {}
        for task in plan['tasks']:
            entry, settings = state['tasks'][task['id']], task['settings']
            if entry['status'] not in STATUSES:
                raise ValueError('Unknown task status')
            key = (task['preset'], settings['model'])
            cell = grid.setdefault(key, {'preset': key[0], 'model': key[1], 'total': 0, 'completed': 0, 'failed': 0})
            cell['total'] += 1
            cell['completed'] += entry['status'] == 'completed'
            cell['failed'] += entry['status'] in ('failed', 'interrupted')
            rows.append({'id': task['id'], 'preset': task['preset'], 'kind': task['kind'], 'model': settings['model'],
                         'frame_ms': settings['frame_ms'], 'n_filters': settings['n_filters'],
                         'blocks': settings['blocks'], 'portion': settings['portion'],
                         'snr_db': None if settings['test_snr_quarter_db'] is None else settings['test_snr_quarter_db'] / 4,
                         'trial': settings['partition']['trial'] if settings['partition'] else None,
                         'status': entry['status'], 'attempts': len(entry['attempts'])})
        self.plan, self.state, self.rows = plan, state, rows
        self.counts = dict(Counter(row['status'] for row in rows))
        self.coverage = list(grid.values())
        self.plan_signature, self.state_signature = plan_signature, state_signature

    def detail(self, task_id):
        row = next((row for row in self.rows if row['id'] == task_id), None)
        if row is None:
            raise ValueError('Unknown task ID')
        entry = self.state['tasks'][task_id]
        result = {**row, 'error': entry.get('error'), 'history': entry['attempts'][-50:],
                  'history_truncated': len(entry['attempts']) > 50, 'progress': None,
                  'log': {'text': 'This task has no attempt yet.', 'available': False, 'truncated': False}}
        if not entry['attempts']:
            return result
        attempt = entry['attempts'][-1]
        directory = inside(self.output, attempt['directory'])
        progress = inside(self.output, str(directory / 'progress.json'))
        if progress.exists():
            result['progress'] = read_json(progress, 16384)
        # Basenames allow the same logs to be remounted at another container path.
        path = inside(self.logs, Path(attempt['log']).name)
        result['log'] = log_tail(path)
        result['started_at'] = attempt.get('started_at')
        result['finished_at'] = attempt.get('finished_at')
        return result

    def status(self, selected=None):
        with self.lock:
            self.refresh()
            now = datetime.now(timezone.utc).isoformat()
            response = {'observed_at': now, 'name': self.output.name, 'output': str(self.output),
                        'status': 'waiting', 'counts': {}, 'total': 0, 'coverage': [], 'selected': None,
                        'runner_pid': None, 'process': None, 'started_at': None, 'finished_at': None}
            if self.state is None:
                return response
            state = self.state
            active = next((row['id'] for row in self.rows if row['status'] == 'running'), None)
            if not selected:
                attempted = [row for row in self.rows if row['attempts']]
                problem = next((row['id'] for row in self.rows if row['status'] in ('failed', 'interrupted')), None)
                selected = active or problem or (attempted[-1]['id'] if attempted else None)
            pid = lock_owner(self.output) if state['status'] == 'running' else None
            status = 'unobserved' if state['status'] == 'running' and pid is None else state['status']
            detail = self.detail(selected) if selected else None
            active_detail = self.detail(active) if active and active != selected else detail
            response.update(status=status, ledger_status=state['status'], total=len(self.rows), counts=self.counts,
                            coverage=self.coverage, started_at=state['started_at'], finished_at=state.get('finished_at'),
                            runner_pid=pid, process=process_metrics(pid), active_task=active,
                            selected=detail, current_progress=active_detail['progress'] if active_detail else None)
            return response

    def tasks(self, status='', model='', search='', offset=0):
        with self.lock:
            self.refresh()
            rows = [row for row in self.rows if (not status or row['status'] == status)
                    and (not model or row['model'] == model)
                    and (not search or search.lower() in json.dumps(row).lower())]
            return {'total': len(rows), 'offset': offset, 'limit': 50, 'rows': rows[offset:offset + 50]}


def make_server(monitor, port=8765, token=None):
    token = token or secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Polls must not produce an unbounded access log.

        def respond(self, status, payload, content_type='application/json'):
            body = json.dumps(payload, allow_nan=False).encode() if content_type == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            host = self.headers.get('Host', '')
            try:
                hostname = urlsplit('http://' + host).hostname
            except ValueError:
                hostname = None
            if hostname not in ('127.0.0.1', 'localhost'):
                self.respond(403, {'error': 'Use the loopback URL or an SSH local port forward'})
                return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + host:
                self.respond(403, {'error': 'Cross-origin requests are not allowed'})
                return
            url = urlsplit(self.path)
            if url.path in ASSETS:
                filename, mime = ASSETS[url.path]
                self.respond(200, (Path(__file__).parent / 'web' / filename).read_bytes(), mime)
                return
            if url.path not in ('/api/status', '/api/tasks'):
                self.respond(404, {'error': 'Unknown endpoint'})
                return
            if not secrets.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer ' + token).encode()):
                self.respond(401, {'error': 'Open the complete URL printed by drone-auth monitor, including its token fragment'})
                return
            try:
                params = parse_qs(url.query)
                if url.path == '/api/status':
                    payload = monitor.status(params.get('task', [None])[0])
                else:
                    offset = int(params.get('offset', ['0'])[0])
                    if offset < 0:
                        raise ValueError('Offset must be nonnegative')
                    payload = monitor.tasks(*(params.get(k, [''])[0] for k in ('status', 'model', 'search')), offset=offset)
                self.respond(200, payload)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.respond(503, {'error': f'Cannot read a consistent sweep snapshot: {error}'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.monitor_token = token
    return server


def serve(output, logs, port):
    if not 1 <= port <= 65535:
        raise ValueError('Monitor port must be between 1 and 65535')
    with make_server(Monitor(output, logs), port) as server:
        print(f'Monitor: http://127.0.0.1:{server.server_port}/#{server.monitor_token}', flush=True)
        print('Read-only, localhost only. Polls every 3 seconds. Keep the token URL private.', flush=True)
        print('SSH: forward this port to 127.0.0.1 on the training host. Ctrl-C stops only the monitor.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('Monitor stopped; the training process was not modified.', flush=True)
