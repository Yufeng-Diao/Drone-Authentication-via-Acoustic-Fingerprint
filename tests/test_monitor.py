"""Read-only monitoring, real Linux liveness, bounded data and HTTP access."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import http.client
import json
import os
from pathlib import Path
import threading

import pytest

from drone_fingerprint.monitor import Monitor, lock_owner, make_server, process_metrics, read_json
from drone_fingerprint.storage import atomic_json


def fixture(root, count=2):
    output, logs = root / 'run', root / 'logs'
    output.mkdir(); logs.mkdir()
    settings = {'model': 'GNB', 'frame_ms': 1000, 'n_filters': 50, 'blocks': 1,
                'portion': 3, 'partition': None, 'test_snr_quarter_db': None}
    tasks = [{'id': f'task-{i}', 'preset': 'table2', 'kind': 'fit', 'settings': settings} for i in range(count)]
    state = {'schema_version': 1, 'started_at': datetime.now(timezone.utc).isoformat(),
             'status': 'pending', 'tasks': {task['id']: {'status': 'pending', 'attempts': []} for task in tasks}}
    atomic_json(output / 'plan.json', {'schema_version': 1, 'tasks': tasks})
    atomic_json(output / 'sweep.json', state)
    return Monitor(output, logs), state


def attempt(monitor, state, key='task-0'):
    directory = monitor.output / key / 'attempt-001'
    directory.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    state['status'] = 'running'
    state['tasks'][key] = {'status': 'running', 'attempts': [
        {'directory': str(directory.relative_to(monitor.output)), 'log': '/other/mount/current.log',
         'status': 'running', 'started_at': now}]}
    atomic_json(directory / 'progress.json', {'phase': 'training', 'status': 'running', 'at': now, 'pid': 999999})
    atomic_json(monitor.output / 'sweep.json', state)
    (monitor.logs / 'current.log').write_text('training: started\n')
    return directory


def test_waiting_does_not_create_run_or_logs(tmp_path):
    monitor = Monitor(tmp_path / 'future', tmp_path / 'future-logs')
    assert monitor.status()['status'] == 'waiting'
    assert monitor.tasks()['rows'] == []
    assert list(tmp_path.iterdir()) == []


def test_actual_lock_owner_metrics_and_released_runner(tmp_path):
    monitor, state = fixture(tmp_path)
    attempt(monitor, state)
    with (monitor.output / '.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        assert lock_owner(monitor.output) == os.getpid()
        snapshot = monitor.status()
        assert snapshot['status'] == 'running' and snapshot['runner_pid'] == os.getpid()
        assert snapshot['process']['cpu_seconds'] >= 0 and snapshot['process']['rss_bytes'] > 0
        assert snapshot['selected']['progress']['phase'] == 'training'
        assert lock_owner(monitor.output) == os.getpid()  # Observation never releases the runner's lock.
    assert monitor.status()['status'] == 'unobserved'
    assert monitor.status()['process'] is None
    assert process_metrics(999999999) is None


def test_live_phase_log_and_completed_updates_without_writes(tmp_path):
    monitor, state = fixture(tmp_path)
    directory = attempt(monitor, state)
    monitor.status()
    atomic_json(directory / 'progress.json', {'phase': 'evaluation', 'status': 'running', 'at': state['started_at']})
    (monitor.logs / 'current.log').write_text('evaluation: started\n<script>bad()</script>\n')
    updated = monitor.status()
    assert updated['selected']['progress']['phase'] == 'evaluation'
    assert '<script>' in updated['selected']['log']['text']  # Browser renders this with textContent.
    state['tasks']['task-0']['status'] = 'completed'
    state['tasks']['task-0']['attempts'][-1]['status'] = 'completed'
    state['status'] = 'completed'
    atomic_json(monitor.output / 'sweep.json', state)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob('*') if p.is_file()}
    assert monitor.status()['counts'] == {'completed': 1, 'pending': 1}
    assert monitor.status()['status'] == 'completed'
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in before}


def test_full_grid_pagination_filters_and_cached_plan(tmp_path, monkeypatch):
    monitor, state = fixture(tmp_path, 8440)
    state['tasks']['task-100']['status'] = 'failed'
    atomic_json(monitor.output / 'sweep.json', state)
    assert monitor.status()['total'] == 8440
    import drone_fingerprint.monitor as module
    def forbidden(*args):
        pytest.fail('Unchanged plan and ledger must not be reread')
    monkeypatch.setattr(module, 'read_json', forbidden)
    assert len(monitor.tasks(offset=50)['rows']) == 50
    assert monitor.tasks(status='failed')['rows'][0]['id'] == 'task-100'
    assert monitor.tasks(model='RF')['total'] == 0
    assert monitor.tasks(search='task-8439')['total'] == 1


def test_log_tail_and_attempt_history_are_bounded(tmp_path):
    monitor, state = fixture(tmp_path)
    attempt(monitor, state)
    entry = state['tasks']['task-0']
    entry['attempts'] = [dict(entry['attempts'][0], status='interrupted') for _ in range(60)]
    entry['attempts'][-1]['status'] = 'running'
    atomic_json(monitor.output / 'sweep.json', state)
    (monitor.logs / 'current.log').write_text('Earlier line\n' * 8000 + 'Newest line\n')
    result = monitor.status()['selected']
    assert result['history_truncated'] and len(result['history']) == 50
    assert result['log']['truncated'] and len(result['log']['text'].encode()) <= 32768
    assert result['log']['text'].endswith('Newest line\n')


@pytest.mark.parametrize('target', ['attempt', 'progress', 'log'])
def test_paths_cannot_escape_configured_roots(tmp_path, target):
    monitor, state = fixture(tmp_path)
    directory = attempt(monitor, state)
    outside = tmp_path / 'private.json'; outside.write_text('{}')
    if target == 'attempt':
        state['tasks']['task-0']['attempts'][0]['directory'] = '../outside'
        atomic_json(monitor.output / 'sweep.json', state)
    elif target == 'progress':
        (directory / 'progress.json').unlink()
        (directory / 'progress.json').symlink_to(outside)
    else:
        (monitor.logs / 'current.log').unlink()
        (monitor.logs / 'current.log').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        monitor.status()


def test_corruption_is_not_reported_as_a_fresh_cached_success(tmp_path):
    monitor, state = fixture(tmp_path)
    monitor.status()
    (monitor.output / 'sweep.json').write_text('{')
    with pytest.raises(ValueError):
        monitor.status()
    atomic_json(monitor.output / 'sweep.json', state)
    assert monitor.status()['status'] == 'pending'
    with pytest.raises(ValueError, match='exceeds'):
        read_json(monitor.output / 'sweep.json', limit=10)


def test_resume_validation_failure_is_selected_before_old_later_completions(tmp_path):
    monitor, state = fixture(tmp_path)
    attempt(monitor, state, 'task-0')
    attempt(monitor, state, 'task-1')
    state['tasks']['task-1']['status'] = 'completed'
    state['tasks']['task-0'].update(status='failed', error='Completed artifact checksum mismatch')
    state['status'] = 'failed'
    atomic_json(monitor.output / 'sweep.json', state)
    assert monitor.status()['selected']['id'] == 'task-0'
    assert monitor.status()['selected']['error'] == 'Completed artifact checksum mismatch'


@contextmanager
def serving(monitor):
    with make_server(monitor, port=0, token='test-secret') as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown(); thread.join()


def request(server, path, headers=None, method='GET'):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_http_token_origin_routes_and_read_only_methods(tmp_path):
    monitor, state = fixture(tmp_path)
    with serving(monitor) as server:
        assert server.server_address[0] == '127.0.0.1'
        for route in ('/', '/monitor.css', '/monitor.js'):
            status, headers, body = request(server, route)
            assert status == 200 and body and headers['Cache-Control'] == 'no-store'
        auth = {'Authorization': 'Bearer test-secret'}
        assert request(server, '/api/status')[0] == 401
        assert request(server, '/api/status', {'Authorization': 'Bearer wrong'})[0] == 401
        assert request(server, '/api/status', {'Authorization': 'Bearer é'})[0] == 401
        assert request(server, '/api/status', {**auth, 'Host': 'evil.example'})[0] == 403
        assert request(server, '/api/status', {**auth, 'Host': '['})[0] == 403
        assert request(server, '/api/status', {**auth, 'Origin': 'https://evil.example'})[0] == 403
        assert request(server, '/../../etc/passwd', auth)[0] == 404
        assert request(server, '/api/status', auth, 'POST')[0] == 501
        assert json.loads(request(server, '/api/status', auth)[2])['status'] == 'pending'
        assert request(server, '/api/status', {**auth, 'Host': 'localhost:8876'})[0] == 200
        assert request(server, '/api/tasks?offset=-1', auth)[0] == 503
        assert request(server, '/api/status?task=missing', auth)[0] == 503
        state['status'] = 'failed'
        atomic_json(monitor.output / 'sweep.json', state)
        assert json.loads(request(server, '/api/status', auth)[2])['status'] == 'failed'
