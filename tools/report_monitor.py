"""Validate the external live-monitor evidence and render an offline review page."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from drone_fingerprint.storage import external_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    evidence = external_path(args.evidence)

    def read(name):
        return json.loads((evidence / name).read_text())

    browser = read('browser-acceptance.json')
    final_ui = read('browser-ui-final.json')
    package = read('package-container-final.json')
    integrity = read('metadata-science-checks.json')
    for result in (browser, final_ui, package, integrity):
        assert result['status'] == 'passed'
    for result in (browser, final_ui):
        assert all(result['checks'].values()) and not result['browser_errors']
        assert all(command['exit_code'] == 0 for command in result['commands'])
    assert package['wheel_assets_match'] and package['image_assets_match']
    assert package['container_runner_observed_on_host'] and package['container_exit_code'] == 0
    assert package['monitor_imports_no_numpy']
    assert integrity['raw_wavs_unchanged_and_readonly'] == 257
    assert integrity['dependency_lock_unchanged']
    assert all(row['matches_p4_exactly'] for row in integrity['scientific_comparison'])
    perf = integrity['performance']
    assert perf['task_count'] == 8440 and perf['run_files_unchanged']
    tests = {}
    for version in ('314', '313'):
        tree = ET.parse(evidence / f'release-tests{version}.xml')
        assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
        tests[version] = len(tree.findall('.//testcase'))
        assert tests[version] == 154
    snapshots = browser['snapshots']
    assert {'training', 'evaluation'} <= {row.get('phase', {}).get('phase') for row in snapshots}
    assert any(row.get('process', {}).get('cpu_seconds', 0) > 0 for row in snapshots if row.get('process'))
    commands = [json.loads(line) for line in (evidence / 'commands.jsonl').read_text().splitlines()]
    exits = {row['name']: row['exit_code'] for row in commands if row['event'] == 'finished'}
    required = ('release-tests314', 'release-tests313', 'browser-live-final', 'browser-ui-final',
                'metadata-science-checks', 'ui-release-sync313', 'ui-release-build',
                'ui-release-container-build', 'ui-package-container-retry')
    assert all(exits[name] == 0 for name in required)
    comparisons = ''.join(
        f"<tr><td>{escape(row['model'])}</td><td>{row['training_rows']:,}</td>"
        f"<td>{row['test_rows']:,}</td><td>{row['timings_s']['training']:.2f} s</td>"
        f"<td>{row['timings_s']['evaluation']:.2f} s</td><td>Exact P4 match</td></tr>"
        for row in integrity['scientific_comparison'])
    screenshots = ''.join(
        f'<figure><a href="{name}"><img src="{name}" alt="{caption}" loading="lazy"></a>'
        f'<figcaption>{caption}</figcaption></figure>'
        for name, caption in [
            ('live-desktop.png', 'Real RBF-SVM training: task counts, phase, CPU, memory and external logs.'),
            ('live-mobile.png', 'The same live run at a 390 px mobile viewport. Wide tables scroll inside their panel.'),
            ('disconnected-desktop.png', 'Monitor deliberately stopped during fitting: the page labels stale data; the trainer continues.'),
            ('completed-desktop.png', 'Both real SVM tasks completed after the monitor was restarted.'),
            ('final-ui-desktop.png', 'Final packaged UI after the bounded-disconnection check and stylesheet formatting.')])
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P5 addendum: live training monitor</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#203538;background:#f3f6f3;max-width:1180px;margin:auto;padding:32px 24px}}
h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px}}a{{color:#117864}}code{{overflow-wrap:anywhere}}
.card,figure{{background:white;border:1px solid #dce5e2;border-radius:10px;padding:20px;margin:18px 0}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card strong{{display:block;font-size:25px}}
img{{max-width:100%;max-height:1000px;display:block;margin:auto}}figcaption{{margin-top:14px;color:#607477}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dce5e2}}
.scroll{{overflow:auto}}.note{{border-left:4px solid #bd7d21;padding:16px;background:#fff1dc}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style></head><body><p>P5 ADDENDUM · UBUNTU · LOCAL ACCEPTANCE COMPLETE</p>
<h1>Watch training without interrupting it</h1>
<p>The new <code>drone-auth monitor</code> serves a read-only browser dashboard on localhost.
SSH port forwarding makes the same page available remotely. It observes existing sweep files and
Linux process counters; the trainer has no dependency on the webpage. No Python dependency was added.</p>
<div class="cards"><div class="card"><strong>154 + 154</strong>Tests passed on Python 3.14.8 and 3.13.16</div>
<div class="card"><strong>{len(snapshots)} live snapshots</strong>Real LSVM and RBF-SVM subset training</div>
<div class="card"><strong>8,440 tasks</strong>Full-plan metadata test; no full-grid training</div></div>
<p><a href="../../../Drone-Authentication-refactor/docs/LIVE_MONITOR.md">Local, SSH and Docker instructions</a> ·
<a href="../../../REFACTOR_PLAN.md">Updated refactor plan</a> · <a href="commands.jsonl">Command ledger</a> ·
<a href="source-manifest.json">Source revision and hashes</a></p>
<h2>What the browser shows</h2>
<p>Task completion and coverage, current phase and elapsed time, six minutes of CPU/RSS samples,
filtered task pages, latest attempts, bounded log tails and explicit failure/disconnection states.
Task count is not a percentage of elapsed computation. Native fits expose no trustworthy internal
percentage, convergence estimate or ETA; the monitor does not invent one.</p>
{screenshots}
<h2>Evidence and scientific noninterference</h2>
<p>The browser was exercised at desktop and mobile sizes with real public-data training.
Stopping and restarting the monitor left the trainer alive. Tests cover filtering, task selection,
text-only log rendering, no external assets, stale snapshots and token replacement after restart.
The final browser check simulates 150 failed refreshes and verifies a 120-sample upper bound and reconnection.</p>
<div class="scroll"><table><tr><th>Classifier</th><th>Train rows</th><th>Test rows</th><th>Fit</th><th>Evaluate</th><th>Outputs</th></tr>{comparisons}</table></div>
<p>Predictions, labels, metrics, confusion matrices, segments and scientific settings match the earlier
P4 runs exactly. Timings above are individual subset observations, not speed comparisons. RBF-SVM's
existing <code>max_iter=1000</code> convergence warning and scikit-learn deprecation warning remain in the
results and logs. Completion does not establish convergence.</p>
<p>A non-root, network-disabled, read-only-root Docker training run also completed while the native host
monitor observed its lock owner, process counters and remapped log files. Final wheel and image monitor
assets match source byte for byte. Image: <code>{escape(package['image_id'])}</code>.</p>
<ul><li><a href="browser-acceptance.json">Real browser acceptance, process lifetimes and samples</a></li>
<li><a href="browser-ui-final.json">Final UI, bounded-disconnection and reconnect checks</a></li>
<li><a href="release-tests314.xml">Python 3.14 tests</a> · <a href="release-tests313.xml">Python 3.13 installed-package tests</a></li>
<li><a href="package-container-final.json">Final packaging and real container monitoring</a></li>
<li><a href="metadata-science-checks.json">Scientific comparison, full-plan measurement and source/data integrity</a></li></ul>
<h2>Bounded metadata reads</h2>
<p>The saved full plan is {perf['plan_bytes']:,} bytes. First snapshot: {perf['first_status_ms']:.2f} ms.
Across {perf['samples']} cached calls, median status-plus-task-page time: {perf['cached_status_and_page_median_ms']:.2f} ms.
Status JSON: {perf['status_json_bytes']:,} bytes; 50-row page: {perf['task_page_json_bytes']:,} bytes.
These are local metadata-only timings with a pending ledger, not remote-network or full-training benchmarks.</p>
<p>Plans and ledgers are cached by file identity, time and size. Each log read is limited to 32 KiB,
progress to 16 KiB, and history to 50 attempts. No WAV, array or model is opened by the monitor.
No access log is emitted per poll. Tests verify unchanged run files. All 257 read-only WAV filesystem
signatures match the P3 inventory; this phase did not rehash audio contents. Both original repositories
remain clean. The dependency lock is unchanged.</p>
<h2>Corrections retained in the record</h2>
<p>The first browser check caught a fragment-only token update after restart; a hash-change handler fixed it.
An initial package comparison correctly rejected artifacts built before the last source edit; final packages
were rebuilt and verified. A cached GNB container run finished between 200 ms probe samples;
the acceptance probe was repeated at 10 ms to observe that short-lived runner. The final review bounded samples during prolonged disconnection, formatted the
stylesheet for editing, and made failed tasks take priority over older completed tasks in automatic selection.</p>
<p><a href="browser-acceptance-initial.json">Initial browser record</a> ·
<a href="logs/package-container-check.log">Initial stale-package rejection</a> ·
<a href="source.patch">Reviewable source patch</a> · <a href="acceptance/README.md">Archived acceptance scripts</a></p>
<h2>P6 handoff and limits</h2><div class="note">
<p>P6 has not started. Use a fresh full-run directory from the final validated revision and launch the
monitor separately on the Ubuntu training host. Forward its localhost port over SSH and open the printed
token URL. Closing the browser does not detach a foreground trainer: use the documented persistent launch.</p>
<p>One monitor observes one <code>paper</code> sweep. Run it on the native host for Docker training.
Restricted PID views, permissions and network filesystems may hide liveness. The server is intended for
localhost/SSH, not public hosting. Remote SSH transport from the user's own computer has not been exercised;
loopback HTTP, alternate forwarded Host port handling and real browser behavior were verified locally.</p>
</div><p>Temporary preview servers and browsers were stopped. No push, remote CI or full reproduction claim.</p>
<footer>Generated {datetime.now(timezone.utc).isoformat()}. All evidence and runtime artifacts are outside the source repository.</footer>
</body></html>'''
    for target in re.findall(r'(?:href|src)="([^"]+)"', html):
        assert (evidence / target).is_file(), target
    (evidence / 'report.html').write_text(html)
    print(f'Validated live-monitor report: {evidence / "report.html"}')


if __name__ == '__main__':
    main()
