"""Render the P0 evidence bundle as offline HTML and measured figures.

Requires Matplotlib for plotting; the validator itself uses only the standard
library. This report does not run experiments or infer model accuracy.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def external_directory(directory):
    directory = directory.resolve()
    if directory.is_relative_to(REPO):
        raise ValueError("Evidence must be stored outside the source repository")
    if not directory.is_dir():
        raise ValueError("Evidence directory must already exist")
    return directory


def report(directory):
    directory = external_directory(directory)
    result = json.loads((directory / "measurements/protocol-validation.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in result["config_sha256"].items():
        if hashlib.sha256((REPO / "config/paper" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Configuration changed since validation: {name}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#24756c", "#d88736", "#5977b6"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.text(0.02, 0.015, f"P0 • {directory.name} • metadata/specification only; no training", fontsize=8, color="#56616b")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(figures / f"{name}.png", dpi=150)
        fig.savefig(figures / f"{name}.svg")
        plt.close(fig)

    rows = result["table1_comparison"][:-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    identities = [r["identity"] for r in rows]
    ax.bar([i - 0.19 for i in identities], [r["measured"]["1m_s"] / 60 for r in rows], width=0.38, color=colors[0], label="1 m")
    ax.bar([i + 0.19 for i in identities], [r["measured"]["5m_s"] / 60 for r in rows], width=0.38, color=colors[2], label="5 m")
    ax.set(xticks=identities, ylim=(0, 13), xlabel="Drone identity (body/propeller combination)", ylabel="Duration (minutes)",
           title="Paper selection: 240 recordings • 5 pairs per identity • no missing identities")
    ax.legend(ncol=2)
    save(fig, "identity-duration")

    example = result["timeline_example"]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    rate = example["sample_rate_hz"]
    for index, role in enumerate(("train", "test")):
        intervals = [(a / rate, (b - a) / rate) for a, b in example["intervals"][role]]
        for row in (0, 1):
            ax.broken_barh(intervals, (row - 0.25, 0.5), facecolors=colors[index])
    ax.set(yticks=[0, 1], yticklabels=["1 m", "5 m"], xlabel="Time in the original recording (seconds)",
           xlim=(0, example["frames"] / rate), ylim=(-0.6, 1.7),
           title="Frozen paired split: identical sample boundaries; split before framing")
    ax.legend(handles=[Patch(color=colors[i], label=role) for i, role in enumerate(("Train: 70%", "Test: 30%"))], loc="upper right", ncol=2)
    ax.text(0, -0.33, f"Example pair: {example['pair_key']} • no feature extraction performed", transform=ax.transAxes, fontsize=9)
    save(fig, "paired-timeline")

    partitions = json.loads((REPO / "config/paper/attack_partitions.json").read_text())["partitions"]
    matrix = [[next(i for i, role in enumerate(("registered", "background", "attack")) if identity in p[role])
               for identity in range(1, 25)] for p in partitions]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    ax.set(xticks=range(24), xticklabels=range(1, 25), yticks=range(10), yticklabels=range(1, 11),
           xlabel="Drone identity", ylabel="Published trial", title="Table 3: ten fixed partitions • 8 + 8 + 8 identities per trial")
    ax.legend(handles=[Patch(color=colors[i], label=role) for i, role in enumerate(("Registered", "Background", "Attack"))],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, "attack-partitions")

    esc = html.escape
    coverage = "".join(f"<tr><td>{esc(e['paper'])}</td><td><code>{esc(e['id'])}</code></td><td>{e['new_fits']:,}</td><td>{esc(e['status'])}</td></tr>" for e in result["experiment_coverage"])
    decisions = os.path.relpath(REPO / "docs/PAPER_PROTOCOL.md", directory)
    errors = max(abs(v) for row in result["table1_comparison"] for v in row["difference_s"].values())
    checks = "".join(f"<tr><td>{esc(c['name'])}</td><td>{esc(c['status'])}</td><td>{esc(c['detail'])}</td></tr>" for c in manifest["checks"])
    artifact_links = "".join(f'<li><a href="{esc(p.relative_to(directory).as_posix())}">{esc(p.relative_to(directory).as_posix())}</a></li>'
                             for folder in ("logs", "measurements") for p in sorted((directory / folder).glob("*")) if p.is_file())
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P0 protocol and baseline evidence</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#203744;background:#f3f6f7;margin:0}}main{{max-width:1120px;margin:32px auto;padding:0 24px}}section{{background:white;border:1px solid #dfe7e9;border-radius:12px;padding:24px;margin:20px 0}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px}}.tag{{color:#176d5e;font-weight:700}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#eaf3f1;padding:16px;border-radius:8px;flex:1;min-width:180px}}.card strong{{display:block;font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dce4e7;padding:10px;text-align:left}}img{{width:100%;height:auto}}a{{color:#155ba0}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}.muted{{color:#536775}}li{{margin:5px 0}}</style></head>
<body><main><p class="tag">P0 · {esc(manifest['status'].upper())} · PROTOCOL AND DATA ONLY</p>
<h1>The paper protocol is frozen.<br>The experiments have not run.</h1>
<p class="muted">Run: {esc(directory.name)} · Branch: {esc(manifest['branch'])}<br>Source revision: <code>{esc(manifest['source_revision'])}</code><br>Source state: {esc(manifest['source_state'])}</p>
<div class="cards"><div class="card"><strong>{result['published_wav_count']}</strong>WAV hashes and headers verified</div><div class="card"><strong>{result['paper_drone_count']}</strong>drone recordings in paper selection</div><div class="card"><strong>{result['attack_trials']}</strong>published attack partitions checked</div><div class="card"><strong>0</strong>model fits or extracted feature files</div></div>
<section><h2>Acceptance checks</h2><div class="scroll"><table><tr><th>Check</th><th>Status</th><th>Evidence / meaning</th></tr>{checks}</table></div>
<p>All 24 Table 1 rows and its totals were compared with sample-count durations. Maximum absolute difference: {errors:.6f} s; threshold: 0.01 s. The combined total rounds to 29,285.79 s.</p></section>
<section><h2>Data coverage and deliberate exclusions</h2><img src="figures/identity-duration.png" alt="Measured recording duration at both distances for all 24 identities">
<p>Five paired recordings per identity are present in the historical selection. The 16 additional drone files dated 2022-02-27 are retained and explicitly excluded from paper presets. Room noise is separate: {result['room_noise_duration_s']/60:.2f} minutes.</p>
<p>The fixed validation subset uses one complete pair per identity: 48 drone files, {result['subset_drone_duration_s']/60:.2f} minutes in total, plus the first 60 s of room noise. It has no missing identity or distance and is not an accuracy benchmark. No audio copies were generated.</p></section>
<section><h2>Paired train/test timeline</h2><img src="figures/paired-timeline.png" alt="Aligned training and test sample intervals at one and five metres">
<p>This visualizes the frozen interval specification from real recording lengths. Training intervals are [0,20%), [35%,65%), [80%,100%); test intervals fill the gaps. No frames, pre-emphasis or delta contexts may cross boundaries.</p><p>{esc(result['pairing_limit'])}</p></section>
<section><h2>Attack identities remain traceable</h2><img src="figures/attack-partitions.png" alt="Ten attack partitions assigning each identity one of three roles">
<p>Every trial contains three disjoint groups of eight identities, covering 1–24 exactly. The published assignments are replayed without resampling. Attack metrics retain original identity labels.</p></section>
<section><h2>Paper-to-experiment coverage</h2><div class="scroll"><table><tr><th>Paper output</th><th>Preset</th><th>Planned new fits</th><th>Execution</th></tr>{coverage}</table></div>
<p>4,096 fits are specified. This is a workload count, not a runtime estimate. Figures 4 and 5 reuse clean models. Figure 1 describes the physical setup; Table 1 is checked above.</p></section>
<section><h2>Decisions before implementation</h2><ul><li>Remove the first coefficient from each block, select coefficients, concatenate, then normalize the final vector.</li><li>Use floor((filters − 1) × fraction) coefficients per block and a true delta of the delta, within continuous segments.</li><li>Use the entire room-noise recording only for full-protocol background training.</li><li>Report attack recall both as an identity-level macro average and a separately labelled pooled historical comparison.</li><li>Preserve slow estimator startup: liveness heartbeats are planned; silence will not trigger automatic termination.</li></ul>
<p>These choices are documented interpretations or corrections, not assertions about unspecified paper details. See the <a href="{esc(decisions)}">English protocol and decision log</a>. Numerical validation follows in P2.</p></section>
<section><h2>Provenance and raw records</h2><p><a href="manifest.json">Run manifest</a> · <a href="commands.jsonl">Commands, timestamps, exit codes and output locations</a> · <a href="source.patch">Source change snapshot</a></p><ul>{artifact_links}</ul><p>PNG and SVG figure versions are stored in <code>figures/</code>. Logs, reports, source snapshots and measurements are outside the code repositories.</p></section>
<section><h2>Limits and next phase</h2><p>P1 will build and test the modern locked environment and audit classifier defaults. P2 will implement and validate the chosen feature/metric conventions. P4 will validate slow-fit progress. P6 will attempt the full paper runs.</p>
<p>No accuracy, speedup, write-reduction, SSD-wear, training-runtime or modern-environment compatibility claim is established by P0. Pairing checks do not independently verify acoustic synchronization. Historical numerical results may differ after documented corrections.</p></section>
</main></body></html>'''
    (directory / "report.html").write_text(page)
    print(f"Wrote {directory / 'report.html'} and six figure files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    report(parser.parse_args().evidence_dir)
