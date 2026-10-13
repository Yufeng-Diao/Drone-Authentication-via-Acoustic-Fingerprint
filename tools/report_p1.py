"""Render an external P1 evidence bundle without running models again."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(root: Path):
    root = root.resolve()
    if any((parent / ".git").exists() or (parent / "pyproject.toml").exists() for parent in (root, *root.parents)):
        raise ValueError("Report output must be outside source projects")
    manifest = json.loads((root / "manifest.json").read_text())
    runs = {name: json.loads((root / "measurements" / filename).read_text()) for name, filename in manifest["run_measurements"].items()}
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.text(0.02, 0.015, f"P1 • {root.name} • smoke evidence only", fontsize=8, color="#536775")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=150)
        plt.close(fig)

    names = list(runs)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    left = np.zeros(len(names))
    for phase, color in zip(("loading", "extraction", "training", "evaluation", "serialization"),
                            ("#547ab2", "#227c71", "#d8943e", "#9b6495", "#7e8894")):
        values = np.array([runs[name]["timings_s"][phase] for name in names])
        ax.barh(names, values, left=left, label=phase.title(), color=color)
        left += values
    ax.set(xlabel="Observed stage time (seconds)", title="Single smoke observations • no speedup or full-workload estimate")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
    save(fig, "stage-timings")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, name in zip(axes, ("Real / Python 3.14", "Synthetic / pip wheel")):
        result = runs[name]
        matrix = np.asarray(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (row, column), count in np.ndenumerate(matrix):
            ax.text(column, row, str(count), ha="center", va="center", color="white" if count > matrix.max()/2 else "#193749", fontsize=16)
        labels = [f"D{i}" for i in result["labels"]]
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted identity", ylabel="True identity", title=f"{name}\nGNB • {result['test_rows']} test frames")
    save(fig, "example-confusion")

    esc = html.escape
    platforms = "".join(f"<tr><td>{esc(row['platform'])}</td><td>{esc(row['python'])}</td><td>{esc(row['status'])}</td><td>{esc(row['evidence'])}</td></tr>" for row in manifest["platform_checks"])
    run_rows = "".join(f"<tr><td>{esc(name)}</td><td>{r['training_rows']}</td><td>{r['test_rows']}</td><td>{r['feature_columns']}</td><td>{r['metrics']['accuracy']:.4f}</td><td>{len(r['warnings'])}</td></tr>" for name, r in runs.items())
    failures = "".join(f"<li>{esc(item)}</li>" for item in manifest["resolved_failures"])
    links = "".join(f'<li><a href="{esc(path.relative_to(root).as_posix())}">{esc(path.relative_to(root).as_posix())}</a></li>'
                    for folder in ("measurements", "logs") for path in sorted((root / folder).glob("*")) if path.is_file())
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P1 environment and workflow evidence</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#203744;background:#f3f6f7;margin:0}}main{{max-width:1120px;margin:32px auto;padding:0 24px}}section{{background:white;border:1px solid #dfe7e9;border-radius:12px;padding:24px;margin:20px 0}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px}}.status{{color:#876111;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dce4e7;padding:10px;text-align:left}}img{{width:100%;height:auto}}a{{color:#155ba0}}code{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}li{{margin:5px 0}}</style></head><body><main>
<p class="status">P1 · LINUX VALIDATED · WINDOWS EXECUTION PENDING</p><h1>The modern environment runs.<br>Windows acceptance remains open.</h1>
<p>Run: {esc(root.name)}<br>Revision: <code>{esc(manifest['source_revision'])}</code><br>{esc(manifest['source_state'])}</p>
<section><h2>What was established</h2><p>Python 3.14.8 and 3.13.16 install from one uv lock. The installed package loads WAV files, extracts base MFCCs, trains GNB and saves predictions, metrics and a model. A separate venv/pip installation uses hash-checked requirements exported from the same lock.</p>
<p>NumPy 2.5.4 · SciPy 1.18.1 · scikit-learn 1.9.1 · Matplotlib 3.11.2 · uv 0.13.0. The 28-test suite passes on both Linux interpreters. All eight constructor/default audits pass; full eight-model fit acceptance is still P4 work.</p>
<div class="scroll"><table><tr><th>Platform</th><th>Python</th><th>Status</th><th>Evidence</th></tr>{platforms}</table></div>
<p>Windows CI configuration is ready, but no Windows runner was available locally and no workflow has been published or executed. Wheel resolution alone does not prove Windows runtime compatibility. P1 therefore remains partial against its full platform acceptance criteria.</p></section>
<section><h2>Measured stage times</h2><img src="figures/stage-timings.png" alt="Stacked stage timings for actual smoke workflows"><p>Individual observations include checksum/decode loading, MFCC extraction, fitting, prediction/metrics and serialization. Imports, environment setup and final JSON output are outside these stage timings. Small fitting/evaluation segments may be difficult to see; exact values are linked below.</p></section>
<section><h2>Example confusion matrices</h2><img src="figures/example-confusion.png" alt="Real two-identity and synthetic example confusion matrices with test frame counts">
<div class="scroll"><table><tr><th>Run</th><th>Train rows</th><th>Test rows</th><th>Features</th><th>Accuracy</th><th>Warnings</th></tr>{run_rows}</table></div>
<p>The real example uses four complete WAVs: identities 1 and 2 at both distances, selected from the frozen P0 subset. It uses overlapping frames from the inherited within-recording split. Even a perfect score here is not the paper's 24-class result or evidence of recording-held-out generalization. Synthetic scores only validate the pipeline.</p></section>
<section><h2>Environment and scientific boundaries</h2><ul><li>Reuse the original MFCC core; remove eager legacy plotting imports and 22 tracked bytecode files.</li><li>Preserve LinearSVC dual=True, map RF auto to sqrt, keep SVC probability disabled, retain DT seed 1 and record seed 2022 for previously unseeded estimators.</li><li>Validate configuration fields, file hashes, PCM format, paired membership and external output placement. Existing runs cannot be overwritten.</li><li>Print phase/recording progress and 10-second elapsed/CPU heartbeats without an application wall-time cutoff. A silent-operation test passes; a genuine slow-fit trace remains P4.</li><li>Write no intermediate audio or feature files. P3 has not yet demonstrated I/O improvement or established any cause of past SSD damage.</li></ul></section>
<section><h2>Retained failed attempts and resolutions</h2><ul>{failures}</ul><p>Failures remain in the command ledger and logs; successful reruns are separate records.</p></section>
<section><h2>Raw evidence and reproducibility</h2><p><a href="manifest.json">Manifest, hashes and acceptance criteria</a> · <a href="commands.jsonl">Exact commands and exit records</a> · <a href="source.patch">Source patch</a></p><ul>{links}</ul><p>Model and prediction files remain in external artifacts and are referenced by path/checksum. No generated evidence is committed to source.</p></section>
<section><h2>Next work</h2><p>Execute the prepared Windows 3.13/3.14 matrix before declaring P1 fully complete. P2 will validate numerical corrections and metric conventions; P3 will measure storage behavior; P4 will validate all estimators and actual slow training. No paper reproduction, speedup, SSD-wear reduction or full training runtime is claimed here.</p></section>
</main></body></html>'''
    (root / "report.html").write_text(page, encoding="utf-8")
    print(f"Wrote {root / 'report.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    render(parser.parse_args().evidence_dir)
