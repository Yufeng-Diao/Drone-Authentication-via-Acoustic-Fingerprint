"""Validate saved P2/P3 benchmark artifacts and render an external HTML report."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drone_fingerprint.data import sha256
from drone_fingerprint.storage import external_path

MODES = ('p2', 'p3-empty', 'p3-populated')
LABELS = ('P2', 'P3 empty cache', 'P3 populated cache')
COLORS = ('#8999a4', '#be7838', '#218e82')


def check_equivalence(rows):
    checks=[]
    for repetition in range(4):
        group={r['mode']:r for r in rows if r['repetition']==repetition}
        if set(group)!=set(MODES):
            raise ValueError('Expected a warm-up plus three complete measured groups')
        reference=Path(group['p2']['run'])
        old=json.loads((reference/'result.json').read_text())
        for mode in MODES[1:]:
            candidate=Path(group[mode]['run'])
            new=json.loads((candidate/'result.json').read_text())
            assert old['segments']==new['segments'] and old['metrics']==new['metrics']
            maximum=0.0
            with np.load(reference/'features.npz',allow_pickle=False) as a, np.load(candidate/'features.npz',allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:
                    if 'labels' in key: np.testing.assert_array_equal(a[key],b[key])
                    else:
                        np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
                        maximum=max(maximum,float(np.max(abs(a[key]-b[key]))))
            with np.load(reference/'predictions.npz',allow_pickle=False) as a, np.load(candidate/'predictions.npz',allow_pickle=False) as b:
                for key in a.files: np.testing.assert_array_equal(a[key],b[key])
            assert len(old['noise'])==len(new['noise'])
            for a,b in zip(old['noise'],new['noise']):
                assert a['seed']==b['seed'] and a['requested_snr_db']==b['requested_snr_db']
                assert abs(a['measured_full_recording_snr_db']-b['measured_full_recording_snr_db'])<1e-12
            for directory,result in ((reference,old),(candidate,new)):
                for name,metadata in result['artifacts'].items():
                    assert sha256(directory/name)==metadata['sha256']
            assert new['cache']['directory_walks']==0
            checks.append({'repetition':repetition,'mode':mode,'max_absolute_feature_error':maximum,
                           'labels_intervals_predictions_metrics_exact':True})
        assert group['p3-empty']['task_id']==group['p3-populated']['task_id']
        assert group['p3-populated']['cache_before']==group['p3-populated']['cache_after']
        assert group['p3-populated']['cache']['selected_hits']==2
        assert group['p3-populated']['cache']['hashed_recordings']==0
        assert len(group['p3-empty']['cache_after'])==10
    return checks


def summarize(rows):
    result={}
    for mode in MODES:
        measured=[r for r in rows if r['mode']==mode and r['measured']]
        assert len(measured)==3 and all(r['exit_code']==0 for r in measured)
        result[mode]={key:statistics.median(r[key] for r in measured) for key in ('wall_s','peak_rss_kib')}
        result[mode]['io']={key:statistics.median(r['io_delta'][key] for r in measured) for key in ('rchar','wchar','read_bytes','write_bytes')}
        result[mode]['cache_files']=len(measured[0]['cache_after'])
        result[mode]['cache_bytes']=sum(v['bytes'] for v in measured[0]['cache_after'].values())
        result[mode]['run_bytes']=sum(v['bytes'] for v in measured[0]['run_inventory'].values())
        result[mode]['run_files']=len(measured[0]['run_inventory'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args()
    output=external_path(args.evidence)
    criteria=json.loads((output/'acceptance.json').read_text())
    test_counts={}
    for version in ('314','313'):
        tree=ET.parse(output/'measurements'/f'final-tests{version}.xml')
        assert not tree.findall('.//failure') and not tree.findall('.//error')
        test_counts[version]=len(tree.findall('.//testcase'))
    text=(output/'logs/full-recording-index.log').read_text()
    inventory=json.loads(text[text.index('{'):])
    assert inventory=={'status':'passed','recordings':257,'directory_walks':1,'hashed_recordings':257}
    assert criteria['baseline_commit'].startswith('3885fbc')
    datasets={name:json.loads((output/'measurements'/name/'measurements.json').read_text())
              for name in ('open-set-refined','closed-noisy-refined')}
    checks={name:check_equivalence(rows) for name,rows in datasets.items()}
    summaries={name:summarize(rows) for name,rows in datasets.items()}
    base,cold,warm=(summaries['open-set-refined'][mode] for mode in MODES)
    ratios={'empty_peak_rss':cold['peak_rss_kib']/base['peak_rss_kib'],
            'empty_read_syscalls':cold['io']['rchar']/base['io']['rchar'],
            'empty_wall_time':cold['wall_s']/base['wall_s'],
            'populated_wall_time':warm['wall_s']/base['wall_s']}
    limits={'empty_peak_rss':0.60,'empty_read_syscalls':0.75,'empty_wall_time':1.50,'populated_wall_time':0.50}
    assert all(ratios[name]<=limit for name,limit in limits.items()),(ratios,limits)
    (output/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','svg'):fig.savefig(output/'figures'/f'{name}.{extension}',dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for axis,(name,rows) in zip(axes.T,datasets.items()):
        for ax,field,scale,label in [(axis[0],'wall_s',1,'Process wall time (s)'),(axis[1],'peak_rss_kib',1024,'Peak RSS (MiB)')]:
            values=[[r[field]/scale for r in rows if r['mode']==mode and r['measured']] for mode in MODES]
            ax.bar(LABELS,[statistics.median(v) for v in values],color=COLORS)
            for i,observations in enumerate(values):
                ax.scatter([i-.08,i,i+.08],observations,color='#183942',s=20,zorder=3)
            ax.set(ylabel=label,title=name.replace('-refined','').replace('-',' ').title())
            ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('Bars: medians; dots: three measured repetitions; warm-up excluded',fontsize=12)
    save(fig,'time-memory')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for axis,(name,summary) in zip(axes.T,summaries.items()):
        for ax,field,label in [(axis[0],'rchar','Read syscall bytes (MiB)'),(axis[1],'write_bytes','Process storage write bytes (MiB)')]:
            bars=ax.bar(LABELS,[summary[m]['io'][field]/2**20 for m in MODES],color=COLORS)
            ax.bar_label(bars,fmt='%.1f',padding=3)
            ax.set(ylabel=label,title=name.replace('-refined','').replace('-',' ').title())
            ax.margins(y=.16);ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('OS cache uncontrolled; syscall counts exclude mmap; storage counters are not SSD wear',fontsize=11)
    save(fig,'io')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(name,rows) in zip(axes,datasets.items()):
        for mode,label,color in zip(MODES,LABELS,COLORS):
            run=next(r for r in rows if r['repetition']==1 and r['mode']==mode)
            ax.plot([p['elapsed_s'] for p in run['rss_trace']],[p['rss_kib']/1024 for p in run['rss_trace']],label=label,color=color)
        ax.set(title=name.replace('-refined',''),xlabel='Elapsed process time (s)',ylabel='RSS sampled at 100 ms (MiB)');ax.legend(fontsize=9)
    save(fig,'memory-trace')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,scale,ylabel in [(axes[0],'bytes',2**20,'Retained artifact bytes (MiB)'),(axes[1],'files',1,'Retained files')]:
        summary=summaries['open-set-refined']
        run=[summary[m]['run_'+key]/scale for m in MODES]
        cache=[summary[m]['cache_'+key]/scale for m in MODES]
        ax.bar(LABELS,run,label='Run incl. optional feature snapshot',color='#8097a3')
        ax.bar(LABELS,cache,bottom=run,label='Shared cache (already exists on hit)',color='#238d82')
        ax.set(ylabel=ylabel,title='Open-set validation',ylim=(0,max(a+b for a,b in zip(run,cache))*1.35));ax.legend(fontsize=8)
    save(fig,'artifacts')
    first=summarize(json.loads((output/'measurements/open-set-final/measurements.json').read_text()))
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=['Initial P3 implementation','Final P3 implementation']
    values=[first['p3-empty']['io']['rchar']/base['io']['rchar'],ratios['empty_read_syscalls']]
    bars=ax.bar(labels,values,color=['#be7838','#238d82']);ax.bar_label(bars,fmt='%.3f',padding=10)
    ax.axhline(.75,color='#942c33',linestyle='--',label='Predeclared maximum: 0.75 × P2')
    ax.set(ylabel='Read syscall bytes / P2',ylim=(0,1),title='Original acceptance threshold retained');ax.legend()
    save(fig,'refinement')

    manifest={'phase':'P3','status':'passed','generated_at':datetime.now(timezone.utc).isoformat(),
              'source_commit':'pending local phase commit','baseline_commit':criteria['baseline_commit'],
              'tests':test_counts,'recording_index':inventory,'numerical_checks':checks,'summaries':summaries,'resource_ratios':ratios,'resource_limits':limits,
              'platform':'Ubuntu 24.04, CPython 3.14.8 benchmark; 3.13 compatibility checked separately',
              'scope':'Fixed subset and GNB. No full paper/all-model fit claim. No physical cold-disk claim.',
              'counters':'rchar/wchar exclude mmap; read_bytes/write_bytes are process storage accounting. Device wear not measured.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def link(target,label):return f'<a href="{escape(str(target),quote=True)}">{escape(label)}</a>'
    rows=''.join(f'<tr><td>{escape(name)}</td><td>{LABELS[i]}</td><td>{s[mode]["wall_s"]:.3f}</td><td>{s[mode]["peak_rss_kib"]/1024:.1f}</td><td>{s[mode]["io"]["rchar"]/2**20:.1f}</td><td>{s[mode]["io"]["write_bytes"]/2**20:.1f}</td><td>{s[mode]["cache_files"]}</td></tr>' for name,s in summaries.items() for i,mode in enumerate(MODES))
    budget_rows=''.join(f'<tr><td>{key}</td><td>{ratios[key]:.3f} × P2</td><td>≤ {limit:.2f} × P2</td><td>Passed</td></tr>' for key,limit in limits.items())
    maximum=max(c['max_absolute_feature_error'] for cases in checks.values() for c in cases)
    artifact_links=''.join(f'<li>{link(os.path.relpath(Path(r["run"])/"result.json",output),name+" / "+r["mode"]+" / repetition 1")}</li>' for name,runs in datasets.items() for r in runs if r['repetition']==1)
    captions=[('time-memory','Elapsed time and peak memory','The empty cache pays for verification, bounded extraction and durable cache publication. It is slower on this one-shot workload; populated-cache reuse is substantially faster. All bars are medians of three measured runs.'),('io','Read/write volume with explicit counter limits','Read syscall counts exclude mmap page accesses. Process storage writes include dirty mapped cache pages, which wchar omits. Cache construction adds about one shared array set; warm hits do not rewrite it. OS page cache is uncontrolled, and these counters cannot diagnose SSD wear.'),('memory-trace','Memory over time','The first measured repetition is shown. RSS sampling at 100 ms can miss short peaks; acceptance uses process ru_maxrss instead. Room-noise selection no longer requires decoding all 3,628 seconds to float64.'),('artifacts','The cost of a reusable binary cache','Each split has one base and one selected NPY array plus metadata. Both splits together have eight entry files, one recording index and one lock. The benchmark additionally retains an NPZ snapshot per run to prove equivalence. P2 already avoids per-frame files; P3 adds a small shared cache rather than claiming a lower total file count.'),('refinement','A failed resource gate led to another optimization','The initial implementation exceeded the read threshold. It remains archived. Final code reuses PCM overlap and avoids rehashing arrays immediately after the writer has hashed and atomically published them under the same lock. Existing-cache hits still verify checksums. The original threshold was not relaxed.')]
    figures=''.join(f'<section><h2>{title}</h2><p>{caption}</p><img src="figures/{name}.png" alt="{escape(title)}"><p>{link("figures/"+name+".svg","SVG download")}</p></section>' for name,title,caption in captions)
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P3 — Data pipeline and resource use</title><style>
body{{margin:0;background:#edf3f5;color:#173642;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:30px 22px 70px}}header,section{{background:#fff;border:1px solid #d6e2e5;border-radius:12px;margin:20px 0;padding:26px 30px}}header{{border-top:6px solid #238d82}}h1{{font-size:2.35rem;line-height:1.15}}h2{{font-size:1.4rem}}a{{color:#096d91}}img{{max-width:100%;height:auto}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{background:#e9f5f1;padding:16px;border-radius:8px}}.card strong{{display:block;font-size:1.6rem}}table{{width:100%;border-collapse:collapse;font-size:.93rem}}th,td{{padding:10px;border-bottom:1px solid #d4e0e4;text-align:left;vertical-align:top}}.scroll{{overflow:auto}}.notice{{padding:16px;background:#fff4df;border-left:4px solid #bf7e35}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}</style></head><body><main>
<header><p>P3 / PASSED / UBUNTU</p><h1>Read less audio. Reuse verified features.</h1><p>Bounded extraction, indexed WAV validation, shared float64 NumPy arrays and atomic cache publication. P2 scientific definitions are preserved.</p><div class="cards"><div class="card"><strong>{cold['peak_rss_kib']/1024:.0f} MiB</strong>Empty-cache peak RSS; P2 {base['peak_rss_kib']/1024:.0f} MiB</div><div class="card"><strong>{warm['wall_s']:.2f} s</strong>Populated-cache median; P2 {base['wall_s']:.2f} s</div><div class="card"><strong>{maximum:.2e}</strong>Maximum feature difference</div><div class="card"><strong>10 files</strong>One shared train/test cache configuration</div></div><p>One warm-up + three measured repetitions for each mode and dataset. Source and checks recorded in {link('manifest.json','manifest.json')}.</p></header>
<section><h2>Acceptance is numerical and operational</h2><p>All saved labels, interval boundaries, predictions and metrics match P2 exactly. Features pass atol=rtol=1e-12, including noisy testing; original per-recording noise seeds and empirical SNR are preserved. The same P3 task has the same identity with empty or populated caches. Cache files remain byte- and mtime-identical on hits.</p><div class="scroll"><table><tr><th>Open-set gate</th><th>Measured median ratio</th><th>Predeclared limit</th><th>Result</th></tr>{budget_rows}</table></div><p>{link('acceptance.json','Criteria declared before measurement')} · {link('measurements/acceptance-checks.json','Final tests, inventory and artifact checks')}</p><div class="notice">A first run is not free: empty-cache open-set time is {cold['wall_s']/base['wall_s']-1:.1%} higher than P2, within the declared 50% maximum regression. The cache adds retained files and storage writes. Its benefit comes from lower extraction memory and repeated reuse across models and coefficient selections.</div></section>
{figures}
<section><h2>Measurements</h2><div class="scroll"><table><tr><th>Subset</th><th>Mode</th><th>Wall s</th><th>RSS MiB</th><th>Read syscall MiB</th><th>Storage-write MiB</th><th>Cache files</th></tr>{rows}</table></div><p>Open-set trial 1 uses 48 drone recordings plus a declared 60-second room-noise interval; closed-noisy uses four recordings, three MFCC blocks and 0 dB test AWGN. GNB is the estimator in both benchmarks. No full-paper accuracy claim is made.</p></section>
<section><h2>What was validated</h2><p>Ubuntu Python 3.14: {test_counts["314"]} tests passed. Python 3.13: {test_counts["313"]} tests passed. The complete 257-WAV inventory passed checksum and format verification. {link("measurements/hardware.json","Hardware and filesystem record")}</p><ul><li>One directory walk per full inventory build; indexed tasks use no directory walk.</li><li>Model-independent cache keys are exercised for all eight model names. Changing the coefficient fraction reuses base MFCCs; test AWGN reuses clean training arrays.</li><li>Two processes serialize publication and the second reuses the first result. A forcibly exited writer leaves no completed entry; the next execution removes its staging data and rebuilds.</li><li>Bad WAV content, modified input signatures, corrupt index metadata, corrupt arrays, invalid paths and insufficient disk space fail explicitly.</li><li>The index trusts unchanged local filesystem signatures by default; <code>--verify-inputs</code> forces hashes. This tradeoff is documented rather than described as a fresh checksum on every hit.</li><li>Output completion and index JSON updates are atomic. An interrupted estimator fit is still a failed/incomplete task; restart orchestration belongs to P4.</li></ul></section>
<section><h2>Records and remaining limits</h2><ul><li>{link('commands.jsonl','Phase command ledger')}</li><li>{link('measurements/open-set-refined/measurements.json','Open-set measurements and nested command ledger directory')}</li><li>{link('measurements/closed-noisy-refined/measurements.json','Noisy closed-set measurements')}</li><li>{link('measurements/open-set-final/measurements.json','Initial implementation measurements (read gate failed)')}</li><li>{link('logs/benchmark-open-set.log','Initial benchmark launcher failure')} — resolving the virtualenv executable symlink lost the installed package; the tool now preserves the virtualenv path.</li><li>{link('source.patch','Source patch')} · {link('../../../Drone-Authentication-refactor/docs/DATA_PIPELINE.md','Researcher guide')} · {link('../../../REFACTOR_PLAN.md','Updated plan')}</li></ul><p>Representative saved results:</p><ul>{artifact_links}</ul><p>Not run: physical cold-disk tests, full sweeps/all-model fits, remote CI, network filesystem tests or publication. mmap does not bound sklearn fitting memory. No decoded-audio cache, database or float32 conversion was introduced. Storage counters are not SSD wear measurements.</p></section></main></body></html>'''
    (output/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'status':'passed','resource_ratios':ratios,'max_feature_difference':maximum,'report':str(output/'report.html')},indent=2))


if __name__=='__main__':
    main()
