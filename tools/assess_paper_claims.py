"""Assess paper claims using frozen P6 measurements, without fitting new models.

All grid averages below are descriptive summaries of related configurations,
not independent replications, significance tests or newly selected benchmarks.
"""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drone_fingerprint.config import MODELS
from drone_fingerprint.data import sha256
from drone_fingerprint.storage import atomic_json, create_output


# Numerical annotations and claim interpretations refer to this frozen P6 bundle.
EXPECTED_INPUTS = {'plot-data.json': '63c440cdd7d1101aae34c2837d97cc24774032c035c2e5d817df24e282a6ba63', 'tables.json': '60a9f8237d068115ade7c4f56984dc52f9fd455fa40c89ea3cc0b80aaf9bffc8', 'metrics.json': '6801c123a604691255909c957eb2cb7985ad442c101ed3250c1e639382ce9dd2'}

def rank(values, model):
    """Competition rank: ties share their rank; no model-order tie breaking."""
    return 1 + sum(value > values[model] + 1e-10 for value in values.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.full_evidence; gallery = source / 'gallery'
    manifest = json.loads((gallery / 'manifest.json').read_text())
    assert manifest['status'] == 'passed' and manifest['full_grid'] and manifest['full_dataset']
    signatures = {}
    for name in ('plot-data.json', 'tables.json', 'metrics.json'):
        signatures[name] = sha256(gallery / name)
        assert signatures[name] == manifest['artifacts'][name]['sha256']
    if signatures != EXPECTED_INPUTS:
        raise ValueError('This claim audit is bound to the frozen P6 results; review the claim text before assessing another run')
    series = json.loads((gallery / 'plot-data.json').read_text())['series']
    tables = json.loads((gallery / 'tables.json').read_text())['tables']
    published = json.loads((source / 'published-tables.json').read_text())['tables']
    metrics = json.loads((gallery / 'metrics.json').read_text())['results']
    diagnostic = json.loads((source / 'label-order-summary.json').read_text())
    assert diagnostic['status'] == 'passed'
    rows = {(r['preset'], r['blocks'], r['portion'], r['model']): r for r in series}
    assert len(rows) == len(series) == 160
    # Independently reconcile plot values against the exported per-task metrics.
    points = {}
    for result in metrics:
        preset = result['experiment']; c = result['configuration']
        if preset not in ('figure2', 'figure3', 'figure4', 'figure5'):
            continue
        x = c['frame_ms'] if preset == 'figure2' else c['test_snr_quarter_db']/4 if preset == 'figure5' else c['n_filters']
        key = (preset, c['blocks'], c['portion'], result['model'], x)
        assert key not in points
        points[key] = result['metrics']['accuracy']*100
    for key, row in rows.items():
        assert len(row['x']) == len(row['y']) and all(v is not None for v in row['y'])
        for x, y in zip(row['x'], row['y']):
            assert abs(points[(*key, x)] - y) < 1e-10
    assert len(points) == 8352
    assert rank({'a': 1, 'b': 1, 'c': 0}, 'b') == 1
    assert rank({'a': 1, 'b': 1, 'c': 0}, 'c') == 3
    output = create_output(args.output)
    def save(fig, name):
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(output/f'{name}.{suffix}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    def values(preset, blocks, portion, model):
        return np.asarray(rows[preset, blocks, portion, model]['y'])
    paired, dominance, plateau = [], {}, []
    for preset in ('figure3', 'figure4'):
        cube = np.array([[values(preset,b,p,m) for m in MODELS] for b in (1,2,3) for p in (1,2,3)])
        maxima = cube.max(axis=1)
        dominance[preset] = {'configurations': 450,
            'co_highest_counts': {m: int(np.isclose(cube[:,i,:],maxima,atol=1e-10,rtol=0).sum()) for i,m in enumerate(MODELS)},
            'top_group_always_contains_a_winner': bool(np.all(cube[:,:3,:].max(axis=1) >= maxima-1e-10)),
            'qda_lda_lsvm_strictly_above_other_models': int((cube[:,:3,:].min(axis=1) > cube[:,3:,:].max(axis=1)+1e-10).sum())}
        for model in MODELS:
            for blocks in (2,3):
                differences = np.array([values(preset,blocks,p,model)-values(preset,1,p,model) for p in (1,2,3)])
                assert differences.shape == (3,50)
                higher, lower = int((differences>1e-10).sum()), int((differences< -1e-10).sum())
                paired.append({'preset':preset,'model':model,'blocks':blocks,'comparisons':150,
                    'higher':higher,'lower':lower,'equal':150-higher-lower,
                    'mean_difference_pp':float(differences.mean()),'minimum_pp':float(differences.min()),'maximum_pp':float(differences.max()),
                    'at_201_filters_all_coefficients_pp':float(differences[2,35]),
                    'differences_pp_by_portion_and_filter':differences.tolist()})
            for b in (1,2,3):
                for p in (1,2,3):
                    row=rows[preset,b,p,model]; i=row['x'].index(201); assert i==35
                    plateau.append({'preset':preset,'model':model,'blocks':b,'portion':p,
                        'gain_26_to_271_pp':row['y'][-1]-row['y'][0],
                        'best_gain_above_201_pp':max(row['y'][i:])-row['y'][i]})
    x=np.array(rows['figure5',1,3,'QDA']['x']); noise=np.array([values('figure5',1,3,m) for m in MODELS])
    margin=noise[4]-np.delete(noise,4,axis=0).max(axis=0)
    noise_summary={'knn_below_minus3':{'tested_points':int((x< -3).sum()),'strict_wins':int((margin[x< -3]>1e-10).sum()),'minimum_margin_pp':float(margin[x< -3].min())},
        'at_2_db':{m:float(noise[i,list(x).index(2)]) for i,m in enumerate(MODELS)},
        'minimum_above_2_db':{m:float(noise[i,x>2].min()) for i,m in enumerate(MODELS)},
        'gain_10_to_15_db_pp':{m:float(noise[i,-1]-noise[i,list(x).index(10)]) for i,m in enumerate(MODELS)}}
    closed={r['model']:r for r in tables['table2']['rows']}
    opened={name:{m:tables[name]['rows'][-2][m] for m in MODELS} for name in ('table4','table5')}
    old_open={name:{m:published[name]['rows'][-1][m] for m in MODELS} for name in ('table4','table5')}
    ranks={name:{m:rank(opened[name],m) for m in MODELS} for name in opened}
    dominated=[m for m in MODELS if m!='QDA' and all(opened[t]['QDA']>opened[t][m] for t in opened)]
    qda=next(r for r in metrics if r['experiment']=='table2' and r['model']=='QDA')['sources'][0]
    result_path=Path(qda['attempt'])/'result.json'; assert sha256(result_path)==qda['result_sha256']
    result=json.loads(result_path.read_text()); assert result['labels']==list(range(1,25))
    cm=np.asarray(result['confusion_matrix']); recalls=cm.diagonal()/cm.sum(axis=1)*100
    groups={name:float(recalls[start:start+8].mean()) for name,start in [('Original propellers',0),('Replacement set x',8),('Replacement set y',16)]}
    audit={'status':'passed','generated_at':datetime.now(timezone.utc).isoformat(),'input_sha256':signatures,'verified_points':8352,
        'table2':closed,'open_set':opened,'published_open_set':old_open,'open_set_ranks':ranks,'qda_dominates_on_both_means':dominated,
        'qda_per_identity_recall':recalls.tolist(),'qda_propeller_group_macro_recall':groups,
        'grid_dominance':dominance,'derivative_paired_comparisons':paired,'filter_plateau':plateau,'noise':noise_summary,
        'label_order_diagnostic':diagnostic,'scope':'Descriptive audit of the frozen primary run. Grid points and ten partitions are not independent replicates. No significance or generalization test is claimed.'}
    atomic_json(output/'measurements.json',audit)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,name,title in zip(axes,('table4','table5'),('Registered macro recall','Unknown-drone macro rejection')):
        ticks=np.arange(8);ax.bar(ticks-.18,[old_open[name][m] for m in MODELS],.36,label='Published')
        ax.bar(ticks+.18,[opened[name][m] for m in MODELS],.36,label='Frozen primary rerun')
        ax.set(xticks=ticks,xticklabels=MODELS,ylim=(0,105),ylabel='Percent',title=title);ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8)
    save(fig,'open-set')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,preset,title in zip(axes,('figure3','figure4'),('Clean DS1','DS1 with 0 dB test AWGN')):
        matrix=np.array([[next(r['mean_difference_pp'] for r in paired if (r['preset'],r['model'],r['blocks'])==(preset,m,b)) for b in (2,3)] for m in MODELS])
        im=ax.imshow(matrix,cmap='RdBu',vmin=-13,vmax=13,aspect='auto')
        for i in range(8):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>7 else 'black')
        ax.set(xticks=[0,1],xticklabels=['MFCC + delta','MFCC + delta + delta-delta'],yticks=range(8),yticklabels=MODELS,title=title)
    fig.colorbar(im,ax=axes,label='Mean accuracy change vs MFCC only (pp), 150 matched configurations')
    save(fig,'derivatives')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    low=x< -3; axes[0].plot(x[low],margin[low],'o-');axes[0].axhline(0,color='black',linestyle='--');axes[0].set(xlabel='SNR (dB)',ylabel='KNN minus best other classifier (pp)',title='KNN leads at every tested SNR below −3 dB')
    for model in ('QDA','LDA','LSVM','KNN'):
        i=MODELS.index(model);keep=(x>=1.5)&(x<=3);axes[1].plot(x[keep],noise[i,keep],'.-',label=model)
    axes[1].axhline(80,color='black',linestyle='--');axes[1].axvline(2,color='gray',linestyle=':');axes[1].set(xlabel='SNR (dB)',ylabel='Accuracy (%)',title='The 80% threshold: 2 dB vs strictly above 2 dB');axes[1].legend()
    save(fig,'noise-thresholds')
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained'); xfilter=rows['figure4',1,3,'LDA']['x']
    for b,label in [(1,'MFCC only'),(2,'MFCC + delta'),(3,'MFCC + delta + delta-delta')]:ax.plot(xfilter,values('figure4',b,3,'LDA'),label=label)
    ax.axvline(201,color='gray',linestyle=':');ax.set(xlabel='Number of filters',ylabel='Accuracy (%)',ylim=(40,100),title='Counterexample to “derivatives have no positive effect”: LDA, 0 dB, all coefficients');ax.legend()
    save(fig,'lda-counterexample')
    claims = [
        ('C01', 'Core feasibility', '§1 p.659; §5.1 p.663; §7 p.667', 'Supported within the evaluated protocol',
         'MFCC fingerprints distinguish individual same-model drone configurations in the closed-set experiment.',
         f"QDA accuracy is {closed['QDA']['accuracy']:.4f}%. All four Table 2 metrics exceed 96%. The data support feasibility in this indoor, within-recording evaluation, not universal physical uniqueness or deployment security."),
        ('C02', 'Body and propeller combinations', '§1 p.659; §5.1 p.663; §7 p.667', 'Supported as observed separability',
         'A new combination of body and propellers can be treated as a distinct identity.',
         f"QDA macro recall within the original, replacement-x and replacement-y groups is {groups['Original propellers']:.2f}%, {groups['Replacement set x']:.2f}% and {groups['Replacement set y']:.2f}%. The 24-class result still supports separability. This is not an isolated causal test of motor versus propeller contributions or a held-out-session test."),
        ('C03', 'Clean 8-drone classifier preference', '§4.2 p.663; §7 p.667', 'Supported as a dominant group',
         'QDA, LDA and LSVM outperform the alternatives without AWGN.',
         f"One of these three wins in all 450 clean configurations. They strictly occupy all top-three positions in {dominance['figure3']['qda_lda_lsvm_strictly_above_other_models']}/450. At 26 filters with one-third coefficients, RF enters the top three in two configurations. A universal strict ranking would be too strong."),
        ('C04', 'Clean 24-drone metrics', 'Table 2 p.663; §7 p.667', 'Supported',
         'QDA, LDA, LSVM and KNN each exceed 90% on accuracy, precision, recall and F1; QDA is best.',
         'All 16 model/metric comparisons pass the stated 90% threshold, and QDA leads all four metrics. Accuracies: QDA 96.2034%, LDA 92.4298%, LSVM 93.6781%, KNN 90.4970%. This is a numerical ranking, not a new statistical significance claim.'),
        ('C05', 'KNN at low SNR', '§5.2 p.664; §7 p.667', 'Supported across the tested range',
         'KNN is the most effective method when SNR is below −3 dB.',
         f"KNN wins all {noise_summary['knn_below_minus3']['strict_wins']}/20 sampled points from −8 to −3.25 dB. Its smallest lead over the best other classifier is {noise_summary['knn_below_minus3']['minimum_margin_pp']:.4f} pp. This does not extrapolate below −8 dB or to untested continuous SNR values."),
        ('C06', 'LDA under 0 dB AWGN', '§4.2.4 p.663; §7 p.667', 'Dominance retained; universal wording needs qualification',
         'LDA has the highest accuracy in all configurations under 0 dB AWGN.',
         f"LDA is highest in {dominance['figure4']['co_highest_counts']['LDA']}/450 configurations. Ten low-filter/one-third configurations are exceptions. The conclusion that LDA is a strong choice remains supported; the word all is not supported by the new run."),
        ('C07', 'QDA on registered drones', 'Abstract p.658; §5.4 p.665–666; §7 p.667', 'Supported',
         'QDA has the highest mean recall on registered drones, about 94.19%.',
         f"Ten-trial macro recall is {opened['table4']['QDA']:.4f}%, rank {ranks['table4']['QDA']}. The statement concerns the ten-trial mean: QDA is not first in every individual trial."),
        ('C08', 'QDA on unknown drones', 'Abstract p.658; §5.4.2 p.666; §7 p.667', 'Magnitude retained; exact rank and number change',
         'QDA has the third-highest mean recall on unregistered drones, 82.35%.',
         f"Macro rejection is {opened['table5']['QDA']:.4f}% (−0.1527 pp from the printed value), rank {ranks['table5']['QDA']} in the frozen primary run. Pooled rejection is 82.2340%. QDA remains strong, but third and 82.35% cannot label the new measurements. The complementary macro miss rate is about 17.80%; rejection is not perfect."),
        ('C09', 'QDA as an overall choice', '§5.4.2 p.666; §7 p.667', 'Supported as a trade-off, not a universal optimum',
         'QDA offers the best overall authentication performance.',
         f"QDA has higher registered recall and unknown rejection than {len(dominated)} of the other seven classifiers on the two ten-trial means. RF still rejects more unknowns (84.4099% versus 82.1973%), while recognizing fewer registered drones (75.4722% versus 94.1884%). QDA is a strong balanced choice; no predeclared utility or error-cost function establishes an unconditional single best model."),
        ('C10', 'Derivative effect on RBF-SVM and KNN', '§4.2.2–4 p.662–663', 'Clean-set direction retained; magnitude and generality change',
         'Adding derivatives reduces RBF-SVM and KNN accuracy; KNN works well only with MFCC.',
         'In the clean grid, each added-feature variant is lower than its matched MFCC-only result in all 150/150 configurations for both classifiers. For KNN, mean losses are 1.8470 pp and 2.1647 pp. At 201 filters/all coefficients it remains 95.1630% or 95.0589%, versus 96.1165% with MFCC only. Thus the direction survives but good only with MFCC overstates the new degradation. At 0 dB there are also configurations with small gains; the decrease is not universal.'),
        ('C11', 'No positive effect from derivatives', '§4.2.4 p.663', 'Not supported as a general claim',
         'DMFCC and DDMFCC do not have a positive effect on drone authentication.',
         'MFCC + delta + delta-delta improves LDA and LSVM in every matched configuration: 150/150 each on clean data and 150/150 each at 0 dB. At 201 filters/all coefficients, LDA at 0 dB rises from 93.7240% to 97.0700% (+3.3460 pp); LSVM rises from 90.6207% to 92.6318% (+2.0111 pp). These are observed gains under the revised feature definitions, not evidence of statistical significance or an attribution to one individual code fix.'),
        ('C12', 'Practical feature configuration', '§4.1–4.2 p.662–663; §7 p.667', 'Effective baseline retained; optimality not established',
         'Use 1000 ms frames, 50% overlap, 201 filters and MFCC coefficients 2–201.',
         'The unchanged DS2 setting still produces QDA accuracy 96.2034% and strong open-set results. It remains a useful baseline. The derivative results do not establish MFCC-only as optimal. Figure 2 samples 970 and 1020 ms, not exactly 1000 ms; it supports a broad longer-frame trend. Overlap was held fixed, so its optimality was not tested.'),
        ('C13', 'More filters and a plateau near 200', '§4.2 p.662–663', 'Broad trend retained; plateau needs qualification',
         'More filters improve accuracy, with a bottleneck around 200 filters.',
         'All 72 curves in each clean/noisy grid end higher at 271 than at 26 filters; this is not strict stepwise monotonicity. In the clean grid, the largest improvement available beyond 201 filters is 2.3232 pp across the scanned configurations. At 0 dB, QDA with one-third MFCC coefficients gains 10.4196 pp from 201 to 271 filters. A universal noisy-data plateau at 200 is not supported.'),
        ('C14', 'The 2 dB and high-SNR thresholds', '§5.2 p.664; §6 p.666', 'Broad robustness retained; boundaries need qualification',
         'QDA/LDA/LSVM/KNN exceed 80% from 2 dB, with almost no further growth after 10 dB.',
         'At exactly 2 dB, accuracies are 80.0104%, 84.1118%, 79.8723% and 79.6192%, respectively. All four exceed 80% at every sampled point strictly above 2 dB (2.25–15 dB). Section 6 says greater than 2 dB and remains supported in that sampled sense; at least 2 dB is slightly too strong. From 10 to 15 dB, gains range from 0.1150 pp (RBF-SVM) to 4.6422 pp (DT), so a universal no-growth statement is too strong.'),
        ('C15', 'RBF-SVM rejects almost everything', '§5.4.1–2 p.665–666', 'Not supported for the frozen primary encoding',
         'RBF-SVM achieves 98.10% unknown rejection by classifying almost all drones as unknown, with only 13.05% registered recall.',
         'The primary run instead gives 79.2360% registered recall and 30.9563% rejection. A separate background-last control gives 13.0362% and 97.9357%, close to the historical behavior. This establishes a configured-procedure sensitivity to class order. It is not a replacement result, a converged-SVM guarantee, or a rerun of all eight models under the alternate encoding.'),
        ('C16', 'Novelty, physical cause and deployment scope', '§1 p.659; §6 p.666–667', 'Not established by this rerun',
         'Novelty and physical-source interpretation; wear/reassembly observations; outdoor and replay robustness identified as future work.',
         'A computational rerun cannot verify first-work priority or isolate the physical source of the fingerprint. Wear/reassembly observations were not repeated. Outdoor noise, held-out recording sessions, other drone models and replay attacks were not tested; the paper itself identifies several of these as future work. These limits are not newly demonstrated failures caused by the refactor.'),
    ]
    # Direct threshold checks support the key accepted statements without inventing a tolerance.
    assert all(closed[m][metric] > 90 for m in ('QDA','LDA','LSVM','KNN') for metric in ('accuracy','macro_precision','macro_recall','macro_f1'))
    assert all(closed['QDA'][metric] > max(closed[m][metric] for m in MODELS if m!='QDA') for metric in ('accuracy','macro_precision','macro_recall','macro_f1'))
    assert all(noise_summary['minimum_above_2_db'][m] > 80 for m in ('QDA','LDA','LSVM','KNN'))
    assert ranks['table4']['QDA']==1 and ranks['table5']['QDA']==2 and len(dominated)==6
    records=[dict(zip(('id','topic','paper_location','assessment','paper_claim_paraphrase','evidence_and_limit'), row)) for row in claims]
    atomic_json(output/'claims.json', {'scope':'Assessment of original claims, not an equivalence or significance certificate.', 'claims':records})
    summary='The central feasibility result and QDA as a strong balanced choice remain supported within the original evaluation setting. An unqualified statement that every conclusion is unchanged is not supported: the derivative-feature interpretation, exact rankings, RBF-SVM open-set behavior and several universal or boundary statements require revision.'
    method='All 8,352 plot points were checked against per-task exported metrics and gallery hashes. Paired derivative comparisons match filter count, coefficient fraction, classifier, test noise and dataset, changing the feature-block setting; each contrast covers 150 related configurations. This audits the maintained implementation and does not isolate individual old-to-new code changes. Grid averages and ten shared-data partitions are descriptive, not independent replicates. No statistical equivalence margin, independent multi-seed replication, significance test or new training is introduced.'
    table=''.join('<tr>'+''.join('<td>'+escape(value)+'</td>' for value in row)+'</tr>' for row in claims)
    charts=''.join(f'<figure><a href="{name}.svg"><img src="{name}.png" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>' for name,caption in [
        ('open-set','QDA remains close to its published means. Primary RBF-SVM behavior changes sharply, and QDA unknown-rejection rank is now second.'),
        ('derivatives','Mean percentage-point differences from matched MFCC-only settings. KNN still loses modestly, while LDA and LSVM gain when both derivatives are included. These grid means are not independent statistical replicates.'),
        ('lda-counterexample','A concrete counterexample to a blanket no-benefit claim: at 201 filters/all coefficients, LDA improves by 3.3460 pp at 0 dB.'),
        ('noise-thresholds','KNN leads at all 20 sampled low-SNR points. At exactly 2 dB, LSVM and KNN remain just below 80%.')])
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper conclusions: claim-by-claim assessment</title><style>
body{{font:16px/1.6 system-ui;background:#f4f6f7;color:#234;max-width:1500px;margin:30px auto;padding:0 20px}}.note{{background:#fff0d8;border-left:4px solid #a77325;padding:18px}}table{{border-collapse:collapse;width:100%;background:white;min-width:1000px}}td,th{{border-bottom:1px solid #dde3e7;padding:12px;text-align:left;vertical-align:top}}td:nth-child(6){{width:40%}}.scroll{{overflow-x:auto}}figure{{background:white;padding:18px;margin:24px 0}}img{{width:100%;height:auto}}a{{color:#17627d}}figcaption{{font-size:14px}}code{{overflow-wrap:anywhere}}</style></head><body><h1>Paper conclusions: claim-by-claim assessment</h1><p class="note">{summary}</p><p>{method}</p><p>Paper locations use printed page numbers; PDF page = printed page − 657. References are to the author-provided ACSAC manuscript. Numerical differences are descriptive; no predeclared equivalence tolerance exists for this audit.</p><p><a href="claims.json">Claim ledger</a> · <a href="measurements.json">Computed evidence</a></p><div class="scroll"><table><thead><tr><th>ID</th><th>Topic</th><th>Paper location</th><th>Assessment</th><th>Original claim (paraphrased)</th><th>Current evidence and limitation</th></tr></thead><tbody>{table}</tbody></table></div>{charts}<h2>Suggested research statement</h2><blockquote>The updated implementation continues to support the paper's central feasibility findings and QDA as a strong balanced choice for the evaluated closed-set and open-set tasks. Some feature-extraction findings, exact rankings and numerical thresholds differ; these changes are documented rather than treated as exact reproduction of every published conclusion.</blockquote><p>This statement does not certify real-world security, new-environment generalization, statistical significance or historical novelty. No published result or frozen primary measurement was replaced.</p></body></html>'''
    (output/'index.html').write_text(html)
    markdown='# Paper conclusions: claim-by-claim assessment\n\n'+summary+'\n\n'+method+'\n'
    for row in records:
        markdown+=f"\n## {row['id']} — {row['topic']}\n\nPaper: {row['paper_location']}\n\nClaim: {row['paper_claim_paraphrase']}\n\nAssessment: **{row['assessment']}**\n\n{row['evidence_and_limit']}\n"
    (output/'assessment.md').write_text(markdown)
    print(json.dumps({'status':'passed','verified_points':8352,'qda_ranks':{k:v['QDA'] for k,v in ranks.items()},'noise':noise_summary},indent=2))


if __name__=='__main__':
    main()
