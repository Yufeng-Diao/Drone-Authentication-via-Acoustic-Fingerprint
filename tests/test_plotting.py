"""Scientific series, explicit missingness, trusted result collection and exports."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from drone_fingerprint.config import MODELS, Settings
from drone_fingerprint.data import sha256
from drone_fingerprint.experiments import paper_tasks
from drone_fingerprint.features import feature_metadata
from drone_fingerprint.metrics import classification_metrics, open_set_metrics
from drone_fingerprint.results import PRESETS, collect_results, figure_series, paper_tables
from drone_fingerprint.plotting import draw_paper_figure, render_results

PAPER = Path(__file__).resolve().parents[1] / 'config/paper'
PROTOCOL = json.loads((PAPER / 'protocol.json').read_text())
PARTITIONS = json.loads((PAPER / 'attack_partitions.json').read_text())


def result_fixture(root, presets=None, models=None, wrong_prediction=False):
    tasks = paper_tasks(PROTOCOL, PARTITIONS, presets or {'table2': {}}, models or ['GNB'])
    inputs = {key: {'path': str(PAPER / name), 'sha256': sha256(PAPER / name)} for key, name in
              (('protocol', 'protocol.json'), ('recordings_manifest', 'recordings.json'),
               ('subset_manifest', 'validation_subset.json'), ('partitions_manifest', 'attack_partitions.json'))}
    entries = {}
    for task in tasks:
        directory = root / task['id'] / 'attempt-001'
        directory.mkdir(parents=True)
        settings = Settings(**task['settings'])
        labels = list(settings.identities) if not settings.partition else [0] + settings.partition['registered']
        original = np.asarray(settings.identities if not settings.partition else settings.partition['registered'] + settings.partition['attack'])
        truth = original.copy() if not settings.partition else np.where(np.isin(original, settings.partition['attack']), 0, original)
        prediction = truth.copy()
        if wrong_prediction:
            prediction[0] = labels[-1]
        np.savez(directory / 'predictions.npz', truth=truth, prediction=prediction, original_identity=original)
        (directory / 'model.joblib').write_bytes(b'Not a pickle: the plotter must never deserialize models.')
        metrics = classification_metrics(truth, prediction, labels)
        if settings.partition:
            metrics.update(open_set_metrics(original, prediction, settings.partition))
        result = {'schema_version': 2, 'status': 'passed', 'task_id': task['id'],
                  'configuration': asdict(settings), 'labels': labels, 'feature_metadata': feature_metadata(settings),
                  'provenance': {'experiment': task['preset'], 'config_id': task['id'], **{k+'_sha256': v['sha256'] for k,v in inputs.items()}},
                  'environment': {'packages': {'drone-acoustic-fingerprint': 'fixture'}}, 'metrics': metrics,
                  'test_rows': len(truth), 'confusion_matrix': confusion_matrix(truth, prediction, labels=labels).tolist(),
                  'segments': [], 'model_parameters': {}, 'warnings': [], 'timings_s': {},
                  'artifacts': {name: {'bytes': (directory/name).stat().st_size, 'sha256': sha256(directory/name)} for name in ('model.joblib','predictions.npz')}}
        (directory / 'result.json').write_text(json.dumps(result))
        entries[task['id']] = {'status': 'completed', 'task_id': task['id'], 'attempts': [{'status':'completed','directory':str(directory.relative_to(root))}]}
    (root/'plan.json').write_text(json.dumps({'schema_version':1,'tasks':tasks,'inputs':inputs}))
    (root/'sweep.json').write_text(json.dumps({'schema_version':1,'plan_id':'fixture','tasks':entries}))
    return root


def test_saved_results_need_no_audio_features_or_model_deserialization(tmp_path, monkeypatch):
    import joblib
    from drone_fingerprint import models, features, data
    def forbidden(*args, **kwargs):
        pytest.fail('Plotting must not deserialize models, train or read audio')
    for module, name in ((joblib,'load'),(models,'make_model'),(features,'extract_segment'),(data,'read_recording')):
        monkeypatch.setattr(module,name,forbidden)
    root=result_fixture(tmp_path/'run')
    bundle=collect_results([root,root],PAPER)
    assert len(bundle['results'])==1 and bundle['duplicates']==1 and not bundle['full_dataset']
    assert len(next(iter(bundle['sources'].values())))==2


@pytest.mark.parametrize('mutation,message', [('metrics','metrics'),('configuration','configuration'),('provenance','provenance')])
def test_inconsistent_saved_metadata_rejected(tmp_path,mutation,message):
    root=result_fixture(tmp_path/'run')
    path=next(root.glob('*/attempt-001/result.json'));result=json.loads(path.read_text())
    if mutation=='metrics':result['metrics']['accuracy']=0.1
    elif mutation=='configuration':result['configuration']['n_filters']=50
    else:result['provenance']['subset_manifest_sha256']='wrong'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match=message):collect_results([root],PAPER)


def test_conflicting_repetitions_rejected(tmp_path):
    first=result_fixture(tmp_path/'first')
    second=result_fixture(tmp_path/'second',wrong_prediction=True)
    with pytest.raises(ValueError,match='Conflicting repetitions'):collect_results([first,second],PAPER)


def test_mixed_runtime_rejected(tmp_path):
    a=result_fixture(tmp_path/'a',models=['GNB']);b=result_fixture(tmp_path/'b',models=['DT'])
    path=next(b.glob('*/attempt-001/result.json'));r=json.loads(path.read_text());r['environment']['python']='different';path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='runtimes'):collect_results([a,b],PAPER)


def test_failed_latest_attempt_never_falls_back_to_old_success(tmp_path):
    root=result_fixture(tmp_path/'run',models=['GNB','DT'])
    path=root/'sweep.json';s=json.loads(path.read_text());entry=next(iter(s['tasks'].values()))
    entry['status']='failed';entry['attempts'].append({'status':'failed','directory':'failed-attempt'})
    path.write_text(json.dumps(s))
    bundle=collect_results([root],PAPER)
    assert len(bundle['results'])==1 and bundle['skipped'][0]['status']=='failed'


def test_path_escape_and_checksum_corruption_rejected(tmp_path):
    root=result_fixture(tmp_path/'run')
    path=root/'sweep.json';s=json.loads(path.read_text());entry=next(iter(s['tasks'].values()));entry['attempts'][-1]['directory']='../outside'
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='escapes'):collect_results([root],PAPER)
    root=result_fixture(tmp_path/'corrupt');next(root.glob('*/attempt-001/predictions.npz')).write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum|size'):collect_results([root],PAPER)


def test_gaps_units_and_nine_panel_order(tmp_path):
    import matplotlib.pyplot as plt
    root=result_fixture(tmp_path/'frames',presets={'figure2':{'frame_ms':[20,120]}})
    bundle=collect_results([root],PAPER)
    rows=figure_series(bundle,'figure2');line=next(r for r in rows if r['model']=='GNB')
    assert line['x'][:3]==[20,70,120] and line['y'][:3]==[100.0,None,100.0]
    fig,_=draw_paper_figure(bundle,'figure2')
    assert np.isnan(fig.axes[0].lines[-1].get_ydata()[1]) and len(fig.axes[0].lines)==8
    plt.close(fig)
    root=result_fixture(tmp_path/'filters',presets={'figure3':{'n_filters':[26]}})
    bundle=collect_results([root],PAPER)
    rows=figure_series(bundle,'figure3')
    assert len(rows)==72 and [(r['blocks'],r['portion']) for r in rows[::8]]==[(b,p) for b in (1,2,3) for p in (1,2,3)]
    assert all(r['x'][0]==26 and r['x'][-1]==271 and r['x_unit']=='filters' for r in rows)
    fig,_=draw_paper_figure(bundle,'figure4')
    assert len(fig.axes)==9 and all(len(ax.lines)==8 for ax in fig.axes)
    assert fig.axes[3].get_title().startswith('(d) MFCC + delta · one third')
    plt.close(fig)
    snr=figure_series(bundle,'figure5')
    assert len(snr)==8 and snr[0]['x'][0]==-8 and snr[0]['x'][-1]==15 and len(snr[0]['x'])==93


def test_ten_trial_mean_requires_ten_and_zero_is_not_missing(tmp_path):
    root=result_fixture(tmp_path/'run',presets={'tables4_5':{}})
    bundle=collect_results([root],PAPER)
    results=list(bundle['results'].values())
    results[0]['metrics']['registered_macro_recall']=0.0
    tables=paper_tables(bundle)
    assert tables['table4']['rows'][-2]['GNB']==90.0
    assert tables['table4']['rows'][-2]['QDA'] is None
    bundle['results'].pop(next(iter(bundle['results'])))
    tables=paper_tables(bundle)
    assert tables['table4']['rows'][-2]['GNB'] is None
    assert tables['table4']['rows'][-1]['GNB']=='9/10'
    assert tables['table5']['metric']=='attack_macro_rejection'
    assert tables['table5_pooled_comparison']['metric']=='attack_pooled_rejection'
    assert tables['table3']['rows']==PARTITIONS['partitions']


def test_headless_gallery_exports_and_no_input_changes(tmp_path,monkeypatch):
    root=result_fixture(tmp_path/'run')
    before={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
    monkeypatch.delenv('DISPLAY',raising=False)
    output=tmp_path/'gallery'
    manifest=render_results([root],PAPER,output)
    assert not manifest['full_grid'] and manifest['verified_results']==1
    assert manifest['scope'].startswith('INCOMPLETE PREVIEW')
    for name in ('figure2','figure3','figure4','figure5','diagnostic_confusion','diagnostic_open_recall'):
        for ext in ('png','svg','pdf'):assert (output/f'{name}.{ext}').stat().st_size>100
    assert len(json.loads((output/'plot-data.json').read_text())['series'])==160
    assert (output/'table2.md').read_text().count('| —')==28
    assert 'INCOMPLETE PREVIEW' in (output/'figure3.svg').read_text()
    assert before=={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):render_results([root],PAPER,output)


def test_relative_model_references_survive_moving_a_run_tree(tmp_path):
    from drone_fingerprint.synthetic import create_fixture
    from drone_fingerprint.data import select_recordings
    from drone_fingerprint.workflow import run_experiment
    from drone_fingerprint.artifacts import completed_result
    manifest,subset=create_fixture(tmp_path/'audio');records=select_recordings(manifest,subset,(1,2))
    tree=tmp_path/'original';settings=Settings()
    run_experiment(records,manifest.parent,settings,tree/'clean',{})
    result=run_experiment(records,manifest.parent,replace(settings,test_snr_quarter_db=0),tree/'noisy',{},trained_from=tree/'clean')
    assert result['model_reference']['path']=='../clean'
    moved=tmp_path/'moved';shutil.move(tree,moved)
    assert completed_result(moved/'noisy')['task_id']==result['task_id']
