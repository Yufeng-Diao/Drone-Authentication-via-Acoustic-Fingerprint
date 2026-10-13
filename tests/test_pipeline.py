"""Streaming equivalence, cache identity, corruption and interrupted publication."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from drone_fingerprint.config import Settings
from drone_fingerprint.data import inspect_recording, PCMReader, read_recording, select_recordings, sha256
from drone_fingerprint.features import extract_base, extract_segment, iter_base_frames, iter_selected_frames
from drone_fingerprint.noise import add_noise, NoisyReader
from drone_fingerprint.pipeline import array_writer, build_recording_index, cache_lock, prepare_features
from drone_fingerprint.storage import atomic_json
from drone_fingerprint.synthetic import create_fixture


@pytest.fixture
def dataset(tmp_path):
    manifest, subset = create_fixture(tmp_path/'audio')
    records = select_recordings(manifest, subset, (1, 2))
    return manifest, records


class ArrayReader:
    def __init__(self, data):
        self.data = data
        self.maximum_read = 0
        self.info = {'frames': len(data), 'signal_power': float(np.mean(data**2))}

    def read(self, start, end):
        self.maximum_read = max(self.maximum_read, end-start)
        return self.data[start:end]


@pytest.mark.parametrize('frame_ms', [20, 70, 1000, 2520])
@pytest.mark.parametrize('batch', [1, 3, 32])
def test_batched_mfcc_preserves_preemphasis_overlap_and_final_padding(frame_ms, batch):
    rng = np.random.default_rng(21)
    signal = rng.normal(size=160123)
    settings = Settings(frame_ms=frame_ms, n_filters=26, blocks=3, portion=2)
    reader = ArrayReader(signal)
    base = np.concatenate(list(iter_base_frames(reader, 3, len(signal)-2, settings, batch)))
    reference = extract_base(signal[3:-2], settings)
    np.testing.assert_allclose(base, reference, atol=1e-12, rtol=1e-12)
    selected = np.concatenate(list(iter_selected_frames(base, settings, batch_frames=3)))
    np.testing.assert_allclose(selected, extract_segment(signal[3:-2], settings), atol=1e-12, rtol=1e-12)
    assert reader.maximum_read <= (batch + 1) * frame_ms * 44100 / 1000 + 1


def test_chunked_noise_matches_full_timeline_across_skips_and_overlap():
    signal = np.random.default_rng(1).integers(-2000,2001,size=900_013).astype(float)
    full, reference = add_noise(signal, 'a'*64, 2022, -17)
    reader = NoisyReader(ArrayReader(signal), 'a'*64, 2022, -17)
    for start, end in [(700, 1123), (999, 125678), (125677, 420001), (700000, 899900)]:
        np.testing.assert_array_equal(reader.read(start,end), full[start:end])
    metadata = reader.finish()
    assert metadata['seed'] == reference['seed']
    assert metadata['signal_power'] == reference['signal_power']
    assert metadata['measured_full_recording_snr_db'] == pytest.approx(reference['measured_full_recording_snr_db'],abs=1e-12)


def test_recording_index_verifies_once_and_bounded_reader_matches(dataset):
    manifest, records = dataset
    record = records[0]
    info = inspect_recording(manifest.parent,record)
    reference = read_recording(manifest.parent,record)
    assert info['signal_power'] == np.mean(reference**2)
    reader = PCMReader(manifest.parent,record,info)
    try:
        np.testing.assert_array_equal(reader.read(13,200001),reference[13:200001])
        with pytest.raises(ValueError,match='bounds'):
            reader.read(-1,100)
    finally:
        reader.close()


def test_cache_hit_shared_across_models_and_base_shared_across_selections(tmp_path,dataset,monkeypatch):
    manifest,records=dataset
    cache=tmp_path/'cache'
    settings=Settings(blocks=3,portion=1)
    first=prepare_features(records,manifest.parent,settings,cache)
    assert first[-1]['base_misses'] == 2 and first[-1]['selected_misses'] == 2
    signatures={str(p.relative_to(cache)):(sha256(p),p.stat().st_mtime_ns) for p in cache.rglob('*') if p.is_file()}
    def forbidden(*args,**kwargs):
        pytest.fail('A populated cache must not read/decode audio or extract MFCCs')
    monkeypatch.setattr('drone_fingerprint.pipeline.PCMReader',forbidden)
    second=prepare_features(records,manifest.parent,replace(settings,model='RF'),cache)
    assert second[-1]['selected_hits'] == 2 and second[-1]['hashed_recordings'] == 0
    assert signatures=={str(p.relative_to(cache)):(sha256(p),p.stat().st_mtime_ns) for p in cache.rglob('*') if p.is_file()}
    from drone_fingerprint.config import MODELS
    for name in MODELS:
        reused=prepare_features(records,manifest.parent,replace(settings,model=name),cache)
        assert reused[-1]['selected_hits']==2 and reused[-1]['cache_keys']==first[-1]['cache_keys']
    for split in ('train','test'):
        assert isinstance(second[0][split],np.memmap) and not second[0][split].flags.writeable
        np.testing.assert_array_equal(first[0][split],second[0][split])
    third=prepare_features(records,manifest.parent,replace(settings,portion=2),cache)
    assert third[-1]['base_hits'] == 2 and third[-1]['base_misses'] == 0
    assert len([p for p in cache.rglob('*') if p.is_file()]) == 14  # Base + two selections per split, index, lock.


def test_noisy_cache_reuses_clean_training(tmp_path,dataset):
    manifest,records=dataset
    cache=tmp_path/'cache'
    clean=prepare_features(records,manifest.parent,Settings(),cache)
    noisy=prepare_features(records,manifest.parent,Settings(test_snr_quarter_db=0),cache)
    assert noisy[-1]['selected_hits'] == 1 and noisy[-1]['base_misses'] == 1
    assert clean[-1]['cache_keys']['train']==noisy[-1]['cache_keys']['train']
    np.testing.assert_array_equal(clean[0]['train'],noisy[0]['train'])
    assert not np.array_equal(clean[0]['test'],noisy[0]['test'])


def test_changed_raw_and_corrupt_feature_cache_are_rejected(tmp_path,dataset):
    manifest,records=dataset
    cache=tmp_path/'cache'
    first=prepare_features(records,manifest.parent,Settings(),cache)
    array=cache/first[-1]['cache_keys']['test']['selected']/'features.npy'
    with array.open('r+b') as f:
        f.seek(-1,2); f.write(b'\xff')
    with pytest.raises(ValueError,match='Invalid feature cache'):
        prepare_features(records,manifest.parent,Settings(),cache)
    audio=manifest.parent/records[0]['path']
    with audio.open('r+b') as f:
        f.seek(-1,2); f.write(b'\xff')
    with pytest.raises(ValueError,match='checksum'):
        prepare_features(records,manifest.parent,Settings(),cache)


def test_inventory_uses_one_walk_and_tasks_use_none(tmp_path,dataset,monkeypatch):
    manifest,records=dataset
    import os
    walk=os.walk
    count=[]
    def counted(*args,**kwargs):
        count.append(1)
        return walk(*args,**kwargs)
    monkeypatch.setattr('drone_fingerprint.pipeline.os.walk',counted)
    result=build_recording_index(json.loads(manifest.read_text()),manifest.parent,tmp_path/'cache')
    assert result['directory_walks']==1 and len(count)==1
    prepared=prepare_features(records,manifest.parent,Settings(),tmp_path/'cache')
    assert prepared[-1]['hashed_recordings']==0 and len(count)==1


def test_interrupted_array_publication_recovers_without_a_false_hit(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    script='''import os,sys
from pathlib import Path
from drone_fingerprint.pipeline import array_writer
with array_writer(Path(sys.argv[1]), ('a'*64), {'test': True}, (8,3)) as (array, metadata):
    array[:]=42
    array.flush()
    os._exit(91)
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(cache)])
    assert result.returncode==91 and not (cache/('a'*64)).exists()
    assert (cache/('.building-'+'a'*64)).exists()
    with cache_lock(cache,tmp_path/'audio'):
        assert not (cache/('.building-'+'a'*64)).exists()
        with array_writer(cache,('a'*64),{'test':True},(8,3)) as (array,metadata):
            array[:]=1
    with (cache/('a'*64)/'metadata.json').open() as f:
        assert json.load(f)['status']=='complete'
    assert np.load(cache/('a'*64)/'features.npy').sum()==24


def test_atomic_completion_preserves_previous_json_on_failed_replace(tmp_path,monkeypatch):
    path=tmp_path/'result.json';atomic_json(path,{'status':'before'})
    def failed(*args): raise OSError('Injected publication failure')
    monkeypatch.setattr('drone_fingerprint.storage.os.replace',failed)
    with pytest.raises(OSError,match='Injected'):
        atomic_json(path,{'status':'after'})
    assert json.loads(path.read_text())=={'status':'before'}
    assert list(tmp_path.iterdir())==[path]


def test_explicit_verification_and_index_corruption(tmp_path,dataset):
    manifest,records=dataset
    cache=tmp_path/'cache'
    prepare_features(records,manifest.parent,Settings(),cache)
    verified=prepare_features(records,manifest.parent,Settings(),cache,verify_inputs=True)
    assert verified[-1]['hashed_recordings']==4 and verified[-1]['selected_hits']==2
    index=next(cache.glob('recordings-*.json'))
    raw=json.loads(index.read_text());raw['recordings'][records[0]['path']]['signal_power']=0
    index.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Invalid recording index'):
        prepare_features(records,manifest.parent,Settings(),cache)


def test_two_processes_share_one_cache_publication(tmp_path,dataset):
    manifest,records=dataset
    script='''import json,sys
from pathlib import Path
from drone_fingerprint.config import Settings
from drone_fingerprint.pipeline import prepare_features
manifest=Path(sys.argv[1])
records=json.loads(manifest.read_text())['recordings']
result=prepare_features(records,manifest.parent,Settings(),Path(sys.argv[2]))
print(json.dumps(result[-1]))
'''
    args=[sys.executable,'-I','-B','-c',script,str(manifest),str(tmp_path/'cache')]
    processes=[subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    reports=[]
    for process in processes:
        stdout,stderr=process.communicate(timeout=45)
        assert process.returncode==0,stderr
        reports.append(json.loads(stdout.strip().splitlines()[-1]))
    assert sum(r['base_misses'] for r in reports)==2
    assert sum(r['selected_hits'] for r in reports)==2
    assert not list((tmp_path/'cache').glob('.building-*'))


def test_low_disk_space_fails_before_array_allocation(tmp_path,monkeypatch):
    from collections import namedtuple
    disk=namedtuple('disk','total used free')
    monkeypatch.setattr('drone_fingerprint.pipeline.shutil.disk_usage',lambda _:disk(1,1,0))
    with pytest.raises(ValueError,match='Insufficient free cache space'):
        with array_writer(tmp_path,'e'*64,{},(100,10)):
            pytest.fail('Insufficient disk should reject before yielding')
    assert not list(tmp_path.iterdir())


def test_cache_staging_cleanup_is_limited_to_owned_keys(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    unrelated=cache/'.building-not-a-cache-key';unrelated.mkdir()
    marker=unrelated/'keep.txt';marker.write_text('Keep this unrelated file.')
    with cache_lock(cache,tmp_path/'audio'):
        assert marker.exists()
    with pytest.raises(ValueError,match='Cache key'):
        with array_writer(cache,'../outside',{},(1,1)):
            pytest.fail('Invalid keys must be rejected before allocation')


def test_numerical_thread_limits_are_applied_and_restored(tmp_path,dataset,monkeypatch):
    from sklearn.naive_bayes import GaussianNB
    from threadpoolctl import threadpool_info, threadpool_limits
    from drone_fingerprint.workflow import run_experiment
    model=GaussianNB()
    original_fit=model.fit
    def checked_fit(features,labels):
        assert all(p['num_threads']==1 for p in threadpool_info())
        return original_fit(features,labels)
    monkeypatch.setattr('drone_fingerprint.workflow.make_model',lambda *args:model)
    monkeypatch.setattr(GaussianNB,'fit',lambda self,features,labels: checked_fit(features,labels))
    manifest,records=dataset
    with threadpool_limits(limits=2):
        result=run_experiment(records,manifest.parent,Settings(),tmp_path/'run',{})
        assert result['thread_limit']==1
        assert all(p['num_threads']==2 for p in threadpool_info())


def test_recording_identities_join_to_paper_body_propeller_combinations():
    config=Path(__file__).resolve().parents[1]/'config/paper'
    rows=json.loads((config/'identities.json').read_text())['identities']
    recordings=json.loads((config/'recordings.json').read_text())['recordings']
    assert [r['identity'] for r in rows]==list(range(1,25))
    assert {r['identity'] for r in recordings if r['identity'] is not None}=={r['identity'] for r in rows}
    assert len({(r['body'],r['propeller_set']) for r in rows})==24
    assert [r['body'] for r in rows]==list('ABCDEFGH')*3
    assert [r['propeller_set'] for r in rows[:8]]==[f'{p}1-{p}4' for p in 'abcdefgh']
    assert {r['propeller_set'] for r in rows[8:16]}=={'x1-x4'}
    assert {r['propeller_set'] for r in rows[16:]}=={'y1-y4'}
