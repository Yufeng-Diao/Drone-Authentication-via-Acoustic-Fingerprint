"""Compare P2 and P3 in isolated Ubuntu processes, with external evidence.

Application-cache states are explicit. OS cache is uncontrolled; this tool never
requires root, clears system caches, or claims physical cold-disk performance.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import threading
import time


def file_inventory(root):
    import hashlib
    if not root.exists():
        return {}
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': checksum, 'mtime_ns': path.stat().st_mtime_ns}
    return result


def proc_io():
    return {k: int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def swap_counters():
    return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}


def child(measurement, arguments):
    before, started, points = proc_io(), time.perf_counter(), []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            status = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines())
            points.append({'elapsed_s':time.perf_counter()-started,'rss_kib':int(status['VmRSS'].split()[0])})
            stop.wait(0.1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    code=1
    try:
        from drone_fingerprint.cli import main
        sys.argv=['drone-auth',*arguments]
        code=main()
    finally:
        stop.set();thread.join()
        after=proc_io();usage=resource.getrusage(resource.RUSAGE_SELF)
        result={'exit_code':code,'wall_s':time.perf_counter()-started,'peak_rss_kib':usage.ru_maxrss,
                'user_cpu_s':usage.ru_utime,'system_cpu_s':usage.ru_stime,
                'io_delta':{k:after[k]-before[k] for k in before},'rss_trace':points,
                'scope':'CLI plus numerical imports; 100 ms RSS sampling; measurement JSON write excluded; OS page cache uncontrolled'}
        measurement.write_text(json.dumps(result,indent=2)+'\n')
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-python',type=Path,required=True)
    parser.add_argument('--candidate-python',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--audio-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh external evidence directory')
    parser.add_argument('--artifacts',type=Path,required=True,help='Fresh external run/cache directory')
    parser.add_argument('--repetitions',type=int,default=3)
    args=parser.parse_args()
    if args.repetitions < 3:
        parser.error('At least three measured repetitions are required')
    from drone_fingerprint.storage import create_output
    output=create_output(args.output,args.audio_root)
    artifacts=create_output(args.artifacts,args.audio_root)
    if output.is_relative_to(artifacts) or artifacts.is_relative_to(output):
        raise ValueError('Evidence and runtime artifacts must be separate')
    environment={key:'1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE')}
    environment['PYTHONPATH']=''
    env=dict(os.environ,**environment)
    results=[]
    for repetition in range(args.repetitions+1):
        purpose='warmup' if repetition==0 else f'repetition-{repetition}'
        cache=artifacts/f'{purpose}-cache'
        for mode in ('p2','p3-empty','p3-populated'):
            name=f'{purpose}-{mode}'
            run_dir=artifacts/name
            measurement=output/f'{name}.json'
            executable=args.baseline_python if mode=='p2' else args.candidate_python
            command=[str(executable.absolute()),'-I','-B',str(Path(__file__).resolve()),'child',str(measurement),
                     'run','--config',str(args.config.resolve()),'--audio-root',str(args.audio_root.resolve()),
                     '--output',str(run_dir),'--save-features']
            if mode!='p2':command+=['--cache-root',str(cache)]
            cached_before=file_inventory(cache) if mode=='p3-populated' else {}
            swap_before=swap_counters()
            record={'name':name,'mode':mode,'measured':repetition>0,'repetition':repetition,
                    'command':command,'cwd':str(Path.cwd()),'environment_overrides':environment,
                    'started_at':datetime.now(timezone.utc).isoformat(),'cache_before':cached_before}
            with (output/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(record,event='started'))+'\n')
            with (output/f'{name}.log').open('w') as log:
                completed=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            record.update(exit_code=completed.returncode,finished_at=datetime.now(timezone.utc).isoformat(),event='finished')
            swap_after=swap_counters()
            record['system_swap_pages_delta']={k:swap_after[k]-swap_before[k] for k in swap_before}
            with (output/'commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(f'{name}: exit {completed.returncode}',flush=True)
            if completed.returncode:
                raise RuntimeError(f'Benchmark child failed; see {output/name}.log')
            row=json.loads(measurement.read_text())
            run_result=json.loads((run_dir/'result.json').read_text())
            cache_after=file_inventory(cache) if mode!='p2' else {}
            if mode=='p3-populated' and cached_before!=cache_after:
                raise ValueError('A populated cache was modified during the measured run')
            record.update(row,run=str(run_dir),task_id=run_result['task_id'],metrics=run_result['metrics'],
                          cache=run_result.get('cache'),cache_after=cache_after,
                          run_inventory=file_inventory(run_dir))
            results.append(record)
            (output/'measurements.json').write_text(json.dumps(results,indent=2)+'\n')
    summaries={mode:{field:statistics.median(r[field] for r in results if r['measured'] and r['mode']==mode)
                     for field in ('wall_s','peak_rss_kib')} for mode in ('p2','p3-empty','p3-populated')}
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child':
        sys.exit(child(Path(sys.argv[2]),sys.argv[3:]))
    main()
