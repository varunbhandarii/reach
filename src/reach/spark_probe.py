"""Explicit saved-image measurements; never opens a camera or microphone."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import time

import numpy as np

from .config import load_config
from .log import EventLog
from .spark import ReadBackClient


def probe(args):
    if not 1 <= args.count <= 20:
        raise ValueError('Probe count must be between 1 and 20')
    cfg = load_config(args.config)
    images = [('correct',args.image,True)] if args.wrong_image else [('unrated',args.image,None)]
    if args.wrong_image:
        images.append(('wrong',args.wrong_image,False))
    combinations = [(e,d) for e in ('minimal','low') for d in ('low','auto')] if args.matrix else [(args.effort,args.detail)]
    root = args.log_dir/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    rows, token_counts, summaries = [], [], []
    with EventLog(root,save_frames=False) as log:
        log.write_manifest({'source':'saved_crops','phrase':args.phrase,'combinations':combinations,
                            'count':args.count,'privacy':{'standard_tier':True,'store':False}})
        counted = set()
        for effort,detail in combinations:
            cfg.setdefault('spark',{}).update(reasoning_effort=effort,detail=detail,cache='off')
            client = ReadBackClient(cfg,log)
            try:
                for label,path,expected in images:
                    jpeg = path.read_bytes()
                    if args.count_tokens and detail not in counted:
                        try:
                            tokens = client.count_tokens(jpeg,args.phrase)
                            token_counts.append({'detail':detail,'response':tokens})
                        except Exception as exc:
                            token_counts.append({'detail':detail,'error':f'token count failed ({type(exc).__name__})'})
                        counted.add(detail)
                    for _ in range(args.count):
                        client.request(len(rows)+1,jpeg,args.phrase,0)
                        while client.stats()['busy']:
                            time.sleep(.005)
                        result, = client.poll()
                        row = {**asdict(result),'effort':effort,'detail':detail,'crop':label,'expected_match':expected,
                               'correct':result.matches == expected if expected is not None and result.error is None else None}
                        rows.append(row)
                        print(json.dumps(row),flush=True)
                summaries.append({'effort':effort,'detail':detail,**client.stats()})
            finally:
                client.stop()
        groups = []
        for effort,detail in combinations:
            selected = [r for r in rows if r['effort']==effort and r['detail']==detail]
            values = [r['latency_ms'] for r in selected if not r['error']]
            groups.append({'effort':effort,'detail':detail,'n':len(selected),'valid':len(values),
                'correct':sum(r['correct'] is True for r in selected),'rated':sum(r['correct'] is not None for r in selected),
                'p50_ms':float(np.median(values)) if values else None,'p95_ms':float(np.percentile(values,95)) if values else None})
        result = {'completed':all(r['error'] is None for r in rows),'groups':groups,'trials':rows,
                  'usage_and_cost':summaries,'token_counts':token_counts,
                  'latency_note':'Saved-image API request to parsed response; excludes local speech and audio output delay.'}
        (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(groups,indent=2))
    print(f'Read-back probe saved: {root}')
    return 0 if result['completed'] else 1
