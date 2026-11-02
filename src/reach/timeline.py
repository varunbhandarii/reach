"""Self-contained SVG timelines from event logs, including events-only exports."""

import html
import json
import math
from pathlib import Path


PALETTE={'IDLE':'#435066','SEARCHING':'#bb8a35','GUIDING':'#288b67','HAND_LOST':'#a66141',
    'TARGET_LOST':'#9a597b','DISAMBIGUATING':'#6a68b5','CONFIRMING':'#407fa7','DONE':'#598547','DESCRIBING':'#477cac'}


def write_html(run_dir,output):
    root,output=Path(run_dir),Path(output)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (root/'events.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    stripped=manifest.get('replay_only') or manifest.get('source')=='events_only'
    # Stripped exports keep effects inside brain_out rather than separate live events.
    kinds={r['type'] for r in rows}
    extra=[]
    for row in rows:
        if row['type']=='brain_in':
            for result in row['data'].get('results',()):
                kind='scene_resp' if 'objects' in result else 'readback_resp'
                if kind not in kinds:
                    extra.append({'t_mono':row['t_mono'],'type':kind,'data':result})
            if stripped:
                for command in row['data'].get('commands',()):
                    if command.get('source') in ('voice','voice_fast','ptt','spark_parse'):
                        extra.append({'t_mono':command.get('t_mono',row['t_mono']),
                            'type':'voice_effect','data':{'source':command.get('source'),'kind':command['kind']}})
        if row['type']!='brain_out':
            continue
        if 'decision' not in kinds and row['data'].get('decision'):
            extra.append({'t_mono':row['t_mono'],'type':'decision','data':row['data']['decision']})
        for effect in row['data'].get('effects',[]):
            kind=effect.get('key') if effect.get('kind')=='log' else 'speech_effect' if effect.get('kind')=='speak' else None
            if effect.get('kind') in ('readback','describe'):
                kind='readback_req' if effect['kind']=='readback' else 'scene_req'
            if kind and kind not in kinds:
                extra.append({'t_mono':row['t_mono'],'type':kind,'data':effect.get('payload') or {'text':effect.get('text')}})
    rows=sorted(rows+extra,key=lambda r:r['t_mono'])
    times=[r['t_mono'] for r in rows]
    start=manifest.get('active_start_mono') or (min(times) if times else 0)
    end=manifest.get('active_end_mono') or (max(times) if times else start+1)
    duration=max(.001,end-start)
    width,left,right=1400,165,1370
    lane_names=['States','Distance d','Pan','Pitch Hz','Voice turns','Spoken phrases','SAM requests','Read-back','Contact','Trials']
    ys={name:85+i*68 for i,name in enumerate(lane_names)}
    height=ys['Trials']+90
    x=lambda stamp:left+max(0,min(duration,stamp-start))/duration*(right-left)
    escape=lambda value:html.escape(str(value),quote=True)
    svg=[]
    details=[]
    def label(value,px,py,color='#d8e3f0',size=12):
        svg.append(f'<text x="{px:.2f}" y="{py:.2f}" fill="{color}" font-size="{size}">{escape(value)}</text>')
    def bar(a,b,y,color,title,text=''):
        a,b=max(start,a),min(end,b)
        if b<a:
            return
        details.append((a-start,b-start,title))
        svg.append(f'<rect x="{x(a):.2f}" y="{y}" width="{max(2,x(b)-x(a)):.2f}" height="22" rx="3" fill="{color}"><title>{escape(title)}</title></rect>')
        available=max(0,int((x(b)-x(a)-6)/6))
        if text and available>=5:
            label(text if len(text)<=available else text[:available-1]+'…',x(a)+3,y+38,size=11)
    label('Event timeline — seconds from active start',left,30,size=18)
    for i in range(9):
        stamp=start+duration*i/8
        px=x(stamp)
        svg.append(f'<path d="M{px:.2f},55 V{height-30}" stroke="#263447"/>')
        label(f'{duration*i/8:.1f}s',px,52,size=11)
    for name,y in ys.items():
        label(name,12,y+15,size=14)
        svg.append(f'<path d="M{left},{y+49} H{right}" stroke="#263447"/>')
    states=[r for r in rows if r['type']=='state']
    for i,row in enumerate(states):
        state=row['data'].get('to','unknown')
        stop=states[i+1]['t_mono'] if i+1<len(states) else end
        bar(row['t_mono'],stop,ys['States'],PALETTE.get(state,'#536477'),state,state)
    decisions=[r for r in rows if r['type']=='decision']
    for name,key,low,high,color in [('Distance d','d',0,1,'#62d3a6'),('Pan','pan',-1,1,'#76b9ff'),('Pitch Hz','freq_hz',220,880,'#efbf6b')]:
        points=[]
        segments=[]
        stride=max(1,math.ceil(len(decisions)/10000))
        for row in decisions[::stride]:
            value=row['data'].get(key) if key=='d' else (row['data'].get('cue') or {}).get(key)
            if isinstance(value,(float,int)) and math.isfinite(value):
                py=ys[name]+40-40*max(0,min(1,(value-low)/(high-low)))
                points.append(f'{x(row["t_mono"]):.2f},{py:.2f}')
            elif points:
                segments.append(points);points=[]
        if points:segments.append(points)
        for points in segments:
            svg.append(f'<polyline points="{" ".join(points)}" stroke="{color}" stroke-width="1.5" fill="none"/>')
        label(f'{low}…{high}; source: Decision.{"cue." if key!="d" else ""}{key}',left,ys[name]+62,size=10)
    open_requests={}
    speech={}
    trials={}
    counts={}
    for row in rows:
        kind,data,stamp=row['type'],row['data'],row['t_mono']
        counts[kind]=counts.get(kind,0)+1
        if kind=='voice_turn' and not stripped:
            local=data.get('t_local_speech_end') if data.get('local_latency_valid') else None
            stop=data.get('t_complete') or stamp
            note=(f"local energy EOS to complete: {data.get('latency_local_ms'):.1f} ms"
                  if local is not None and isinstance(data.get('latency_local_ms'),(float,int)) else 'local endpoint unmeasured')
            if data.get('mode')=='PUSH_TO_TALK':
                local=data.get('t_speech_end')
                note=f"second tap to final: {data.get('latency_ms')} ms"
            bar(local or stamp,stop,ys['Voice turns'],'#9589df',note,note)
        elif kind=='voice_effect':
            note=f"{data['source']} command accepted: {data['kind']}; endpoint latency unavailable"
            bar(stamp,stamp,ys['Voice turns'],'#9589df',note)
        elif kind=='speech' and not stripped:
            ident=data.get('id')
            if data.get('event')=='start':
                speech[ident]=(stamp,data.get('text',''))
            elif data.get('event') in ('end','cancel') and ident in speech:
                a,text=speech.pop(ident)
                bar(a,stamp,ys['Spoken phrases'],'#6b8fc0',text,text)
        elif kind=='speech_effect':
            bar(stamp,stamp,ys['Spoken phrases'],'#6b8fc0','Reducer speech effect: '+str(data.get('text','')),str(data.get('text','')))
        elif kind in ('sam_req','readback_req','scene_req'):
            family=kind.split('_')[0]
            ident=data.get('request_id',data.get('req_id',data.get('id')))
            open_requests[family,ident]=stamp
        elif kind in ('sam_resp','readback_resp','scene_resp'):
            family=kind.split('_')[0]
            ident=data.get('request_id',data.get('req_id',data.get('id')))
            a=open_requests.pop((family,ident),stamp)
            name='SAM requests' if family=='sam' else 'Read-back'
            latency=data.get('latency_ms')
            measured=f'{latency:.1f} ms' if isinstance(latency,(int,float)) else 'unmeasured'
            note=f"{family} {data.get('status',data.get('error') or 'answer')}; client {measured}"
            bar(a,stamp,ys[name],'#5e9d99' if not data.get('error') else '#bd6f69',note,note)
        elif kind=='contact' and (data.get('event')=='fire' or data.get('fired')):
            bar(stamp,stamp,ys['Contact'],'#f0bf60','Automatic contact heuristic; not proof of physical touch','chime')
        elif kind=='trial_start':
            trials[data.get('index')]=(stamp,data.get('phrase',''))
        elif kind=='trial_end':
            a,phrase=trials.pop(data.get('index'),(stamp,data.get('phrase','')))
            elapsed=data.get('elapsed_s')
            measured=f'{elapsed:.2f} s' if isinstance(elapsed,(int,float)) else 'unmeasured'
            note=f"{phrase}: {data.get('reason','outcome')}, {measured} (reported outcome)"
            bar(a,stamp,ys['Trials'],'#459578' if data.get('success') else '#a56763',note,note)
    for a,text in speech.values():
        bar(a,end,ys['Spoken phrases'],'#6b8fc0',text,text)
    legend=' '.join(f'<span style="border-left:12px solid {color};padding-left:5px;margin-right:15px">{name}</span>' for name,color in PALETTE.items())
    title='Events-only run' if stripped else root.name
    table=''.join(f'<tr><td>{a:.2f}</td><td>{b:.2f}</td><td>{escape(note)}</td></tr>' for a,b,note in details)
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reach timeline — {escape(title)}</title><style>body{{background:#101924;color:#d8e3f0;font-family:system-ui;margin:24px}}svg{{width:100%;min-width:1000px}}.chart{{overflow:auto}}p{{max-width:1100px;line-height:1.5}}.legend{{line-height:2.5}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #263447}}summary{{cursor:pointer;padding:12px 0}}</style>
<h1>Reach timeline</h1><p>{escape(title)} · {duration:.2f} active seconds · {len(decisions)} logged decisions.</p>
<div class="legend">{legend}</div><div class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Run events over time">{''.join(svg)}</svg></div>
<p>Source: saved event timestamps and reducer decisions. Hover over bars for details. Voice latency uses the capture-clocked local energy endpoint to receipt of the completed turn; tap-to-talk uses the second tap. These exclude acoustic output delay. API labels use client-reported durations. Speech bars show callback playback; events-only exports show reducer speech effects. Automatic chimes do not establish physical touch. Plot values are clipped to the labeled range and sampled to at most 10,000 decisions.</p>
<p>Raw microphone transcript text is omitted. Missing lanes indicate absent measurements; they are not zero-valued results.</p></html>'''
    document=document.replace('</html>',f'<details><summary>Event details ({len(details)}) — seconds from active start</summary><table><tr><th>Start</th><th>End</th><th>Logged observation</th></tr>{table}</table></details></html>')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(document,encoding='utf-8')
    return {'path':str(output),'events':len(rows),'decisions':len(decisions),'stripped':bool(stripped)}
