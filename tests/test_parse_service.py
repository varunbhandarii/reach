import asyncio
import json
import time
from types import SimpleNamespace as NS

import httpx2
from openai import AsyncOpenAI
import pytest

from reach.spark_services import ParseClient,ParsedRequest,should_parse
from reach.types import Command
from reach.voice import VoiceClient


def test_chatter_filter_and_non_english_request():
    assert not should_parse('It is a sunny afternoon.')
    assert should_parse('can you help me find my reading glasses')
    assert should_parse('मेरा बटुआ कहाँ है')


def test_text_only_parse_uses_no_image_and_maps_structured_request():
    bodies=[]
    async def handler(request):
        body=json.loads(request.content);bodies.append(body)
        value={'intent':'find','target_phrase':'reading glasses','spoken_phrase':'reading glasses','language':'en'}
        return httpx2.Response(200,json={'id':'synthetic','object':'response','created_at':0,'status':'completed',
            'output':[{'type':'message','id':'m','role':'assistant','status':'completed','content':[
                {'type':'output_text','annotations':[],'text':json.dumps(value)}]}]})
    factory=lambda **kw:AsyncOpenAI(**kw,http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    client=ParseClient({},client_factory=factory)
    assert not client.request(1,'Lovely weather today')
    assert client.request(2,'can you help me find my reading glasses')
    deadline=time.perf_counter()+5
    while client.stats()['busy'] and time.perf_counter()<deadline:time.sleep(.005)
    row,=client.poll()
    assert row.intent=='find' and row.target_phrase=='reading glasses' and row.error is None
    assert bodies[0]['store'] is False
    assert all(c['type']=='input_text' for c in bodies[0]['input'][0]['content'])
    client.stop()


def test_late_fallback_find_cannot_undo_stop():
    posted=[]
    class Fake:
        def __init__(self,*args):self.rows=[]
        def request(self,ident,text):self.ident=ident
        def poll(self):rows,self.rows=self.rows,[];return rows
        def stop(self):pass
    voice=VoiceClient({},None,posted.append,parse_factory=Fake)
    voice._post(Command('unknown',None,'help me find glasses',1,1))
    ident=voice._parse_client.ident
    voice._post(Command('stop',None,'stop',2,2))
    voice._parse_client.rows=[ParsedRequest(ident,'find','glasses','glasses','en',100)]
    voice._poll_parse()
    assert [c.kind for c in posted]==['stop']


def test_keyboard_cancel_invalidates_pending_voice_fallback():
    from reach.runner import Runner
    class Fake:
        def __init__(self,*args):self.rows=[]
        def request(self,ident,text):self.ident=ident
        def poll(self):rows,self.rows=self.rows,[];return rows
        def stop(self):pass
    runner=Runner.__new__(Runner);runner._trial=None
    posted=[];runner.inbox=NS(post=posted.append)
    voice=VoiceClient({},None,runner.post_command,parse_factory=Fake);runner.voice=voice
    voice._post(Command('unknown',None,'help me find glasses',1,1))
    ident=voice._parse_client.ident
    runner.post_command(Command('cancel',None,'cancel',None,2,'key'))
    voice._parse_client.rows=[ParsedRequest(ident,'find','glasses','glasses','en',100)]
    voice._poll_parse()
    assert [c.kind for c in posted]==['cancel']


def test_timeout_posts_nothing_and_help_notice_is_once():
    posted,notices=[],[]
    class Fake:
        def __init__(self,*args):self.rows=[]
        def request(self,ident,text):self.ident=ident
        def poll(self):rows,self.rows=self.rows,[];return rows
        def stop(self):pass
    voice=VoiceClient({},None,posted.append,parse_factory=Fake,on_parse_notice=lambda:notices.append(True))
    for error in ('timeout',None,None):
        voice._post(Command('unknown',None,'help me with something',1,1))
        voice._parse_client.rows=[ParsedRequest(voice._parse_client.ident,'none','','','en',3000,error)]
        voice._poll_parse()
    assert posted==[] and notices==[True]


def test_local_voice_server_dispatches_fallback_without_uploading_chatter():
    from websockets.asyncio.server import serve
    async def scenario():
        commands,texts=[],[]
        class FakeParse:
            def __init__(self,*args):self.rows=[]
            def request(self,ident,text):
                texts.append(text)
                phrase='wallet' if 'बटुआ' in text else 'reading glasses'
                self.rows.append(ParsedRequest(ident,'find',phrase,phrase,'hi' if phrase=='wallet' else 'en',20))
            def poll(self):rows,self.rows=self.rows,[];return rows
            def stats(self):return {}
            def stop(self):pass
        async def source():
            while True:
                yield bytes(3840)
                await asyncio.sleep(.01)
        async def handler(ws):
            await ws.recv()
            await ws.send(json.dumps({'sessionId':'synthetic'}))
            for index,text in enumerate(('Lovely weather today','can you help me find my reading glasses','मेरा बटुआ कहाँ है'),1):
                await ws.send(json.dumps({'type':'speechComplete','turnId':index,'transcript':text}))
                await asyncio.sleep(.12)
            async for message in ws:
                if isinstance(message,str) and json.loads(message).get('type')=='endStream':
                    await ws.close();return
        async with serve(handler,'127.0.0.1',0) as server:
            voice=VoiceClient({},None,commands.append,parse_factory=FakeParse,
                parser=lambda text,*args:Command('unknown',None,text,None,0),audio_source=source(),
                uri=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",api_key='synthetic')
            voice.start()
            try:
                async with asyncio.timeout(5):
                    while len([c for c in commands if c.kind=='find'])<2:await asyncio.sleep(.01)
                actions=[c for c in commands if c.kind!='unknown']
                assert [c.phrase for c in actions]==['reading glasses','wallet']
                assert all(c.source=='spark_parse' for c in actions)
                assert actions[-1].language=='hi' and len(texts)==2
            finally:
                await asyncio.to_thread(voice.stop)
    asyncio.run(scenario())
