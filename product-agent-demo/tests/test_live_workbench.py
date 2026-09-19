import asyncio
import json
import unittest
from unittest.mock import patch
import httpx
from app.lab import normalize_identity, parse_json, retrieve, run_models, sources_from, validate_report

class WorkbenchLiveTests(unittest.TestCase):
    def test_identity_classifier_keeps_one_post_small_and_marks_collage_for_separate_batch_items(self):
        one = normalize_identity({'input_type':'ugc_social_post','batch_detected':False,
                                  'brand':'薇诺娜','product_name':'特护面膜','ocr_text':'x'*3000,
                                  'claims':['x'*500 for _ in range(20)],'confidence':1.2})
        self.assertFalse(one['batch_detected'])
        self.assertLessEqual(len(one['ocr_text']),700)
        self.assertLessEqual(len(one['claims']),8)
        self.assertTrue(all(len(c)<=180 for c in one['claims']))
        many = normalize_identity({'input_type':'multi_post_collage','brand':'薇诺娜'})
        self.assertTrue(many['batch_detected'])

    def test_report_validation_caps_repeated_model_output(self):
        report = validate_report({'summary':'x'*1000,
            'claims':['x']*30,
            'official_facts':[{'fact':f'f{i}','source_ids':['s1']} for i in range(20)],
            'claim_evidence_audit':[{'claim':f'c{i}','status':'待核验','source_ids':[]} for i in range(20)],
            'evidence_gaps':['g']*20,'image_observations':['o']*20},
            [{'source_id':'s1','url':'https://example.com'}])
        self.assertLessEqual(len(report['summary']),500)
        self.assertLessEqual(len(report['claims']),8)
        self.assertLessEqual(len(report['official_facts']),5)
        self.assertLessEqual(len(report['claim_evidence_audit']),8)
        self.assertLessEqual(len(report['evidence_gaps']),6)
        self.assertLessEqual(len(report['image_observations']),8)

    def test_report_validation_rejects_untrusted_sources_as_official_facts(self):
        report = validate_report({
            'summary': 'summary',
            'official_facts': [{'fact': '营销页面说法', 'source_ids': ['s1']}],
            'claim_evidence_audit': [{'claim': '用户说法', 'status': '有资料支持', 'source_ids': ['s1']}],
            'evidence_gaps': [],
        }, [{'source_id': 's1', 'url': 'https://brand.example/product', 'trusted': False}])
        self.assertEqual(report['official_facts'], [])
        self.assertTrue(any('监管公开来源' in gap for gap in report['evidence_gaps']))
        self.assertEqual(report['claim_evidence_audit'][0]['status'], '待核验')

    def test_report_validation_downgrades_malformed_status(self):
        report = validate_report({
            'summary': 'summary',
            'official_facts': [],
            'claim_evidence_audit': [{'claim': 'x', 'status': [], 'source_ids': []}],
            'evidence_gaps': [],
        }, [])
        self.assertEqual(report['claim_evidence_audit'][0]['status'], '待核验')

    def test_explicit_official_source_metadata_is_preserved(self):
        sources = sources_from([{'url': 'https://brand.example/product', 'title': '官方产品页', 'source_level': 'official'}])
        report = validate_report({'official_facts': [{'fact': '官方事实', 'source_ids': ['s1']}], 'claim_evidence_audit': []}, sources)
        self.assertTrue(sources[0]['trusted'])
        self.assertEqual(report['official_facts'][0]['source_ids'], ['s1'])

    def test_report_json_accepts_one_fenced_object_after_explanation(self):
        text='报告如下。来源为空，因此全部待核验。\n```json\n{"summary":"实际模型报告","claims":[]}\n```'
        self.assertEqual(parse_json(text)['summary'],'实际模型报告')
        for invalid in ['报告尚未生成', '```json\n{"summary":"截断',
                        '```json\n{"summary":"a"}\n```\n```json\n{"summary":"b"}\n```']:
            with self.assertRaises(ValueError):parse_json(invalid)

    def test_search_stream_keeps_completed_sources_and_progress(self):
        seen=[];progress=[]
        payload={'status':'completed','output':[{'type':'web_search_call','action':{'sources':[{'url':'https://brand.example/a','title':'Brand'}]}},{'type':'message','content':[{'text':'Retrieved actual text','annotations':[]}]}]}
        async def handler(req):
            seen.append(json.loads(req.content))
            events=[{'type':'response.created'}, {'type':'response.output_item.done','item':payload['output'][0]}, {'type':'response.completed','response':payload}]
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
        async def notify(message):progress.append(message)
        real=httpx.AsyncClient
        with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
            sources,material=asyncio.run(retrieve({'base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key':'x'},'product',{}, {'provider':'bailian'},progress=notify))
        self.assertTrue(seen[0]['stream'])
        self.assertEqual(sources[0]['url'],'https://brand.example/a')
        self.assertIn('Retrieved actual text',material)
        self.assertTrue(progress)

    def test_incomplete_search_stream_is_not_a_successful_empty_result(self):
        async def handler(req):
            return httpx.Response(200,headers={'content-type':'text/event-stream'},text='data: {"type":"response.created"}\n\n')
        real=httpx.AsyncClient
        with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
            with self.assertRaisesRegex(ValueError,'完整'):
                asyncio.run(retrieve({'base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key':'x'},'product',{}, {'provider':'bailian'}))

    def test_three_agents_run_concurrently_with_distinct_roles_and_image(self):
        async def scenario():
            started=[];bodies=[];gate=asyncio.Event()
            async def handler(request):
                body=json.loads(request.content)
                if not body.get('stream'):
                    return httpx.Response(200,json={'choices':[{'message':{'content':'{"product_name":"test","confidence":1}'}}]})
                bodies.append(body)
                started.append(body['messages'][0]['content'])
                if len(started)==3:gate.set()
                await asyncio.wait_for(gate.wait(),1)
                report={'summary':'result','official_facts':[],'image_observations':['visible packaging'],'evidence_gaps':[]}
                return httpx.Response(200,text='data: '+json.dumps({'id':'req-test','choices':[{'delta':{'content':json.dumps(report)}}]})+'\n\ndata: [DONE]\n\n')
            real=httpx.AsyncClient
            models=[{'id':i,'agent_role':i,'model':'qwen3.8-max','base_url':'https://model.example/v1','api_key':'x'} for i in ['facts','review','visual']]
            with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
                events=[json.loads(s.removeprefix('data: ')) async for s in run_models({'models':models,'query':'product','search_enabled':False},'data:image/png;base64,a')]
            self.assertEqual(len([e for e in events if e['type']=='report']),3)
            self.assertEqual(len(set(started)),3)
            self.assertTrue(all(any(p.get('type')=='image_url' for p in b['messages'][1]['content']) for b in bodies))
            self.assertEqual({e['model_id'] for e in events if e['type']=='report'},{'facts','review','visual'})
        asyncio.run(scenario())

    def test_qwen38_search_uses_responses_and_preserves_actual_sources(self):
        requests=[]
        async def handler(req):
            requests.append(req)
            return httpx.Response(200,json={'output':[{'type':'web_search_call','action':{'sources':[{'url':'https://brand.example/a','title':'Official page'}]}},{'type':'message','content':[{'type':'output_text','text':'Retrieved material','annotations':[]}]}]})
        real=httpx.AsyncClient
        with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
            sources,_=asyncio.run(retrieve({'base_url':'https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1','api_key':'x','model':'qwen3.8-max'},'cream',{}, {'provider':'bailian','model':'qwen3.8-max'}))
        self.assertTrue(str(requests[0].url).endswith('/responses'))
        self.assertEqual(json.loads(requests[0].content)['tools'],[{'type':'web_search'}])
        self.assertEqual(sources[0]['url'],'https://brand.example/a')

    def test_tavily_search_returns_clickable_deduplicated_sources(self):
        seen=[]
        async def handler(request):
            seen.append(request)
            return httpx.Response(200,json={'results':[{'title':'Brand product','url':'https://brand.example/product','content':'Ingredients from source'}, {'url':'https://brand.example/product'}, {'url':'javascript:alert(1)'}]})
        real=httpx.AsyncClient
        def client(**kwargs):return real(transport=httpx.MockTransport(handler),**kwargs)
        with patch('app.lab.httpx.AsyncClient',side_effect=client):
            sources,text=asyncio.run(retrieve({'base_url':'https://model.example/v1','api_key':'model-key'},'test product',{}, {'provider':'tavily','api_key':'search-key'}))
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]['snippet'],'Ingredients from source')
        self.assertEqual(seen[0].headers['authorization'],'Bearer search-key')
        self.assertEqual(json.loads(seen[0].content)['query'],'test product')

    def test_configured_vision_model_then_search_then_grounded_report(self):
        bodies=[]
        async def handler(request):
            body=json.loads(request.content);bodies.append(body)
            if body.get('stream'):
                report={'summary':'Verified only against returned source','official_facts':[{'fact':'Supported','source_ids':['s1']}], 'claim_evidence_audit':[], 'evidence_gaps':[]}
                chunk={'choices':[{'delta':{'content':json.dumps(report)}}]}
                return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'brand':'Test','product_name':'Cream','confidence':.92})}}]})
        real=httpx.AsyncClient
        def client(**kwargs):return real(transport=httpx.MockTransport(handler),**kwargs)
        async def search(*args,**kwargs):return ([{'source_id':'s1','url':'https://www.nifdc.org.cn/product','title':'Product','trusted':True}],'Source text')
        async def collect():
            return [json.loads(s.removeprefix('data: ')) async for s in run_models({'models':[{'id':'report','base_url':'https://model.example/v1','api_key':'x','model':'report-model'}],'vision_model':'vision-model','search_enabled':True},'data:image/png;base64,a')]
        with patch('app.lab.httpx.AsyncClient',side_effect=client),patch('app.lab.retrieve',side_effect=search):events=asyncio.run(collect())
        kinds=[e['type'] for e in events]
        self.assertLess(kinds.index('product_identified'),kinds.index('search_started'))
        self.assertLess(kinds.index('sources'),kinds.index('report'))
        self.assertEqual(bodies[0]['model'],'vision-model')
        report=next(e['report'] for e in events if e['type']=='report')
        self.assertEqual(report['official_facts'][0]['source_ids'],['s1'])

    def test_low_confidence_stops_before_search(self):
        async def handler(request):return httpx.Response(200,json={'choices':[{'message':{'content':'{"confidence":0.2,"product_name":"unclear"}'}}]})
        real=httpx.AsyncClient
        async def collect():return [json.loads(s.removeprefix('data: ')) async for s in run_models({'models':[{'base_url':'https://model.example/v1','api_key':'x','model':'vision'}]},'data:image/png;base64,a')]
        with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)),patch('app.lab.retrieve',side_effect=AssertionError('must not search')):events=asyncio.run(collect())
        self.assertIn('needs_input',[e['type'] for e in events]);self.assertNotIn('search_started',[e['type'] for e in events])
