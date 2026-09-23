import asyncio
import json
import unittest
from unittest.mock import patch
import httpx
from app.lab import ReportFormatError, error_text, normalize_identity, parse_json, retrieve, run_models, sources_from, stable_input_id, validate_report

class WorkbenchLiveTests(unittest.TestCase):
    def test_upstream_status_errors_explain_key_and_quota_failures(self):
        self.assertIn('API Key', error_text(httpx.HTTPStatusError('x', request=httpx.Request('POST', 'https://example.com'), response=httpx.Response(401))))
        self.assertIn('权限', error_text(httpx.HTTPStatusError('x', request=httpx.Request('POST', 'https://example.com'), response=httpx.Response(403))))
        self.assertIn('额度', error_text(httpx.HTTPStatusError('x', request=httpx.Request('POST', 'https://example.com'), response=httpx.Response(429))))

    def test_vision_report_retries_with_compact_schema_after_truncated_json(self):
        async def scenario():
            requests=[]
            async def handler(request):
                body=json.loads(request.content); requests.append(body)
                if len(requests) == 1:
                    truncated=json.dumps({'input_type':'product_packaging','brand':'Test'})[:-1]
                    return httpx.Response(200, json={'choices':[{'message':{'content':truncated}}]})
                if body.get('stream'):
                    report={'summary':'fixture report','official_facts':[],'claim_evidence_audit':[],'evidence_gaps':[]}
                    chunk={'choices':[{'delta':{'content':json.dumps(report)}}]}
                    return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
                return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps({'input_type':'product_packaging','brand':'Test','product_name':'Cream','confidence':.95})}}]})
            real=httpx.AsyncClient
            with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
                events=[json.loads(s.removeprefix('data: ')) async for s in run_models({
                    'models':[{'id':'facts','agent_role':'facts','model':'report-model','base_url':'https://model.example/v1','api_key':'x'}],
                    'vision_model':'vision-model','search_enabled':False,
                },'data:image/png;base64,a')]
            self.assertEqual(len(requests),3)
            self.assertEqual(requests[0]['max_tokens'],1400)
            self.assertEqual(requests[1]['max_tokens'],1200)
            self.assertEqual(next(event['product']['brand'] for event in events if event['type']=='product_identified'),'Test')
            self.assertNotIn('error',[event['type'] for event in events])
        asyncio.run(scenario())

    def test_structured_claim_contract_preserves_conditions_entities_and_stable_id(self):
        raw = {
            'brand': '示例品牌', 'product_name': '示例精华', 'specification': '30ml',
            'claims': [{
                'claim_id': 'model-random-id',
                'text': '受试者30人，连续使用28天，细纹指标改善20%，敏感肌适用',
                'claim_type': 'efficacy',
                'metric': {'value': '细纹', 'original_text': '细纹指标'},
                'value': {'value': 20, 'unit': '%', 'original_text': '改善20%'},
                'duration': {'value': 28, 'unit': '天', 'original_text': '连续使用28天'},
                'audience': {'value': '敏感肌', 'original_text': '敏感肌适用'},
                'sample_size': {'value': 30, 'unit': '人', 'original_text': '受试者30人'},
                'endorsements': [{'entity_type': 'expert', 'name': '张三', 'original_text': '专家：张三'}],
            }],
            'confidence': .98,
        }
        first = normalize_identity(raw, 'input-fixed')
        second = normalize_identity(raw, 'input-fixed')
        claim = first['claims_structured'][0]
        self.assertEqual(claim['claim_id'], second['claims_structured'][0]['claim_id'])
        self.assertNotEqual(claim['claim_id'], 'model-random-id')
        self.assertEqual(claim['upstream_claim_id'], 'model-random-id')
        self.assertEqual(first['claims'][0], claim['original_text'])
        self.assertEqual(claim['conditions']['value']['unit'], '%')
        self.assertEqual(claim['conditions']['time']['value'], 28)
        self.assertEqual(claim['conditions']['sample_size']['value'], 30)
        self.assertEqual(claim['conditions']['endorsements'][0]['name'], '张三')
        self.assertEqual(first['claim_coverage']['status'], 'complete')

    def test_compound_claims_keep_parent_relationship(self):
        result = normalize_identity({
            'claims': ['连续使用28天；细纹改善20%；敏感肌适用'],
            'confidence': .9,
        }, 'input-compound')
        self.assertEqual(result['claim_coverage']['candidate_count'], 4)
        parent, *children = result['claims_structured']
        self.assertIsNone(parent['parent_claim_id'])
        self.assertTrue(children)
        self.assertTrue(all(child['parent_claim_id'] == parent['claim_id'] for child in children))
        self.assertEqual({child['input_id'] for child in children}, {'input-compound'})

    def test_budget_overflow_is_explicit_and_not_silently_dropped(self):
        result = normalize_identity({'claims': [f'声明 {index}' for index in range(10)]}, 'input-budget')
        coverage = result['claim_coverage']
        self.assertEqual(coverage['candidate_count'], 10)
        self.assertEqual(coverage['processed_count'], 8)
        self.assertEqual(coverage['excluded_count'], 2)
        self.assertEqual(coverage['status'], 'partial')
        self.assertTrue(all(item['reason'] == 'budget_exceeded' for item in coverage['excluded']))
        self.assertTrue(all(item['parse_status'] == 'budget_excluded' for item in coverage['excluded']))
        self.assertEqual(len(result['claims']), 8)

    def test_uncertain_ocr_and_missing_fields_are_explicit(self):
        result = normalize_identity({'claims': [{
            'text': '改善效果约为？', 'ocr_confidence': .2, 'claim_type': 'efficacy',
        }]}, 'input-uncertain')
        claim = result['claims_structured'][0]
        self.assertEqual(claim['parse_status'], 'partially_parsed')
        self.assertTrue(claim['uncertainty_reasons'])
        self.assertIsNone(claim['conditions']['value'])
        self.assertIsNone(claim['conditions']['time'])
        self.assertEqual(result['claim_coverage']['unparsed_count'], 1)

    def test_input_id_fallback_is_deterministic_and_namespaced(self):
        self.assertEqual(stable_input_id('same'), stable_input_id('same'))
        self.assertNotEqual(stable_input_id('same'), stable_input_id('other'))
        first = normalize_identity({'claims': ['同一声明']}, stable_input_id('image-a'))
        second = normalize_identity({'claims': ['同一声明']}, stable_input_id('image-b'))
        self.assertNotEqual(first['claims_structured'][0]['claim_id'], second['claims_structured'][0]['claim_id'])

    def test_report_validation_keeps_valid_claim_id_and_rejects_unknown_id(self):
        identity = normalize_identity({'claims': ['待核验声明']}, 'input-report')
        valid_id = identity['claims_structured'][0]['claim_id']
        report = validate_report({
            'claim_evidence_audit': [
                {'claim_id': valid_id, 'claim': '待核验声明', 'status': '待核验', 'source_ids': []},
                {'claim_id': 'input-report:claim:not-real', 'claim': '伪造声明', 'status': '有资料支持', 'source_ids': []},
            ]
        }, [], identity)
        self.assertEqual(len(report['claim_evidence_audit']), 1)
        self.assertEqual(report['claim_evidence_audit'][0]['claim_id'], valid_id)
        self.assertIn('不存在的 claim_id', report['evidence_gaps'][0])
        self.assertEqual(report['claim_coverage']['input_id'], 'input-report')

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
        duplicate='```json\n{"summary":"实际模型报告","claims":[]}\n```\n{"claims":[],"summary":"实际模型报告"}'
        self.assertEqual(parse_json(duplicate)['summary'],'实际模型报告')
        for invalid in ['报告尚未生成', '```json\n{"summary":"截断',
                        '```json\n{"summary":"a"}\n```\n```json\n{"summary":"b"}\n```']:
            with self.assertRaises(ReportFormatError):parse_json(invalid)

    def test_report_json_rejects_nested_object_in_truncated_json(self):
        with self.assertRaisesRegex(ReportFormatError,'完整'):
            parse_json('{"report":{"summary":"截断"}')
        with self.assertRaises(ReportFormatError):
            parse_json('[{"summary":"不是顶层对象"}]')

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

    def test_review_agent_receives_structured_claims_and_returns_claim_id(self):
        async def scenario():
            requests = []
            claim_id = 'input-review:claim:fixture'
            async def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if not body.get('stream'):
                    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
                        'input_type': 'official_product_page', 'brand': 'Test', 'product_name': 'Cream',
                        'confidence': .92, 'claims': [{'claim_id': 'upstream', 'text': '7天修护', 'claim_type': 'efficacy'}]
                    })}}]})
                audit = {'claim_id': claim_id, 'claim': '7天修护', 'status': '待核验', 'reason': '暂无来源', 'source_ids': []}
                report = {'summary': 'review result', 'official_facts': [], 'claim_evidence_audit': [audit], 'evidence_gaps': []}
                chunk = {'choices': [{'delta': {'content': json.dumps(report)}}]}
                return httpx.Response(200, text='data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n')
            identity = normalize_identity({'brand': 'Test', 'product_name': 'Cream', 'confidence': .92,
                                           'claims': [{'claim_id': 'upstream', 'text': '7天修护', 'claim_type': 'efficacy'}]}, 'input-review')
            claim_id = identity['claims_structured'][0]['claim_id']
            real = httpx.AsyncClient
            with patch('app.lab.httpx.AsyncClient', side_effect=lambda **kw: real(transport=httpx.MockTransport(handler), **kw)):
                events = [json.loads(s.removeprefix('data: ')) async for s in run_models({
                    'input_id': 'input-review',
                    'models': [{'id': 'review', 'agent_role': 'review', 'model': 'review-model',
                                'base_url': 'https://model.example/v1', 'api_key': 'x'}],
                    'vision_model': 'vision-model', 'search_enabled': False,
                }, 'data:image/png;base64,a')]
            report = next(event['report'] for event in events if event['type'] == 'report')
            self.assertEqual(report['claim_evidence_audit'][0]['claim_id'], claim_id)
            report_task = next(json.loads(next(part['text'] for part in body['messages'][1]['content'] if part.get('type') == 'text')) for body in requests if body.get('stream'))
            self.assertEqual(report_task['input_id'], 'input-review')
            self.assertEqual(report_task['claims_structured'][0]['claim_id'], claim_id)
            self.assertEqual(report_task['claim_coverage']['processed_count'], 1)
            self.assertNotIn('claims_structured', report_task['image_reading'])
        asyncio.run(scenario())

    def test_bailian_report_stream_uses_json_mode_but_custom_endpoint_does_not(self):
        async def collect(base_url):
            bodies=[]
            async def handler(request):
                body=json.loads(request.content);bodies.append(body)
                report={'summary':'result','official_facts':[],'claim_evidence_audit':[],'evidence_gaps':[]}
                chunk={'choices':[{'delta':{'content':json.dumps(report)}}]}
                return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
            real=httpx.AsyncClient
            with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
                events=[json.loads(s.removeprefix('data: ')) async for s in run_models(
                    {'models':[{'id':'report','base_url':base_url,'api_key':'x','model':'qwen3.8-max'}],
                     'query':'product','search_enabled':False},None)]
            self.assertIn('report',[event['type'] for event in events])
            return bodies[0]
        bailian=asyncio.run(collect('https://dashscope.aliyuncs.com/compatible-mode/v1'))
        custom=asyncio.run(collect('https://model.example/v1'))
        self.assertEqual(bailian['response_format'],{'type':'json_object'})
        self.assertFalse(bailian['enable_thinking'])
        self.assertNotIn('response_format',custom)
        self.assertNotIn('enable_thinking',custom)

    def test_report_format_retry_reuses_context_without_repeating_vision_or_search(self):
        async def scenario():
            requests=[];retrieval_calls=[]
            async def handler(request):
                body=json.loads(request.content);requests.append(body)
                if body['model']=='vision-model':
                    return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'brand':'Test','product_name':'Cream','confidence':.92})}}]})
                if body.get('stream'):
                    truncated='{"summary":"first","official_facts":[],"claim_evidence_audit":['
                    chunk={'id':'stream-request','choices':[{'delta':{'content':truncated}}]}
                    return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
                repaired={'summary':'repaired','official_facts':[],'claim_evidence_audit':[],'evidence_gaps':[]}
                return httpx.Response(200,json={'id':'repair-request','choices':[{'message':{'content':json.dumps(repaired)}}]})
            async def search(*args,**kwargs):
                retrieval_calls.append(True)
                return ([{'source_id':'s1','url':'https://www.nifdc.org.cn/product','title':'Product','trusted':True}],'Source text')
            real=httpx.AsyncClient
            with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)),patch('app.lab.retrieve',side_effect=search):
                events=[json.loads(s.removeprefix('data: ')) async for s in run_models(
                    {'models':[{'id':'report','base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key':'x','model':'report-model'}],
                     'vision_model':'vision-model','search_enabled':True},'data:image/png;base64,a')]
            self.assertEqual([event['type'] for event in events].count('model_retry'),1)
            report=next(event['report'] for event in events if event['type']=='report')
            self.assertEqual(report['summary'],'repaired')
            self.assertEqual(len(retrieval_calls),1)
            self.assertEqual(len([body for body in requests if body['model']=='vision-model']),1)
            report_requests=[body for body in requests if body['model']=='report-model']
            self.assertEqual(len(report_requests),2)
            self.assertTrue(report_requests[0]['stream'])
            self.assertFalse(report_requests[1]['stream'])
            self.assertIn('修复要求',report_requests[1]['messages'][0]['content'])
            self.assertIn('必须优先返回完整 JSON',report_requests[1]['messages'][0]['content'])
            self.assertEqual(report_requests[0]['max_tokens'],1200)
            self.assertEqual(report_requests[1]['max_tokens'],2000)
            self.assertTrue(all(body['response_format']=={'type':'json_object'} for body in report_requests))
        asyncio.run(scenario())

    def test_failed_format_repair_only_fails_the_affected_agent(self):
        async def scenario():
            requests=[]
            async def handler(request):
                body=json.loads(request.content);requests.append(body)
                model=body['model']
                if not body.get('stream'):
                    self.assertEqual(model,'bad-model')
                    return httpx.Response(200,json={'choices':[{'message':{'content':'still not JSON'}}]})
                if model=='bad-model':
                    text='{"summary":"first"}\n{"summary":"second"}'
                else:
                    text=json.dumps({'summary':model,'official_facts':[],'claim_evidence_audit':[],'evidence_gaps':[]})
                chunk={'choices':[{'delta':{'content':text}}]}
                return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
            real=httpx.AsyncClient
            models=[{'id':'bad','base_url':'https://model.example/v1','api_key':'x','model':'bad-model'},
                    {'id':'facts','base_url':'https://model.example/v1','api_key':'x','model':'facts-model'},
                    {'id':'visual','base_url':'https://model.example/v1','api_key':'x','model':'visual-model'}]
            with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(handler),**kw)):
                events=[json.loads(s.removeprefix('data: ')) async for s in run_models({'models':models,'query':'product','search_enabled':False},None)]
            self.assertEqual({event['model_id'] for event in events if event['type']=='report'},{'facts','visual'})
            errors=[event for event in events if event['type']=='model_error']
            self.assertEqual([event['model_id'] for event in errors],['bad'])
            self.assertIn('初始报告格式错误',errors[0]['message'])
            self.assertEqual([event['model_id'] for event in events if event['type']=='model_retry'],['bad'])
            self.assertEqual(len([body for body in requests if body['model']=='bad-model' and not body.get('stream')]),1)
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
