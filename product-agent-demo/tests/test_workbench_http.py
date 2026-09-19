import json
import unittest
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from app.main import app

class WorkbenchHTTPTests(unittest.TestCase):
    def setUp(self): self.client=TestClient(app)

    def test_empty_initial_page_has_no_mock_playback_or_fake_results(self):
        r=self.client.get('/')
        self.assertEqual(r.status_code,200)
        self.assertIn('product-workbench.js',r.text)
        self.assertNotIn('Collaboration Playback',r.text)
        self.assertNotIn('62%',r.text)
        self.assertNotIn('La Roche-Posay',r.text)

    def test_missing_key_returns_actionable_error_before_upstream_call(self):
        r=self.client.post('/api/lab/run',data={'options':json.dumps({'models':[{'model':'qwen3.8-max','base_url':'https://dashscope.aliyuncs.com/compatible-mode/v1','api_key':''}]})})
        self.assertEqual(r.status_code,400)
        self.assertIn('API Key',r.json()['detail'])

    def test_unsupported_file_is_rejected(self):
        r=self.client.post('/api/lab/run',data={'options':json.dumps({'models':[{'model':'test','base_url':'http://127.0.0.1:9999/v1'}]})},files={'upload':('test.txt',b'hello','text/plain')})
        self.assertEqual(r.status_code,400)
        self.assertIn('PNG',r.json()['detail'])

    def test_invalid_connection_url_has_safe_error(self):
        r=self.client.post('/api/lab/test-model',json={'base_url':'file:///tmp/test','api_key':'must-not-be-disclosed'})
        self.assertEqual(r.json()['status'],'error')
        self.assertNotIn('must-not-be-disclosed',r.text)

    def test_multipart_image_reaches_all_three_agents(self):
        seen=[]
        async def upstream(request):
            body=json.loads(request.content);seen.append(body)
            if not body.get('stream'):
                return httpx.Response(200,json={'choices':[{'message':{'content':'{"brand":"FIXTURE","product_name":"HTTP pipeline fixture","confidence":0.9}'}}]})
            report={'summary':'TEST FIXTURE ONLY','official_facts':[],'claim_evidence_audit':[],'evidence_gaps':['No external source test']}
            chunk={'id':'fixture-request','choices':[{'delta':{'content':json.dumps(report)}}]}
            return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
        real=httpx.AsyncClient
        models=[{'id':id,'agent_role':id,'base_url':'http://127.0.0.1:9999/v1','model':'qwen3.8-max'} for id in ['facts','review','visual']]
        with patch('app.lab.httpx.AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(upstream),**kw)):
            response=self.client.post('/api/lab/run',data={'options':json.dumps({'models':models,'search_enabled':False})},files={'upload':('fixture.png',b'fixture-image-data','image/png')})
        self.assertEqual(response.status_code,200)
        events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        self.assertEqual({e['model_id'] for e in events if e['type']=='report'},{'facts','review','visual'})
        self.assertEqual(len(seen),4)
        self.assertTrue(all(body['messages'][1]['content'][0]['image_url']['url'].startswith('data:image/png;base64,') for body in seen))
