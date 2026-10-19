"""Behavioral checks for safe offline CSV ingestion, export and HTTP boundary."""
import base64
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import zipfile
from http.server import ThreadingHTTPServer

import app
from pumpguard.features import CHANNELS, FEATURE_NAMES, FEATURE_VERSION


def fixture_model():
    return {
        'schema_version':2,'feature_version':FEATURE_VERSION,'feature_names':FEATURE_NAMES,
        'channels':list(CHANNELS),'feature_indices':[2],
        'predictor':{'kind':'logistic','intercept':-2,'coef':[1],'center':[0],'scale':[1]},
        'calibration':{'intercept':0,'coef':1},'threshold':0.5,
        'max_window':10,'max_gap_seconds':0.15,'expected_sample_interval_seconds':0.1,
        'expected_sampling_hz':10,'model_id':'test-only-model',
        'review_absolute_limits':[100,100,100],
        'alarm_policy':{'window':3,'minimum_history':3,'votes_required':2},
    }


def csv_input(current=1, n=8):
    rows=['TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current,Equipment_state']
    for i in range(n):
        rows.append(f'2026-01-01 00:00:00.{i}00,{0.1*i},{-0.1*i},{current},0')
    return ('\n'.join(rows)+'\n').encode()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        model=self.root/'model.json'
        model.write_text(json.dumps(fixture_model()),encoding='utf-8')
        self.service=app.Service(model)

    def tearDown(self):
        self.service.stopped.set()
        self.temp.cleanup()

    def test_missing_columns_and_empty_file_rejected(self):
        for data in (b'x,y\n1,2',b'TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current\n'):
            with self.assertRaises(ValueError): self.service.analyze(data,'bad.csv')
        self.assertIsNone(self.service.result)

    def test_bad_file_clears_previous_normal_result(self):
        self.service.analyze(csv_input(),'good.csv')
        with self.assertRaises(ValueError): self.service.analyze(b'not,csv','bad.csv')
        self.assertIsNone(self.service.result)
        self.assertTrue(self.service.error)

    def test_label_does_not_affect_prediction(self):
        a=self.service.analyze(csv_input(3),'a.csv')
        b=self.service.analyze(csv_input(3).replace(b',3,0',b',3,1'),'b.csv')
        self.assertEqual([r['score'] for r in a['records']],[r['score'] for r in b['records']])
        self.assertTrue(any(r['alarm'] for r in a['records']))

    def test_korean_mapping_and_cp949(self):
        text=csv_input().decode().replace('TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current','시각,상부,하부,전류')
        mapping=dict(zip(app.DEFAULT_MAPPING,('시각','상부','하부','전류')))
        result=self.service.analyze(text.encode('cp949'),'한글.csv',mapping)
        self.assertEqual(result['summary']['rows'],8)

    def test_invalid_numeric_is_not_normal(self):
        text=csv_input().replace(b',0.0,-0.0,1,',b',NaN,-0.0,1,')
        result=self.service.analyze(text,'bad_number.csv')
        self.assertEqual(result['records'][0]['status'],'INVALID')
        json.dumps(result,allow_nan=False)

    def test_export_contains_lineage_and_feedback(self):
        self.service.analyze(csv_input(),'good.csv')
        self.service.feedback=[{'observation':'=HYPERLINK("example")','verified':False}]
        with zipfile.ZipFile(io.BytesIO(self.service.export())) as z:
            self.assertIn('predictions.csv',z.namelist())
            audit=json.loads(z.read('audit.json'))
            self.assertEqual(len(audit['source_sha256']),64)
            self.assertNotIn('source_file',audit)
            self.assertEqual(audit['field_validation'],'NOT_PERFORMED')
            self.assertIn("'=HYPERLINK",z.read('operator_feedback.csv').decode('utf-8-sig'))

    def test_export_contains_segment_priority(self):
        self.service.analyze(csv_input(current=3),'good.csv')
        with zipfile.ZipFile(io.BytesIO(self.service.export())) as z:
            rows=list(csv.DictReader(io.StringIO(z.read('segment_priority.csv').decode('utf-8-sig'))))
        self.assertEqual([r['rank'] for r in rows],['1'])
        self.assertEqual(rows[0]['judged_rows'],'6')

    def test_feedback_survives_next_analysis(self):
        # a watched folder can replace the displayed result at any time; unsaved notes must stay
        token='test-token'
        server=ThreadingHTTPServer(('127.0.0.1',0),app.make_handler(self.service,token))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            self.service.analyze(csv_input(),'first.csv')
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/feedback',data=json.dumps({'note':'체결 이상 없음'}).encode(),
                                           headers={'X-PressGuard-Token':token,'Content-Type':'application/json'})
            self.assertTrue(json.load(urllib.request.urlopen(request))['ok'])
            self.service.analyze(csv_input(current=2),'second.csv')
            with zipfile.ZipFile(io.BytesIO(self.service.export())) as z:
                notes=list(csv.DictReader(io.StringIO(z.read('operator_feedback.csv').decode('utf-8-sig'))))
            self.assertEqual([(n['observation'],n['source_name']) for n in notes],[('체결 이상 없음','first.csv')])
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_watch_requires_separate_output(self):
        with self.assertRaises(ValueError):self.service.watch(self.root)
        with self.assertRaises(ValueError):self.service.watch(self.root,output_folder=self.root)

    def test_stable_watch_archives_once_and_never_changes_input(self):
        folder=self.root/'input';folder.mkdir()
        out=self.root/'results'
        file=folder/'sensor.csv';data=csv_input();file.write_bytes(data)
        self.service.watch(folder,output_folder=out)
        deadline=time.monotonic()+9
        while not list(out.glob('*.zip')) and time.monotonic()<deadline:time.sleep(0.2)
        self.assertEqual(len(list(out.glob('*.zip'))),1)
        self.assertEqual(file.read_bytes(),data)
        self.assertEqual(self.service.watch_processed,1)

    def test_http_auth_host_origin_and_export(self):
        token='test-token'
        server=ThreadingHTTPServer(('127.0.0.1',0),app.make_handler(self.service,token))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with self.assertRaises(urllib.error.HTTPError) as ex:
                urllib.request.urlopen(url+'/api/state')
            self.assertEqual(ex.exception.code,403)
            for bad_header in ({'Origin':'https://evil.invalid'},{'Host':'evil.invalid'}):
                headers={'X-PressGuard-Token':token,**bad_header}
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(urllib.request.Request(url+'/api/state',headers=headers))
            payload={'name':'input.csv','data':base64.b64encode(csv_input()).decode()}
            request=urllib.request.Request(url+'/api/analyze',data=json.dumps(payload).encode(),headers={'X-PressGuard-Token':token,'Content-Type':'application/json'})
            self.assertTrue(json.load(urllib.request.urlopen(request))['ok'])
            request=urllib.request.Request(url+'/api/state',headers={'X-PressGuard-Token':token})
            self.assertEqual(json.load(urllib.request.urlopen(request))['result']['summary']['rows'],8)
        finally:
            server.shutdown();server.server_close();thread.join(2)


    def test_mode_switch_reanalyzes_same_input_and_rejects_unsupported(self):
        self.service.analyze(csv_input(current=2.2),'a.csv')
        self.assertEqual(self.service.result['metadata']['operating_mode'],'standard')
        self.assertEqual([m['key'] for m in self.service.modes()],['standard'])
        with self.assertRaises(ValueError):self.service.set_mode('screening')
        model=fixture_model();model['operating_modes']={'unmanned':{'threshold':0.6},'standard':{'threshold':0.5},'screening':{'threshold':0.4}}
        path=self.root/'modes.json';path.write_text(json.dumps(model),encoding='utf-8')
        service=app.Service(path)
        try:
            service.analyze(csv_input(current=1.8),'b.csv')
            before=service.revision;standard=service.result['summary']['alarm_rows']
            service.set_mode('screening')
            self.assertGreater(service.revision,before)
            self.assertEqual(service.result['metadata']['operating_mode'],'screening')
            self.assertEqual(service.result['metadata']['source_name'],'b.csv')
            self.assertGreaterEqual(service.result['summary']['alarm_rows'],standard)
            self.assertEqual(service.state()['op_mode'],'screening')
            with self.assertRaises(ValueError):service.set_mode('nonexistent')
        finally:
            service.stopped.set()


if __name__=='__main__':unittest.main()
