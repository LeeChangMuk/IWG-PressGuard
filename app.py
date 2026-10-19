"""PressGuard Light. Offline, read-only CSV decision-support application.

The portable binary only needs the Python standard library and the bundled JSON
model. It neither connects to a PLC nor sends factory data over the internet.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from pumpguard.inference import analyze_csv, load_model, mode_threshold, OPERATING_MODES, MODE_LABELS

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
VERSION = '1.2.1'
MAX_FEEDBACK = 1000
MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100000
DEFAULT_MAPPING = {k: k for k in ('TimeStamp', 'AI0_Vibration', 'AI1_Vibration', 'AI2_Current')}
MODE_DESCRIPTIONS = {
    'unmanned': '야간·무인 시간대용. 개발 구간 교차검증에서 정상 표본 오경보율 0.1% 이하 중 F1이 가장 높은 기준입니다. 경보가 적은 대신 놓치는 이상이 늘어납니다.',
    'standard': '일상 운전 기본값. 개발 구간 교차검증에서 오경보율 1% 이하 중 F1이 가장 높은 기준입니다.',
    'screening': '정기점검 전 점검 후보 선별용. 개발 구간 교차검증에서 이상 표본 95% 이상을 잡는 기준이며 오경보가 많습니다. 경보가 아니라 점검 우선순위로 사용하세요.',
}
DISCLAIMER = '현장 검증 전 참고용입니다. 이상 점수는 KAMP 데이터 기준이며 실제 고장 확률이나 고장 전 남은 시간이 아닙니다. 설비 조작은 현장 담당자가 결정합니다.'


def clean_cell(v):
    if isinstance(v, (list, dict)):
        v = json.dumps(v, ensure_ascii=False)
    if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')):
        return "'" + v
    return v


def csv_bytes(records):
    if not records:
        return b'\xef\xbb\xbf'
    out = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for r in records for k in r))
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()
    for r in records:
        w.writerow({k: clean_cell(v) for k, v in r.items()})
    return out.getvalue().encode('utf-8-sig')


def normalize_csv(data, mapping):
    if len(data) > MAX_BYTES:
        raise ValueError('파일이 16MB를 넘습니다. 수집 시간별로 나누어 주세요.')
    text = None
    for enc in ('utf-8-sig', 'cp949'):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise ValueError('UTF-8 또는 CP949 CSV 파일이 필요합니다.')
    if '\x00' in text:
        raise ValueError('CSV 텍스트가 아닙니다. 엑셀 파일은 CSV로 저장하세요.')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    names = reader.fieldnames or []
    if len(names) != len(set(names)):
        raise ValueError('동일한 컬럼 이름이 중복되어 있습니다.')
    source_names = [mapping.get(k, k) for k in DEFAULT_MAPPING]
    if len(set(source_names)) != 4:
        raise ValueError('시간·상부진동·하부진동·전류는 서로 다른 컬럼이어야 합니다.')
    missing = [s for s in source_names if s not in names]
    if missing:
        raise ValueError('필수 컬럼이 없습니다: ' + ', '.join(missing))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(DEFAULT_MAPPING))
    writer.writeheader()
    count = 0
    for r in reader:
        count += 1
        if count > MAX_ROWS:
            raise ValueError('100,000행을 넘습니다. 시간별 파일로 나누어 주세요.')
        if None in r:
            raise ValueError(f'{count+1}번째 줄의 컬럼 수가 맞지 않습니다.')
        writer.writerow({k: r.get(mapping.get(k, k), '') for k in DEFAULT_MAPPING})
    if not count:
        raise ValueError('데이터 행이 없는 파일입니다.')
    return out.getvalue().encode('utf-8'), count


class Service:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or ROOT / 'pumpguard' / 'model.json')
        self.model = load_model(self.model_path)
        self.model_hash = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.lock = threading.RLock()
        self.busy = False
        self.result = None
        self.error = ''
        self.revision = 0
        self.watch_folder = None
        self.watch_output = None
        self.watch_mapping = dict(DEFAULT_MAPPING)
        self.watch_seen = {}
        self.watch_stable = {}
        self.watch_processed = 0
        self.watch_errors = []
        self.stopped = threading.Event()
        self.feedback = []
        self.op_mode = 'standard'
        self.last_input = None
        threading.Thread(target=self._watch_loop, daemon=True).start()

    def modes(self):
        out = []
        for key in OPERATING_MODES:
            try:
                threshold = mode_threshold(self.model, key)
            except ValueError:
                continue
            out.append({'key': key, 'label': MODE_LABELS[key], 'threshold': threshold, 'description': MODE_DESCRIPTIONS[key]})
        return out

    def set_mode(self, op_mode):
        mode_threshold(self.model, op_mode)  # validates key and model support
        with self.lock:
            self.op_mode = op_mode
            last = self.last_input
        if last is not None:
            data, name, mapping, source_mode = last
            self.analyze(data, name, mapping, source_mode)

    def analyze(self, data, name, mapping=None, mode='현장 CSV'):
        with self.lock:
            if self.busy:
                raise ValueError('이전 분석이 진행 중입니다. 잠시 후 다시 시도하세요.')
            self.busy = True
            self.error = ''
        try:
            normalized, rows = normalize_csv(data, mapping or DEFAULT_MAPPING)
            with self.lock:
                op_mode = self.op_mode
            with tempfile.TemporaryDirectory(prefix='pressguard_') as td:
                p = Path(td) / 'input.csv'
                p.write_bytes(normalized)
                result = analyze_csv(p, self.model, op_mode)
            result.setdefault('metadata', {}).update({
                'app_version': VERSION,
                'source_name': Path(name).name,
                'source_sha256': hashlib.sha256(data).hexdigest(),
                'model_sha256': self.model_hash,
                'mode': mode,
                'processed_at_utc': datetime.now(timezone.utc).isoformat(),
                'input_rows': rows,
                'column_mapping': mapping or DEFAULT_MAPPING,
                'field_validation': 'NOT_PERFORMED',
                'disclaimer': DISCLAIMER,
            })
            result['metadata'].pop('source_file', None)
            with self.lock:
                self.last_input = (data, name, mapping, mode)
                self.result = result
                # Operator notes are kept across analyses (each note carries its source hash), so a
                # new file arriving in the watched folder never discards unsaved notes.
                self.revision += 1
            return result
        except Exception as ex:
            with self.lock:
                self.error = str(ex)
                self.result = None  # Never leave an old normal result after a failed input.
                self.revision += 1
            raise
        finally:
            with self.lock:
                self.busy = False

    def watch(self, folder, mapping=None, output_folder=None):
        p = Path(folder).expanduser()
        if not p.is_dir():
            raise ValueError('존재하는 CSV 폴더의 전체 경로를 입력하세요.')
        if not output_folder:
            raise ValueError('분석 결과를 보관할 별도 폴더를 입력하세요.')
        out = Path(output_folder).expanduser().resolve()
        if out == p.resolve():
            raise ValueError('입력과 결과 보관 폴더는 다르게 선택하세요.')
        out.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.watch_folder = p.resolve()
            self.watch_output = out
            self.watch_mapping = mapping or dict(DEFAULT_MAPPING)
            self.watch_seen = {}
            self.watch_stable = {}
            self.watch_processed = 0
            self.watch_errors = []

    def _watch_loop(self):
        while not self.stopped.wait(2):
            with self.lock:
                folder, mapping, output = self.watch_folder, dict(self.watch_mapping), self.watch_output
            if folder is None:
                continue
            try:
                files = sorted(folder.glob('*.csv'), key=lambda p: p.name.lower())[:1000]
                for p in files:
                    with self.lock:
                        if self.watch_folder != folder or self.busy:
                            break
                    stat = p.stat()
                    sig = (stat.st_size, stat.st_mtime_ns)
                    if self.watch_seen.get(p) == sig:
                        continue
                    if self.watch_stable.get(p) != sig:
                        self.watch_stable[p] = sig
                        continue
                    self.watch_seen[p] = sig
                    try:
                        if stat.st_size > MAX_BYTES:
                            raise ValueError('16MB 초과 파일')
                        data = p.read_bytes()
                        after = p.stat()
                        if sig != (after.st_size, after.st_mtime_ns):
                            self.watch_seen.pop(p, None)
                            continue
                        analyzed = self.analyze(data, p.name, mapping, '폴더 감시 / 현장 CSV')
                        # Retain every completed analysis, even after the dashboard advances.
                        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        digest = hashlib.sha256(data).hexdigest()[:12]
                        archive = output / f'PressGuard_{stamp}_{digest}.zip'
                        pending = archive.with_suffix('.tmp')
                        pending.write_bytes(self.export(analyzed, []))
                        pending.replace(archive)
                        with self.lock:
                            self.watch_processed += 1
                    except Exception as ex:
                        with self.lock:
                            if str(ex).startswith('이전 분석이 진행 중'):
                                self.watch_seen.pop(p, None)
                                continue
                            self.watch_errors = (self.watch_errors + [f'{p.name}: {ex}'])[-10:]
                            self.error = str(ex)
                            self.result = None
                            self.revision += 1
            except Exception as ex:
                with self.lock:
                    self.error = f'폴더 감시 오류: {ex}'
                    self.watch_folder = None

    def state(self):
        with self.lock:
            return {
                'busy': self.busy, 'error': self.error, 'revision': self.revision,
                'watching': self.watch_folder is not None,
                'watch_folder': str(self.watch_folder or ''),
                'watch_output': str(self.watch_output or ''),
                'watch_processed': self.watch_processed,
                'watch_errors': list(self.watch_errors),
                'result': self.result, 'feedback_count': len(self.feedback),
                'model_version': self.model.get('model_version', self.model.get('version', VERSION)),
                'app_version': VERSION,
                'model_hash': self.model_hash[:16],
                'op_mode': self.op_mode,
                'modes': self.modes(),
                'disclaimer': DISCLAIMER,
            }

    def export(self, result=None, feedback=None):
        with self.lock:
            if result is None:
                result = self.result
            if result is None:
                raise ValueError('먼저 파일을 분석하세요.')
            feedback = list(self.feedback) if feedback is None else feedback
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('predictions.csv', csv_bytes(result['records']))
            z.writestr('segment_priority.csv', csv_bytes(result.get('segments', [])))
            z.writestr('summary.json', json.dumps(result['summary'], ensure_ascii=False, indent=2))
            z.writestr('audit.json', json.dumps(result['metadata'], ensure_ascii=False, indent=2))
            z.writestr('operator_feedback.csv', csv_bytes(feedback))
            z.writestr('READ_ME.txt', 'IWG 프레스가드 라이트\n' + DISCLAIMER + '\n관측된 이상 구간과 실제 고장 발생은 다릅니다. 시간 공백·준비·판단보류를 정상으로 집계하지 마세요.\nsegment_priority.csv는 연속 구간별 점검 우선순위(현재 운영 모드 기준 초과 비율 순)입니다.\n작업자 의견은 확인되지 않은 관찰이며 자동 재학습에 사용되지 않습니다. 기록마다 대상 파일명·해시가 함께 저장됩니다.\n')
        return out.getvalue()


def make_handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, content_type='application/json; charset=utf-8', attachment=None):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            if isinstance(body, str):
                body = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'none'")
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(body)

        def permitted(self, auth=True):
            expected = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected:
                self.send(403, {'error': '허용하지 않은 호스트입니다.'})
                return False
            origin = self.headers.get('Origin')
            if origin is not None and origin != 'http://' + expected:
                self.send(403, {'error': '다른 사이트의 요청은 허용하지 않습니다.'})
                return False
            if auth and not secrets.compare_digest(self.headers.get('X-PressGuard-Token', ''), token):
                self.send(403, {'error': '실행 링크를 통해 접속하세요.'})
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if not self.permitted(auth=path not in ('/', '/favicon.ico')):
                return
            if path == '/':
                self.send(200, (ROOT / 'dashboard.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.send(200, service.state())
            elif path == '/api/export':
                try:
                    self.send(200, service.export(), 'application/zip', 'PressGuard_results.zip')
                except ValueError as ex:
                    self.send(400, {'error': str(ex)})
            elif path == '/favicon.ico':
                self.send(204, b'', 'image/x-icon')
            else:
                self.send(404, {'error': '찾을 수 없습니다.'})

        def do_POST(self):
            if not self.permitted():
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= MAX_BYTES * 2:
                    raise ValueError('요청 크기가 허용 범위를 벗어났습니다.')
                payload = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == '/api/analyze':
                    data = base64.b64decode(payload['data'], validate=True)
                    service.analyze(data, str(payload.get('name', 'input.csv')), payload.get('mapping'))
                elif path == '/api/demo':
                    name = {'normal': 'normal_test.csv', 'anomaly': 'anomaly_test.csv'}.get(payload.get('kind'))
                    if not name:
                        raise ValueError('알 수 없는 데모입니다.')
                    service.analyze((ROOT / 'demo' / name).read_bytes(), name, mode='KAMP 공개 데이터 재생 / 실제 현장 아님')
                elif path == '/api/watch':
                    service.watch(str(payload.get('folder', '')), payload.get('mapping'), payload.get('output_folder'))
                elif path == '/api/mode':
                    service.set_mode(str(payload.get('mode', '')))
                elif path == '/api/stop-watch':
                    with service.lock:
                        service.watch_folder = None
                elif path == '/api/feedback':
                    note = str(payload.get('note', '')).strip()[:2000]
                    if not note:
                        raise ValueError('관찰 내용을 입력하세요.')
                    with service.lock:
                        if not service.result:
                            raise ValueError('먼저 파일을 분석하세요.')
                        service.feedback.append({'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'observation': note, 'verified': False, 'source_name': service.result['metadata']['source_name'], 'source_sha256': service.result['metadata']['source_sha256']})
                        del service.feedback[:-MAX_FEEDBACK]
                elif path == '/api/shutdown':
                    self.send(200, {'ok': True})
                    service.stopped.set()
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                else:
                    self.send(404, {'error': '찾을 수 없습니다.'})
                    return
                self.send(200, {'ok': True})
            except (ValueError, KeyError, OSError, TypeError) as ex:
                self.send(400, {'error': str(ex)})
            except Exception:
                self.send(500, {'error': '분석 중 오류가 발생했습니다. 입력 형식과 모델 파일을 확인하세요.'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description='IWG PressGuard Light offline CSV monitor')
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--connection-file', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--mode', choices=OPERATING_MODES, default='standard',
                        help='unmanned=무인 감시, standard=표준, screening=정기점검 선별')
    args = parser.parse_args()
    service = Service(args.model)
    service.set_mode(args.mode)
    if args.analyze:
        if not args.output:
            parser.error('--analyze requires --output')
        args.output.mkdir(parents=True, exist_ok=True)
        service.analyze(args.analyze.read_bytes(), args.analyze.name)
        (args.output / 'PressGuard_results.zip').write_bytes(service.export())
        (args.output / 'summary.json').write_text(json.dumps(service.result['summary'], ensure_ascii=False, indent=2), encoding='utf-8')
        service.stopped.set()
        return
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service, token))
    url = f'http://127.0.0.1:{server.server_port}/#{token}'
    if args.connection_file:
        args.connection_file.parent.mkdir(parents=True, exist_ok=True)
        args.connection_file.write_text(json.dumps({'url': url, 'port': server.server_port, 'token': token}), encoding='utf-8')
    if sys.stdout:
        print('PressGuard running at', url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        service.stopped.set()
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if getattr(sys, 'frozen', False):
            # A GUI binary has no console. Preserve a local diagnostic, no telemetry.
            logdir = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'IWG_PressGuard'
            logdir.mkdir(parents=True, exist_ok=True)
            (logdir / 'startup_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
