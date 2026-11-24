"""Check that your Gemini key, model name and network work, and list the models your key can use.

    export GEMINI_API_KEY=...   (or set it in Colab)
    python tools/check_gemini.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from agents.llm import GeminiProvider  # noqa: E402

p = GeminiProvider()
print("model asked for:", p.model)
ok, msg = p.check()
print("OK  " if ok else "FAIL", msg)
if not ok:
    alt = p.discover_model()
    print("a model your key can use:", alt or "(none found: check the key, or that the Generative Language API is enabled)")
    print("set it with:  export GSC_LLM_MODEL=" + (alt or "<model>"))
sys.exit(0 if ok else 1)
