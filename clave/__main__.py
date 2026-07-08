"""Standalone runner: ``python -m clave`` serves Clave alone on port 6019.

Normally Clave runs inside the GLADE webui (third nav tab, mounted at
``/clave``); this entry point keeps the old standalone workflow working.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "glafic2", "python"),
          os.path.join(_ROOT, "Rhongomyniad")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask  # noqa: E402

from clave import bp  # noqa: E402

app = Flask(__name__)
app.register_blueprint(bp)

if __name__ == "__main__":
    port = int(os.environ.get("CLAVE_PORT", 6019))
    print(f"\n{'=' * 52}")
    print("  Clave -- Gravitational Lens Calculator")
    print(f"  URL  : http://localhost:{port}/clave/")
    print(f"{'=' * 52}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
