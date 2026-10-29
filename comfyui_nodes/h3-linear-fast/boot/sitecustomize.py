"""2 台組の worker（sp_worker.py はカスタムノードを読まない）でも差し替えを効かせるための起動フック（2026-10-28）。
使い方: worker を起動するときだけ PYTHONPATH=<このフォルダ> H3_LINEAR_FAST=1 を付ける。
comfy_kitchen が import された後に差し替える必要があるので、import の仕組みに小さなフックを掛けて待つ。
元の sitecustomize（Ubuntu の /usr/lib/python3.12/sitecustomize.py）があれば先に実行する。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    _cand = os.path.join(_p, "sitecustomize.py")
    if _p and os.path.abspath(_p) != _here and os.path.exists(_cand):
        try:
            exec(compile(open(_cand, encoding="utf-8").read(), _cand, "exec"), {"__name__": "sitecustomize_orig"})
        except Exception:
            pass
        break

if os.environ.get("H3_LINEAR_FAST", "0") == "1":
    _node = os.path.dirname(_here)
    if _node not in sys.path:
        sys.path.insert(0, _node)
    import importlib.abc
    import importlib.machinery

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "comfy_kitchen.backends.cuda":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                import h3_linear_fast_patch
                h3_linear_fast_patch.install()

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
