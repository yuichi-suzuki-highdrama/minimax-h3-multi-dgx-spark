"""2 台組の worker（sp_worker.py はカスタムノードを読まない）に h3-fuse-graph を効かせる起動口（2026-10-28）。
使い方: python -s h3_fuse_launch.py <sp_worker.py> [sp_worker の引数...]
comfy.ldm.minimax.model の読み込み直後に DiTBlock.forward を差し替え（H3_FUSE=1 のとき）、
計測（H3_FG_PROFILE / H3_FG_TIMING）があれば sp_forward を包んでから、sp_worker.py を __main__ として走らせる。
h3-linear-fast の sitecustomize（PYTHONPATH の boot）とはぶつからない。
"""
import os
import runpy
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_node = os.path.dirname(_here)
if _node not in sys.path:
    sys.path.append(_node)

import h3_fg_profile  # noqa: E402
import h3_fuse_patch  # noqa: E402


def _on_model(_module):
    h3_fuse_patch.install()
    h3_fg_profile.install()


h3_fuse_patch.after_import(lambda n: n == "comfy.ldm.minimax.model", _on_model)
if h3_fg_profile._enabled():
    h3_fuse_patch.after_import(lambda n: n.endswith("minimax_sp.sp_forward"), h3_fg_profile.install_sp)

script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name="__main__")
