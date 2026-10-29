"""h3-fuse-graph: H3 の DiT ブロックの細かい処理（norm・adaLN の変調・ゲート・残差）を Triton で融合する（2026-10-28）。
環境変数 H3_FUSE=1 のときだけ、起動時に comfy.ldm.minimax.model.DiTBlock.forward を差し替える（出力はビット一致）。
H3_FG_PROFILE / H3_FG_TIMING は計測用（h3_fg_profile.py）。
ノードは追加しない。ComfyUI 本体と comfy-kitchen のファイルは書き換えない。
2 台組の worker には boot/h3_fuse_launch.py から効かせる（start_sp_worker.sh）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import h3_fg_profile  # noqa: E402
import h3_fuse_patch  # noqa: E402

h3_fg_profile.install()
if h3_fg_profile._enabled():
    h3_fuse_patch.after_import(lambda n: n.endswith("minimax_sp.sp_forward"), h3_fg_profile.install_sp)
h3_fuse_patch.install()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
