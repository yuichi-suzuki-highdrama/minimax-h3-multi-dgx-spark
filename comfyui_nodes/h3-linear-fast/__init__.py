"""h3-linear-fast: H3 の INT8 ConvRot Linear の GEMM を Triton に差し替える（2026-10-28）。
環境変数 H3_LINEAR_FAST=1 のときだけ、起動時に comfy_kitchen の CUDA backend の int8_linear を差し替える。
ノードは追加しない。ComfyUI 本体と comfy-kitchen のファイルは書き換えない。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import h3_linear_fast_patch  # noqa: E402

h3_linear_fast_patch.install()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
