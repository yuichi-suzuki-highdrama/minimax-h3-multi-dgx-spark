"""LoRA などの weight patch を head から worker へ渡す（2 台版）。

buqi 版は同じ機械の一時ファイルの道を worker に読ませていたが、別ホストでは読めないので
torch.save したバイト列を gloo の broadcast_object_list で送る。送り元は自分の head プロセス
（直結リンク越し）だけなので、worker 側の torch.load は weights_only=False で ComfyUI の
weight_adapter（LoRAAdapter など）をそのまま復元する。信用できない相手には使わないこと。
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import io

import torch
import comfy.model_management


@dataclass(frozen=True)
class PatchManifest:
    protocol: int
    sha256: str
    version: str
    keys: tuple[str, ...]
    blob: bytes = b""

    def message(self):
        d = asdict(self)
        d["op"] = "patches"
        return d

    @classmethod
    def from_message(cls, message):
        if message.get("protocol") != 2:
            raise RuntimeError("unsupported MiniMax SP patch protocol")
        return cls(2, message["sha256"], message["version"], tuple(message["keys"]), message.get("blob", b""))

    def ready(self, rank):
        return {"rank": rank, "ready": True, "version": self.version,
                "sha256": self.sha256, "keys": list(self.keys)}


def validate_replies(manifest, replies, world):
    if len(replies) != world or any(reply != manifest.ready(rank) for rank, reply in enumerate(replies)):
        raise RuntimeError(f"MiniMax SP patch synchronization failed; denoise prohibited: {replies}")


def validate_patcher(patcher):
    if patcher.weight_wrapper_patches or patcher.hook_patches:
        raise RuntimeError("MiniMax SP dynamic weight wrappers/hooks are not synchronized")
    if patcher.get_additional_models_with_key("multigpu"):
        raise RuntimeError("Do not combine ComfyUI thread-based MultiGPU clones with MiniMax SP workers")
    for entries in patcher.patches.values():
        if any(len(entry) != 5 or entry[4] is not None for entry in entries):
            raise RuntimeError("MiniMax SP supports native static patches without custom patch functions")


def patch_version(patcher):
    validate_patcher(patcher)
    return str(patcher.patches_uuid)


def memory_budget(patcher):
    if patcher.is_dynamic():
        return 0
    return max(0, comfy.model_management.get_total_memory(patcher.load_device)
               - patcher.loaded_size() - comfy.model_management.extra_reserved_memory())


@contextmanager
def export_patches(patcher):
    version = patch_version(patcher)
    keys = tuple(sorted(patcher.patches))
    if not keys:
        yield PatchManifest(2, hashlib.sha256(b"").hexdigest(), version, keys)
        return
    buf = io.BytesIO()
    torch.save(patcher.patches, buf)
    blob = buf.getvalue()
    yield PatchManifest(2, hashlib.sha256(blob).hexdigest(), version, keys, blob)


class WorkerPatches:
    def __init__(self, patcher, rank):
        self.patcher = patcher
        self.rank = rank
        self.memory_required = 0
        self.version = None
        self.ready = False

    def prepare(self, memory_required, version):
        self.ready = False
        self.memory_required = memory_required
        comfy.model_management.load_models_gpu([self.patcher], memory_required=memory_required)
        self.ready = self.version == version
        return {"ready": True}

    def apply(self, manifest):
        self.ready = False
        patcher = self.patcher
        if not manifest.keys:
            patches = {}
            if self.version is None and not patcher.patches:
                patcher.patches_uuid = manifest.version
                patcher.model.current_weight_patches_uuid = manifest.version
                self.version = manifest.version
                self.ready = True
                return manifest.ready(self.rank)
        else:
            if hashlib.sha256(manifest.blob).hexdigest() != manifest.sha256:
                raise RuntimeError("patch blob digest mismatch")
            # 送り元は自分の head だけ（信用できる）。weight_adapter の物体を含むので weights_only=False。
            patches = torch.load(io.BytesIO(manifest.blob), map_location="cpu", weights_only=False)
        if tuple(sorted(patches)) != manifest.keys:
            raise RuntimeError("patch manifest keys mismatch")
        patcher.unpatch_model(patcher.offload_device)
        patcher.patches = patches
        patcher.patches_uuid = manifest.version
        comfy.model_management.load_models_gpu([patcher], memory_required=self.memory_required)
        if patcher.model.current_weight_patches_uuid != patcher.patches_uuid:
            raise RuntimeError("ModelPatcher did not apply the requested patch version")
        self.version = manifest.version
        self.ready = True
        return manifest.ready(self.rank)

    def require_ready(self):
        if not self.ready:
            raise RuntimeError("MiniMax SP forward before successful patch verification")
