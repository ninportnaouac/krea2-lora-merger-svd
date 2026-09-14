
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class Pair:
    raw_base: str
    down: torch.Tensor
    up: torch.Tensor
    alpha: float | None
    down_suffix: str
    up_suffix: str


@dataclass
class LokrPair:
    raw_base: str
    w1: torch.Tensor
    w2: torch.Tensor


@dataclass
class ScanResult:
    filename: str
    lora_layers: int = 0
    lokr_layers: int = 0
    incomplete: list[str] = field(default_factory=list)
    conventions: set[str] = field(default_factory=set)


def load_state(path: str) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def load_metadata(path: str) -> dict[str, str] | None:
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.metadata()


def _split_role(key: str):
    patterns = [
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
        (".lora_A.default.weight", "down"),
        (".lora_B.default.weight", "up"),
        (".alpha", "alpha"),
        (".lokr_w1", "lokr_w1"),
        (".lokr_w2", "lokr_w2"),
    ]
    for suffix, role in patterns:
        if key.endswith(suffix):
            return key[:-len(suffix)], suffix, role
    return None, None, None


def extract_groups(state: dict[str, torch.Tensor]):
    groups: dict[str, dict[str, Any]] = {}
    for key, t in state.items():
        base, suffix, role = _split_role(key)
        if role is None:
            continue
        g = groups.setdefault(base, {})
        g[role] = t
        g[role + "_suffix"] = suffix
    return groups


_TOP_LEVEL = [
    (re.compile(r"^transformer\.transformer_blocks\.(\d+)\."), r"blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.layerwise_blocks\.(\d+)\."), r"txtfusion.layerwise_blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.refiner_blocks\.(\d+)\."), r"txtfusion.refiner_blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.projector$"), "txtfusion.projector"),
    (re.compile(r"^transformer\.img_in$"), "first"),
    (re.compile(r"^transformer\.final_layer\.linear$"), "last.linear"),
    (re.compile(r"^transformer\.time_embed\.linear_1$"), "tmlp.0"),
    (re.compile(r"^transformer\.time_embed\.linear_2$"), "tmlp.2"),
    (re.compile(r"^transformer\.txt_in\.linear_1$"), "txtmlp.1"),
    (re.compile(r"^transformer\.txt_in\.linear_2$"), "txtmlp.3"),
    (re.compile(r"^transformer\.time_mod_proj$"), "tproj.1"),
]
_COMPONENT = [
    (re.compile(r"\.attn\.to_out\.0$"), ".attn.wo"),
    (re.compile(r"\.attn\.to_q$"), ".attn.wq"),
    (re.compile(r"\.attn\.to_k$"), ".attn.wk"),
    (re.compile(r"\.attn\.to_v$"), ".attn.wv"),
    (re.compile(r"\.attn\.to_gate$"), ".attn.gate"),
    (re.compile(r"\.ff\.up$"), ".mlp.up"),
    (re.compile(r"\.ff\.down$"), ".mlp.down"),
]


def canonical_base(raw: str) -> str:
    """Canonical Krea2-ish name so Diffusers/native LoRAs can be matched."""
    x = raw

    # common prefixes
    for p in ("base_model.model.", "lora_unet_", "lora_transformer_"):
        if x.startswith(p):
            x = x[len(p):]
            break

    # Diffusers Krea2 -> native Krea2
    if x.startswith("transformer."):
        for pat, repl in _TOP_LEVEL:
            y = pat.sub(repl, x)
            if y != x:
                x = y
                break
        for pat, repl in _COMPONENT:
            x = pat.sub(repl, x)

    # normalize model prefix
    if x.startswith("model.diffusion_model."):
        x = x[len("model.diffusion_model."):]

    # Kohya underscore form: keep conservative; only normalize separators for comparison
    return re.sub(r"[._]+", "_", x.lower()).strip("_")


def pairs_from_state(state: dict[str, torch.Tensor]):
    loras: dict[str, Pair] = {}
    lokrs: dict[str, LokrPair] = {}
    incomplete: list[str] = []

    for raw, g in extract_groups(state).items():
        canon = canonical_base(raw)
        if "down" in g and "up" in g:
            alpha = None
            if "alpha" in g:
                try:
                    alpha = float(g["alpha"].item())
                except Exception:
                    alpha = None
            loras[canon] = Pair(
                raw_base=raw,
                down=g["down"],
                up=g["up"],
                alpha=alpha,
                down_suffix=g["down_suffix"],
                up_suffix=g["up_suffix"],
            )
        elif "lokr_w1" in g and "lokr_w2" in g:
            lokrs[canon] = LokrPair(raw_base=raw, w1=g["lokr_w1"], w2=g["lokr_w2"])
        else:
            incomplete.append(raw)

    return loras, lokrs, incomplete


def scan_file(path: str) -> ScanResult:
    state = load_state(path)
    loras, lokrs, incomplete = pairs_from_state(state)
    conventions = set()
    for p in loras.values():
        if p.down_suffix == ".lora_down.weight":
            conventions.add("Kohya/ComfyUI")
        elif ".default." in p.down_suffix:
            conventions.add("PEFT default")
        else:
            conventions.add("Diffusers/PEFT")
    if lokrs:
        conventions.add("LoKr")
    return ScanResult(
        filename=Path(path).name,
        lora_layers=len(loras),
        lokr_layers=len(lokrs),
        incomplete=incomplete,
        conventions=conventions,
    )


def _compress_factors_svd(
    up_cat: torch.Tensor,
    down_cat: torch.Tensor,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compress U@D without materializing the full delta matrix."""
    exact_rank = up_cat.shape[1]
    target_rank = max(1, min(int(target_rank), exact_rank))
    if target_rank >= exact_rank:
        return up_cat, down_cat, 1.0

    # U = Qu Ru ; D^T = Qv Rv ; U D = Qu (Ru Rv^T) Qv^T
    qu, ru = torch.linalg.qr(up_cat, mode="reduced")
    qv, rv = torch.linalg.qr(down_cat.T, mode="reduced")
    core = ru @ rv.T
    u, s, vh = torch.linalg.svd(core, full_matrices=False)

    total = torch.sum(s * s)
    kept = torch.sum(s[:target_rank] * s[:target_rank])
    retained = float((kept / total).item()) if float(total) > 0 else 1.0

    sroot = torch.sqrt(s[:target_rank].clamp_min(0))
    up = (qu @ u[:, :target_rank]) * sroot.unsqueeze(0)
    down = sroot.unsqueeze(1) * (vh[:target_rank, :] @ qv.T)
    return up.contiguous(), down.contiguous(), retained


def merge_standard_loras(
    paths: list[str],
    weights: list[float],
    output_path: str,
    out_dtype: torch.dtype = torch.float16,
    preserve_first_metadata: bool = True,
    compress: bool = False,
    target_rank: int = 64,
    trigger_words: str = "",
    write_trigger_txt: bool = True,
) -> dict[str, Any]:
    """Merge standard LoRAs exactly or compress the exact merge with SVD."""
    if len(paths) != len(weights):
        raise ValueError("paths/weights length mismatch")
    if not paths:
        raise ValueError("No LoRA files supplied")

    all_loras = []
    all_lokrs = []
    metadata = load_metadata(paths[0]) if preserve_first_metadata else None

    for p in paths:
        state = load_state(p)
        loras, lokrs, _ = pairs_from_state(state)
        all_loras.append(loras)
        all_lokrs.append(lokrs)

    lokr_total = sum(len(x) for x in all_lokrs)
    if lokr_total:
        names=[]
        for path,lokrs in zip(paths,all_lokrs):
            if lokrs:
                names.append(f"{Path(path).name}: {len(lokrs)} LoKr")
        raise RuntimeError(
            "LoKr détecté (" + ", ".join(names) + "). "
            "Cette version fusionne et compresse les LoRA standards uniquement."
        )

    all_keys = sorted(set().union(*(set(d.keys()) for d in all_loras)))
    out: dict[str, torch.Tensor] = {}
    merged_layers = 0
    contributors_total = 0
    skipped_shape=[]
    energies=[]
    exact_ranks=[]
    final_ranks=[]

    for canon in all_keys:
        contributors=[]
        template: Pair | None = None
        expected_out = expected_in = None

        for loras,user_w,path in zip(all_loras,weights,paths):
            pair = loras.get(canon)
            if pair is None or float(user_w)==0:
                continue
            if pair.down.dim()!=2 or pair.up.dim()!=2:
                skipped_shape.append(f"{canon}: {Path(path).name} tensors non-2D")
                continue
            rank=pair.down.shape[0]
            if pair.up.shape[1]!=rank:
                skipped_shape.append(f"{canon}: {Path(path).name} rank mismatch up{tuple(pair.up.shape)} down{tuple(pair.down.shape)}")
                continue
            out_dim=pair.up.shape[0]; in_dim=pair.down.shape[1]
            if expected_out is None:
                expected_out,expected_in=out_dim,in_dim; template=pair
            elif out_dim!=expected_out or in_dim!=expected_in:
                skipped_shape.append(f"{canon}: {Path(path).name} shape ({out_dim},{in_dim}) != ({expected_out},{expected_in})")
                continue

            scale=(pair.alpha/rank) if pair.alpha is not None else 1.0
            scale*=float(user_w)
            contributors.append((pair.up.to(torch.float32)*scale, pair.down.to(torch.float32)))

        if not contributors or template is None:
            continue

        up_cat=torch.cat([x[0] for x in contributors],dim=1)
        down_cat=torch.cat([x[1] for x in contributors],dim=0)
        exact_rank=int(down_cat.shape[0])

        if compress:
            up_out,down_out,retained=_compress_factors_svd(up_cat,down_cat,target_rank)
        else:
            up_out,down_out,retained=up_cat,down_cat,1.0

        rank_out=int(down_out.shape[0])
        raw=template.raw_base
        out[raw+template.down_suffix]=down_out.to(out_dtype).contiguous()
        out[raw+template.up_suffix]=up_out.to(out_dtype).contiguous()
        out[raw+'.alpha']=torch.tensor(float(rank_out),dtype=torch.float32)

        merged_layers+=1
        contributors_total+=len(contributors)
        energies.append(retained)
        exact_ranks.append(exact_rank)
        final_ranks.append(rank_out)

    if not out:
        raise RuntimeError("Aucune couche LoRA standard compatible n'a été fusionnée.")

    meta=dict(metadata or {})
    meta['merge_type']='svd_compressed_lora' if compress else 'exact_lora_concatenation'
    meta['merged_from']=' | '.join(f"{Path(p).name}@{w:g}" for p,w in zip(paths,weights))
    meta['merged_layers']=str(merged_layers)
    if compress:
        meta['target_rank']=str(target_rank)

    trigger_words=(trigger_words or "").strip()
    if trigger_words:
        meta['trigger_words']=trigger_words
        meta['ss_trigger_words']=trigger_words
        meta['modelspec.trigger_words']=trigger_words

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out,output_path,metadata=meta)

    trigger_txt=None
    if trigger_words and write_trigger_txt:
        trigger_txt=str(Path(output_path).with_suffix(".txt"))
        Path(trigger_txt).write_text("Trigger: "+trigger_words+"\n", encoding="utf-8")

    return {
        'output':output_path,
        'merged_layers':merged_layers,
        'contributors_total':contributors_total,
        'shape_warnings':skipped_shape,
        'exact_rank_min':min(exact_ranks) if exact_ranks else None,
        'exact_rank_max':max(exact_ranks) if exact_ranks else None,
        'final_rank_min':min(final_ranks) if final_ranks else None,
        'final_rank_max':max(final_ranks) if final_ranks else None,
        'average_energy':(sum(energies)/len(energies)) if energies else 1.0,
        'compressed':compress,
        'target_rank':target_rank,
        'trigger_words':trigger_words,
        'trigger_txt':trigger_txt,
    }
