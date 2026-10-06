#!/usr/bin/env python3
"""Gate 0: the Anthropic kit's fused triangle kernels against our AF3 fork's own modules, one layer at a time.

Runs inside the AlphaPulldown AF3 image (jax 0.9.1, tokamax 0.0.11, alphafold3 = the KosinskiLab fork, v3.0.3), with the kit's
core (opt_core @ f4f62fa6, a frozen export) on sys.path. No AlphaFold model is built: each cell is one Haiku module on random or
DeepMind inputs.

Cell = (variant, C, N, mask, weights). Variants: trimul_out / trimul_in (TriangleMultiplication), att_start / att_end
(GridSelfAttention, transpose False / True). C = 128 (trunk, MSA stack, confidence head) or 64 (template stack, head dim 16).
N = an AF3 token bucket; the mask pads it to the midpoint of the bucket below it (`pad`), or is an asymmetric random pair mask
(`asym`, orientation test). Weights: `rand` (bf16, like the shipped ones) or `dm:<stack>:<layer>` (DeepMind's).

Arms, all on the same params and inputs:
  stock          the fork's module as the model runs it (tokamax GLU / flash attention, pair_attention_chunk_size rows)
  stock_nochunk  attention only: the same module with row chunking off (separates "fusion" from "no chunking")
  fused          the kit's Pallas prologue / core / epilogue, tile rows from its serve table (DeepMind convention:
                 ending_bias_transposed=False)
  fused_tokcore  attention only: the kit's prologue / epilogue around tokamax's flash-attention core

Per arm and cell:
  accuracy     error against an f64 numpy reference of the stock module's math (bf16 weights and inputs as the model has them),
               on a sampled grid of pixels (trimul) or whole attention lines (attention). Metrics on the real region:
                 rel = ||y - ref||_F / ||ref||_F        max = max|y - ref| / rms(ref)
               Pre-registered gate (fused and fused_tokcore against stock on the same cell):
                 rel_fused <= 1.2 * rel_stock + 1e-3   and   max_fused <= 2 * max_stock + 1e-2
  padded       every output in the padded region finite (it never reaches real outputs, see leak; NaN would, via 0 * NaN)
  leak         real-region outputs bit-identical when the padded region's inputs are replaced by other finite values
  determinism  two calls of one executable bit-identical
  speed        AOT-compiled executable, warmed up, timed with block_until_ready; arms interleaved per repetition; small N run
               K layers in one fori_loop (residual update) so per-call dispatch does not dominate. Compile time separate.
  memory       XLA's memory_analysis of the single-layer executable (temp / output / argument bytes)

One JSON line per (cell, arm) to --out. A cell that runs out of memory stops that arm for larger N of the same (variant, C).
"""
import argparse
import functools
import json
import math
import os
import platform
import sys
import time
import traceback

import numpy as np

BUCKETS = (256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)   # AF3's default token buckets
EQ = {"trimul_out": "ikc,jkc->ijc", "trimul_in": "kjc,kic->ijc"}
GATE = dict(rel_ratio=1.2, rel_floor=1e-3, max_ratio=2.0, max_floor=1e-2)
DM_SCOPES = {   # weights=dm:<stack>:<layer> -> the module scope prefix in af3.bin (layer = index on its stacked axis)
    "trunk": "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/",
    "template": "diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/",
}
MODULE_NAME = {"trimul_out": "triangle_multiplication_outgoing", "trimul_in": "triangle_multiplication_incoming",
               "att_start": "pair_attention1", "att_end": "pair_attention2"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="JSON-lines file, appended to")
    p.add_argument("--kit", required=True, help="directory holding the kit's opt_core package (frozen export)")
    p.add_argument("--variants", default="trimul_out,trimul_in,att_start,att_end")
    p.add_argument("--channels", default="128,64")
    p.add_argument("--sizes", default=",".join(map(str, BUCKETS)))
    p.add_argument("--asym_sizes", default="256,512", help="sizes that also get an asymmetric random pair mask")
    p.add_argument("--dm_weights", default="", help="AF3 weights dir (af3.bin); adds DeepMind-weight cells")
    p.add_argument("--dm_sizes", default="512,1536")
    p.add_argument("--dm_layers", default="trunk:0,trunk:24,trunk:47,template:0,template:1")
    p.add_argument("--reps", type=int, default=15, help="timed repetitions per arm (interleaved)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--lines", type=int, default=12, help="reference lines (attention) / grid side (trimul)")
    p.add_argument("--tile_cc", default="", help="tile table to use (default: the device's compute capability)")
    p.add_argument("--interpret", action="store_true", help="CPU smoke test: Pallas interpret mode, XLA attention")
    p.add_argument("--no_timing", action="store_true")
    p.add_argument("--negative_control", action="store_true",
                   help="add arm fused_biasT on att_end: the OpenFold bias orientation, which the gate must fail")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------------------- f64 reference
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ln(x, scale, offset, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = (x * x).mean(-1, keepdims=True) - m * m
    return (x - m) / np.sqrt(v + eps) * scale + offset


def ref_trimul(act_rows, act_cols, act_pix, mask, P, variant, I, J):
    """f64 output at pixels I x J (|I|, |J|, C). act_rows(idx) -> (len, N, C) rows of act, act_cols(idx) -> (len, N, C) with
    [j, k] = act[k, j], act_pix(I, J) -> (|I|, |J|, C). mask (N, N) f64."""
    def glu(x):
        return (x @ P["w_proj"]) * _sig(x @ P["w_gate"])            # tokamax: activation(x @ w_gate) * (x @ w_up)
    ln_in = functools.partial(_ln, scale=P["ln_in_scale"], offset=P["ln_in_offset"])
    if variant == "trimul_out":                                     # out[i,j,c] = sum_k a(i,k)_c b(j,k)_c
        a = glu(ln_in(act_rows(I)))[..., 0::2] * mask[I, :, None]
        b = glu(ln_in(act_rows(J)))[..., 1::2] * mask[J, :, None]
        tri = np.einsum("ikc,jkc->ijc", a, b)
    else:                                                           # out[i,j,c] = sum_k a(k,j)_c b(k,i)_c
        a = glu(ln_in(act_cols(J)))[..., 0::2] * mask[:, J].T[:, :, None]
        b = glu(ln_in(act_cols(I)))[..., 1::2] * mask[:, I].T[:, :, None]
        tri = np.einsum("jkc,ikc->ijc", a, b)
    y = _ln(tri, P["ln_c_scale"], P["ln_c_offset"]) @ P["w_out"]
    g = _sig(ln_in(act_pix(I, J)) @ P["w_gl"])
    return y * g


def ref_attention(act_chunk, act_line, mask, P, transpose, lines, N):
    """f64 module output on whole lines: {b: (N, C)} = pixels (b, s) (starting) or (s, b) (ending); NaN where the line has no
    valid key. DeepMind convention: bias[h, s, t] = pair_bias_projection(LN(act)[s, t]) for both orientations; key t of line b
    is valid iff pair_mask[t, b] > 0 (the stock module's swapped mask)."""
    H, D, C = P["wq"].shape
    bias = np.empty((H, N, N))
    ch = max(1, min(N, (1 << 26) // (N * C)))
    for q0 in range(0, N, ch):
        x = _ln(act_chunk(q0, min(N, q0 + ch)), P["ln_scale"], P["ln_offset"])
        bias[:, q0:q0 + ch, :] = np.moveaxis(x @ P["wb"], -1, 0)
    out = {}
    for b in lines:
        x = _ln(act_line(b, transpose), P["ln_scale"], P["ln_offset"])          # (N, C): position s of line b
        q = np.einsum("sc,hdc->shd", x, P["wq"]) / math.sqrt(D)
        k = np.einsum("sc,hdc->shd", x, P["wk"])
        v = np.einsum("sc,chd->shd", x, P["wv"])
        valid = mask[:, b] > 0
        o = np.empty((N, H, D))
        if not valid.any():
            out[b] = np.full((N, C), np.nan)
            continue
        for h in range(H):
            lg = q[:, h, :] @ k[:, h, :].T + bias[h]
            lg[:, ~valid] = -np.inf
            lg -= lg.max(axis=1, keepdims=True)
            w = np.exp(lg)
            w /= w.sum(axis=1, keepdims=True)
            o[:, h, :] = w @ v[:, h, :]
        g = _sig(x @ P["wg"].T)                                                  # gating_query stored (HD, C)
        out[b] = (o.reshape(N, H * D) * g) @ P["wo"]
    return out


def err_metrics(y, ref):
    y = np.asarray(y, np.float64)
    ok = np.isfinite(ref)
    if not ok.any():
        return dict(rel=None, max=None, n=0)
    d, r = (y - ref)[ok], ref[ok]
    rms = float(np.sqrt(np.mean(r * r)))
    return dict(rel=float(np.linalg.norm(d) / np.linalg.norm(r)), max=float(np.max(np.abs(d)) / rms) if np.isfinite(d).all() else float("inf"),
                n=int(ok.sum()))


def gate_ok(m, s):
    if m.get("rel") is None or s.get("rel") is None:
        return None
    return bool(m["rel"] <= GATE["rel_ratio"] * s["rel"] + GATE["rel_floor"] and m["max"] <= GATE["max_ratio"] * s["max"] + GATE["max_floor"])


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    args = parse_args()
    from absl import flags as _absl_flags                           # tokamax reads absl flags from sys.argv on first use
    _absl_flags.FLAGS([sys.argv[0]])
    if args.interpret:
        import jax.experimental.pallas as _pl
        _orig = _pl.pallas_call
        _pl.pallas_call = lambda *a, **k: _orig(*a, **{**k, "interpret": True})   # the kit calls pl.pallas_call at call time
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import tokamax
    from alphafold3.model import model_config
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.components import utils
    from alphafold3.model.network import modules
    sys.path.insert(0, args.kit)
    from opt_core.kernels.fpf_pallas import trimul_pallas as K, triattn_pallas as A
    from opt_core.kernels import fpf_pallas_serve as S

    dev = jax.devices()[0]
    cc = str(getattr(dev, "compute_capability", "") or "")
    tile_cc = args.tile_cc or cc
    import importlib.metadata as _md
    try:
        mem_limit_gb = round(dev.memory_stats()["bytes_limit"] / 2**30, 2)
    except Exception:  # noqa: BLE001
        mem_limit_gb = None
    env = dict(host=platform.node(), device=getattr(dev, "device_kind", str(dev)), cc=cc, jax=jax.__version__,
               jaxlib=_md.version("jaxlib"), tokamax=_md.version("tokamax"), mem_limit_gb=mem_limit_gb,
               xla_env={k: v for k, v in os.environ.items() if k.startswith(("XLA_", "JAX_", "TF_"))},
               job=os.environ.get("SLURM_JOB_ID", ""), interpret=args.interpret)
    try:
        tile_cc_, _, own = S.tables_for(tile_cc)
        env["tiles"] = S.tiles_label(tile_cc_, own)
        tiles_err = None
    except S.Refusal as r:
        env["tiles"] = f"refused:{r.kind}"
        tiles_err = str(r)
    print("ENV", json.dumps(env), flush=True)
    attn_impl = "xla" if args.interpret else "triton"
    gc = model_config.GlobalConfig(flash_attention_implementation=attn_impl)
    gc_nochunk = model_config.GlobalConfig(flash_attention_implementation=attn_impl, pair_attention_chunk_size=((None, None),))
    out_f = open(args.out, "a")

    def emit(rec):
        out_f.write(json.dumps({**env, **rec}) + "\n")
        out_f.flush()
        brief = {k: rec.get(k) for k in ("variant", "C", "N", "mask", "weights", "arm", "status", "ms", "rel", "max", "gate", "leak_ok", "det_ok",
                                          "pad_finite", "temp_mb")}
        print("CELL", json.dumps(brief), flush=True)

    # ------------------------------------------------------------------ the fused classes: the port's prototype (same params as stock)
    def tokamax_core(q4, k4, v4, bias, mask2):
        return tokamax.dot_product_attention(q4, k4, v4, bias=bias[None], mask=mask2[:, None, None, :], implementation=attn_impl)

    class FusedTriangleMultiplication(modules.TriangleMultiplication):
        def __call__(self, act, mask):
            c, n = act.shape[-1], act.shape[0]
            w_proj, _ = hm.haiku_linear_get_params(act, num_output=2 * c, name="projection")
            w_gate, _ = hm.haiku_linear_get_params(act, num_output=2 * c, initializer=self.global_config.final_init, name="gate")
            with hk.name_scope("left_norm_input"):
                s_in = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones)
                o_in = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            with hk.name_scope("center_norm"):
                s_c = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones)
                o_c = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            w_out, _ = hm.haiku_linear_get_params(act, num_output=c, initializer=self.global_config.final_init, name="output_projection")
            w_gl, _ = hm.haiku_linear_get_params(act, num_output=c, name="gating_linear")
            p = dict(ln_in_scale=s_in, ln_in_offset=o_in, w_proj=w_proj, w_gate=w_gate, ln_c_scale=s_c, ln_c_offset=o_c, w_out=w_out, w_gl=w_gl)
            return K.triangle_multiplication_fused(act, mask, p, equation=self.config.equation, cfg=S.trimul_cfg(n, tile_cc)).astype(act.dtype)

    def make_fused_attention(use_tokamax_core, ending_bias_transposed=False):
        class FusedGridSelfAttention(modules.GridSelfAttention):
            def __call__(self, act, pair_mask):
                h, c, n, dt = self.config.num_head, act.shape[-1], act.shape[0], act.dtype
                d = S.head_dim(c, h)
                with hk.name_scope("act_norm"):
                    s = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones)
                    o = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)

                def w(name, shape):
                    with hk.name_scope(name):
                        return hk.get_parameter("weights", shape, dt, init=jnp.zeros)
                hp = {"act_norm": {"scale": s, "offset": o}, "pair_bias_projection": {"weights": w("pair_bias_projection", (c, h))},
                      "q_projection": {"weights": w("q_projection", (h, d, c))}, "k_projection": {"weights": w("k_projection", (h, d, c))},
                      "v_projection": {"weights": w("v_projection", (c, h, d))}, "gating_query": {"weights": w("gating_query", (h * d, c))},
                      "output_projection": {"weights": w("output_projection", (h * d, c))}}
                kp = A.attn_params_from_haiku(hp)
                cfg = S.attn_cfg(n, tile_cc)
                if not use_tokamax_core:
                    return A.grid_self_attention_fused(act, pair_mask, kp, transpose=self.transpose,
                                                       ending_bias_transposed=ending_bias_transposed, cfg=cfg).astype(dt)
                c2 = {**A.DEFAULT_ATTN_CFG, **A.ATTN_CFG_BY_N.get(n, {}), **cfg}       # the kit's fused_with_core, DeepMind convention
                H, D = kp["H"], kp["D"]
                q, k, v, braw = A.attn_prologue(act, kp["ln_scale"], kp["ln_offset"], kp["wq_t"], kp["wk_t"], kp["wv2"], kp["wb16"],
                                                transpose=self.transpose, t=c2["t1"], num_warps=c2["w1"])
                bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))
                mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0
                o4 = tokamax_core(q.reshape(n, n, H, D), k.reshape(n, n, H, D), v.reshape(n, n, H, D), bias, mask2)
                return A.attn_epilogue(o4.reshape(n, n, H * D), act, kp["ln_scale"], kp["ln_offset"], kp["wg_t"], kp["wo"],
                                       transpose=self.transpose, t=c2["t2"], num_warps=c2["w2"]).astype(dt)
        return FusedGridSelfAttention

    FUSED_ATT = make_fused_attention(False)
    FUSED_ATT_TOK = make_fused_attention(True)
    FUSED_ATT_BIAST = make_fused_attention(False, ending_bias_transposed=True)

    def module_fn(variant, arm):
        is_tm = variant.startswith("trimul")

        def fn(act, mask):
            with utils.bfloat16_context():
                if is_tm:
                    cls = FusedTriangleMultiplication if arm == "fused" else modules.TriangleMultiplication
                    return cls(modules.TriangleMultiplication.Config(equation=EQ[variant]), gc, name="layer")(act, mask)
                cls = {"stock": modules.GridSelfAttention, "stock_nochunk": modules.GridSelfAttention, "fused": FUSED_ATT,
                       "fused_tokcore": FUSED_ATT_TOK, "fused_biasT": FUSED_ATT_BIAST}[arm]
                g = gc_nochunk if arm == "stock_nochunk" else gc
                return cls(modules.GridSelfAttention.Config(num_head=4), g, transpose=(variant == "att_end"), name="layer")(act, mask)
        return hk.without_apply_rng(hk.transform(fn))

    # ------------------------------------------------------------------ params
    dm_params = None

    def make_params(variant, C, N, weights, key):
        shapes = jax.eval_shape(module_fn(variant, "stock").init, key, jax.ShapeDtypeStruct((N, N, C), jnp.bfloat16),
                                jax.ShapeDtypeStruct((N, N), jnp.bfloat16))
        leaves = []
        flat = [(m, n, s) for m, d in shapes.items() for n, s in d.items()]
        if weights == "rand":
            ks = jax.random.split(key, len(flat))
            out = {}
            for (m, n, s), k in zip(flat, ks):
                z = jax.random.normal(k, s.shape, jnp.float32)
                if n == "scale":
                    v = 1.0 + 0.2 * z
                elif n == "offset":
                    v = 0.2 * z
                else:                                             # every weight matrix here has C inputs (H * D == C)
                    v = (z / math.sqrt(C)).astype(jnp.bfloat16)   # stored bf16, like af3.bin
                out.setdefault(m, {})[n] = v
            return out
        _, stack, layer = weights.split(":")
        prefix = DM_SCOPES[stack] + MODULE_NAME[variant]
        out = {}
        for m, n, s in flat:
            src = dm_params[prefix + m[len("layer"):]][n][int(layer)]
            assert src.shape == s.shape, (m, n, src.shape, s.shape)
            out.setdefault(m, {})[n] = jnp.asarray(src)
        return out

    def params_f64(variant, params):
        g = lambda m, n: np.asarray(jnp.asarray(params["layer" + m][n]).astype(jnp.float32), np.float64)   # noqa: E731
        if variant.startswith("trimul"):
            return dict(ln_in_scale=g("/left_norm_input", "scale"), ln_in_offset=g("/left_norm_input", "offset"),
                        w_proj=g("/projection", "weights"), w_gate=g("/gate", "weights"), ln_c_scale=g("/center_norm", "scale"),
                        ln_c_offset=g("/center_norm", "offset"), w_out=g("/output_projection", "weights"), w_gl=g("/gating_linear", "weights"))
        return dict(ln_scale=g("/act_norm", "scale"), ln_offset=g("/act_norm", "offset"), wb=g("/pair_bias_projection", "weights"),
                    wq=g("/q_projection", "weights"), wk=g("/k_projection", "weights"), wv=g("/v_projection", "weights"),
                    wg=g("/gating_query", "weights"), wo=g("/output_projection", "weights"))

    # ------------------------------------------------------------------ inputs
    def make_inputs(C, N, mask_kind, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        act = (jax.random.normal(k1, (N, N, C), jnp.bfloat16) * (1.0 + 0.5 * jax.random.uniform(k2, (C,), jnp.bfloat16))
               + 0.3 * jax.random.normal(k3, (C,), jnp.bfloat16)).astype(jnp.bfloat16)
        if mask_kind == "pad":                                       # real tokens: the midpoint of the bucket below N and N
            prev = max([b for b in BUCKETS if b < N], default=0)
            n_real = (prev + N) // 2 + 1 if N in BUCKETS else N - N // 4
            tok = np.arange(N) < n_real
            mask = np.outer(tok, tok).astype(np.float32)
        else:
            n_real = N
            tok = np.ones(N, bool)
            mask = (np.asarray(jax.random.uniform(k4, (N, N))) < 0.85).astype(np.float32)
        return act, mask, tok, n_real

    def sample_lines(n_real, N, rng):
        base = {0, 1, n_real // 3, n_real // 2, (2 * n_real) // 3, n_real - 2, n_real - 1}
        extra = [int(x) for x in rng.choice(n_real, size=min(n_real, args.lines), replace=False)]
        real = sorted({x for x in base if 0 <= x < n_real} | set(extra[: max(0, args.lines - len(base))]))[: args.lines]
        pad = [n_real, N - 1] if N > n_real else []
        return real, pad

    # ------------------------------------------------------------------ one cell
    is_oom = lambda e: any(s in str(e) for s in ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory", "OOM"))   # noqa: E731
    stopped = {}                                                    # (variant, C, arm) -> smallest N at which a call ran out of memory

    def fingerprint(y, real):
        """Two position-weighted sums of the output's raw bits over the real region: bitwise equality without a second full copy."""
        bits = jax.lax.bitcast_convert_type(y, jnp.uint16).astype(jnp.uint32)
        pos = jax.lax.broadcasted_iota(jnp.uint32, y.shape, 0) * jnp.uint32(y.shape[1] * y.shape[2]) + \
            jax.lax.broadcasted_iota(jnp.uint32, y.shape, 1) * jnp.uint32(y.shape[2]) + jax.lax.broadcasted_iota(jnp.uint32, y.shape, 2)
        b = jnp.where(real, bits, jnp.uint32(0))
        return jnp.stack([jnp.sum(b * (pos % jnp.uint32(65521) + 1)), jnp.sum(b * (pos % jnp.uint32(65519) + 7) ^ pos)])
    fingerprint = jax.jit(fingerprint)

    def run_cell(variant, C, N, mask_kind, weights):
        is_tm = variant.startswith("trimul")
        arms = ["stock", "fused"] if is_tm else ["stock", "stock_nochunk", "fused", "fused_tokcore"]
        if args.negative_control and variant == "att_end":
            arms.append("fused_biasT")
        base0 = dict(variant=variant, C=C, N=N, mask=mask_kind, weights=weights)
        if all(N >= stopped.get((variant, C, a), 1 << 30) for a in arms):
            for a in arms:
                emit({**base0, "arm": a, "status": "skipped_after_oom", "oom_at": stopped[(variant, C, a)]})
            return
        key = jax.random.PRNGKey(args.seed * 1000003 + N * 7 + C)
        kp_, ki = jax.random.split(key)
        params = make_params(variant, C, N, weights, kp_)
        act, mask_np, tok, n_real = make_inputs(C, N, mask_kind, ki)
        mask = jnp.asarray(mask_np, jnp.bfloat16)                    # the model's pair mask is in the activation dtype
        if tiles_err:
            arms = [a for a in arms if not a.startswith("fused")]
        base = dict(variant=variant, C=C, N=N, n_real=n_real, mask=mask_kind, weights=weights)
        served = S.served_reason("trimul" if is_tm else "triattn", (N, N, C), "bfloat16", (N, N), num_head=4, cc=tile_cc if not tiles_err else None,
                                 pad=False)
        base["served_reason"] = served
        if tiles_err:
            emit({**base, "arm": "fused", "status": "refused", "error": tiles_err})

        # padded region; the leak input is built only around its own call
        pad_pix = ~np.outer(tok, tok)
        pad_dev = jnp.asarray(pad_pix)[..., None]
        has_pad = bool(pad_pix.any())
        everywhere = jnp.ones((1, 1, 1), bool)

        def make_leak():
            noise = 8.0 * jax.random.normal(jax.random.PRNGKey(N + 17), act.shape, jnp.bfloat16)
            return jnp.where(pad_dev, noise, act)

        # the f64 reference first, so each arm's full output can be dropped after its sampled pixels are taken
        rng = np.random.default_rng(N * 31 + C)
        real_lines, _ = sample_lines(n_real, N, rng)
        P = params_f64(variant, params)
        f64 = lambda x: np.asarray(jnp.asarray(x).astype(jnp.float32), np.float64)   # noqa: E731
        ref_set, ref_error = None, None
        t0 = time.perf_counter()
        if is_tm:
            I = J = np.array(real_lines)
            pick = lambda y: f64(y[np.ix_(I, J)])               # noqa: E731
            try:
                ref_set = ref_trimul(lambda idx: f64(act[np.asarray(idx)]), lambda idx: f64(jnp.swapaxes(act[:, np.asarray(idx)], 0, 1)),
                                     lambda I_, J_: f64(act[np.ix_(I_, J_)]), mask_np.astype(np.float64), P, variant, I, J)
            except Exception as e:  # noqa: BLE001
                ref_error = repr(e)[:300] + " | " + traceback.format_exc()[-800:]
        else:
            transpose = variant == "att_end"
            lines = real_lines
            pick = lambda y: np.stack([f64(y[:, b] if transpose else y[b]) for b in lines])   # noqa: E731
            try:
                refd = ref_attention(lambda a, b: f64(act[a:b]), lambda b, tr: f64(act[:, b] if tr else act[b]), mask_np.astype(np.float64),
                                     P, transpose, lines, N)
                ref_set = np.stack([refd[b] for b in lines]) * np.where(tok[None, :, None], 1.0, np.nan)   # real query positions only
            except Exception as e:  # noqa: BLE001
                ref_error = repr(e)[:300] + " | " + traceback.format_exc()[-800:]
        ref_s = round(time.perf_counter() - t0, 1)

        recs, picks, timers = {}, {}, {}
        kk = 16 if N <= 768 else 4 if N <= 1536 else 1               # K residual layers per timed call; the same body at every N
        for arm in arms:
            if N >= stopped.get((variant, C, arm), 1 << 30):
                emit({**base, "arm": arm, "status": "skipped_after_oom", "oom_at": stopped[(variant, C, arm)]})
                continue
            rec = recs[arm] = dict(base, arm=arm)
            apply = module_fn(variant, arm).apply
            try:
                t0 = time.perf_counter()
                exe = jax.jit(apply).lower(params, act, mask).compile()
                rec["compile_s"] = round(time.perf_counter() - t0, 3)  # indicative: XLA's GEMM autotune cache is warm after the first arm
            except Exception as e:  # noqa: BLE001  e.g. a Triton shared-memory overflow: not an OOM, never stops the arm
                rec.update(status="compile_error", error=repr(e)[:1500])
                jax.clear_caches()
                continue
            try:
                ma = exe.memory_analysis()
                rec.update(temp_mb=round(ma.temp_size_in_bytes / 2**20, 1), out_mb=round(ma.output_size_in_bytes / 2**20, 1),
                           arg_mb=round(ma.argument_size_in_bytes / 2**20, 1))
            except Exception as e:  # noqa: BLE001
                rec["memory_analysis_error"] = repr(e)[:200]
            try:                                                    # the first call: accuracy and finiteness, then the output is freed
                y = exe(params, act, mask)
                y.block_until_ready()
                picks[arm] = pick(y)
                if ref_set is not None:
                    rec.update(err_metrics(picks[arm], ref_set))
                rec["real_finite"] = bool(jnp.all(jnp.where(pad_dev, True, jnp.isfinite(y))))
                if has_pad:
                    rec["pad_finite"] = bool(jnp.all(jnp.where(pad_dev, jnp.isfinite(y), True)))
                fp_all, fp_real = fingerprint(y, everywhere), fingerprint(y, ~pad_dev)
                del y
                rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                rec.update(status="oom" if is_oom(e) else "error", error=repr(e)[:600] + " | " + traceback.format_exc()[-900:])
                if rec["status"] == "oom":
                    stopped[(variant, C, arm)] = min(N, stopped.get((variant, C, arm), N))
                jax.clear_caches()
                continue
            try:                                                    # checks: a failure here is recorded; the first call's numbers stand
                rec["det_ok"] = bool(jnp.array_equal(fingerprint(exe(params, act, mask), everywhere), fp_all))
                if has_pad:
                    act_leak = make_leak()
                    rec["leak_ok"] = bool(jnp.array_equal(fingerprint(exe(params, act_leak, mask), ~pad_dev), fp_real))
                    del act_leak
            except Exception as e:  # noqa: BLE001
                rec["check_error"] = repr(e)[:300]
            if not args.no_timing:
                try:
                    def loop(params, act, mask, apply=apply):
                        return jax.lax.fori_loop(0, kk, lambda i, a: (a + apply(params, a, mask)).astype(a.dtype), act)
                    timers[arm] = jax.jit(loop).lower(params, act, mask).compile()
                except Exception as e:  # noqa: BLE001
                    rec["timing_error"] = repr(e)[:300]
            jax.clear_caches()

        # timing, arms interleaved per repetition
        if timers:
            t = {a: [] for a in timers}
            for r in range(args.warmup + args.reps):
                for arm, exe in list(timers.items()):
                    try:
                        t0 = time.perf_counter()
                        exe(params, act, mask).block_until_ready()
                        if r >= args.warmup:
                            t[arm].append((time.perf_counter() - t0) / kk * 1e3)
                    except Exception as e:  # noqa: BLE001
                        recs[arm]["timing_error"] = repr(e)[:300]
                        timers.pop(arm)
            for arm, v in t.items():
                if v:
                    recs[arm].update(ms=round(float(np.median(v)), 4), ms_p10=round(float(np.percentile(v, 10)), 4),
                                     ms_p90=round(float(np.percentile(v, 90)), 4), loop_k=kk, reps=len(v))
        timers.clear()

        # against stock on the same sampled pixels (real region), and the pre-registered gate
        if "stock" in picks:
            ok = np.isfinite(ref_set) if ref_set is not None else np.ones_like(picks["stock"], bool)
            s = picks["stock"][ok]
            for arm, y in picks.items():
                if arm != "stock":
                    recs[arm]["vs_stock_rel"] = float(np.linalg.norm(y[ok] - s) / np.linalg.norm(s))
                if arm.startswith("fused") and recs[arm].get("rel") is not None:
                    recs[arm]["gate"] = gate_ok(recs[arm], recs["stock"])
        for arm in arms:
            if arm in recs:
                recs[arm].update(ref_s=ref_s, ref_lines=len(real_lines))
                if ref_error:
                    recs[arm]["ref_error"] = ref_error
                emit(recs[arm])
        del picks, act, mask, params
        jax.clear_caches()

    # ------------------------------------------------------------------ the sweep
    variants = args.variants.split(",")
    channels = [int(x) for x in args.channels.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    asym = {int(x) for x in args.asym_sizes.split(",") if x}
    cells = []
    for C in channels:
        for v in variants:
            for N in sizes:
                cells.append((v, C, N, "pad", "rand"))
                if N in asym:
                    cells.append((v, C, N, "asym", "rand"))
    if args.dm_weights:
        import pathlib
        from alphafold3.model import params as af3_params
        dm_params = af3_params.get_model_haiku_params(pathlib.Path(args.dm_weights))
        for spec in args.dm_layers.split(","):
            stack, layer = spec.split(":")
            C = 128 if stack == "trunk" else 64
            if C not in channels:
                continue
            for v in variants:
                for N in [int(x) for x in args.dm_sizes.split(",")]:
                    cells.append((v, C, N, "pad", f"dm:{stack}:{layer}"))
    cells.sort(key=lambda c: (c[4] != "rand", c[2], c[1], c[0], c[3]))     # smallest first, random weights first
    print(f"CELLS {len(cells)}", flush=True)
    for cell in cells:
        t0 = time.perf_counter()
        try:
            run_cell(*cell)
        except Exception as e:  # noqa: BLE001
            emit(dict(variant=cell[0], C=cell[1], N=cell[2], mask=cell[3], weights=cell[4], arm="*", status="cell_error",
                      error=repr(e)[:400] + " | " + traceback.format_exc()[-1500:]))
        print(f"DONE {cell} {time.perf_counter() - t0:.1f}s", flush=True)
    out_f.close()


if __name__ == "__main__":
    main()
