"""Gate 0b: the kit's fused triangle kernels bound into AlphaFold 3's modules for one process (a pilot, not the port).

install() rebinds alphafold3.model.network.modules.{TriangleMultiplication, GridSelfAttention} to subclasses that read the stock
parameters and call the kit's Pallas kernels (opt_core @ f4f62fa6 on sys.path), DeepMind convention (ending_bias_transposed=False).
Every call the kernels do not serve runs the stock class; served and fallback calls are counted at trace time and printed at exit.

Environment:
  G0B_TRIMUL   1 | 0                     fused triangle multiplication (default 1)
  G0B_ATT      auto | kit | tokcore | off  attention: the kit's flash core, or its prologue/epilogue around tokamax's core.
                                          auto = kit on compute capability 8.0, 9.x, 10.x, 12.x (own or 9.0 tiles), tokcore on 8.6 / 8.9
                                          (Gate 0: the kit core on generic safe tiles is 0.62-0.87x there)
  G0B_ATT_MAX_N  0 = no limit; else attention above this pair size stays stock (memory: the fused path drops row chunking)
"""
import atexit
import collections
import json
import os
import sys

STATS = collections.Counter()


def _cc():
    import jax
    return str(getattr(jax.devices()[0], "compute_capability", "") or "")


def config():
    cc = _cc()
    att = os.environ.get("G0B_ATT", "auto")
    if att == "auto":
        att = "tokcore" if cc in ("8.6", "8.9") else "kit"
    return dict(cc=cc, trimul=os.environ.get("G0B_TRIMUL", "1") == "1", att=att, att_max_n=int(os.environ.get("G0B_ATT_MAX_N", "0") or 0))


class _LazyConfig(dict):
    """Resolved at the first trace, so install() does not start the jax backend before AlphaPulldown has configured it."""
    def __getitem__(self, k):
        if not dict.__len__(self):
            self.update(config())
            if self["cc"].startswith("12."):
                os.environ.setdefault("OPT_CORE_PALLAS_ALLOW_FALLBACK_CC", "1")   # sm_120: the 9.0 table (Gate 0: passes, 1.15-1.47x)
            print("G0B_CONFIG " + json.dumps(dict(self)), file=sys.stderr, flush=True)
        return dict.__getitem__(self, k)


def install():
    cfg = _LazyConfig()
    import haiku as hk
    import jax.numpy as jnp
    import tokamax
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import modules
    from opt_core.kernels.fpf_pallas import trimul_pallas as K, triattn_pallas as A
    from opt_core.kernels import fpf_pallas_serve as S
    stock_tm, stock_ga = modules.TriangleMultiplication, modules.GridSelfAttention

    def served(kind, act, mask, num_head=None):
        if act.dtype != jnp.bfloat16:
            return "dtype"
        return S.served_reason(kind, act.shape, act.dtype, mask.shape, num_head=num_head, cc=cfg["cc"], pad=False)

    class FusedTriangleMultiplication(stock_tm):
        def __call__(self, act, mask):
            n, c = act.shape[0], act.shape[-1]
            why = None if cfg["trimul"] else "off"
            why = why or served("trimul", act, mask)
            STATS[f"trimul C={c} N={n} " + ("fused" if why is None else f"stock:{why}")] += 1
            if why is not None:
                return super().__call__(act, mask)
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
            return K.triangle_multiplication_fused(act, mask, p, equation=self.config.equation, cfg=S.trimul_cfg(n, cfg["cc"])).astype(act.dtype)

    class FusedGridSelfAttention(stock_ga):
        def __call__(self, act, pair_mask):
            h, n, c, dt = self.config.num_head, act.shape[0], act.shape[-1], act.dtype
            why = None if cfg["att"] in ("kit", "tokcore") else "off"
            if why is None and cfg["att_max_n"] and n > cfg["att_max_n"]:
                why = "over_max_n"
            why = why or served("triattn", act, pair_mask, num_head=h)
            STATS[f"attention C={c} N={n} " + (f"fused:{cfg['att']}" if why is None else f"stock:{why}")] += 1
            if why is not None:
                return super().__call__(act, pair_mask)
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
            tiles = S.attn_cfg(n, cfg["cc"])
            if cfg["att"] == "kit":
                return A.grid_self_attention_fused(act, pair_mask, kp, transpose=self.transpose, ending_bias_transposed=False,
                                                   cfg=tiles).astype(dt)
            c2 = {**A.DEFAULT_ATTN_CFG, **A.ATTN_CFG_BY_N.get(n, {}), **tiles}
            H, D = kp["H"], kp["D"]
            q, k, v, braw = A.attn_prologue(act, kp["ln_scale"], kp["ln_offset"], kp["wq_t"], kp["wk_t"], kp["wv2"], kp["wb16"],
                                            transpose=self.transpose, t=c2["t1"], num_warps=c2["w1"])
            bias = jnp.transpose(braw[:, :, :H], (2, 0, 1))
            mask2 = jnp.swapaxes(pair_mask, -1, -2) > 0
            o4 = tokamax.dot_product_attention(q.reshape(n, n, H, D), k.reshape(n, n, H, D), v.reshape(n, n, H, D), bias=bias[None],
                                               mask=mask2[:, None, None, :], implementation=self.global_config.flash_attention_implementation)
            return A.attn_epilogue(o4.reshape(n, n, H * D), act, kp["ln_scale"], kp["ln_offset"], kp["wg_t"], kp["wo"],
                                   transpose=self.transpose, t=c2["t2"], num_warps=c2["w2"]).astype(dt)

    modules.TriangleMultiplication = FusedTriangleMultiplication
    modules.GridSelfAttention = FusedGridSelfAttention
    atexit.register(lambda: print("G0B_SERVED " + json.dumps(dict(sorted(STATS.items()))), file=sys.stderr, flush=True))
    return cfg
