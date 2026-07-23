"""Cube browser — the spaxel workspace's 'Forward model' stand-in.  [AI-Claude]

The spaxel family has NO emulator (the flow trains on raw THOR cubes), so live
params→cube sliders aren't possible. Instead: browse the held-out THOR cubes across
inclination, with true parameters, channel maps and the 3-D wind — a real-data
intuition builder for how geometry and kinematics imprint on the observable.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import core
from views.upload_cube import _channel_fig, _moment_fig


def render(ctx):
    ex = core.load_cube_examples(ctx.config_path)
    phys = ctx.prior.from_z(ex["z"])
    j_incl = ctx.names.index("incl")
    order = np.argsort(phys[:, j_incl])
    extent = float((ctx.cube_meta or {}).get("cube_extent_kpc", 60.0))

    st.caption("No cube emulator exists (the flow deliberately trains on raw THOR cubes, "
               "not a surrogate) — so this tab browses REAL held-out simulations instead of "
               "synthesizing them. Slide through viewing angles and watch the biconical "
               "morphology turn.")
    k = st.select_slider("held-out example, sorted by inclination",
                         options=list(order),
                         format_func=lambda i: f"i = {phys[i, j_incl]:.0f}°",
                         key=f"pg_cube_{ctx.config_path}")
    p = dict(zip(ctx.names, phys[k]))
    st.markdown("  ·  ".join(f"**{core.PARAM_META[n][0]}** = {p[n]:.3g} {core.PARAM_META[n][1]}"
                             for n in ctx.names))
    st.plotly_chart(_channel_fig(ex["cubes"][k], ctx.vel, extent),
                    use_container_width=True, key="pg_channels")
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.plotly_chart(_moment_fig(ex["cubes"][k], ctx.vel, extent),
                        use_container_width=True, key="pg_moments")
    with c2:
        fig3d = core.cached_biconical(*core.round_pv(p["theta"], p["incl"], p["av"],
                                                     p["vexp_kms"], p["logN"], 100.0),
                                      disk_hh_kpc=0.5, disk_on=True, preview=True)
        st.plotly_chart(fig3d, use_container_width=True, key="pg_wind3d")
