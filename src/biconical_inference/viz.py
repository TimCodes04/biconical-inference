"""Interactive 3D rendering of the biconical MgII wind + static disk.  [AI-Claude]

Geometry/physics faithful to THOR's `biconical_shellmodel`
(branch biconical_model_w/disk; src/datasets/BiconicalShellDataset.cpp,
src/helpers_geometry.h):

  - Double cone (both ±z nappes) of half-opening angle θ measured FROM the +z axis;
    the gas fills the polar wedge within θ of the axis, over a spherical shell
    r_inner ≤ r ≤ r_outer (box center at origin here).
  - Radial outflow  v(r) = v_max · (r / r_outer)^a_v   (a_v = powerlaw_index_velocity),
    purely radial, anchored to v_max at r_outer.
  - Mass-conserving density  ρ(r) ∝ r^−(2 + a_v), normalized to the wind column logN.
  - Static toroidal disk ⟂ axis: outer radius R_disk, full thickness h, central hole
    R_hole = (h/2)·tan θ (cone and disk tangent).
  - Line of sight  n̂ = [sin i, 0, cos i]  at inclination i from +z.

Drawn in kpc, in the PRODUCTION geometry (sherlock.yaml: disk-ON, outer radius 100 kpc).
The figure is a plotly Figure (drag to rotate / zoom), so it drops straight into
Streamlit via st.plotly_chart. The wind is shown as a translucent, lit cone surface
plus radial streamlines whose colour is a smooth velocity gradient (slow near the
launch radius → fast at the cone edge for a_v > 0).
"""

from __future__ import annotations

import numpy as np

from .thor_sim import config as _cfg
from .thor_sim.constants import BOXSIZE_KPC, INNER_RADIUS_BOX

# Fixed geometry in kpc, matching the PRODUCTION forward model (configs/sherlock.yaml,
# the config that generated library.h5 the model trained on) — NOT configs/default.yaml,
# which is a wind-only/125-kpc reference TEMPLATE. Production overrode THOR's default
# outer_radius (0.5 box = 125 kpc) to 0.4 box = 100 kpc, and turned the disk ON.
R_INNER_KPC = INNER_RADIUS_BOX * BOXSIZE_KPC               # 2 kpc  (0.008 box)
PROD_OUTER_RADIUS_KPC = 0.4 * BOXSIZE_KPC                  # 100 kpc (sherlock.yaml outer_radius_box)
DISK_RADIUS_KPC = _cfg.DISK_RADIUS_BOX * BOXSIZE_KPC       # 10 kpc
DISK_HALF_H_KPC = 0.5 * _cfg.DISK_HEIGHT_BOX * BOXSIZE_KPC  # 1 kpc (full thickness 2 kpc)

# Bright speed scale (blue → teal → gold → orange) — every value is visible on the dark
# background, unlike Plasma/Viridis whose low end is near-black (slow wind would vanish).
_COLORSCALE = [[0.0, "#4895ef"], [0.35, "#43c59e"], [0.70, "#f9c74f"], [1.0, "#f3722c"]]
# Slightly glossier than before (specular/fresnel up) for a "lit gas" sheen — cosmetic.
_LIGHT = dict(ambient=0.55, diffuse=0.85, specular=0.22, roughness=0.5, fresnel=0.18)


def _cone_surface(theta_deg, r_inner, r_outer, sign, a_v, vmax, n_s=32, n_phi=72):
    """Lateral mantle of one cone nappe (sign=+1 for +z, −1 for −z), as a smooth
    plotly Surface whose colour is the local outflow speed v(r)."""
    th = np.radians(theta_deg)
    s = np.linspace(r_inner, r_outer, n_s)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    S, P = np.meshgrid(s, phi, indexing="ij")
    x = S * np.sin(th) * np.cos(P)
    y = S * np.sin(th) * np.sin(P)
    z = sign * S * np.cos(th)
    speed = vmax * (S / r_outer) ** a_v
    return x, y, z, speed


def _wind_directions(theta_deg, n_lines):
    """Evenly spread radial unit directions inside the bicone (both nappes), via a
    golden-angle / equal-area spiral so streamlines fan out without clumping."""
    th = np.radians(theta_deg)
    ga = np.pi * (3.0 - np.sqrt(5.0))
    half = max(1, n_lines // 2)
    dirs = []
    for sign in (1.0, -1.0):
        for i in range(half):
            cz = 1.0 - (i + 0.5) / half * (1.0 - np.cos(th))     # equal-area in polar cap
            sz = np.sqrt(max(0.0, 1.0 - cz * cz))
            az = i * ga
            dirs.append(np.array([sz * np.cos(az), sz * np.sin(az), sign * cz]))
    return dirs


def _disk_traces(go, theta_deg, r_disk, half_h, color="#8893a3", opacity=0.5):
    """Annular-cylinder disk (top, bottom, outer rim, inner conical-tangent hole)."""
    th = np.radians(theta_deg)
    r_hole = min(half_h * np.tan(th), 0.98 * r_disk)
    phi = np.linspace(0, 2 * np.pi, 80)
    cs = [[0, color], [1, color]]
    traces = []
    for zf in (half_h, -half_h):                      # top + bottom annular faces
        rr = np.linspace(r_hole, r_disk, 12)
        R, P = np.meshgrid(rr, phi, indexing="ij")
        traces.append(go.Surface(x=R * np.cos(P), y=R * np.sin(P), z=np.full_like(R, zf),
                                 showscale=False, opacity=opacity, colorscale=cs,
                                 surfacecolor=np.zeros_like(R), lighting=_LIGHT,
                                 hoverinfo="skip", name="disk"))
    zz = np.linspace(-half_h, half_h, 6)
    for rad in (r_disk, r_hole):                      # outer rim + inner hole wall
        Z, P = np.meshgrid(zz, phi, indexing="ij")
        traces.append(go.Surface(x=rad * np.cos(P), y=rad * np.sin(P), z=Z,
                                 showscale=False, opacity=opacity, colorscale=cs,
                                 surfacecolor=np.zeros_like(Z), lighting=_LIGHT,
                                 hoverinfo="skip", name="disk"))
    return traces, r_hole


def _starfield_trace(go, n, radius, seed=7):
    """A faint, deterministic star shell for observatory atmosphere (cosmetic only).
    Points sit just inside the axis range so they read as a backdrop without
    occluding the wind. Returns a single Scatter3d trace."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = radius * rng.uniform(0.82, 0.99, n)
    s = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    x, y, z = r * s * np.cos(phi), r * s * np.sin(phi), r * u
    return go.Scatter3d(x=x, y=y, z=z, mode="markers",
                        marker=dict(size=rng.uniform(0.6, 2.0, n),
                                    color="rgba(220,228,240,0.5)"),
                        hoverinfo="skip", showlegend=False, name="stars")


def biconical_figure(theta_deg, incl_deg, av, vexp_kms, logN=None, sigmaran_kms=None,
                     r_inner_kpc=R_INNER_KPC, r_outer_kpc=PROD_OUTER_RADIUS_KPC,
                     disk_on=True, disk_radius_kpc=DISK_RADIUS_KPC,
                     disk_half_height_kpc=DISK_HALF_H_KPC, show_los=True, n_streamlines=34,
                     colorscale=None, lighting=None, starfield=False, transparent=False,
                     show_colorbar=True, rings=True, n_s=32, n_phi=72,
                     camera_eye=(1.5, 1.5, 0.85), uirevision=None):
    """Build an interactive plotly Figure of the biconical wind for given parameters.

    Required physical params: theta_deg (cone half-angle), incl_deg (LOS inclination),
    av (velocity power-law index), vexp_kms (v_max). logN / sigmaran_kms are shown in
    the title if provided. Returns a plotly.graph_objects.Figure.

    All keyword-only options below default to the original behaviour (so existing
    callers are unaffected):
      colorscale/lighting — override the wind surface look (default the module wind
        speed ramp / lit material).
      starfield — True (~420 pts) / int N draws a faint star backdrop for atmosphere.
      transparent — transparent paper + axis backgrounds so the scene floats on a
        dark page (else the opaque rgb(8,10,16) scene).
      show_colorbar — show the wind-speed colorbar (off for compact previews).
      rings — draw the faint iso-velocity rings.
      n_s/n_phi — cone mesh resolution (lower = cheaper preview).
      camera_eye — default camera position.
      uirevision — when set, persists the user's camera/zoom across Streamlit reruns.
    """
    import plotly.graph_objects as go

    cs = colorscale if colorscale is not None else _COLORSCALE
    light = lighting if lighting is not None else _LIGHT
    vmax = float(vexp_kms)
    av = float(av)
    speed_max = max(vmax, 1.0)
    fig = go.Figure()

    if starfield:
        n_stars = 420 if starfield is True else int(starfield)
        fig.add_trace(_starfield_trace(go, n_stars, 1.3 * r_outer_kpc))

    # --- wind cones: solid, lit, coloured by the outflow speed v(r). The gradient runs
    #     from slow near the launch radius up to v_max at the cone edge — a clean glowing
    #     bicone with no spiky glyphs at any opening angle. ---
    for sign in (+1, -1):
        x, y, z, sp = _cone_surface(theta_deg, r_inner_kpc, r_outer_kpc, sign, av, vmax,
                                    n_s=n_s, n_phi=n_phi)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=sp, colorscale=cs, cmin=0.0, cmax=speed_max,
            opacity=0.985, showscale=(sign == +1 and show_colorbar), lighting=light,
            colorbar=dict(title="wind speed<br>[km/s]", len=0.55, x=0.0, thickness=14, outlinewidth=0),
            hovertemplate="v = %{surfacecolor:.0f} km/s<extra></extra>", name="wind cone"))
        # inner spherical cap (closes the cone visually at the launch radius)
        th = np.radians(theta_deg)
        pol = np.linspace(0, th, 10)
        az = np.linspace(0, 2 * np.pi, 48)
        PL, AZ = np.meshgrid(pol, az, indexing="ij")
        cx = r_inner_kpc * np.sin(PL) * np.cos(AZ)
        cy = r_inner_kpc * np.sin(PL) * np.sin(AZ)
        cz = sign * r_inner_kpc * np.cos(PL)
        fig.add_trace(go.Surface(
            x=cx, y=cy, z=cz, surfacecolor=np.zeros_like(cx), colorscale=cs,
            cmin=0.0, cmax=speed_max, opacity=0.985, showscale=False, lighting=light,
            hoverinfo="skip", name="wind base"))

    # --- faint iso-velocity rings on the cone (constant radius = constant speed) so the
    #     gradient reads as discrete contours, not just a smear ---
    th = np.radians(theta_deg)
    ring_phi = np.linspace(0, 2 * np.pi, 64)
    if rings:
        for frac in (0.4, 0.65, 0.9):
            rs = frac * r_outer_kpc
            for sign in (+1, -1):
                fig.add_trace(go.Scatter3d(
                    x=rs * np.sin(th) * np.cos(ring_phi), y=rs * np.sin(th) * np.sin(ring_phi),
                    z=sign * rs * np.cos(th) * np.ones_like(ring_phi), mode="lines",
                    line=dict(color="rgba(255,255,255,0.30)", width=1.5),
                    showlegend=False, hoverinfo="skip"))

    # --- bipolar outflow direction (two clean axis-aligned arrows, not a spiky fan) ---
    for sign in (+1, -1):
        fig.add_trace(go.Cone(
            x=[0], y=[0], z=[sign * 1.12 * r_outer_kpc], u=[0], v=[0], w=[float(sign)],
            colorscale=[[0, "#f3722c"], [1, "#f3722c"]], showscale=False,
            sizemode="absolute", sizeref=0.09 * r_outer_kpc, anchor="tail", opacity=0.85,
            hoverinfo="skip", name="outflow"))

    # --- static disk ---
    if disk_on:
        disk_traces, _ = _disk_traces(go, theta_deg, disk_radius_kpc, disk_half_height_kpc)
        for t in disk_traces:
            fig.add_trace(t)

    # --- central continuum source + cone axis ---
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=5, color="gold", symbol="diamond", line=dict(color="orange", width=1)),
        name="continuum source", hoverinfo="name"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-r_outer_kpc, r_outer_kpc], mode="lines",
        line=dict(color="rgba(140,140,150,0.4)", width=2, dash="dot"),
        name="wind axis", hoverinfo="name"))

    # --- line of sight (observer direction) ---
    if show_los:
        i = np.radians(incl_deg)
        los = np.array([np.sin(i), 0.0, np.cos(i)])
        L = 1.25 * r_outer_kpc
        fig.add_trace(go.Scatter3d(x=[0, L * los[0]], y=[0, L * los[1]], z=[0, L * los[2]],
            mode="lines", line=dict(color="#22d3ee", width=6),
            name=f"line of sight (i = {incl_deg:.0f}°)"))
        fig.add_trace(go.Cone(x=[L * los[0]], y=[L * los[1]], z=[L * los[2]],
            u=[los[0]], v=[los[1]], w=[los[2]], colorscale=[[0, "#22d3ee"], [1, "#22d3ee"]],
            showscale=False, sizemode="absolute", sizeref=0.10 * r_outer_kpc, anchor="tip",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[L * los[0]], y=[L * los[1]], z=[L * los[2]], mode="text",
            text=["observer"], textposition="top center",
            textfont=dict(color="#22d3ee", size=13), hoverinfo="skip", showlegend=False))

    title = f"θ = {theta_deg:.0f}°,  i = {incl_deg:.0f}°,  a_v = {av:.2f},  v_max = {vmax:.0f} km/s"
    if logN is not None:
        title += f",  logN = {logN:.2f}"
    if sigmaran_kms is not None:
        title += f",  σ = {sigmaran_kms:.0f} km/s"
    rng = 1.3 * r_outer_kpc
    bg = "rgb(8,10,16)"
    axis_bg = "rgba(0,0,0,0)" if transparent else bg
    paper_bg = "rgba(0,0,0,0)" if transparent else bg
    ex, ey, ez = camera_eye
    scene = dict(
        xaxis=dict(title="x [kpc]", range=[-rng, rng], backgroundcolor=axis_bg,
                   gridcolor="rgba(120,120,140,0.18)", zerolinecolor="rgba(120,120,140,0.3)"),
        yaxis=dict(title="y [kpc]", range=[-rng, rng], backgroundcolor=axis_bg,
                   gridcolor="rgba(120,120,140,0.18)", zerolinecolor="rgba(120,120,140,0.3)"),
        zaxis=dict(title="z [kpc] (wind axis)", range=[-rng, rng], backgroundcolor=axis_bg,
                   gridcolor="rgba(120,120,140,0.18)", zerolinecolor="rgba(120,120,140,0.3)"),
        aspectmode="cube", camera=dict(eye=dict(x=ex, y=ey, z=ez)),
    )
    if uirevision is not None:
        scene["uirevision"] = uirevision
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color="#d8dee9")),
        scene=scene, paper_bgcolor=paper_bg, font=dict(color="#d8dee9"),
        margin=dict(l=0, r=0, t=34, b=0), showlegend=True,
        legend=dict(x=0.0, y=1.0, font=dict(size=10), bgcolor="rgba(8,9,15,0.45)"))
    return fig
