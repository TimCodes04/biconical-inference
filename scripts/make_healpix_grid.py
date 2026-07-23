"""Generate the static HealPix Nside=4 geometry the Sky-survey tab renders.  [AI-Claude]

Dev-time ONLY (healpy is not a project dependency — the app reads the committed JSON):

    uv run --with healpy python scripts/make_healpix_grid.py

Writes app/static/healpix_nside4.json with, per pixel in RING order:
  corners : (4, 3) unit vectors of the pixel boundary (step=1)
  center  : (3,) unit vector
  lonlat  : (lon_deg, lat_deg)
plus nest2ring (192,) — nest2ring[i_nest] = i_ring, for reordering NESTED uploads.
"""

import json
import os

import healpy as hp
import numpy as np

NSIDE = 4
NPIX = hp.nside2npix(NSIDE)          # 192

out = {
    "nside": NSIDE,
    "npix": int(NPIX),
    "order": "ring",
    "corners": [],
    "centers": [],
    "lonlat": [],
    "nest2ring": [int(hp.nest2ring(NSIDE, i)) for i in range(NPIX)],
}
for i in range(NPIX):
    b = hp.boundaries(NSIDE, i, step=1)              # (3, 4) xyz corner columns
    out["corners"].append(np.round(b.T, 6).tolist())
    vec = hp.pix2vec(NSIDE, i)
    out["centers"].append([round(float(v), 6) for v in vec])
    lon, lat = hp.pix2ang(NSIDE, i, lonlat=True)
    out["lonlat"].append([round(float(lon), 3), round(float(lat), 3)])

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "app", "static", "healpix_nside4.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(out, open(path, "w"))
print(f"[healpix] {NPIX} pixels -> {path} ({os.path.getsize(path) / 1e3:.0f} KB)")
