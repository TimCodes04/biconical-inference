"""Edge-calibration audit of the spaxel6m NPE on reserved held-out rows. [AI-Claude]

Question: near prior bounds, are the tight/railed posteriors users see calibrated
(honest physics) or a flow artifact (overconfident/biased)?

Protocol: stratified sample of reserved rows (per-param bottom-10% / top-10% of the
prior range + an all-interior control pool), one fit per row, per-fit per-param
coverage / SBC-rank / width / rail-mass stats, aggregated per param x stratum.
"""
import os, sys, json, time
os.chdir('/Users/jarvis/Documents/biconical-inference')
sys.path.insert(0, 'app'); sys.path.insert(0, 'src')
import numpy as np, torch, core
import h5py, yaml
from biconical_inference import splits as _splits

SCRATCH = 'validation/spaxel6m'
CFG = 'configs/spaxel6m.yaml'
OUT = os.path.join(SCRATCH, 'edge_calibration.json')
RAW = os.path.join(SCRATCH, 'edge_calibration_perfit.json')

rng = np.random.default_rng(42)

_cfg, prior, _em, post, _dev, _co, _n = core.load_models(CFG)
names = list(prior.names)
lo = np.asarray(prior.lo, float); hi = np.asarray(prior.hi, float)
rng_j = hi - lo
print('params:', names, flush=True)
print('lo:', lo, 'hi:', hi, flush=True)

cfg = yaml.safe_load(open(CFG))
f = h5py.File('library/library_spaxel.h5', 'r')
z = f['params_z'][:]
mask = _splits.test_mask(z, run_id=f['run_id'][:], aperture_kpc=f['aperture_kpc'][:],
                         path=cfg.get('splits', _splits.DEFAULT_PATH))
rows = np.nonzero(mask)[0]
phys = prior.from_z(z)                 # (N, 6) physical, full library
u = (phys - lo) / rng_j                # normalized truth
print(f'reserved rows: {len(rows)}', flush=True)

# ---- stratified selection (on reserved rows only) ----
u_res = u[rows]
sel = {}   # row_index -> set of (param, stratum) memberships
def add(idxs, tag):
    for i in idxs:
        sel.setdefault(int(i), set()).add(tag)

N_EDGE, N_INT = 40, 120
for j, nm in enumerate(names):
    cand_lo = rows[u_res[:, j] < 0.10]
    cand_hi = rows[u_res[:, j] > 0.90]
    pick_lo = rng.choice(cand_lo, size=min(N_EDGE, len(cand_lo)), replace=False)
    pick_hi = rng.choice(cand_hi, size=min(N_EDGE, len(cand_hi)), replace=False)
    add(pick_lo, (nm, 'edge_lo')); add(pick_hi, (nm, 'edge_hi'))
    print(f'{nm}: edge_lo cand={len(cand_lo)} picked={len(pick_lo)}; '
          f'edge_hi cand={len(cand_hi)} picked={len(pick_hi)}', flush=True)

interior_mask = np.all((u_res > 0.25) & (u_res < 0.75), axis=1)
cand_int = rows[interior_mask]
pick_int = rng.choice(cand_int, size=min(N_INT, len(cand_int)), replace=False)
add(pick_int, ('*', 'interior'))
print(f'interior cand={len(cand_int)} picked={len(pick_int)}', flush=True)

all_rows = sorted(sel.keys())
print(f'total unique rows to fit: {len(all_rows)}', flush=True)

# ---- fit each row once ----
perfit = []
t0 = time.time()
for k, i in enumerate(all_rows):
    cube = np.asarray(f['cubes'][i], np.float32)
    samp, _ = core.cached_infer(cube, 30.0, 0.0, CFG)
    samp = np.asarray(samp)            # (5000, 6) physical
    truth = phys[i]
    rec = {'row': int(i), 'truth': truth.tolist(),
           'u': u[i].tolist(),
           'strata': sorted([f'{p}|{s}' for (p, s) in sel[i]])}
    q = np.percentile(samp, [5, 16, 50, 84, 95], axis=0)
    rec['q'] = q.tolist()
    rec['cov68'] = ((q[1] <= truth) & (truth <= q[3])).tolist()
    rec['cov90'] = ((q[0] <= truth) & (truth <= q[4])).tolist()
    rec['rank'] = np.mean(samp < truth[None, :], axis=0).tolist()
    rec['w68'] = ((q[3] - q[1]) / rng_j).tolist()
    rec['rail_hi'] = np.mean(samp > (hi - 0.01 * rng_j)[None, :], axis=0).tolist()
    rec['rail_lo'] = np.mean(samp < (lo + 0.01 * rng_j)[None, :], axis=0).tolist()
    perfit.append(rec)
    if (k + 1) % 25 == 0 or k == len(all_rows) - 1:
        el = time.time() - t0
        print(f'  fit {k+1}/{len(all_rows)}  ({el:.0f}s, {el/(k+1):.2f}s/fit)', flush=True)
        with open(RAW, 'w') as fh:
            json.dump(perfit, fh)

# ---- aggregate ----
def agg(recs, j):
    if not recs:
        return {'n': 0}
    c68 = np.mean([r['cov68'][j] for r in recs])
    c90 = np.mean([r['cov90'][j] for r in recs])
    w68 = np.median([r['w68'][j] for r in recs])
    ranks = np.array([r['rank'][j] for r in recs])
    rail = np.mean([(r['rail_lo'][j] > 0.5 or r['rail_hi'][j] > 0.5) for r in recs])
    return {'n': len(recs), 'cov68': round(float(c68), 3), 'cov90': round(float(c90), 3),
            'median_w68': round(float(w68), 4), 'rail_rate': round(float(rail), 3),
            'rank_mean': round(float(ranks.mean()), 3),
            'rank_frac_lt05': round(float(np.mean(ranks < 0.05)), 3),
            'rank_frac_gt95': round(float(np.mean(ranks > 0.95)), 3)}

results = {'config': CFG, 'n_fits': len(perfit), 'params': names,
           'lo': lo.tolist(), 'hi': hi.tolist(), 'per_param': {}}
for j, nm in enumerate(names):
    interior = [r for r in perfit if f'*|interior' in r['strata']]
    elo = [r for r in perfit if f'{nm}|edge_lo' in r['strata']]
    ehi = [r for r in perfit if f'{nm}|edge_hi' in r['strata']]
    results['per_param'][nm] = {'interior': agg(interior, j),
                                'edge_lo': agg(elo, j), 'edge_hi': agg(ehi, j)}
    # rail-CONSISTENCY at edges: when truth is at the bound, does the posterior rail
    # toward the correct side (good) and how often toward the WRONG side (bad)?
    if elo:
        results['per_param'][nm]['edge_lo']['correct_rail_rate'] = round(
            float(np.mean([r['rail_lo'][j] > 0.5 for r in elo])), 3)
        results['per_param'][nm]['edge_lo']['wrong_side_rail_rate'] = round(
            float(np.mean([r['rail_hi'][j] > 0.5 for r in elo])), 3)
    if ehi:
        results['per_param'][nm]['edge_hi']['correct_rail_rate'] = round(
            float(np.mean([r['rail_hi'][j] > 0.5 for r in ehi])), 3)
        results['per_param'][nm]['edge_hi']['wrong_side_rail_rate'] = round(
            float(np.mean([r['rail_lo'][j] > 0.5 for r in ehi])), 3)

# false-rail rate: interior-control rows (ALL params in middle 50%) — any rail is false
interior = [r for r in perfit if '*|interior' in r['strata']]
any_rail = [any((r['rail_lo'][j] > 0.5 or r['rail_hi'][j] > 0.5) for j in range(len(names)))
            for r in interior]
per_param_false = {nm: round(float(np.mean([(r['rail_lo'][j] > 0.5 or r['rail_hi'][j] > 0.5)
                                            for r in interior])), 4)
                   for j, nm in enumerate(names)}
results['false_rail'] = {'n_interior': len(interior),
                         'any_param_rate': round(float(np.mean(any_rail)), 4),
                         'per_param': per_param_false}

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print(json.dumps(results, indent=2), flush=True)
print('DONE ->', OUT, flush=True)
