# Reserved test split — DO NOT TRAIN ON THESE ROWS

`reserved_test.json` defines the **held-out test set**: the rows of `library.h5`
reserved exclusively for **evaluation**. The emulator and NPE must never train on
them.

- **Definition:** `seed=0`, `test_frac=0.1` permutation prefix (identical to the
  original emulator training split), so it matches the model under comparison.
- **Keyed to the library:** `library_hash` (sha1 of `params_z`) + `n_rows` pin the
  exact library; if the library is re-aggregated, regenerate this file and the
  training/validation guards will catch the mismatch.
- **Enforced:** `emulator.data.make_datasets` asserts its computed test split equals
  this file (refuses to train otherwise); `scripts/validate_holdout.py` and the
  retrain/validation scripts condition on exactly this set.

Regenerate with:

```bash
uv run python -m biconical_inference.splits --config configs/default.yaml
```

The split is loaded via `biconical_inference.splits.{load, test_mask, compute_test_idx}`.
