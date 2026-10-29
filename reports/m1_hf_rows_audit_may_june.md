# May–June 2026: fresh Hugging Face row audit

**Result: PASS for every uploaded game and row in the audited months.**

Pinned revision: `a461d1be5e8afe599ce68fa9a7c9e315df0bd598`.
Games: 10,000; training rows: 752,271; chess states including final boards: 762,271; mismatches: 0.

| Month | Games | Rows | Both players ≥2000 | Days | Result |
|---|---:|---:|---:|---:|---|
| 2026-05 | 5000 | 375511 | 3016 | 31 | PASS |
| 2026-06 | 5000 | 376760 | 3020 | 30 | PASS |

## Validation

- Full downloaded payload SHA256/size
- Exact Parquet schemas and metadata
- All compact game metadata and actions
- Every materialized field and feature byte against fresh PGN replay
- Full Pgx/python-chess per-ply legality, state/history/status/reward parity including final boards
- Filters, side-to-move WDL, timeout weights, horizon, order, counts and daily/rating coverage
- Unique game identities across the audited months
- Original logical/feature/selected-ID/shard digests
- Daily accepted counts and strong-rating counts match each approved monthly manifest.
- The previously interrupted uploads left no missing, extra or changed rows in these audited artifacts.

Every metadata field, action/UCI, FEN/history, position hash, WDL label, value/policy weight, plane/context byte and legal-mask byte matched fresh full-game replay.

## Resources

Elapsed: 17.69 minutes. Downloaded 424,672,605 bytes. Peak temporary audit disk: 5.34 MiB. Peak RSS: 726.58 MiB. All temporary audit downloads removed.

## Reproduce

```bash
.venv/bin/python scripts/audit_hf_rows.py --revision a461d1be5e8afe599ce68fa9a7c9e315df0bd598 --months 2026-05 2026-06 --output reports/m1_hf_rows_audit_may_june.json
```

Requires the retained frozen run configuration and cached HF authentication. No uploads or source extraction are performed.

## Limitations

- Only May and June uploaded objects at the pinned revision were freshly downloaded; full Lichess archives were not rescanned.
- The audit uses the accepted canonical adapter/encoder with a python-chess differential oracle, not an independently implemented encoder.
- Dataset D overlap and final validation/test/50K acceptance remain pending. Human moves are not guaranteed optimal.
