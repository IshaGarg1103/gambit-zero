# April 2026 — fresh Hugging Face row audit

**Result: PASS for every audited April game and row.**

Pinned revision: `8e8fab23e7b6711d787d6fb64b59aba949444241`.

Games: 5,000. Position rows: 380,334. Chess states including final boards: 385,334. Mismatches: 0.

## Checks

- Full downloaded payload SHA256/size
- Exact Parquet schemas and metadata
- All compact game metadata and actions
- Every materialized field and feature byte against fresh PGN replay
- Full Pgx/python-chess per-ply legality, state/history/status/reward parity including final boards
- Filters, side-to-move WDL, timeout weights, horizon, order, counts and daily/rating coverage
- Unique game identities across the audited months
- Original logical/feature/selected-ID/shard digests
- Every day quota and the strong-rating count match the approved April manifest.

Both players ≥2000: 3,020/5,000 (60.40%). Calendar days represented: 30. Time-forfeit games: 819; their 75,112 rows retain zero value weight.

## Measured resources

Elapsed: 7.64 minutes. Downloaded: 213,968,941 bytes across 83 objects. Peak temporary disk: 5.38 MiB. Peak process RSS: 783.30 MiB. All temporary downloads removed.

## Reproduce

```bash
.venv/bin/python scripts/audit_hf_rows.py --revision 8e8fab23e7b6711d787d6fb64b59aba949444241 --months 2026-04 --output reports/m1_hf_rows_audit_april.json
```

Uses the existing frozen run configuration and cached HF authentication. No uploads or archive extraction.

## Limitations

- Full Lichess archive not downloaded again; this is a fresh audit of uploaded game inputs and every output row.
- Uses the accepted canonical adapter/encoder and python-chess differential oracle, not an independent encoder implementation.
- Dataset D overlap, remaining validation/test splits and final 50K acceptance are still pending. Human demonstrations are not guaranteed optimal moves.
