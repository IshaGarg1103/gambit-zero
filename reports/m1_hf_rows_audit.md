# Fresh Hugging Face row audit — January–March 2026

**Result: PASS for all audited uploaded rows.** This is not final M1/50K acceptance.

Pinned repository revision: `74cd47d07e12ee4562515554456bc5b6433e8410`.
Freshly replayed games: 15,000; materialized rows compared: 1,141,765; Pgx/python-chess states checked: 1,156,765; mismatches: 0.

| Month | Games | Position rows | Strong games (both ≥2000) | Calendar days | Result |
|---|---:|---:|---:|---:|---|
| 2026-01 | 5000 | 380274 | 3016 | 31 | PASS |
| 2026-02 | 5000 | 381007 | 3012 | 28 | PASS |
| 2026-03 | 5000 | 380484 | 3016 | 31 | PASS |

## Checks performed

- Full downloaded payload SHA256/size
- Exact Parquet schemas and metadata
- All compact game metadata and actions
- Every materialized field and feature byte against fresh PGN replay
- Full Pgx/python-chess per-ply legality, state/history/status/reward parity including final boards
- Filters, side-to-move WDL, timeout weights, horizon, order, counts and daily/rating coverage
- Unique game identities across the audited months
- Original logical/feature/selected-ID/shard digests
- Actual day counts and strong-rating counts also match the approved monthly manifests.

Every uploaded field was compared with a fresh complete-game replay, including exact planes/context bytes, packed legal masks, actions/UCI, FEN/history, position hashes, WDL perspective, timeout masks and metadata. Original game/feature/order/shard digests matched. No whole-game duplicates were found among the 15,000 audited games.

## Measured resources

Elapsed: 29.26 minutes. Downloaded 642,830,786 bytes across 249 objects, one shard pair at a time. Peak temporary audit disk: 5.35 MiB. Peak process RSS: 599.58 MiB. Temporary downloads removed: yes.

Five audit regression tests passed, including changed action/label/feature/mask detection and Parquet list-child name normalization. The normalization was verified to preserve exact field types and version metadata; no data correction was needed.

## Limitations

- Audits the pinned January–March uploaded artifacts; April is outside this audit.
- Full Lichess archives were not downloaded again. Original whole-archive SHA256 verification is separate evidence.
- Fresh replay uses the accepted canonical Pgx adapter/encoder and python-chess differential oracle, not an independently reimplemented encoder.
- Dataset D overlap remains unchecked by explicit user deferral; validation/test months and final 50K acceptance remain pending.
- Human moves and source PGN outcomes are demonstrations and noisy labels, not engine-certified strategic targets.

## Reproduce

```bash
.venv/bin/python -m pytest tests/test_hf_row_audit.py -q
.venv/bin/python scripts/audit_hf_rows.py --revision 74cd47d07e12ee4562515554456bc5b6433e8410 --output reports/m1_hf_rows_audit.json
```

The command uses the existing frozen run configuration/source manifest and cached Hugging Face authentication. It does not upload, change the dataset, or start archive extraction.
