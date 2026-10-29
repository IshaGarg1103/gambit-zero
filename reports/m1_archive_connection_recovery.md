# Archive connection recovery

The September process stopped after a premature compressed-body EOF (about
70 MB), despite external power remaining connected. No archive checksum or
September dataset completion was claimed. The prior 42,500 games are unchanged.

## Implemented recovery

Within one live process, read timeouts, connection resets and premature EOF can
reconnect up to eight times, using bounded exponential backoff and the existing
runtime deadline. The next request starts at the last byte actually delivered
to the decoder. HTTP 206, exact contiguous Content-Range, frozen remaining
Content-Length, identity encoding and the unchanged strong ETag are mandatory.
A server returning HTTP 200 for a suffix or changing the snapshot stops the run.

The same SHA256 object, frame decoder, PGN parser and sampler stay in memory.
A suffix is never independently decompressed; this is not random-access PGN
sampling. Original compressed bytes must still reach exact EOF and match the
official full-file SHA256. No complete compressed/decompressed archive is saved.

Closing the process, charger-triggered termination, or exhausting retries still
requires a fresh month scan. A partial status is never an accepted checkpoint.
Transport/TLS overhead and bytes received internally before a failed read are
not metered; logical delivered source bytes remain contiguous and unrepeated.
The source allowance is reserved once per process attempt. Hub upload timeout
behavior is unchanged and retains its safe local recovery files.

## Verification

- Focused HTTP recovery/archive tests: **43 passed in 3.33s**.
- Full regression suite: **204 passed in 48.81s**.
- A live September suffix-range probe read only **64 KiB**, with HTTP 206,
  unchanged ETag, correct size and exact range. This is not full-archive
  verification; see `m1_archive_recovery_live_probe.json`.
- Fixtures interrupt inside frame headers and compressed bodies, preserve partial
  reads, exercise timeouts and bounded retries, and reject wrong ranges, changed
  ETags, content transforms and bad final SHA256. Resumed sampling equals an
  uninterrupted scan, including frame indices.

Frozen stopped runtime updated with checksummed operational amendments. No new
scan was started and no transfer allowance was increased by this fix. Another
whole September attempt needs 29,224,520,887 bytes of additional allowance.
No M2, training, Git mutation or changes to chess/schema/encoder policy.

```bash
.venv/bin/python -m pytest tests/test_archive_http_recovery.py tests/test_archive_scan.py -q
.venv/bin/python -m pytest -q
```
