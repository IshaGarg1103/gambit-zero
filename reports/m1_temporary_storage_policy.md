# Temporary-output cap waiver

Prepared `enforce_temporary_storage_caps: false` for January. The worker sets
`TINYCHESS_ENFORCE_TEMP_STORAGE_CAPS=0`, disabling managed-output disk-size caps
for candidates, upload spool and audit metadata. The 2.5 GiB figure remains an
estimate, not an enforced total. Actual use may exceed it.

The 20 GiB free-disk reserve, memory/frame/shard bounds, source checksum checks,
transfer allowance and 8-hour runtime limit remain in effect. Verified shard
copies continue to be removed incrementally; no full archive is saved.

Tests: 35 passed (storage policy, charger policy, archive scan).

The user explicitly approved restarting January. Restart completed at 2026-10-25T17:13:43.072406+00:00.
Live status confirms charger enforcement and temporary-output caps are disabled.
