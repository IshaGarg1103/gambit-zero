# M1 evidence archive

Detailed journals and historical PR snapshots are archived outside the source tree.
The published Dataset A `main` revision remains `b7ddb96d8ce8cfe55298e01a7d928b0f763e1c4c`.

- [Download the immutable evidence bundle](https://huggingface.co/datasets/ishagarg1103/gambit-zero/resolve/5edc8bec33e5a8b398f5a4af5b501e846998a93e/evidence/pr-2/pr-2-cb5ba8bf1759-original.zip)
- Evidence branch: `evidence/pr-2-cb5ba8bf1759`
- Immutable evidence revision: `5edc8bec33e5a8b398f5a4af5b501e846998a93e`
- Bundle size: **631,472 bytes**
- Bundle SHA256: `dc6c38ef0208dde0ccdfb2b8f13c2afaa1a966043e7041859bfb5c77817b034f`
- Original PR source: `cb5ba8bf1759a0b6eef2db0bdd82363d2ceedee7`
- [Per-file byte counts and SHA256 index](m1_evidence_archive.json)

The bundle preserves all 199 original changed files, including retained source.
It is an evidence/source snapshot, not a PGN or Parquet training-data archive.
Download verifies size and SHA256; entry reads verify their own SHA256. Files
are cached under ignored `data/processed/m1_evidence/`; nothing is extracted
back into tracked `reports/`. Current local runtime results take precedence.

```bash
.venv/bin/python scripts/fetch_m1_evidence.py --verify
.venv/bin/python scripts/fetch_m1_evidence.py --verify --offline
.venv/bin/python scripts/fetch_m1_evidence.py --path reports/m1_dataset_d_exclusions.json --output data/processed/m1_evidence/restored/exclusions.json
```

The first command can fetch the small public archive. Subsequent report generation
and archived-input readers use that verified cache. Set `TINYCHESS_EVIDENCE_OFFLINE=1`
to forbid network fallback. A missing or corrupted archive fails explicitly.

## Files scheduled for removal from Git tracking

These exact original filenames are resolved by the evidence reader. Per-file
SHA256 hashes and byte sizes are in the linked index. The full classification
of all original PR files is in [pr2_file_audit.tsv](pr2_file_audit.tsv).
These are local deletions pending the user's staging, commit and push.

| Original path | Archive entry |
|---|---|
| <a id="docs-m1-initial-checklist-md"></a>`docs/m1_initial_checklist.md` | Same path inside the verified ZIP |
| <a id="reports-m0-large-batch-cpu-json"></a>`reports/m0_large_batch_cpu.json` | Same path inside the verified ZIP |
| <a id="reports-m0-large-batch-cpu-repeat-json"></a>`reports/m0_large_batch_cpu_repeat.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-coverage-json"></a>`reports/m1_50k_coverage.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-diagnostics-json"></a>`reports/m1_50k_diagnostics.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-launch-storage-json"></a>`reports/m1_50k_launch_storage.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-launch-tests-txt"></a>`reports/m1_50k_launch_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-preflight-json"></a>`reports/m1_50k_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-preflight-tests-txt"></a>`reports/m1_50k_preflight_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-preflight-unit-tests-txt"></a>`reports/m1_50k_preflight_unit_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-recovery-json"></a>`reports/m1_50k_recovery.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-rejections-json"></a>`reports/m1_50k_rejections.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-resource-gate-json"></a>`reports/m1_50k_resource_gate.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-run-preflight-json"></a>`reports/m1_50k_run_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-run-status-json"></a>`reports/m1_50k_run_status.json` | Same path inside the verified ZIP |
| <a id="reports-m1-50k-runtime-snapshot-json"></a>`reports/m1_50k_runtime_snapshot.json` | Same path inside the verified ZIP |
| <a id="reports-m1-acceptance-gap-review-md"></a>`reports/m1_acceptance_gap_review.md` | Same path inside the verified ZIP |
| <a id="reports-m1-april-preflight-json"></a>`reports/m1_april_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-archive-connection-recovery-json"></a>`reports/m1_archive_connection_recovery.json` | Same path inside the verified ZIP |
| <a id="reports-m1-archive-recovery-live-probe-json"></a>`reports/m1_archive_recovery_live_probe.json` | Same path inside the verified ZIP |
| <a id="reports-m1-archive-recovery-regression-txt"></a>`reports/m1_archive_recovery_regression.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-audit-run1-json"></a>`reports/m1_audit_run1.json` | Same path inside the verified ZIP |
| <a id="reports-m1-audit-run2-json"></a>`reports/m1_audit_run2.json` | Same path inside the verified ZIP |
| <a id="reports-m1-cancelled-transfer-accounting-json"></a>`reports/m1_cancelled_transfer_accounting.json` | Same path inside the verified ZIP |
| <a id="reports-m1-closeout-storage-json"></a>`reports/m1_closeout_storage.json` | Same path inside the verified ZIP |
| <a id="reports-m1-dataset-d-exclusions-json"></a>`reports/m1_dataset_d_exclusions.json` | Same path inside the verified ZIP |
| <a id="reports-m1-dataset-d-exclusions-attempt1-json"></a>`reports/m1_dataset_d_exclusions_attempt1.json` | Same path inside the verified ZIP |
| <a id="reports-m1-dataset-d-upload-diagnostic-json"></a>`reports/m1_dataset_d_upload_diagnostic.json` | Same path inside the verified ZIP |
| <a id="reports-m1-determinism-json"></a>`reports/m1_determinism.json` | Same path inside the verified ZIP |
| <a id="reports-m1-github-handoff-md"></a>`reports/m1_github_handoff.md` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-api-contract-json"></a>`reports/m1_hf_api_contract.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-dataset-card-md"></a>`reports/m1_hf_dataset_card.md` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-implementation-hashes-json"></a>`reports/m1_hf_implementation_hashes.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-live-smoke-json"></a>`reports/m1_hf_live_smoke.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-offline-smoke-json"></a>`reports/m1_hf_offline_smoke.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-regression-tests-txt"></a>`reports/m1_hf_regression_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-json"></a>`reports/m1_hf_rows_audit.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-md"></a>`reports/m1_hf_rows_audit.md` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-april-json"></a>`reports/m1_hf_rows_audit_april.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-april-md"></a>`reports/m1_hf_rows_audit_april.md` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-diagnostic-json"></a>`reports/m1_hf_rows_audit_diagnostic.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-may-june-json"></a>`reports/m1_hf_rows_audit_may_june.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-rows-audit-may-june-md"></a>`reports/m1_hf_rows_audit_may_june.md` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-storage-plan-json"></a>`reports/m1_hf_storage_plan.json` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-unit-tests-txt"></a>`reports/m1_hf_unit_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-hf-upload-md"></a>`reports/m1_hf_upload.md` | Same path inside the verified ZIP |
| <a id="reports-m1-implementation-hashes-json"></a>`reports/m1_implementation_hashes.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-ac-restart-json"></a>`reports/m1_january_ac_restart.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-ac-restart-preflight-json"></a>`reports/m1_january_ac_restart_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-battery-restart-json"></a>`reports/m1_january_battery_restart.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-battery-restart-preflight-json"></a>`reports/m1_january_battery_restart_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-before-battery-restart-json"></a>`reports/m1_january_before_battery_restart.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-before-cap-restart-json"></a>`reports/m1_january_before_cap_restart.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-cap-restart-json"></a>`reports/m1_january_cap_restart.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-finalization-fix-md"></a>`reports/m1_january_finalization_fix.md` | Same path inside the verified ZIP |
| <a id="reports-m1-january-fresh-20261026-preflight-json"></a>`reports/m1_january_fresh_20261026_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-prefinalization-network-ledger-json"></a>`reports/m1_january_prefinalization_network_ledger.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-restart-diagnosis-json"></a>`reports/m1_january_restart_diagnosis.json` | Same path inside the verified ZIP |
| <a id="reports-m1-january-session-preflight-json"></a>`reports/m1_january_session_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-july-august-preflight-json"></a>`reports/m1_july_august_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-local-cleanup-json"></a>`reports/m1_local_cleanup.json` | Same path inside the verified ZIP |
| <a id="reports-m1-manifest-run1-json"></a>`reports/m1_manifest_run1.json` | Same path inside the verified ZIP |
| <a id="reports-m1-manifest-run2-json"></a>`reports/m1_manifest_run2.json` | Same path inside the verified ZIP |
| <a id="reports-m1-may-june-preflight-json"></a>`reports/m1_may_june_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-measurements-run1-json"></a>`reports/m1_measurements_run1.json` | Same path inside the verified ZIP |
| <a id="reports-m1-measurements-run2-json"></a>`reports/m1_measurements_run2.json` | Same path inside the verified ZIP |
| <a id="reports-m1-one-month-tests-txt"></a>`reports/m1_one_month_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-power-policy-change-md"></a>`reports/m1_power_policy_change.md` | Same path inside the verified ZIP |
| <a id="reports-m1-power-policy-tests-txt"></a>`reports/m1_power_policy_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-pzstd-source-probe-json"></a>`reports/m1_pzstd_source_probe.json` | Same path inside the verified ZIP |
| <a id="reports-m1-sample-audit-json"></a>`reports/m1_sample_audit.json` | Same path inside the verified ZIP |
| <a id="reports-m1-sample-audit-run2-json"></a>`reports/m1_sample_audit_run2.json` | Same path inside the verified ZIP |
| <a id="reports-m1-september-first-attempt-stopped-json"></a>`reports/m1_september_first_attempt_stopped.json` | Same path inside the verified ZIP |
| <a id="reports-m1-september-network-fixed-retry-preflight-json"></a>`reports/m1_september_network_fixed_retry_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-september-preflight-json"></a>`reports/m1_september_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-september-retry-20261028-preflight-json"></a>`reports/m1_september_retry_20261028_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-september-retry-preflight-json"></a>`reports/m1_september_retry_preflight.json` | Same path inside the verified ZIP |
| <a id="reports-m1-september-stopped-before-20261028-retry-json"></a>`reports/m1_september_stopped_before_20261028_retry.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-01-json"></a>`reports/m1_session_2026-01.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-02-json"></a>`reports/m1_session_2026-02.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-03-json"></a>`reports/m1_session_2026-03.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-04-json"></a>`reports/m1_session_2026-04.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-05-json"></a>`reports/m1_session_2026-05.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-06-json"></a>`reports/m1_session_2026-06.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-07-json"></a>`reports/m1_session_2026-07.json` | Same path inside the verified ZIP |
| <a id="reports-m1-session-2026-08-json"></a>`reports/m1_session_2026-08.json` | Same path inside the verified ZIP |
| <a id="reports-m1-smoke100-manifest-json"></a>`reports/m1_smoke100_manifest.json` | Same path inside the verified ZIP |
| <a id="reports-m1-smoke100-measurements-json"></a>`reports/m1_smoke100_measurements.json` | Same path inside the verified ZIP |
| <a id="reports-m1-source-handoff-review-json"></a>`reports/m1_source_handoff_review.json` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-fixture-smoke-json"></a>`reports/m1_streaming_fixture_smoke.json` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-header-profile-json"></a>`reports/m1_streaming_header_profile.json` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-implementation-hashes-json"></a>`reports/m1_streaming_implementation_hashes.json` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-plan-md"></a>`reports/m1_streaming_plan.md` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-regression-tests-txt"></a>`reports/m1_streaming_regression_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-storage-plan-json"></a>`reports/m1_streaming_storage_plan.json` | Same path inside the verified ZIP |
| <a id="reports-m1-streaming-unit-tests-txt"></a>`reports/m1_streaming_unit_tests.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-temporary-storage-policy-md"></a>`reports/m1_temporary_storage_policy.md` | Same path inside the verified ZIP |
| <a id="reports-m1-tests-final-txt"></a>`reports/m1_tests_final.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-tests-pre1000-txt"></a>`reports/m1_tests_pre1000.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-tests-safepath-txt"></a>`reports/m1_tests_safepath.txt` | Same path inside the verified ZIP |
| <a id="reports-m1-zstd-prefix-frames-json"></a>`reports/m1_zstd_prefix_frames.json` | Same path inside the verified ZIP |
