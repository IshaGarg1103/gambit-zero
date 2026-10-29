# M1 bounded archive scan: implementation and resource decision

> Storage destination update: the incremental Hugging Face design supersedes the earlier local two-build disk estimate. See [m1_hf_upload.md](m1_hf_upload.md). Historical measurements below are unchanged; the full scan remains stopped.

25 October 2026. **The full archive scan has NOT started.** The user has not
allocated a transfer allowance, runtime allowance or dedicated persistent host.
Full execution remains disabled in `configs/data_human_50k_stream.yaml`.
This is preparation for the 50K pilot, not its acceptance report or a 50K result.
No Git commands, branch changes, commits, pushes, model training or self-play ran.

## Recommendation

Preserve the monthly source population and official full-file verification:
**stream one complete month at a time, retaining only bounded candidate caches
and metadata**. After fully consuming and verifying that month, save its selected
candidate/checkpoint manifest and compression-frame index. Reuse those immutable
artifacts for validation and repeat builds. Never store a full compressed archive
or decompress one to disk.

On this Mac, schedule roughly 2–3 hours of uninterrupted, powered-on network
access per month, rather than relying on one unbroken nine-month session. The
measured nine-month transfer projection is 19.2–21.2 hours, before retries and
later validation. These are short-probe estimates, not promised runtimes. This
approach uses little disk, but it does **not** eliminate the 263.45 GB transfer.
A persistent CPU host would reduce interruption risk; its speed and monetary cost
have not been measured or allocated. No cloud instance or paid service was created.

There is no verified lower-transfer method in the sources inspected that meets
**both** this same month-wide archive sampling design **and our own full original
archive SHA256 verification**. Every original compressed byte must be read
somewhere to perform that comparison. Independent frame checksums, per-game PGN
hashes and a copied published archive hash do not substitute for it.

## Measured local resources and probes

Raw evidence: `m1_50k_preflight.json`, `m1_streaming_storage_plan.json`,
`m1_streaming_header_profile.json`, `m1_pzstd_source_probe.json`.

- macOS arm64, user identifies the computer as MacBook Air M4.
- psutil reports 10 logical/10 physical cores, 16 GiB RAM. Its frequency result
  is implausible and is not treated as measured MHz.
- Available RAM at the storage plan: 4,121,624,576 bytes (~3.84 GiB), a changing
  host measurement. The scanner planning gate requires at least 2 GiB available.
- Free filesystem space: 83,115,008,000 bytes (~77.41 GiB). This already accounts
  for existing environments, source prefixes and prior smoke datasets.
- `quota -v` outside the restricted sandbox reported no user disk quota. This
  does not reserve filesystem capacity against other programs.
- Three new 8 MiB offset-zero probes: 3.80, 3.46 and 3.77 MB/s end-to-end.
  Their summed archive payload was 25,165,824 bytes; the checksum index was
  17,820 bytes. An earlier initial 1 MiB probe and the subsequent 448,915-byte
  frame suffix probe are separate. Known probe/index payload across these
  actions: 26,681,135 bytes. HTTP/TLS overhead and web-document reads are excluded.
- All nine fresh archive sizes and publisher checksums are recorded below and
  in the raw JSON. No monthly full-file checksum is verified.
- Offline new header-filter/reservoir profiling on the nine cached 4 MiB
  prefixes extrapolates to 22,601 s (~6.28 CPU-hours) for all monthly compressed
  bytes. Actual measured prefix timing, candidate counts and rates are retained
  per month in the raw profile. Peak process RSS was 39,043,072 bytes. This is
  not the full-cache peak, not representative game sampling, and not measured
  whole-archive runtime. Network and downstream Pgx replay are excluded.

The transfer likely dominates this measured prefix CPU projection; streaming
can overlap them. Do not add or compare these as fully measured whole-job times.
Allow extra time for validation, output, source retries, thermal changes and
Dataset D preparation. No fixed maximum runtime or transfer allowance is assumed.

## Disk estimate and safeguards

| Planned component | Budget / estimate |
|---|---:|
| Whole compressed archives retained | **0 bytes** |
| Whole decompressed archives retained | **0 bytes** |
| All months' candidate caches, frame indices, ledgers and atomic cache replacements | 1 GiB combined cap |
| Future final build A | 4 GiB planned cap |
| Future reproducibility build B | 4 GiB planned cap |
| Other temporary files | 256 MiB allowance |
| Additional metadata | 128 MiB allowance |
| **Conservative additional-disk planning budget** | **9.375 GiB** |
| Free-space safety reserve | **20 GiB** |
| **Required free space before full workflow** | **29.375 GiB** |

The measured 77.41 GiB free exceeds this planning requirement. If the entire
additional planning budget were used, roughly 68 GiB would remain, assuming no
other process consumes space. This is a budget/estimate, not a measured 50K size.
The accepted 1K result (43,766,401 bytes) scales linearly to about 2.19 GB per
50K build / 4.38 GB for two; game lengths/compression can differ significantly.
The planned 4 GiB/build caps must also be enforced when integrating the eventual
50K materializer; that integration has **not** been run or claimed here.

The actual implemented archive scanner has a 1 GiB **combined** managed cache cap,
checks actual free space on each compression frame and before writes, and refuses
to reduce free space below 20 GiB. Atomic writes count both the old file and the
replacement before creating the temporary file. Logs are aggregate counters and
bounded status snapshots, not a ledger containing every rejected PGN. Only small
candidate caches and metadata are written; no download/decompression spool exists.

Decoded frames are capped at 64 MiB; compressed frames at 8 MiB; candidate PGN
payload in RAM at 128 MiB per active month. These are component bounds rather
than an RSS hard limit: Python objects and JSON serialization add overhead.
Use ~1 GiB as a conservative scanner process planning allowance and remeasure
on a real complete month. An available-memory floor of 512 MiB is checked with
resource checks; falling below it stops the job. Unknown formats, oversized frames,
cache exhaustion or low disk stop safely rather than silently weakening filters.
Do not infer immunity to concurrent disk use, OS kills or sudden power loss.

## Actual source format and resumption

The cached prefixes start with a 12-byte **PZstandard skippable metadata header**.
Earlier inspection only established an empty/skippable frame; interpreting its
payload against the upstream PZstandard specification reveals the advertised
compressed length of the following data frame. This is useful compression
metadata, not a game index. The upstream [PZstandard description](https://raw.githubusercontent.com/facebook/zstd/dev/contrib/pzstd/README.md)
explains this independent-frame layout.

A bounded real January probe completed the advertised first frame using only
448,915 additional bytes. Its 4,643,195 compressed bytes decoded independently to
33,554,432 bytes with an 8,388,608-byte window and a valid frame checksum. The
next PZstandard metadata header was also checked at offset 4,643,207. Neither
compressed nor decoded frame data was saved. This validates the implementation's
first-frame assumptions on a real source; it does not prove every later frame
or every month fits the configured bounds. Unexpected later structure stops.

`PzstdStream` hashes the **original metadata + compressed frame bytes** in order.
It marks verification only after actual transport EOF, exact frozen byte count,
successful frame decoding and comparison to the official full-file SHA256.
A source length/ETag mismatch, incomplete body, corrupt frame or SHA256 mismatch
prevents a verified month checkpoint. No incomplete source is treated as accepted.

**Implemented recovery granularity is a complete month:**

1. A completed month checkpoint contains the verified archive hash/size,
   candidates, sampling config/seed, counters, selected-cache hashes and frame
   offsets/sizes/hashes. It is checksummed and reused without another request.
2. An interrupted month remains `INCOMPLETE` and is rescanned from **byte zero**.
   The small atomic-write temporary file for that month is discarded on restart.
   Previously verified months are preserved.
3. No partial hashlib state is serialized or invented. No blind HTTP Range
   restart is attempted. Restarting an interrupted month can transfer up to
   another ~28–31 GB, and must fit the explicitly allocated retry allowance.
4. The orchestrator conservatively charges a whole-month network allowance to a
   durable ledger **before** opening its socket. Crashes therefore cannot create
   unlimited unaccounted retries. No automatic retry loop exists.

The saved compression-frame index is useful for future verified frame access.
It is **not** an index of complete games; PGNs/UTF-8/header lines may cross frame
boundaries. It is also not a global seek table or a uniform-game sampler.
Future larger sampling may still require rescanning or a further verified
frame/game/day index. The [Zstandard seekable format](https://raw.githubusercontent.com/facebook/zstd/dev/contrib/seekable_format/zstd_seekable_compression_format.md)
is a separate format with a footer/seek table; its existence was not assumed.

## Sleep and session termination

A two-second controlled probe proved only that a detached child can finish after
its immediate parent exits (the child was reparented to PID 1). It did **not** test
Codex app closure, logout, reboot or sleep. `nohup`, `caffeinate` and `launchctl`
are installed; `tmux` is not. No persistent extraction job/service was installed.

Mac sleep suspends useful work and can break the network connection. A socket
failure stops the current month with incomplete status. After waking, rerun the
job: completed months are reused, the interrupted month starts again. Sleep is
not transparent byte-offset resumption. Closing the lid, battery exhaustion,
logout/reboot or force quitting should not be assumed safe for an active socket.

For a later approved local run, use AC power, keep the lid open, and run through
a user-managed persistent job/terminal setup with `caffeinate` as appropriate.
Even `nohup` is not protection against reboot or an OS/app process-group kill.
This is why monthly completion checkpoints matter. A persistent CPU host is the
more reliable option if the Mac cannot remain awake for a complete monthly scan.

## Sampling and validation implementation

`utc-day-rating-bottomk-sha256-v1` is frozen in the new disabled config.
Monthly target counts remain 5,000 each Jan–Jul, 7,500 Aug and 7,500 Sep. Quotas
are split as evenly as possible across actual UTC calendar days; residual slots
go to earlier numbered days, a documented one-game difference, not prefix sampling.

The fast prefilter reads headers only. It checks standard/rated status including
contradictory explicit Rated tags, normal start, both ratings >=1800, base >=300
seconds irrespective of increment, valid results/termination, valid UTC date in
the assigned month, and stable Site game ID where available. Missing/invalid IDs
fall back to SHA256 of the raw PGN, matching the accepted pipeline's identity rule.
Missing, malformed or duplicate required headers and source-month mismatches
are explicitly counted/rejected. Full parsing/Pgx correctness still follows.

Priority is SHA256 of UTF-8 canonical JSON array
`[algorithm_version, seed, source_month, stable_game_id]`. This avoids ambiguous
string concatenation. Ties use game ID and raw-PGN SHA256. The bounded max-heaps
keep the smallest priorities for each UTC-day/rating stratum; the seed is 20261025.
At least ceil(60% of each day's quota) must fully validate with minimum player
rating >=2000. Regular candidates are 1800–1999 by the lower rating. Strong
reserves can fill a regular shortage, explicitly counted, so the minimum is
preserved. Strong shortages, empty days or exhausted reserves stop; no implicit
redistribution or early-month fallback occurs.

Reserves are fixed at factor 3 (strong retains up to 3x the whole daily quota,
regular up to 3x its required quota). Full parser source rejections use the next
ranked reserves. Unexplained Pgx `ParityError` escapes immediately; it is never
caught as an ordinary rejection. A reserve shortfall requires explicit review
and perhaps a new full scan with larger reserves, not automatic budget expansion.
Per-day eligible occurrence counts are distinct from unique-game counts; retained
ID duplicates are deduplicated, conflicting retained payloads stop. No exhaustive
source-wide uniqueness proof is claimed for discarded candidates.

`DaySampler.select` accepts the existing `replay_game` callback. It preserves
full sequential Pgx history, encoder/action/rules versions, WDL and time-forfeit
masks; no rules/search/encoder code was changed. This scanner does not yet create
an accepted 50K position dataset. Dataset D must still be frozen and audited
before real extraction; its required manifest/hash gates remain unset.

## Comparison of acquisition approaches

| Approach | Transfer/storage | Representativeness and verification | Decision |
|---|---|---|---|
| Bounded complete-month streaming | 263.45 GB total, ~19–21 h projected network; no archive disk | Day/rating quotas over all months; own full-file SHA256 possible | **Recommended strict design**, after budget/awake-host review |
| Same scan on a persistent CPU host | Same source bytes at host; only selected artifacts reach Mac | Same design if manifests/logs/source hashes are preserved; remote resource/cost unmeasured | Best reliability if local sleep/uptime is uncertain |
| Monthly full-file spool or torrent | Similar total transfer; retains up to ~31 GB/month | Easier byte-level restart; torrent pieces are not full SHA256 verification until complete | Not implemented; conflicts with user's no-complete-archive retention preference |
| Trusted previously verified selected-game/day cache | Only needed cached games/indices transferred | Good only if selection population, source hash, seed, inclusion rules and verification evidence match | Efficient future reuse; no suitable existing artifact verified here |
| User game export API | Potentially much less; exact amount not measured | Player-selected population; historical/date filters possible, but not the full monthly population or archive SHA256 | Would require explicit research-plan amendment; not silently substituted |
| Export known IDs API | Potentially small, IDs must already be known | Helps retrieve a selected ID list; does not supply an unbiased month-wide ID population or verify monthly files | Useful only with a valid upstream selection manifest |
| Random compressed ranges / frame clusters | Potentially less, not measured | Frames need verified boundaries; compressed-byte/cluster inclusion bias and game-boundary handling differ; no full-file SHA256 | Not a drop-in replacement; not used |
| Start-of-archive prefixes | Small | Severe measured chronology bias; no month-wide coverage/full checksum | Rejected for the 50K pilot |

The official [user export specification](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/games/api-games-user-username.yaml)
supports date/rated filters for a specified user. That can preserve date splits
and be filtered for Elo/clock requirements, but does not make the selected users
representative of all eligible monthly games. The official [known-ID export](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/games/api-games-export-_ids.yaml)
accepts up to 300 supplied IDs; it does not solve ID sampling. Both can retain our
per-ply validation and evaluation exclusions, but change source provenance. The
[Lichess API guidance](https://lichess.org/page/api-tips) requires sequential API
requests and backing off after 429. No bulk API requests, scraping or population
selection by top players/opening-explorer examples was performed.

## Completed checks, pending gates and reproduction

- Full regression results: see `m1_streaming_regression_tests.txt` (final result
  recorded in the companion preparation report).
- `m1_streaming_fixture_smoke.json`: synthetic 31-day archive, 868 source games,
  310 selected, 186 strong (60%), 620 pre-move records / 930 parity states incl.
  final boards. White/Black/draw outcomes and timeout masks checked; no mismatch.
  Immutable checkpoint reuse gave identical selection, only one in-memory fixture
  transport opened, zero real network bytes, no raw/decoded archive file written.
- Tests exercise malformed/missing headers, calendar quotas, rating/time-control
  boundaries, deterministic rank ties, reserve replacement/shortfalls, memory/disk
  caps, truncation/corruption/SHA failures, interrupted-month restart, real corrected
  Pgx replay, and default-disabled execution. Prior 117 acceptance tests retained.
- Real bounded first-frame decode passed; complete real-source compatibility and
  scan reliability remain untested.
- The prior accepted 41-file implementation hash snapshot still matches. No
  accepted M0/M1 implementation, lock or encoder was modified.

Local, no-network reproduction:

```bash
.venv/bin/python scripts/scan_m1_archives.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_streaming_fixture.py
.venv/bin/python benchmarks/bench_archive_headers.py
```

Bounded network probes (not full extraction):

```bash
.venv/bin/python scripts/preflight_m1_50k.py --output reports/m1_50k_preflight.json
.venv/bin/python scripts/probe_pzstd_source.py
```

The scanner's default command is **plan only**. Full execution requires explicit
configuration approval, sufficient cumulative transfer/runtime allowances,
resource gates and the frozen Dataset D manifest/hash. No executable full-scan
command is requested for this handoff; do not enable it until those gates pass.

Pending independently: freeze Dataset D, confirm source/selection design review,
allocate network/runtime and an awake/persistent host, integrate real accepted
50K materialization with its disk caps, then actual full-source scans, per-ply
validation, leakage audit and two-build reproducibility. Blackwell profiling is
also pending; it is independent of CPU scan correctness. No M2 readiness claim.
