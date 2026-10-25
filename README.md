# TinyChess: Can a Small Chess-Specialized Model Outplay Frontier LLMs?

**Research and implementation plan — version 0.4 (24 October 2026)**  
**Status:** Proposed experiment; no results or trained checkpoints are claimed.  
**Hardware:** 1 × NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), with CPU, RAM, local disk, and actual throughput to be measured.  
**Main research model:** ~100M-parameter chess-specific policy–value transformer trained from scratch.  
**Scaling ladder:** ~25–50M lower-capacity comparison; ~600M **gated scaling arm**; ~1B exploratory extension only after evidence at 600M.  
**Compute host:** a remote system equipped with **1 × NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)**. GPU availability is known; CPU core count, host RAM, storage throughput, free disk and network characteristics **must be measured** before setting budgets.
**Primary implementation:** JAX/Flax + Pgx chess + DeepMind mctx (GPU-batched self-play). **Conditional fallback:** PyTorch + Rust `shakmaty` rules via PyO3, only if the Pgx correctness gate cannot be satisfied.

> **Goal:** Build a chess-specialized neural model that achieves a statistically credible positive score in **direct, matched games against a pinned frontier GPT configuration**. We will train primarily on human chess games and **our own self-play**, rather than cloning DeepMind's Searchless Chess pipeline. The model's **network-only** and **search-assisted** strengths must be evaluated and reported separately.

## Version history (research plan)

- **v0.1:** 25M engineering prototype → 100M primary chess transformer; human-game warm-start and simple self-play/MCTS.
- **v0.2:** reviewer-inspired Gumbel/search-efficiency and better targets; elevated 600M main and 100M engineering baseline; added teacher control and stronger evaluation.
- **v0.3:** JAX/Flax, on-device Pgx chess and mctx search; blocking rules-parity M0 and Rust/PyTorch fallback.
- **v0.4 (current):** **100M main**; 25–50M scaling point; **600M gated** and 1B conditional; core order **B0→B1→B2**, then **B3/B4**, optional B5–B7; root-Q blend **full-search only**; explicitly defined remote Blackwell host; rough milestone time boxes.

## Contents

1. [Research objective](#1-research-objective)
2. [What is distinctive about this experiment?](#2-what-is-distinctive-about-this-experiment)
3. [Dataset inventory](#3-dataset-inventory)
4. [Generating Dataset A: human-game supervision](#4-generating-dataset-a-human-game-supervision)
5. [Generating Dataset B: self-play trajectories](#5-generating-dataset-b-self-play-trajectories)
6. [Optional Dataset C: engine-teacher ablation](#6-optional-dataset-c-engine-teacher-ablation)
7. [Dataset D: frozen evaluation suite](#7-dataset-d-frozen-evaluation-suite)
8. [Model architecture](#8-model-architecture)
9. [Training procedure](#9-training-procedure)
10. [Search variants and reviewer-inspired improvements](#10-search-variants-and-reviewer-inspired-improvements)
11. [Evaluation: what does “beats GPT” mean?](#11-evaluation-what-does-beats-gpt-mean)
12. [Experiment and ablation matrix](#12-experiment-and-ablation-matrix)
13. [Compute plan and systems design](#13-compute-plan-and-systems-design)
14. [Proposed repository layout](#14-proposed-repository-layout)
15. [Milestones and exit gates](#15-milestones-and-exit-gates)
16. [Failure modes and safeguards](#16-failure-modes-and-safeguards)
17. [References](#17-references)

---

## 1. Research objective

**Primary question:** Can a chess-specialized transformer with hundreds of millions of parameters outperform much larger general-purpose GPT models at chess, when tested under a fixed, transparent game protocol?

**Secondary questions**

- Does self-play improve playing strength beyond supervised imitation of skilled human games?
- How much benefit comes from neural network weights versus inference-time MCTS?
- At a low simulation budget, does **Gumbel AlphaZero** improve strength faster than ordinary **PUCT**?
- Do **blended outcome/search value targets** improve learning compared with terminal game outcomes alone?
- Does **decaying human-game supervision** help self-play eventually surpass its human imitation policy?
- Under fixed compute, how does the **100M main model** compare with **25–50M**, and when (if ever) does **600M** earn its additional cost?
- As a separately disclosed control, how much improvement does a **Stockfish-labeled teacher dataset** buy compared with self-play-only training?

**Pre-registered primary success condition:** Our selected checkpoint obtains a **game score > 50%** against a named GPT model/version under a fixed protocol, with a prespecified statistical procedure supporting a positive advantage. Each win is 1 point, draw 0.5, loss 0. We will not treat puzzle accuracy, an unrelated leaderboard's Elo, or a few informal games as proof that we “beat GPT.”

**What is *not* the goal:** to beat unrestricted Stockfish; to produce free-form chess explanations; to fine-tune a general language model; or to reimplement Searchless Chess exactly.

## 2. What is distinctive about this experiment?

The approach deliberately combines established ideas—human-game initialization, a neural policy–value network, MCTS, and AlphaZero-style self-play. **Those components are prior art, not new inventions.** Our research contribution, if results support it, would be a controlled study of:

1. **Small-model chess specialization versus frontier general-purpose LLMs**, with identical rule access and explicit accounting for inference-time search.
2. **Search/data efficiency on one GPU**, including Gumbel search, variable playouts, value-target design, and learning curves against GPU-hours.
3. **Engine-free learning versus expert distillation**, using the *same architecture and evaluation harness* for an optional teacher baseline.

**Training-data policy for the main track:** no Stockfish best-move targets, centipawn scores, Leela labels, or external LLM-generated chess analysis. Chess rule enforcement is allowed: it defines the environment rather than providing strategic supervision. Stockfish is permitted for independent evaluation, and teacher labels are permitted **only in Dataset C**, which is never mixed into the engine-free primary run.

**v0.3 systems decision, retained in v0.4:** Replace CPU-side `python-chess` inside the **self-play and MCTS rollout loop** with **[Pgx](https://github.com/sotetsuk/pgx) + [mctx](https://github.com/google-deepmind/mctx)** under JAX. Implement the policy–value transformer and its training in **JAX/Flax**, using Optax for optimization. The main data/learning hypothesis is unchanged. `python-chess` is restricted to **PGN parsing, test oracles, GPT-match validation, and final game adjudication**—never per-node self-play search. If M0 cannot establish rule equivalence or a well-specified termination policy, use the documented **PyTorch + `shakmaty`/PyO3** fallback after benchmarking it. **Verify every integration detail against the installed versions**; a package supporting JAX does not itself guarantee end-to-end speed. See [M0](#15-milestones-and-exit-gates).

**v0.4 research sequencing:** **B0 → B1 → B2** is the required core; **B3 then B4** are the next priority only after that comparison; **B5–B7** are optional. **Value `z/Q` blending applies only to full-search positions** with a valid root `Q`; fast-search positions keep outcome-based value targets. The 600M experiment is never automatic just because weights fit in 96 GB.

---

## 3. Dataset inventory

| ID | Dataset | Where it comes from | What we create | Use | Main track? |
|---|---|---|---|---|---|
| **A** | Human-game policy + outcome data | Public Lichess standard-rated PGN archives | Position/action/outcome records | Supervised warm start | **Yes** |
| **B** | Search-assisted self-play | Games generated by **our model** against itself | Search targets, root value, outcome, metadata | Main self-play training loop | **Yes** |
| **C** | Stockfish teacher data | Selected chess positions scored by Stockfish | Expert action scores and optional best-move labels | Separate teacher ablation | No; optional |
| **D** | Frozen evaluation data | Public Lichess puzzles + generated opening suites + match logs | Held-out tests and direct game results | Checkpoint selection/final assessment | Evaluation only |

**Public sources**

- Lichess standard-rated game exports (CC0): https://database.lichess.org/#standard_games
- Lichess puzzle exports (CC0): https://database.lichess.org/#puzzles
- **Pgx** JAX chess environment (main self-play rules): https://github.com/sotetsuk/pgx and https://www.sotets.uk/pgx/chess/
- **DeepMind mctx** batched JAX tree search (PUCT-style / Gumbel): https://github.com/google-deepmind/mctx
- **JAX + Flax + Optax** (policy–value model and optimization): https://docs.jax.dev/ ; https://flax.readthedocs.io/en/stable/ ; https://optax.readthedocs.io/
- **python-chess** (PGN parsing, test oracle, GPT-match validation, final adjudication only): https://python-chess.readthedocs.io/en/latest/
- **Fallback `shakmaty` + PyO3** (Rust rules if Pgx parity gate fails): https://github.com/niklasf/shakmaty ; https://pyo3.rs/
- Optional Stockfish reference: https://github.com/official-stockfish/Stockfish

### Proposed starting sizes, not final requirements

| Phase | Source size | Derived data | Notes |
|---|---|---|---|
| Parsing smoke test | 100–1,000 complete PGN games | Thousands of position rows | Verify correctness |
| Warm-start pilot | 50K filtered games | Potentially millions of plies | Stream rather than download everything |
| Main human pretraining (100M model) | 1M–5M *accepted* games, conditional | Tens to hundreds of millions of raw plies before sampling | Establish effective unique-position counts; scale only if useful |
| Initial self-play smoke test | 100 completed games | Game-state trajectories | Check reward signs and termination |
| Initial self-play experiment | 1K–5K completed games | Search policy + outcome rows | Scale only after throughput benchmarking |
| Longer self-play | Continuous versioned replay buffer | Depends on model speed and budget | No fixed game count promised |
| Teacher comparison | 100K–1M distinct positions, optional | Engine annotations | Explicitly account for CPU time |
| Evaluation | Frozen test and opening sets | No training targets | Fix before selecting models |

**A million games is not a million training examples.** A single game supplies many plies, and the same opening positions recur widely. Log accepted games, total plies, distinct positions, unique opening families, and effective sample counts separately.

---

## 4. Generating Dataset A: human-game supervision

### 4.1 Which months?

Use a **frozen 2026 snapshot**. A practical initial proposal is:

- **Train:** Lichess **January–July 2026** standard-rated games.
- **Validation:** **August 2026**.
- **Held-out human-game test:** **September 2026**.

These are proposed split boundaries, not months prescribed by a paper. As of 24 October 2026, Lichess has published standard-rated game archives through September 2026. Save archive URLs, sizes, SHA256 checksums, and extraction filters in a manifest. A chronological split prevents a source game from crossing splits but **does not** remove repeated board positions across games. Audit position overlap and report a position-disjoint subset.

Lichess archives are large (~tens of GB compressed per month). **Stream `.pgn.zst`** with `zstandard` and parse PGNs incrementally; do not require full decompression to disk.

### 4.2 Filter the source games

Initial choices to test, not hard-coded truths:

- Standard chess only; exclude Chess960/variants.
- Both players rated **≥1800 Lichess**, with an optional ≥2000 subset for stronger demonstrations.
- Prefer rapid/classical or longer blitz; record the actual time control.
- Successfully parsed, complete games with a valid result (`1-0`, `0-1`, `1/2-1/2`).
- Exclude aborted games, corrupted PGNs, and games with invalid moves.
- Record `Termination` and **exclude or downweight time-forfeit outcomes** for value learning, because clock losses need not mean the position was bad.
- Avoid over-sampling frequently repeated openings; balance opening/middlegame/endgame positions.
- Do **not** read annotated `[%eval]` comments as training targets in the engine-free track.

Record **why each game was accepted/rejected**. Keep raw PGN hashes to enable independent reproduction.

### 4.3 Derive training labels ourselves

For each accepted game:

1. Parse the starting board and iterate through moves in order with `python-chess` **only as the offline PGN parser**; serialize states and history into the Pgx/JAX training representation.
2. Before the human's move, store the current state and history-dependent metadata.
3. Store the move actually played, in **UCI** format.
4. Convert the parsed state/history into our canonical action representation and generate the **training legal-action mask with Pgx**, after validating the FEN/history/side-to-move conversion against `python-chess` in M0. Never use `python-chess` in search or the live self-play loop; its use here is limited to PGN parsing and test assertions.
5. After processing the whole game, assign the final result **from the side-to-move perspective** to every eligible pre-move state: win `+1`, draw `0`, loss `-1`; also store an explicit W/D/L class.
6. Optionally record remaining plies until game end for future moves-left supervision, with caution about games ending on time.
7. Shard output into Parquet (preferred at scale) or compressed JSONL (debugging), with manifest/checksums.

Example record, **illustrative**:

```json
{
  "schema": "human_v2",
  "game_id": "example_game_001",
  "source_month": "2026-03",
  "ply": 4,
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "recent_moves_uci": ["e2e4", "e7e5", "g1f3", "b8c6"],
  "human_move_uci": "f1c4",
  "legal_moves_uci": ["f1c4", "f1e2", "d2d3"],
  "result_stm": 1,
  "result_class_stm": "win",
  "termination": "Normal",
  "white_elo": 2200,
  "black_elo": 2150,
  "label_source": "human_game"
}
```

**Important:** `legal_moves_uci` is deliberately truncated for readability in this example. Real records must store **every legal move** or a precise versioned legal-action mask **mapped to the tested Pgx action indices**. The example's win label assumes a hypothetical White victory; it is not derived from this position.

### 4.4 Data leakage, bias and board-history checks

- Split by **whole game**, not random plies.
- Cap duplicate common positions in sampling; do not collapse all identical-FEN demonstrations into one move label because multiple moves can be good.
- Save **side to move, castling, en passant, half-move clock, and repetition/history features**. FEN by itself does not preserve all repetition information.
- Verify all chosen actions are legal, including castling, underpromotion and en passant.
- Verify result labels flip sign appropriately with perspective.
- Prevent puzzle/evaluation positions from leaking into the primary held-out subsets where feasible; report any unavoidable overlap.
- Maintain opening, side-to-move, phase and WDL balance diagnostics.

### 4.5 What Dataset A teaches

The **policy target** is one human move; the **value target** is the game outcome. Both are noisy: strong humans make mistakes, and a final win is not proof that every earlier move was correct. Dataset A provides an **initial policy**, not an oracle chess policy.

---

## 5. Generating Dataset B: self-play trajectories

This is the project's **main original dataset**, generated after the initial supervised checkpoint is strong enough to complete games.

### 5.1 Self-play loop — Pgx + mctx (JAX), not python-chess

**Intended data path:** `batched Pgx states → Flax policy/value forward pass → mctx PUCT/Gumbel tree search → Pgx state transitions → policy/value/outcome records`. All rule checks and tree expansions **inside self-play** must remain in the JAX execution path; do not call `python-chess` from JIT-compiled rollout/search code. The **JAX/Flax model** runs in the same array framework as Pgx and mctx, reducing unnecessary host/device movement.

mctx offers **JAX-batched MuZero-style PUCT and Gumbel search policies**, but it does **not automatically know chess rules**: our integration must provide a correct deterministic chess transition/recurrent function backed by **Pgx**, legal-action masks, root policy logits, value predictions, and terminal rewards/discounts. The tree must carry full chess history needed to enforce draws. **Verify root/recurrent types, policy outputs, masking conventions, PRNG handling, and the chosen search entry point against the installed mctx version.** No API names/signatures are assumed stable by this README.

For each parallel group of games:

1. Choose valid start states from the **train-only opening pool**, including the normal starting position; establish how to import nonstandard openings and histories into the installed Pgx chess environment (**verify against installed version**). Do not silently reconstruct history-incomplete positions.
2. Hold multiple Pgx chess states **on-device**, using JAX batching/JIT; compute the Flax network's policy logits and WDL values for those roots.
3. Run **mctx PUCT-style search (32 simulations)** for the baseline, and **mctx Gumbel search** for the controlled comparison, using a Pgx-backed recurrent transition. Track JAX compilation, search-tree memory and actual evaluation counts. An `mctx` search policy is not interchangeable with a zero-simulation policy forward pass.
4. Convert mctx's returned action identifiers into the **versioned canonical action map**; apply chosen legal moves using Pgx only. Save policy-improvement weights/visit distributions appropriate to the chosen algorithm, root values and enough metadata to reconstruct the run.
5. Check terminal results and termination reasons **in Pgx during self-play**, with a documented rules/adjudication policy validated at M0 against `python-chess`. Distinguish **threefold/fifty-move *claimability*** from **fivefold/seventy-five-move automatic draws**, and audit any environment-specific max-step or no-progress cutoffs. A Pgx truncation must not silently be labeled a normal FIDE draw.
6. On completion, attach the terminal result to every retained position **from its side-to-move perspective**, along with remaining plies, mask version, rule-engine version, search algorithm/parameters, model checkpoint ID, seed and episode termination cause.
7. Train the next checkpoint in JAX/Flax/Optax, evaluate and optionally promote it to the self-play pool. No game state should move through `python-chess` on each simulation or turn.

**Version-sensitive caution:** The published Pgx chess documentation describes threefold repetition, a **50-halfmove** no-progress cutoff, and a **512-step** cap as termination conditions. Those are **not automatically equivalent** to the usual claimable 50-move rule (100 halfmoves) and automatic 75-move rule (150 halfmoves) used in standard adjudication. Documentation may not match the installed chess environment version: **verify against installed version**, and decide up front whether training follows exact FIDE adjudication or a separately labeled environment convention.

**Correctness is a hard gate:** If Pgx cannot match our required chess/legal/terminal behavior (or be made equivalent through a transparent, validated wrapper), **stop the Pgx path**. Use **PyTorch for the model and `shakmaty` via a PyO3 Rust extension for fast chess state transitions**. This fallback is a deliberately separate implementation, not an invisible backend swap; re-run legality, adjudication, action mapping, throughput and evaluation tests. The fallback will require an appropriate batched search implementation; do not assert that mctx works natively with the Rust/PyTorch stack.

**Performance is empirical:** GPU rules may be faster than CPU Python for large batches, but MCTS may be limited by irregular trees, compilation, network inference, or transfer overhead. Require games/hour and strength-per-GPU-hour measurements rather than assuming a speedup.

### 5.2 Example self-play record

```json
{
  "schema": "selfplay_v3",
  "game_id": "sp_000042",
  "model_checkpoint": "ckpt_0010",
  "rules_backend": "pgx",
  "rules_env_version": "<verified env version>",
  "search_backend": "mctx",
  "action_map_version": "<mapping version>",
  "ply": 37,
  "fen": "<valid FEN string>",
  "recent_moves_uci": ["<recent moves>"],
  "legal_actions": ["<all legal UCI moves>"],
  "chosen_move_uci": "<legal UCI move>",
  "search_algorithm": "gumbel_az",
  "num_simulations": 32,
  "full_search": true,
  "root_q_valid_for_blend": true,
  "policy_target": {"<uci_1>": 0.6, "<uci_2>": 0.4},
  "root_q_stm": 0.35,
  "outcome_stm": 1,
  "remaining_plies": 41,
  "termination": "checkmate",
  "truncated": false,
  "draw_claim_policy": "<pinned and verified policy>",
  "label_source": "self_play",
  "random_seed": 8127
}
```

All entries above are schematic and not a real chess game. For any saved policy target, probabilities must be nonnegative, sum to one (within tolerance) **and refer to the same legal-move ordering**. Gumbel search training targets are **search-improved action weights**, not necessarily vanilla PUCT visit counts. Here `full_search` is a **versioned training-eligibility flag**, not an assertion of an unlimited tree search: it identifies positions whose configured full policy-search budget completed, whose target is valid, and whose root Q is eligible for a blended value target. A failed, truncated, or fast-budget position must set `root_q_valid_for_blend=false` regardless of whether it logged a diagnostic root Q.

### 5.3 Storage and replay strategy

- Store position tensors/action weights in a JAX-friendly batched format while writing durable compact shards asynchronously. Do not serially transfer every MCTS node to the CPU for logging.
- Persist **the Pgx environment version and JAX/mctx versions**, action-map ID, history/termination policy and any explicit adjudication overrides with each run and shard.
- In the Rust fallback, record Rust crate/PyO3/PyTorch versions and identify these results separately.
- Store **episode metadata** separately from per-position records, with unique checkpoint, code and data version IDs.
- Use a **rolling replay buffer** containing a bounded mixture of recent and older self-play positions; avoid training solely on stale weak policies.
- Save the move-selection probability/temperature for reproducibility.
- If using variable playouts, keep **full-search positions** for high-quality policy supervision and potential `z/Q` value blending. **Fast-search positions use terminal-outcome (`z`) value targets only**, never blended root Q; they may contribute valid value supervision when the game truly terminates and is not an unhandled truncation. Save `full_search`, actual simulation count, root-Q validity and target provenance for every position.
- Track win/draw/loss distributions, average game length, opening diversity, policy entropy, search agreement and percentage of games ending by adjudication.

---

## 6. Optional Dataset C: engine-teacher ablation

This is **not** part of the primary engine-free training pipeline. It is a separate control experiment inspired by Searchless Chess.

- Sample **the same source-position distribution** used in our main experiments, or construct and log a matched position set.
- Ask a pinned Stockfish version to evaluate candidate legal moves with a specified **nodes/search budget**, perspective, hardware and options.
- Save all engine-search metadata, top action(s), and values/scores. Scores from different hardware/time budgets are **not automatically comparable**.
- Train a separate copy of the **same chess architecture** using these expert targets.
- Compare **engine-free self-play versus teacher labels** under both equal neural-network training compute and an explicitly reported total-cost accounting that includes CPU labeling.

An engine-supervised student may be stronger or more sample-efficient; that is an empirical question. The teacher dataset must not contaminate the main run.

---

## 7. Dataset D: frozen evaluation suite

Use **independent tests not seen during training or hyperparameter tuning**:

1. **Lichess tactical puzzles:** freeze IDs, difficulty buckets, source snapshot and solution lines. Measure whole-line success, not only first-move accuracy.
2. **Human-game test split:** policy agreement and held-out outcome prediction, clearly distinguished from actual playing strength.
3. **Fixed opening suite:** selected positions from several opening families; play both colors. Do not train on those exact opening sequences as evaluation fixtures.
4. **Engine anchors:** pin Stockfish version and fixed-node/limited-strength configurations; regularly test checkpoint win/draw/loss scores. Engines are opponents/evaluators, not teachers, in the main run.
5. **Previous-checkpoint arena:** paired matches against the previous best model to detect regressions.
6. **Frontier-model match suite:** a fixed GPT model identifier, reasoning budget, move format, time/token limits, and predeclared invalid-response policy.

Archive actual prompts, FENs, legal-move lists, responses, selected moves, tool usage, model IDs, seeds and game logs. **Do not infer cross-platform Elo comparability** from unrelated chess leaderboards.

---

## 8. Model architecture

### 8.1 Size ladder

| Size | Role | Decision rule |
|---|---|---|
| **~25–50M** | Lower-capacity **scaling point** (S0) | Run a deliberately smaller model with the matched architecture family, data, loss and evaluation to measure capacity versus throughput; not the primary success target |
| **~100M** | **Main research model** and all core ablations | Run the B0 → B1 → B2 sequence; next B3/B4; select best validated checkpoint |
| **~600M** | **Gated scaling arm** (S1), not committed | Begin only if the 100M run exhibits sustained evaluation strength gains **and** measured self-play throughput, replay volume, memory and available GPU time make the larger run worthwhile |
| **~1B** | Optional extension (S2) | Consider only if S1 gives a measurable scale benefit and a separate resource/throughput review passes |

The 96 GB GPU may have enough *memory* for a 600M or 1B network; this does **not** imply sufficient compute for MCTS-powered self-play. Before launching S1, publish a gate memo containing (1) working Pgx/mctx correctness, (2) evidence that at least one of B1/B2 improves playing strength over B0 on fixed anchors, (3) warm measured GPU-hours per self-play game and per training update, (4) projected full-run cost and storage, and (5) expected learning benefit from the small-to-main size comparison. If any gate fails, **continue optimizing 100M** rather than spending the budget on 600M.

### 8.2 Input and outputs

**Input:** one embedding for each of 64 board squares, plus global context features (side to move, castling rights, en passant, clocks, repetition/history), extracted from or explicitly supplied alongside **Pgx/JAX game state**. A fixed, explicit **side-to-move perspective convention** must be used everywhere; add tests for board-flipping and action-index mappings. Consider a short stack of history boards to avoid state ambiguity.

**Trunk:** non-causal board transformer written in **JAX/Flax**, initialized **from scratch**. Use Optax for optimization; document the selected Flax module style and pin compatible JAX/CUDA versions. Exact hidden size, number of layers, number of heads, and tokenizer/features will be chosen by a parameter-count calculator and validated with a profiler; don't label a configuration 25–50M, 100M or 600M until its actual parameter count is measured.

**Policy head:** primary candidate is attention-style **from-square → to-square** logits with explicit treatment of promotions/underpromotions. Map its outputs into Pgx's **version-pinned action space** and mask invalid moves using **Pgx legal-action masks** before softmax. Test exact index-to-move round trips (castling orientation and promotions especially). Compare with the classical **8×8×73 action-plane head** as an ablation. The action mask is rule enforcement, not a source of move-quality labels.

**Value head:** output three **W/D/L logits** from the current-player perspective. Convert to a scalar when search needs a value:

\[
V_\theta(s)=P_\theta(\text{win}\mid s)-P_\theta(\text{loss}\mid s),\quad V_\theta\in[-1,1].
\]

**Optional moves-left head:** predict remaining plies. Train only where termination and game-length labels are valid; test whether it improves game conversion instead of assuming it will.

### 8.3 Architecture invariants to test

- The **Flax forward pass** accepts batched Pgx observations/states with no CPU chess-rules calls during self-play; network/application and action-mask shapes match the search integration.
- Every Pgx state carries enough history for the declared repetition/draw policy; where a different FIDE adjudication convention is desired, an explicit validated policy supplies it.
- Legal action mapping is one-to-one for all UCI promotions and castling moves.
- Policies contain zero probability on illegal actions after masking.
- Model outputs remain finite for checkmate, stalemate and drawn positions.
- Perspective transformations change action and result labels consistently.
- The board representation distinguishes states with different castling/en-passant/repetition rights when needed.

---

## 9. Training procedure

### Stage 0 — Systems smoke tests

First satisfy **M0: Pgx ↔ python-chess differential correctness** (see Section 15). Establish compatible JAX, JAX CUDA plugin, Flax, Optax, Pgx and mctx versions on the Blackwell GPU; pin them in a lockfile and record the Pgx **chess environment version**. Build the PGN parser, batched Pgx board encoder, action mapping, losses and JAX/Flax inference runner. Verify mctx-to-Pgx deterministic transitions and terminal values using tiny trees; **verify against installed version** for all library-specific integration details. Train on a tiny dataset and check that the model **can overfit 100–1,000 examples**. Benchmark compiled single-position and batched forward passes, move generation, search, and complete-game throughput **including compilation separately**. If M0 fails, switch to the explicit PyTorch + `shakmaty`/PyO3 branch before pursuing training.

### Stage 1 — Human-game supervised warm start

Train the **JAX/Flax** policy/value network with **Optax** using Dataset A (PGN parsed offline; examples converted into the tested Pgx/action encoding):

\[
\mathcal L_{\rm human}
=\lambda_{\pi}\big[-\log \pi_\theta(a_{\rm human}\mid s)\big]
+\lambda_{\rm wdl}\,\mathrm{CE}(\operatorname{WDL}_\theta(s),y_{\rm game}).
\]

- Human action is a *demonstration*, not ground-truth optimal play.
- Game outcomes are noisy labels, especially when games are decided by clocks; filter/downweight those cases.
- Validate against a held-out date split and fixed low-strength chess opponents.
- Keep a frozen `human_only` checkpoint as the main self-play control.

### Stage 2 — PUCT baseline self-play

Start with **mctx PUCT-style search backed by Pgx at 32 simulations per selected move**, running batched JAX actors; verify the exact mctx policy entry point and recurrent integration against the installed version. Save normalized root visit-count targets for full searches, root Q and eventual results. Evaluate at fixed intervals and promote a model only through a predeclared arena criterion. Initial value targets use outcome-only labels (`z`).

### Stage 3 — Gumbel AlphaZero comparison

From the **same human-only checkpoint**, repeat training with **mctx Gumbel search at 32 simulations**, matching games/search evaluations/GPU-hours as closely as feasible. Its policy improvement uses Gumbel root action selection and an improved policy target (**verify mctx output meaning against installed version**); don't use generic PUCT visit counts as a drop-in replacement. Compare:

- Strength versus fixed anchors and previous checkpoint.
- Improvement versus GPU-hours and model evaluations.
- Root action diversity and search reliability.

If useful, test 16 and 64 simulations as budget ablations. Evaluate Gumbel **without exploration noise** when appropriate at test time and freeze all evaluation settings.

### Stage 4 — Value-target ablation (B3; full-search positions only)

Let `z∈{-1,0,+1}` be the terminal outcome **from the recorded state's player's perspective**, and `Q_root∈[-1,1]` be the root search value in that same perspective. **Only positions with `full_search=true` and `root_q_valid_for_blend=true` can receive blended targets**. The definition of a full search is tied to the configured run budget (e.g., 32 simulations for a fixed PUCT-32 run, 64 for the mixed-budget experiment), and requires a completed, valid search, not just a requested simulation count. For those eligible positions:

\[
\widetilde v(s)=(1-\lambda)z+\lambda Q_{\rm root},\qquad
\lambda\in\{0,0.25,0.5\}.
\]

To preserve a meaningful WDL head, retain a **categorical WDL cross-entropy** target from the actual game result while optionally adding a **scalar expected-value regression** loss:

\[
\mathcal L_{\rm value}
=\mathrm{CE}(\operatorname{WDL}_\theta,y_{\rm game})
+\beta\,\mathbf{1}_{\rm full,validQ}\left(V_\theta-\widetilde v\right)^2.
\]

Here the indicator is 1 **only for completed full-search positions with valid root Q**; the regression term is masked out for fast-search positions. **All positions with valid game outcomes** retain the categorical WDL cross-entropy on `z`, including fast-search positions. In a scalar-value-head variant, fast-search positions must use *unblended* `z` rather than root Q. Never use unverified Q, unhandled truncated games, or incomplete search outputs as blending targets. This avoids pretending a scalar Q is itself a calibrated W/D/L probability vector. Carefully orient all `z` and `Q` values to the **same side to move**, or the training signal will be corrupted. Root Q is bootstrapped from our model and may be biased, so value blending is **an ablation**, not an assumption.

### Stage 5 — Replay-mixture ablation (B4); optional efficiency tests (B5–B7)

**Human-data mixture:** compare fixed 25% human / 75% self-play batches to a decaying **proposed** schedule such as 40% → 25% → 10% → 5% human (conditional on sufficient self-play quality). Keep the total number of optimizer tokens/positions matched. Track loss of opening diversity and catastrophic forgetting.

**Optional B5 — playout cap randomization:** benchmark distinct JIT-compiled fixed search-budget paths (or another version-verified vectorization approach) and compare fixed-budget search to a **proposed** mixed budget: 25% of moves get 64 simulations (eligible for full policy targets), 75% get 8 simulations (primarily to advance games and provide value outcomes). Average is **22 simulations per move**. This is inspired by KataGo, **not a proven optimum** for our chess model. **Only the 64-simulation full-search subset may use blended `z/Q` targets or full-search policy targets** in this configuration; 8-simulation fast-search positions use outcome-only value targets.

**Optional B7 — moves-left head:** add only as a controlled auxiliary experiment. Report conversion rates from winning positions, draws and repetition patterns; don't interpret lower moves-left prediction error as improved playing strength by itself.

### Stage 6 — Capacity comparison and gated scaling

**S0:** Train a ~25–50M point on a matched subset/recipe as a lower-capacity scaling comparison alongside the **~100M main model**. Report both equal-position and equal-GPU-hour learning curves. Run this after B0–B2 have established a usable search pipeline; S0 is not a reason to defer those core results.

**S1 (~600M) is conditional:** launch only after the gate in Section 8.1 passes and a concrete available-compute budget is approved. A successful B0 training run *alone* does not justify S1; we first want evidence of self-play improvement and measured resource costs. **S2 (~1B)** is a separate optional extension only after results at S1 support it. The final selected model may remain 100M.

---

## 10. Search variants and reviewer-inspired improvements

**Implementation note (v0.3 stack retained in v0.4):** All self-play search variants run with **mctx + Pgx in JAX** after M0; no Python loop invoking `python-chess` at every tree node. `mctx` documents batched policies with PUCT-style and Gumbel search, but our chess-specific deterministic recurrent function, legal mask, and target extraction are custom. **Verify against installed version** before committing to any API call or policy-target interpretation. Search selection, action indexing and sign inversion must be unit-tested under batched execution.

| Idea | Baseline / alternative | Scientific reason | When to add |
|---|---|---|---|
| **PUCT → Gumbel AlphaZero** | PUCT-32 vs Gumbel-32 | Search efficiency under few simulations | First major ablation |
| **Playout cap randomization** | Fixed 32 vs 25% × 64 / 75% × 8 | Better policy labels per average search budget; fast-budget positions use `z` only | **Optional B5** |
| **Root Q / terminal z blending** | λ = 0, 0.25, 0.5 on **completed full-search positions only** | Test noisy outcome targets without training on shallow/invalid Q | **B3, after B0–B2** |
| **Decaying human mixture** | Fixed 25/75 vs schedule | Reduce imitation bias as self-play strengthens | **B4, after B3** |
| **Moves-left head** | Off vs on | Test late-game conversion and shuffling | **Optional B7** |
| **Attention policy** | 8×8×73 vs square-to-square + promotion | Architecture efficiency and move representation | **Optional B6** |
| **Teacher-labeled control** | Engine-free vs Stockfish-supervised | Quantify value of expert labels | Separate experiment |
| **Parameter scaling** | **25–50M vs 100M**; gated 600M; optional 1B | Capacity vs data/search tradeoff | S0 after core; S1/S2 only at gates |

**Execution priority:** complete **B0 → B1 → B2** first, then **B3 → B4**. Treat **B5–B7 as optional**, not launch blockers. Avoid introducing multiple search, loss and data changes at once. Restart matched ablations from the **same checkpoint** whenever possible.

---

## 11. Evaluation: what does “beats GPT” mean?

**Headline result is a *match outcome*, not a borrowed Elo number.** Name the GPT version and reasoning-effort configuration that are actually available at evaluation time; don't silently switch models or budgets mid-experiment.

### Primary GPT match protocol

1. Freeze our checkpoint and specify **network-only** or **model + N MCTS simulations** before games begin.
2. Freeze the GPT model ID/configuration, prompt, whether it receives FEN, move history, legal moves, piece list, and any additional board tools. Specify maximum output tokens, thinking effort, cost, and retries.
3. Play **paired-color games from the same opening starts**, with an independently selected opening suite.
4. Supply the same chess rules to both systems: GPT may be provided a legal-move list; our network uses a deterministic legal mask. Explicitly label any unequal assistance.
5. Validate all moves in `python-chess`. For an illegal GPT response, use a **predeclared** retry policy (e.g., one formatting retry) and then apply the designated default, such as a forfeit. Report a separate sensitivity analysis using a random legal replacement if informative. Never quietly repair one model's strategic mistake for free.
6. Report wins, draws, losses, game score, confidence intervals, invalid-move rates, move-selection latency and inference compute/cost for **both** systems.
7. Development can use an **SPRT or other preregistered sequential rule** with defined hypotheses, error rates and stopping limits. A small observed win streak isn't a statistically established result. Reserve a separate final evaluation suite or account for repeated testing.

### Complementary evaluation

- Network-only policy accuracy and legal-move rate.
- Puzzle **complete-line pass@1**, by rating and tactical theme.
- Chess playing strength vs fixed-node Stockfish anchors and previous model checkpoints.
- Position-value calibration, WDL loss and draw detection.
- Strength-vs-compute curves for training and inference: Elo/game score vs GPU-hours, examples, simulations and latency.

**Policy strength with search and no search are different claims.** A 100M model (or gated 600M variant) receiving 64 MCTS simulations at inference should not be described as a network that makes equally strong moves in one forward pass.

---

## 12. Experiment and ablation matrix

**Mandatory, ordered core:** B0 → B1 → B2. **Next priorities:** B3 → B4. **Optional only:** B5–B7; run them if results and GPU budget justify them. All B-series experiments use the **100M main model** unless expressly stated otherwise.

| ID | Priority | Initialization | Difference | Primary question |
|---|---|---|---|---|
| **B0** | **1 · Required** | 100M random init | Human-game supervised policy + WDL only | Can we train a useful baseline? |
| **B1** | **2 · Required** | Frozen B0 | PUCT-32 self-play, terminal `z` | Does self-play help? |
| **B2** | **3 · Required** | **Same frozen B0** | Gumbel-32 self-play, terminal `z` | Does low-budget search improve learning per GPU-hour? |
| **B3** | **4 · Priority** | Best **matched** B1/B2 setup | Blend `z/Q` **only on completed full-search positions with valid Q**; fast-search value data remains `z`-only | Does a reliable search value help? |
| **B4** | **5 · Priority** | Same selected search/model baseline | Fixed vs decaying human-data share; isolate this factor from B3 | Does it improve later learning? |
| **B5** | Optional | Matched search baseline | Playout cap randomization: 64-full vs 8-fast | More strength per GPU-hour? |
| **B6** | Optional | Matched architecture/training | 73-plane vs attention policy head | Does move parameterization matter? |
| **B7** | Optional | Matched baseline | Moves-left head off vs on | Does it help game conversion? |
| **S0** | Capacity point | 25–50M random init | Matched recipe and evaluation against 100M | How much strength is gained per parameter and GPU-hour? |
| **T1** | Optional control | Matched architecture | Stockfish teacher targets, **separate from main self-play** | Cost of excluding expert labels? |
| **S1** | **Gated** | ~600M random init | Best validated 100M training recipe; pass scale gate first | Does the larger model buy meaningful playing strength? |
| **S2** | Optional after S1 | ~1B random init | Same scaled recipe if S1 succeeds | Are further gains worth the throughput cost? |

B3 and B4 should be controlled, separately initialized comparisons, not an unexplained stack of tweaks. B4 may use the selected B1/B2 baseline even if B3 is inconclusive. For search fairness, match total network evaluations/GPU-hours where possible. Report both equal-position and equal-GPU-hour curves for S0/S1. Save preregistered game protocols, seeds and uncertainty-aware results.

## 13. Compute plan and systems design

### Primary stack and GPU memory

**Primary stack:** NVIDIA RTX PRO 6000 Blackwell (96 GB) → **JAX + Flax + Optax** policy–value network → **Pgx GPU-resident chess states** → **mctx batched PUCT-style/Gumbel tree search** → versioned self-play replay dataset. Use `python-chess` only for offline PGN parsing, test oracles, GPT-match validation and final adjudication.

The board rules, search expansions and model inference should remain on-device within compiled/batched sections where supported. mctx provides a JAX search implementation, not the chess dynamics: those must be wired to Pgx, including legal-action masks, terminal states, rewards and the current-player value convention. **Verify against installed version** for specific Pgx/mctx contracts and the feasibility of nonstandard opening-state import.

For ordinary BF16 full-parameter AdamW training, a rough model-state estimate is **~16 bytes/parameter** for weights, gradients, FP32 master weights and Adam moments, **excluding** activations, JAX/XLA temporaries, search trees, Pgx states, buffers, compiler allocations and possible concurrent policy copies:

| Parameters | Approx. model + optimizer state | Interpretation |
|---|---:|---|
| 25–50M | ~0.4–0.8 GB | Lower-capacity scaling point; benchmark throughput |
| 100M | ~1.6 GB | **Main model**; run core ablations |
| 600M | ~9.6 GB | **Gated** scaling; network memory alone is not approval |
| 1B | ~16 GB | Optional after a successful 600M arm |

The major bottleneck may be **MCTS network inference and XLA tree memory**, not parameter storage. If a game averages 80 plies and search performs 32 evaluations per move, **~2,560 model evaluations per game** is a simplified scale estimate—not a runtime prediction. Cache reuse, batching, tree topology and terminal leaves change this count. Full-search actors may compete with training for 96 GB VRAM; measure separate **data-generation → training** phases before attempting concurrent scheduling.

### Mandatory M0 compatibility/correctness benchmarks

1. Record GPU driver, supported JAX/jaxlib CUDA build, Flax, Optax, Pgx, **Pgx chess environment version**, mctx, Python and relevant compiler versions. Pin exact working versions; use `verify against installed version` for any release-specific API.
2. Verify **Pgx ↔ python-chess equivalence** on move sets, next-state transitions, special moves and termination outcomes, including castling, en passant, **all promotions**, checkmate, stalemate, insufficient material, repetition and the 50-move rule. Test 75-move/fivefold automatic rules and any Pgx max-ply truncation too. Explain claimable versus automatic draws explicitly.
3. Test batched state transitions with seeds and FEN/PGN trajectories; compare Pgx/action-map round-trip and legal masks to the oracle, especially under orientation flips and promotions. Treat state history as part of draw/repetition correctness.
4. Build a minimal Pgx-backed mctx recurrence/search smoke test; assert no illegal actions, correct rewards/value-perspective reversal, terminal nodes and normalized policy targets. **Verify against installed version.**
5. Benchmark **GPU rules throughput** (states/sec), **search throughput** (complete games/hour), **network inference throughput** (positions/sec), and **end-to-end strength per GPU-hour**, separately reporting JIT compile time, warm-up and XLA memory. Compare batching levels (e.g., 1, 8, 32, 128) as feasible.
6. If Pgx mismatches rule semantics, attempt a fully tested, explicitly documented game-adjudication policy **only if** it preserves correctness and search dynamics. Otherwise trigger the **PyTorch + `shakmaty` (Rust) via PyO3 fallback**, repeat this gate and benchmark it against the primary stack. Do not switch silently.

### First measured runs on the remote Blackwell GPU host

- GPU model/driver, CUDA/toolkit, JAX versions, BF16 support; **measure** CPU cores, RAM, SSD, free disk, network and remote-host queue/access limits. The host is the single remote RTX PRO 6000 Blackwell system defined at the top of this README; no unstated specs are assumed.
- Compiled Flax inference latency and throughput at practical batch sizes for **25–50M and 100M first**; measure 600M only for a pre-gate feasibility probe, including actual peak HBM usage.
- Batched Pgx environment steps/sec, legal masks/sec and terminal-state checks/sec.
- Batched mctx PUCT-32 and Gumbel-32 games/hour, search node evaluations/game, compilation time, device memory and GPU occupancy.
- JAX/Optax training steps/sec, position throughput, host-to-device input cost, replay write/read throughput and checkpoint load time.
- Full end-to-end cycle: self-play positions produced per GPU-hour; checkpoint playing-strength gain per GPU-hour, measured separately for network-only and search-assisted settings. Before S1, project expected 600M game-production rate, training steps/hour and total GPU-hours from a measured feasibility benchmark, and record **pass/fail** of the Section 8.1 scale gate.

**Systems priorities:** vectorized GPU-resident Pgx game states; batched Flax inference + mctx search; tested action masks and legal-index mapping; bounded JAX search-tree allocations; minimal host/device synchronization; sharded replay data; resumable Flax/Optax checkpoints; deterministic seeds; and exact package/environment pinning. Avoid one Python process calling a CPU rules library for every tree node.

### Contingency if Pgx fails correctness

The **fallback stack is PyTorch + Rust `shakmaty` through PyO3**, with a separately benchmarked batched search/actor architecture. `python-chess` remains an offline oracle and adjudicator only. Because `mctx` is JAX-native, it is **not** retained as the fallback search driver unless a version-verified interop solution exists. A Rust engine may offer fast CPU move generation but will not provide the same on-GPU rollout properties; the fallback's throughput and learning claims must be measured independently.

## 14. Proposed repository layout

This is a **planned layout**, not an existing implementation. `python-chess` must not be imported by `selfplay/`, `search/`, or the JAX network/training hot path; it is limited to the offline parser, differential tests and external game validation/adjudication.

```text
tinychess/
├── README.md
├── pyproject.toml
├── requirements-lock.txt       # exact JAX/CUDA/Flax/Pgx/mctx versions after M0
├── configs/
│   ├── data_human.yaml
│   ├── model_25_50m.yaml       # S0 lower-capacity comparison
│   ├── model_100m.yaml         # primary research model
│   ├── model_600m.yaml         # gated S1 only
│   ├── model_1b.yaml           # optional S2 only
│   ├── search_puct.yaml
│   ├── search_gumbel.yaml
│   ├── ablations.yaml          # B0–B4 mandatory/priority, B5–B7 optional
│   └── scale_gates.yaml        # S1/S2 pass/fail criteria
├── src/tinychess/
│   ├── chess/
│   │   ├── pgx_env.py           # GPU/JAX environment adapter and termination policy
│   │   ├── action_codec.py      # versioned Pgx action index ↔ UCI
│   │   └── state_features.py    # Flax board/history encoding
│   ├── data/
│   │   ├── extract_pgn.py       # offline python-chess PGN parsing only
│   │   ├── build_human.py       # offline preprocessing + Pgx action mapping
│   │   ├── build_teacher.py     # optional, separate Stockfish labels
│   │   ├── manifests.py
│   │   ├── replay_buffer.py
│   │   └── target_eligibility.py # restrict z/Q blend to completed full-search Q
│   ├── model/
│   │   ├── trunk_flax.py        # JAX/Flax transformer
│   │   ├── policy_head_flax.py
│   │   ├── value_head_flax.py
│   │   └── losses.py
│   ├── search/
│   │   ├── mctx_recurrent.py    # Pgx deterministic transition + network value
│   │   ├── puct_mctx.py         # policy wrapper; verify against installed version
│   │   └── gumbel_mctx.py       # policy wrapper; verify against installed version
│   ├── selfplay/
│   │   ├── batched_actors.py    # jax.jit/vmap Pgx states + mctx search
│   │   ├── episode_records.py
│   │   └── arena_jax.py
│   ├── train/
│   │   ├── human_pretrain_jax.py
│   │   └── selfplay_train_jax.py
│   ├── eval/
│   │   ├── puzzles.py
│   │   ├── engine_anchors.py
│   │   ├── gpt_matches.py       # python-chess validates GPT moves
│   │   └── final_adjudication.py# python-chess official game adjudication
│   └── fallback/                # activated only if Pgx M0 fails
│       ├── pytorch_model.py
│       ├── rust_rules/           # Rust crate using shakmaty + PyO3 bindings
│       └── batched_search.py     # separate search path, not mctx by default
├── tests/
│   ├── test_pgx_python_chess_parity.py
│   ├── test_special_moves.py
│   ├── test_draw_semantics.py
│   ├── test_action_codec.py
│   ├── test_mctx_recurrent.py
│   ├── test_value_perspective.py
│   ├── test_full_search_targets.py # fast-search z-only; no invalid-Q blending
│   └── test_game_termination.py
├── benchmarks/
│   ├── bench_pgx_batching.py
│   ├── bench_mctx_search.py
│   └── bench_flax_inference.py
├── data/
│   ├── raw/                    # gitignored
│   ├── processed/              # gitignored
│   ├── replay/                 # gitignored
│   └── manifests/
├── checkpoints/                # gitignored
└── reports/
    ├── m0_rules_parity.md
    ├── evaluation_protocol.md
    ├── run_registry.csv
    ├── scaling_gate_s1.md
    ├── milestone_timeboxes.md
    └── results.md
```

### Planned CLI (illustrative; not implemented)

```bash
python -m tinychess.data.extract_pgn --config configs/data_human.yaml
python -m tinychess.train.human_pretrain_jax --config configs/model_100m.yaml
python -m tinychess.selfplay.batched_actors --config configs/search_puct.yaml
python -m tinychess.train.selfplay_train_jax --config configs/model_100m.yaml
python -m tinychess.eval.engine_anchors --checkpoint checkpoints/latest
```

**Do not treat these commands as working APIs.** Implement and verify each module in the selected environment first. For Pgx/mctx-specific calls and signatures: **verify against installed version**.

## 15. Milestones and exit gates

**Time-box interpretation:** rough **active engineering/research effort** for one researcher with remote access to the Blackwell GPU—not deadlines, guarantees, or measured wall-clock training durations. Compilation problems, data-download bandwidth, CPU/storage limits, queue access, and repeated runs can increase elapsed time substantially. A milestone closes on its **exit criteria**, even if its time box expires; defer optional work rather than weakening correctness checks.

| Milestone | Proposed time box | Required outcome / exit gate |
|---|---|---|
| **M0 — Rules and JAX stack gate** | **4–10 working days** | Pin JAX/Flax/Optax/Pgx/mctx; differential-test Pgx vs `python-chess` for legal moves, castling, en passant, all promotions, checks/checkmate, stalemate, repetition, claimable 50-move/automatic 75-move rules, environment cutoffs, perspective and action mappings. Verify a tiny batched mctx/Pgx recurrent search **against installed version**. Record explicit pass/fail. If unresolved, switch to PyTorch + Rust `shakmaty`/PyO3 and re-run correctness/performance checks. |
| **M1 — Dataset A pilot** | **3–7 working days** | Offline PGN parser, 50K filtered games as a proposed pilot, position/action/WDL manifests, leakage report and valid Pgx-compatible action encoding. |
| **M2 — B0: 100M human warm-start** | **1–3 weeks** | Train Flax/Optax policy + WDL; confirm overfit test, finite losses, held-out performance, working chess games, memory and inference throughput; freeze `human_only` checkpoint. |
| **M3 — Batched GPU self-play integration** | **4–10 working days** | Complete 100 games via Pgx+mctx without `python-chess` in the hot loop; offline outcome checks, coherent policy targets, history and draw semantics, reliable replay and measured warm games/hour. |
| **M4 — B1: PUCT-32 vs B0** | **1–3 weeks** | Self-play/retraining from B0 with `z` value labels, valid full-search policy targets, and fixed-opponent evaluation; report strength and GPU-hours. |
| **M5 — B2: Gumbel-32 vs B1** | **1–3 weeks** | Restart from **the same B0 checkpoint**; verify Gumbel improved-policy target interpretation against installed mctx; compare using matched evaluation and compute conditions. |
| **M6 — B3/B4 controlled ablations** | **1–3 weeks per ablation** | B3 uses `z/Q` **only for completed full-search positions with valid Q**; fast positions remain `z` only. B4 tests a separately controlled human-data decay schedule. Do not combine confounded changes. |
| **M7 — S0: 25–50M size comparison** | **1–2 weeks** | Run a matched smaller-model training/evaluation point; compare strength and games/hour against 100M at equal examples and compute. This can be deferred behind B3/B4 if compute is tight. |
| **M8 — Gated S1: ~600M** | **Pre-gate: 2–4 working days; if approved, exploratory training: 2–6+ weeks** | Write `reports/scaling_gate_s1.md`: M0 green; B1 or B2 demonstrably improves over B0 in fixed anchors; self-play and storage are reliable; actual GPU time/throughput and run budget are known; small-vs-100M capacity trends are analyzed. **If the gate fails, don't launch 600M.** ~1B is optional *after* S1 evidence, not a default milestone. |
| **M9 — Frozen game evaluation** | **4–10 working days plus actual match runtime** | Held-out puzzles, fixed-node Stockfish, previous checkpoints and matched paired-color GPT games; freeze GPT version/effort, illegal-move retry policy and statistical stopping rule before testing. Use `python-chess` for validation/adjudication only. |
| **M10 — Research write-up** | **3–7 working days** | Report all dataset manifests, ablation order/results, model counts, self-play compute, confidence intervals, Pgx correctness, M0 fallback if used, gate decision, and whether the *network-only* or *search-assisted system* outperformed GPT. |

**Optional B5–B7** (playout cap, attention-head comparison, moves-left) are separate budget-approved extensions after B0–B4, with an initial planning allowance of **~1–2 weeks each**, not mandatory milestones. **T1 teacher-data control** and **S2 ~1B scaling** also require separate approvals and measured compute budgets. Time boxes must be revised from measured benchmark results, never represented as promised completion times.

**Hard stops:** An M0 mismatch cannot be waved through; fast-search `Q` cannot enter blended value supervision; and a model-size jump cannot be justified by VRAM capacity alone.

## 16. Failure modes and safeguards

| Failure mode | Diagnostic | Response |
|---|---|---|
| Large-model scaling uses most available compute but few games generated | Games/hour, inference batch utilization and S1 gate | Stay at 100M, improve batching; scale only if the gate passes |
| Self-play never improves | Arena score vs human-only checkpoint | Audit rewards, Q sign, search targets and exploration |
| Draw loops or perpetual shuffling | Draw frequency, game length and repetition types | Fix adjudication, search diversity; test moves-left only afterward |
| Learns human-timeout artifacts | WDL calibration by termination | Filter/downweight clock losses |
| Value signs are reversed | Manually verified test games | Enforce side-to-move perspective tests at every label boundary |
| Model only excels with search | No-search vs MCTS rating gap | Report both honestly; optimize no-search policy separately if desired |
| Data leakage through popular openings | Per-position overlap and opening audits | Stratified, position-disjoint evaluation subset |
| Gumbel implementation quietly uses PUCT targets | Unit tests and search policy comparisons | Verify installed mctx policy output and correct Gumbel-improved targets |
| Fast-search Q contaminates value labels | Check `full_search`, actual simulations, Q validity, and target-provenance masks | Blend z/Q **only** for completed full-search states; train `z`-only on valid fast-search outcomes |
| 600M run starts before 100M results justify it | Review S1 gate memo and measured GPU-hour budget | Block S1 until the gate is signed off; run S0 / 100M studies instead |
| Pgx disagrees on castling/en passant/promotions | M0 differential move-set + transition tests | Stop self-play; fix and revalidate or switch to Rust/PyO3 fallback |
| Pgx training draw/truncation differs from final adjudication | Compare repetition, no-progress and maximum-ply outcomes | Define tested policy, mark truncation separately; fall back if unfixable |
| Pgx and model action indices disagree | Legal mask / UCI round trips / played-action assertions | Pin an action codec and version, block training until correct |
| JAX compilation or search tree storage dominates | Compile time, HBM peaks, games/hour by batch size | Adjust batch/tree shape and scheduling; remeasure |
| Silent fallback contaminates cross-run comparison | Backend manifest and run provenance | Never mix backends without labeling and fresh benchmarks |
| Comparing unfair GPT conditions | Prompt/tool/search budget registry | Freeze assistance and retry rules; disclose mismatch |
| Repeated GPT tests create false-positive claims | Evaluation run log, held-out suite | Use prespecified sequential testing and final confirmation |

---

## 17. References

### Data and implementation

- **Lichess public database** — https://database.lichess.org/ (monthly rated games and puzzles; CC0).
- **Pgx** — https://github.com/sotetsuk/pgx and https://www.sotets.uk/pgx/chess/ (GPU-batched chess rules; exact chess environment version and termination semantics must be verified).
- **DeepMind mctx** — https://github.com/google-deepmind/mctx (batched JAX MCTS; PUCT-style and Gumbel search; chess recurrent integration is ours).
- **JAX documentation** — https://docs.jax.dev/ (GPU compilation, batching and random number handling).
- **Flax documentation** — https://flax.readthedocs.io/en/stable/ (JAX/Flax chess transformer).
- **Optax documentation** — https://optax.readthedocs.io/en/latest/ (JAX optimizer components).
- **python-chess documentation** — https://python-chess.readthedocs.io/en/latest/ (offline PGN parsing, test oracle, GPT-match validation and final adjudication **only**).
- **`shakmaty`** — https://github.com/niklasf/shakmaty and **PyO3** — https://pyo3.rs/ (conditional Rust chess core with PyTorch fallback).
- **Stockfish** — https://github.com/official-stockfish/Stockfish (optional teacher baseline and independent playing-strength anchors).
- **Leela Chess Zero training code** — https://github.com/LeelaChessZero/lczero-training (policy/value/moves-left and attention-policy precedent).

### Research background

- **Silver et al., AlphaZero, 2018** — https://doi.org/10.1126/science.aar6404 (self-play policy and value learning; MCTS).
- **Danihelka et al., “Policy Improvement by Planning with Gumbel,” ICLR 2022** — https://openreview.net/forum?id=bERaNdoegnO.
- **Google DeepMind `mctx`** — https://github.com/google-deepmind/mctx (batched JAX search, verified integration with Pgx required).
- **David J. Wu, “Accelerating Self-Play Learning in Go,” 2019** — https://arxiv.org/abs/1902.10565 (KataGo efficiency methods, including playout cap randomization).
- **DeepMind Searchless Chess, 2024** — https://arxiv.org/abs/2402.04494 (**reference/baseline only**, not the prescribed main methodology).

---

## Project principles

1. **Chess strength is measured in games.** Move imitation loss and puzzle pass rates are diagnostics.
2. **Reproduce a working baseline before optimizing it.** Start simple; change one factor at a time.
3. **Count all compute.** GPU training, self-play inference, CPU labeling and GPT evaluation calls all matter.
4. **Separate supervision sources.** Never blur engine-free self-play with Stockfish-trained baselines.
5. **Publish honest comparisons.** State when performance uses inference-time search, legal-move assistance or external tools.
6. **Scale based on evidence.** **100M is the main model**; 25–50M is a scaling comparison; 600M and 1B require explicit gates.
7. **Prioritize interpretable experiments.** Finish B0 → B1 → B2, then B3/B4; B5–B7 stay optional.
8. **Blend only trustworthy search targets.** Root-Q/value blending is restricted to completed full-search positions with valid Q; fast-search positions use outcome-only values.

**Original pre-implementation plan status: v0.4 methodology specified; Pgx/mctx compatibility, chess correctness, compilation performance, benchmarks and trained checkpoints remain unverified/pending. All time boxes are planning estimates, not measured durations.**

## M0 implementation status

The M0 adapter, action codec, differential tests, dummy mctx integration and
batching benchmark are implemented. Native Pgx requires explicit corrections;
see [the M0 parity report](reports/m0_rules_parity.md) for rules policy, actual
CPU/GPU results, upstream defects and verified Molab Blackwell evidence.
The research plan above remains unchanged. Dataset generation/training/RL have
not begun; the final CPU suite passed 75 tests, and the prior Blackwell suite
passed 69 tests on unchanged environment/search code. All four supplied
conditional-review requirements are addressed; see the commit verification report.

```bash
uv sync --locked --extra test
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/bench_pgx_batching.py --output reports/pgx_batching_cpu.json
```

Molab CUDA setup and every individual test command are in the M0 report.


## GitHub handoff

Repository: [IshaGarg1103/gambit-zero](https://github.com/IshaGarg1103/gambit-zero).
The M0 implementation is prepared on `pgx-mctx-infrastructure`. Final checks and
remaining review limitations are recorded in
[the commit verification report](reports/m0_commit_verification.md).
Push the reviewed branch with:

```bash
git push -u origin pgx-mctx-infrastructure
```

This command does not merge into `main`. Independent review and acceptance of
the declared draw/truncation convention remain distinct from passing tests.


### Adopted M0 training rules

The training convention is `tinychess-auto-fide-pgx-v1`: players do not
voluntarily claim draws. Threefold/50-move claimability is audited separately;
fivefold repetition and 75 moves (150 halfmoves) automatically draw, with
checkmate taking precedence. The 512-ply episode cap is a **truncation**, never
a draw or an outcome-label source. Use
`tinychess.chess.adjudication.final_wdl` only on the actual terminal transition;
it refuses ongoing/truncated states and missing decisive rewards. Future episode
exporters must use this checked boundary before storing any final WDL labels.
These conventions address the supplied review conditions without starting M1.
