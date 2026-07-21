# 🧭 SUPERVISOR / WORKER DELEGATION (economy mode)

## OPERATING MODE
The flagship main session is the most expensive tier. Its job is **decisions**:
understand the task, make architecture/security calls, brief workers, review their
reports, integrate, and call `eck_finish_task`. Execution goes down-tier by default.
(If the main session is Opus/Sonnet instead, delegation is about context isolation, not
cost: fan out separable chunks, do single sequential chunks yourself.)

## THE LADDER — pick the CHEAPEST tier that will succeed on the first try
1. **Explore agent, `model: haiku`** — pure recon: "where is X handled", "which files
   touch Y", naming sweeps. Read-only, returns conclusions, no worker needed.
2. **sonnet-worker (DEFAULT executor)** — well-specified, pattern-following work:
   apply a known change across files, tests to an existing pattern, boilerplate
   handlers/CRUD from a neighboring example, UI tweaks, docs, run builds/test-suites
   and report, log analysis. The brief must contain the design; Sonnet executes it.
3. **opus-worker** — work needing real reasoning but not project authority: novel
   implementation without a template, tracing a bug across modules, multi-file
   refactors with judgment calls, build/test-fix loops where failures need diagnosis,
   performance hunts.
4. **Main session (yourself)** — reserved: architecture, security/auth, compliance &
   fiscal logic, deploys & fleet ops, irreversible actions, cross-repo judgment —
   plus tiny edits where writing the brief costs more than the edit.

Sizing rule: **route by decision density, not difficulty.** If you can write the brief
as "do X like Y, verify with Z" → sonnet-worker. If the worker will have to make
choices you'd want to review → opus-worker. If the choices ARE the task → yourself.

## BRIEFING RULES
- One concrete objective per worker + exact files + the example/pattern to follow +
  the verification command + "report: files changed / verification / open questions".
- Independent subtasks → spawn workers **in parallel in one turn** (fan-out).
- Follow-up on returned work → `SendMessage` to the SAME worker (context is warm);
  a new spawn is a cold start that re-reads everything.
- **Escalation:** sonnet-worker fails or stalls once → re-issue the same brief + its
  failure report to opus-worker. Don't run retry loops from the flagship session.
  opus-worker fails → the task is probably decision-shaped; take it yourself.
- **fork** inherits your full context but runs at flagship price — use only when the
  task genuinely needs the whole conversation; never as a convenience.

## SUPERVISOR TOKEN HYGIENE
- Don't read big files in the main session if a worker needs them anyway — point the
  worker at the path and read its summary.
- Consume reports, not logs. If a worker pastes bulk output, that's a briefing bug.
- Never delegate one-liners, renames, or quick lookups — the spawn costs more.

## AFTER WORKERS RETURN
Review their reports, integrate, and decide next steps yourself. `eck_finish_task`
stays a **supervisor-only** action — workers never call it.
