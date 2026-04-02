# /harden — Adversarial TDD Hardening

Given a target module or function $ARGUMENTS (defaults to the last modified source file),
harden it via adversarial TDD:

## Phase 1 — Spawn adversarial agents in parallel

Launch 3 adversarial sub-agents, each with a different attack angle:

- **Agent A — API contract breaker**: probe the public API for misuse, wrong types,
  missing guards, unexpected kwarg combinations, empty inputs, single-element inputs.
- **Agent B — JAX execution model breaker**: probe interactions with `jax.jit`,
  `jax.grad`, `jax.vmap`, tracing vs eager, tracer leakage, static vs dynamic arg
  violations, re-tracing behaviour.
- **Agent C — Runtime environment breaker**: probe caching edge cases, concurrent
  calls, device mismatches, cache corruption, env var overrides, missing cache dir.

Each agent must:
1. Read the target source thoroughly.
2. Produce a list of concrete breaking scenarios — each with a one-line description
   and the exact call sequence that triggers it.
3. Write pytest tests to a temporary file `tests/test_harden_<angle>.py`. Tests must be
   runnable and either assert the error is raised or pin the correct behaviour. Mark
   genuinely unsupported cases with `pytest.mark.xfail`.

## Phase 2 — Consolidate into existing test files

Collect all tests from the three agent files. Remove exact duplicates (same root cause
tested multiple times — keep the most specific version). Add them into the appropriate existing test files. Only if there is really no good existing home for a test consider opening a new category - with great power comes great responsibility - so make sure it's really needed.

Delete the three agent temp files. Run `uv run pytest tests/ -v` — all must pass.

## Phase 3 — Implement to green

Fix failures one at a time, running `uv run pytest tests/ -v` after each fix.
Do not move to the next fix until the current one is green.

## Phase 4 — Final report

Print a summary table:
| Scenario | Outcome |
|---|---|
| <description> | Fixed / xfail / deleted |

The target for this session is: $ARGUMENTS
