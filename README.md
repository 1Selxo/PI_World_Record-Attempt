# Resumable Pi computation

This project computes decimal digits of π with the Chudnovsky formula and GMP
binary splitting.  It is designed to make forward progress on short-lived
GitHub Actions runners: an interrupted run resumes its current calculation
from a verified chunk checkpoint instead of restarting the target.

## Build and run locally

Install GMP and an OpenMP-capable C compiler, then build:

```sh
make
# or, without make:
gcc -O3 -fopenmp -Wall -Wextra -Wpedantic pi.c -o pi -lgmp
./pi
```

The optional first argument selects the first target digit count.  With no
argument, the program resumes from `state.json`.

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `PI_MAX_RUNTIME_SECONDS` | Stop cleanly at the next checkpoint boundary after this many seconds. |
| `PI_CHUNK_TERMS` | Number of Chudnovsky terms in each resumable work chunk (defaults to 100,000). |
| `PI_MAX_CHUNKS_PER_RUN` | Maximum chunks processed in one invocation; useful for testing. |
| `PI_STATE_FILE` | Override the JSON progress manifest path. |
| `PI_OUTPUT_FILE` | Override the completed decimal-output path. |
| `PI_CHECKPOINT_DIR` | Override the directory holding the binary in-progress checkpoint. |
| `PI_CHECKPOINT_FILE` | Override the checkpoint manifest path (for compatibility). |

For example, this is a small, deliberately interrupted local slice:

```sh
PI_MAX_CHUNKS_PER_RUN=1 PI_CHUNK_TERMS=1000 ./pi 100000
```

Run the same command again to resume it.  The program only advances the
completed-digit count after the complete decimal output has been atomically
written and validated. Each invocation stops after completing one target;
run it again to start the next target (the GitHub workflow does this
automatically).

## Progress files

- `state.json` is the human-readable, Git-tracked manifest.  It records the
  latest completed result and the in-progress target.
- `pi_checkpoint/` is an ignored, atomic GMP checkpoint for the current
  target.  Its small manifest points to immutable partial-result tuples, so a
  new chunk does not rewrite all previously completed work.
- `pi_c_hyperspeed.txt` is an ignored decimal result emitted only after a
  target completes.

Temporary files are written beside their final file and renamed only after a
successful write, so an interrupted write leaves the last valid checkpoint in
place.

## GitHub Actions

The workflow runs a 20-minute compute slice with 100,000 terms per
checkpoint boundary. Its 90-minute job ceiling leaves room for an occasional
large GMP merge and persistence. It commits only `state.json`, restores/saves
only the binary computation checkpoint, and uploads each completed decimal
result as a GitHub Release asset *before* committing the advanced state. This
avoids the much smaller Actions-artifact retention quota at record scale. A
failed compute or result upload still saves a valid checkpoint for retry, but
never advances the Git-tracked state. The workflow uses the built-in GitHub
token to request the next slice after every cached checkpoint, including a
recoverable failed attempt. It therefore runs continuously without manual
restarts; the 15-minute cron schedule is only a recovery fallback.

After an output is written, its final math checkpoint is retained just long
enough for the cache/release/state transaction. The next invocation removes
that now-stale checkpoint before starting the next target.

The workflow cache is a continuation mechanism, not long-term archival.
Completed outputs remain as versioned assets on the `pi-results` GitHub
Release; download or mirror them if you need an independent archive.

At record-scale sizes, GitHub Actions cache snapshots can become expensive and
are subject to repository storage limits.  For very large targets, use durable
object storage or a self-hosted runner with persistent disk for the checkpoint
directory.
