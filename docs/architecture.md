# Architecture and design decisions

## Pipeline

```mermaid
flowchart LR
  V[Video / frame stream] --> P[Preprocess<br/>decode, resize, normalise]
  P --> D[Detector<br/>EventNet CNN, ONNX Runtime]
  D --> A[EventAggregator<br/>runs of frames -> events]
  A --> E[(Event stream<br/>JSON: timestamp, type, confidence, bbox)]
  E --> F[StreamFeaturizer<br/>1 s bins]
  F --> T[Temporal model<br/>1D-CNN autoencoder]
  T --> X[Irregular-play flags]
```

Serving wraps the same stages: `POST /detect` (single frame, micro-batched), `WS /stream`
(frames in, events and flags out), `POST /analyze` (finished event stream).

## ADR-1: synthetic arena footage instead of real VODs

*Context.* Real esports footage is licence-encumbered, large, and has no event labels. *Decision.*
`common/sim.py` renders a seeded top-down arena match with a HUD (kill-feed banner, round
start/end banners, distractor HUD elements) to mp4 and writes exact ground truth, including injected
irregular play. *Consequences.* Every number in this repo is reproducible and exactly labelled, and
the full pipeline runs on a laptop. The cost is realism: the task is far easier than a real game
HUD, so absolute accuracy figures do not transfer. The code paths (decode, classify, aggregate,
featurize, score, serve) are the real ones. Swapping in real footage means adding annotations in
the same JSON schema.

## ADR-2: frame classifier + event aggregator, not an object detector

Events here are HUD banners, so they are frame-level: a banner is on screen or it is not. A small
CNN classifies each frame into `background | kill | round_start | round_end`, and
`EventAggregator` turns runs of identical predictions into one event stamped at the run start
(`min_run` frames of agreement suppress single-frame blips). Banner position is then looked up from
the HUD template (`bbox`) and the kill team is read from the banner colour. For events that are
objects on the map (a goal, a player), replace `EventNet` with YOLO/RT-DETR; the aggregator and
everything downstream are unchanged.

## ADR-3: ONNX Runtime for serving, no PyTorch in the image

Both models export to ONNX with a dynamic batch axis (`serving/onnx_export.py`, which also checks
ONNX output against PyTorch). The service depends on onnxruntime, numpy and OpenCV only. Session
threads are set to 1 intra-op so concurrency comes from batching and the event loop, not from
oversubscribing cores. TensorRT is the next step when a GPU is available.

## ADR-4: micro-batching

Concurrent `/detect` calls and concurrent streams share one `MicroBatcher`: it waits up to
`max_wait_ms` (default 3 ms) or `max_batch` items, then runs one batched ONNX call in a worker
thread. This trades a few ms of queueing for throughput under load; at concurrency 1 the wait is
the cost to measure (`benchmarks/latency_bench.py http`).

## ADR-5: temporal model = reconstruction-based anomaly detection

Irregular play has no abundant labels, so the autoencoder is trained only on windows that overlap
no annotated anomaly (a "baseline of normal play"), and the window's reconstruction error is its
irregularity score. The alert threshold is the 99th percentile of scores on held-out normal
windows. Features per second: kill count, round start/end, in-round flag, time since last kill
(capped). Two anomaly types are injected: *bursts* (6-8 kills at about 1 s spacing) and
*droughts* (a 45-60 s round with no kills). A rule baseline (most kills in 8 s) is reported next to
the model; it cannot see droughts, which is the point of learning the baseline instead.

`StreamFeaturizer` is shared by training and serving, with a 0.5 s lag live so that events the
detector emits a few frames late still land in the right bin.

## Data split (match indices from `make data`)

| Use | Matches |
|---|---|
| Detector train / validation | 0-39 / 40-49 |
| Temporal train (normal windows) / threshold calibration | 0-59 / 60-69 |
| Temporal test (held out) | 70-99 |

The temporal model trains on events the detector produced (`make infer`), so detector errors
propagate into it as they would in production; `--use-gt` trains on ground truth for comparison.
Matches 70-99 are never seen by either model's training.

## Limitations

- Synthetic data; see ADR-1.
- The kill-team and `bbox` fields come from a fixed HUD template, not a learned localiser.
- Anomaly types are the two injected ones; the model flags anything unlike normal windows, but only
  those two are measured.
- Latency numbers are CPU-only.
