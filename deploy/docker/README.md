# YUE2 // GROOVE in Docker (Linux + NVIDIA)

One image, three uses: a desktop or server with an NVIDIA GPU (Linux, or Windows through
Docker Desktop / WSL2), a RunPod pod, and the launcher that creates such pods. It is the
environment `docs/LINUX_CUDA.md` validates — `torch 2.10.0+cu128`, `yue2-infer 0.1.6`, the
SheetSage2 Cover environment — baked once. **Code only:** the YuE2, SheetSage2 and
MERT-v2-FullSong weights are CC BY-NC 4.0 and are downloaded by you at first start into
`/data` (mount it so that happens once).

Not for macOS: Docker Desktop cannot pass Metal/MPS into a Linux container — Mac users take
the Pinokio or native path.

## Requirements

- NVIDIA GPU: 16 GB comfortably runs everything the app can produce; 12 GB runs the
  unquantized model at CFG 1.0 or for shorter songs (`docs/LINUX_CUDA.md` §3).
- A driver that supports CUDA 12.8 (R570 or newer on Linux; the matching WSL driver on Windows).
- Docker with GPU support: NVIDIA Container Toolkit on Linux; Docker Desktop on Windows with
  WSL2 (GPU support is built in).
- ~25 GB of disk: the image (~15 GB) plus ~8 GB of weights in `/data`.

## Run

```bash
docker run --gpus all -p 7860:7860 \
  -v groove-data:/data \
  -e YUE2_GROOVE_AUTH=alice:my-secret \
  ghcr.io/deadjoe/yue2_groove:latest
```

Then open http://localhost:7860. The first start downloads the weights (`weights` step in the
log), verifies their hashes and the GPU (`verify`), and starts the app (`start` → `ready`).
Works and downloaded weights live in the `groove-data` volume (`/data/runs`, `/data/hf`,
`/data/models`) and survive image updates. `docker compose` equivalent:

```yaml
services:
  groove:
    image: ghcr.io/deadjoe/yue2_groove:latest
    ports: ["7860:7860"]
    volumes: ["groove-data:/data"]
    environment:
      YUE2_GROOVE_AUTH: alice:my-secret
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
volumes:
  groove-data: {}
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `YUE2_GROOVE_AUTH` | (none) | `user:password` for the web UI — set it on anything reachable from a network |
| `YUE2_GROOVE_PORT` | `7860` | port the app listens on inside the container |
| `HF_HOME` | `/data/hf` | Hugging Face cache (YuE2 weights, MERT snapshot) |
| `YUE2_GROOVE_RUNS` | `/data/runs` | where works are stored |
| `YUE2_GROOVE_MODELS` | `/data/models` | SheetSage2 snapshot directory |
| `GROOVE_PROGRESS_URL`, `GROOVE_PROGRESS_TOKEN` | (none) | when set, every start step is POSTed as JSON to that URL with a bearer token — used by the pod launcher, ignored otherwise |

## On RunPod

Create a pod from this image with port `7860/http` exposed; RunPod's proxy URL
`https://<pod-id>-7860.proxy.runpod.net` is the app. The base image's `/start.sh` stays the
entrypoint (sshd with `PUBLIC_KEY`, Jupyter with `JUPYTER_PASSWORD`, as on any RunPod pod);
it calls `/pre_start.sh`, which starts `groove-start`. Weights download from Hugging Face in
about 1.5 minutes inside RunPod's datacenters; a network volume mounted at `/data` makes it
instant on later pods in the same datacenter.

## Progress protocol (for launchers)

`groove-start` reports `{"step", "status", "message", "ts", "pod"}` with
`step ∈ {weights, verify, start, ready}` and `status ∈ {started, done, failed}`; `ready/done`
carries the URL in `message` (the RunPod proxy URL when `RUNPOD_POD_ID` is set, else
`http://localhost:<port>`). A `failed` status ends the start; the container stays up so the
log can be read. The same lines are printed to the container log.

## Build

`.github/workflows/image.yml` builds and pushes `ghcr.io/deadjoe/yue2_groove` on every `v*`
tag (also tagged `latest`) and on manual dispatch (tag `main` by default). Local build:

```bash
docker build -f deploy/docker/Dockerfile -t yue2-groove .
```
