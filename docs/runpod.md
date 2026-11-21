# Rendering on RunPod

`--runpod` rents a GPU pod, puts your model files on it, renders, and brings
the images back. You need a RunPod account with credit and an API key.

```bash
# .env
RUNPOD_API_KEY=...
HF_TOKEN=...            # optional: private or gated Hugging Face files
JENA_TARGET=runpod      # optional: render on RunPod without --runpod
```

## What happens

1. **Start.** If no pod of ours is running, one is created from the image in
   [`pod/`](../pod) (`ghcr.io/jenalab-com/jena-image-render-pod`). Community
   Cloud is tried first (cheaper, no SLA); if no GPU is free or the host does
   not start the container within 5 minutes, it is terminated and Secure Cloud
   is used (up to 3 hosts, 8 minutes each). A pod that fails to start is
   always terminated.
2. **Files.** The pod downloads URL files itself (the built-in models, files
   a job names by address, files found in your storage) while files that exist
   only on your computer (the models folder) are uploaded. Both are
   skipped when the pod already has them (verified by SHA-256, or by URL when
   no hash is given). `--runpod --dry-run` shows which file goes which way.
3. **Render.** Images are rendered one by one with live step progress and
   saved locally as they finish.
4. **Stay or stop.** The pod stays up for your next run and **terminates itself
   after 20 minutes without activity**, and in any case after `--max-hours`
   (default 3). `jenair runpod down` stops it now; Ctrl+C during a
   run stops it immediately.

## Your own storage

Uploading big files from home to each new pod is slow. Set a private Hugging
Face repo as storage and local-only files are uploaded there **once**, before
a pod is rented; every pod then downloads them at datacenter speed:

```bash
# .env
JENA_STORAGE=hf://your-name/your-models   # created private if it does not exist
HF_TOKEN=hf_...                            # write access; also lets the pod read it
```

What was uploaded is remembered by SHA-256 in `.jena/storage.json`, so a file
is uploaded again only when it changes; on another computer the repo itself is
checked first, so the same file is not sent twice. A name that is neither built
in nor in your models folder is looked up in the repo too. `jenair push JOB.json`
uploads a job's files without starting a pod. Details: [models.md](models.md#your-own-storage-runpod).

## Security

The pod is protected by a random password created for each pod and stored in
`.jena/session.json` (readable only by you). Port 3000 is the only one exposed.

## Choosing GPUs

Optional `runpod.json` in the project folder (git-ignored):

```json
{
  "community": ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"],
  "secure": ["NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA L4"],
  "host_memory_gb": [62, 48, 32],
  "cuda_versions": ["12.6", "12.7", "12.8", "12.9", "13.0", "13.1", "13.2"],
  "idle_minutes": 20,
  "disk_gb": 100
}
```

GPUs are tried in order; the whole list is tried for each host-RAM tier.
Names are RunPod's GPU type IDs. `qwen` offloads about 32 GB of weights to
system RAM, hence the RAM tiers. The image is built on CUDA 12.6: hosts with
newer drivers are fine, so add new CUDA versions as RunPod hosts upgrade —
otherwise those hosts look like "no stock".

## Costs, roughly

GPU prices change; at the time of writing a Community RTX 3090 / A5000 was
about $0.2–0.3 per hour. The first run on a fresh pod spends a few minutes
starting the container and downloading models (tens of GB for `qwen`); later
runs on the same pod start rendering at once. Check the RunPod console for
current prices, and `jenair runpod status` for your balance.

## Availability tips

- Cheap Community GPUs are scarce during US daytime; `--secure` goes straight
  to datacenter GPUs.
- If nothing starts, `runpod status` shows whether a pod is left over;
  `runpod down` cleans up everything this tool created (pods named
  `jena-render-…`), never your other pods.

## The pod image

[`pod/`](../pod) builds a small image: PyTorch + CUDA 12.6, ComfyUI v0.38.0
(unmodified), Caddy for basic auth, and the `jena_fetch` extension
(`/jena/fetch` downloads, `/jena/upload` chunked uploads, idle self-termination).
No keys, passwords or weights are baked in. `JENA_POD_IMAGE` in `.env` points
the tool at your own build.
