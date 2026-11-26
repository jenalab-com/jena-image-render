# jena-image-render

**ComfyUI without the wires.** Write prompts in a JSON file, and render whole
batches across several models — on your own ComfyUI, or on a RunPod GPU that
starts when you need it and stops itself when you're done.

```json
{
  "output": "model-compare",
  "defaults": { "model": ["zimage", "krea2", "qwen", "klein-4b"], "seeds": [1, 2] },
  "jobs": [
    { "name": "cafe",   "prompt": "A quiet cafe at dawn, warm window light, a cat asleep on a chair." },
    { "name": "street", "prompt": "A rainy neon street at night, cinematic, reflections on the ground." }
  ]
}
```

```bash
jenair configs/basics/03-compare-models.json
```

2 prompts × 4 models × 2 seeds = **16 images**, each saved with a JSON file of
the exact settings that made it. No nodes, no wires, no clicking through a
graph sixteen times.

- **Prompt-only jobs.** Steps, CFG, sampler and size come from tested defaults
  for each engine (the official ComfyUI templates). Override only what you mean to.
- **Combinations for free.** Put a list in `model` or `seeds` and every
  combination is rendered.
- **Checked before you pay.** `--dry-run` validates models, files, LoRA/engine
  matches and output paths without starting a GPU.
- **Resumable.** Images already saved are skipped, so an interrupted batch
  continues where it stopped.
- **Bring your own models.** Drop a file in `models/` and use its name. The
  official releases are built in as links; no weights are bundled.

## Supported engines

| engine | model family | files | default size |
|---|---|---|---|
| `zimage` | Z-Image Turbo | `diffusion`, `text_encoder`, `vae` | 832×1248, 9 steps |
| `krea2` | Krea 2 Turbo | `diffusion`, `text_encoder`, `vae` | 1024×1024, 8 steps |
| `qwen` | Qwen-Image 2.1 | `diffusion`, `text_encoder`, `vae` | 832×1248, 25 steps |
| `klein` | FLUX.2 klein 4B / 9B | `diffusion`, `text_encoder`, `vae` | 832×1248, 4 steps |
| `illustrious` | any SDXL-architecture checkpoint (Illustrious, NoobAI, SDXL) | `checkpoint` | 896×1152, 28 steps |

`jenair engines` prints the same table. Size shorthands
(`"aspect": "2:3"` etc.) work with every engine.

## Quick start

Needs a ComfyUI you can reach (your own computer or another
machine on your network). No NVIDIA card? See [RunPod](#no-gpu-rent-one-on-runpod)
below.

```bash
git clone https://github.com/jenalab-com/jena-image-render
cd jena-image-render
./init.sh
```

`init.sh` finds Python (installing it if it is missing), creates `./.venv`,
installs the `jenair` command into it, copies `.env.example` to `.env` and
checks the first example without rendering. Nothing is installed system-wide,
and it is safe to run again. After that, either activate the environment or let
`init.sh` pass your arguments on:

```bash
source .venv/bin/activate && jenair configs/full.json --dry-run
./init.sh configs/full.json --dry-run        # the same, without activating
```

<details><summary>By hand, or on Windows</summary>

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
jenair configs/basics/01-one-image.json --dry-run
```

</details>

### On your own ComfyUI (the default)

Install ComfyUI (tested with **v0.38.0**, see [docs/local.md](docs/local.md)),
then:

1. Start ComfyUI.
2. Run a job:

```bash
jenair configs/basics/01-one-image.json
```

When ComfyUI runs on this computer, `jenair` asks it where its model folders
are and puts what the job needs there: the built-in models are **downloaded on
the first run** (Z-Image Turbo is about 20 GB, so the first run takes a while;
later runs start at once), and your own files in `./models` are linked in.
Models ComfyUI already has are used by name — nothing is moved or copied.
`--dry-run` shows what would be downloaded or linked, and `--no-download`
turns downloads off.

A ComfyUI on another machine is used by name only; its files must already be
in that machine's `models/` folder:

```bash
jenair configs/basics/01-one-image.json --local http://192.168.0.10:8188
```

### No GPU? Rent one on RunPod

If you have no suitable card, the same job files run on a rented GPU. Put a
RunPod API key in `.env` (`cp .env.example .env`, then `RUNPOD_API_KEY=...`) and
ask for it — a GPU is never rented unless you say so:

```bash
jenair configs/basics/01-one-image.json --runpod
jenair runpod status     # balance, pods, current session
jenair runpod down       # stop the pod now
```

`"target": "runpod"` in a job file, or `JENA_TARGET=runpod` in `.env`, does the
same without the flag. The first run rents a GPU (a cheap Community host first,
Secure Cloud if that fails); the pod downloads the built-in models itself and
receives your own files by upload (or from your storage), then renders. The pod
**stays up for your next run and terminates itself after 20 idle minutes** (and
after `--max-hours`, default 3). Ctrl+C terminates it immediately. Details and
costs: [docs/runpod.md](docs/runpod.md).

## UI

Prefer forms to JSON? `jenair ui` opens a page in your browser for writing and
rendering job files:

```bash
jenair ui            # http://127.0.0.1:8765/, on this computer only
```

Each tab is a set of settings with its prompts (a model with its LoRAs, size,
seeds and sampling, and as many prompts as you like); the fields show each
engine's defaults, so you fill in only what you change. The Render side picks
Local or RunPod, connects (for RunPod with a start-up progress bar, then the
time left until the pod stops itself), checks what would happen, and renders
every tab in one go, with progress and thumbnails. Files are saved as ordinary
job files in `jobs/` — `jenair jobs/<name>.json` renders them too — and any job
file opens in it. *Browse…* picks a model or LoRA file anywhere on this computer
and links it into `models/`, and ⚙ Settings takes your RunPod key and tokens
(it writes `.env`, so you need not edit it). Details and safety:
[docs/ui.md](docs/ui.md).

## Job files

```json
{
  "output": "cafe-test",
  "defaults": { "model": "zimage", "seeds": [1, 2, 3, 4], "aspect": "2:3" },
  "jobs": [
    { "name": "cafe",   "prompt": "a quiet cafe at dawn, warm window light" },
    { "name": "street", "prompt": "a rainy neon street at night", "model": "krea2" },
    {
      "name": "street-anime",
      "model": "illustrious",
      "prompt": "1girl, rainy neon street, night, umbrella",
      "negative": "lowres, bad hands",
      "loras": [{ "name": "illustrious-niji", "scale": 0.8 }]
    }
  ]
}
```

Images land in `outputs/<output>/<model>/<name>_<seed>.png`, each with a
`.json` next to it. Several groups of jobs with their own settings go in
`"batches"` (one sub-folder each; the UI shows them as tabs). All fields:
[docs/config.md](docs/config.md).

**Start from the examples.** [`configs/full.json`](configs/full.json) shows
everything in one file — shared `defaults`, jobs that override them, LoRAs,
several models. [`configs/`](configs) also has one small file per idea
(a batch, model comparison, LoRAs, sizes, sampling, `target`, files by
address, `batches`), one per engine,
and examples for your own checkpoints and LoRAs — each with a `_note` saying
what it shows and what to change. Index: [configs/README.md](configs/README.md).

## Your own models

Two rules. A file **on your computer** goes in the models folder and is used
by its name:

```
models/                       ← ./models (ComfyUI layout)
  checkpoints/               my-anime.safetensors
  diffusion_models/zimage/   my-finetune.safetensors
  loras/                     my-style.safetensors      (or loras/zimage/my-style.safetensors)
```

```json
{ "target": "runpod", "model": "my-anime", "loras": ["my-style"], "prompt": "1girl, cafe" }
```

A file in `checkpoints/` is an SDXL-type checkpoint (`illustrious` engine). A
file in `diffusion_models/<engine>/` is a fine-tune of that engine and keeps
its official text encoder and VAE. A LoRA in `loras/<engine>/` is checked
against that engine; anywhere else in `loras/` it is used with any model.
Names are file names or stems, in any sub-folder. With ComfyUI on this
computer, names are also found in ComfyUI's own model folders (with the same
rules), so a model you already keep there needs no copy here.

A file **online** is written into the job by its address — nothing to
download first:

```json
{
  "model": "hf://OnomaAIResearch/Illustrious-XL-v1.0/Illustrious-XL-v1.0.safetensors",
  "loras": [{ "name": "https://my-storage.example.com/my-style.safetensors", "scale": 0.8 }],
  "prompt": "1girl, cafe"
}
```

An address is `hf://owner/repo[@revision]/file` or any `https://` URL. A model
can also be an object, `{ "engine": "zimage", "diffusion": "hf://…", "name": "my-z" }`,
that swaps just the files you name. See
[`configs/basics/08-from-url.json`](configs/basics/08-from-url.json).

On RunPod, files that are only on your computer are uploaded to the pod — or,
with `JENA_STORAGE=hf://you/your-models` in `.env`, uploaded **once** to your
private Hugging Face repo and fetched from there by every later pod. Any
computer with the same storage finds those files by name.
[docs/models.md](docs/models.md) has the details, the built-in models and
their licenses, private Hugging Face repos and Civitai files.


## Licenses

The code is MIT. The models are not part of this project: each model has its
own license (some are non-commercial or gated) — check it before you use the
results. ComfyUI is GPL-3.0; this tool talks to it over its HTTP API, and the
RunPod image in [`pod/`](pod) installs it unmodified from the official
repository.

Part of the **jena** toolkit.
