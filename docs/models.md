# Models

This project ships **no model weights**. Two rules cover your own files:

- a file **on your computer** goes in the [models folder](#the-models-folder)
  and is used by its name;
- a file **online** is written into the job by its [address](#addresses).

A `model` name is looked up in this order:

1. the **built-in catalog**: the official release of every engine and a few
   public example LoRAs, as pinned Hugging Face links;
2. `checkpoints/` in the models folder, then `diffusion_models/<engine>/`;
3. with a ComfyUI on this computer (local target), ComfyUI's own model
   folders, by the same folder rules ([local.md](local.md));
4. your [storage](#your-own-storage-runpod), when `JENA_STORAGE` is set, by the
   same folder rules;
5. otherwise it is an error that says where the file could go.

A LoRA name goes through the catalog, `loras/`, ComfyUI's folders and the
storage the same way.
So the examples in `configs/` run on a fresh clone, and your own file needs
nothing more than being put in the folder.

## The models folder

`./models` by default (`JENA_MODELS_DIR` in `.env` moves it; usually not
needed). Same layout as ComfyUI's `models/`. With ComfyUI on this computer,
the files a job uses from here are linked into ComfyUI's folders, and names
are also found in ComfyUI's own folders ([local.md](local.md)):

```
models/
  checkpoints/                my-anime.safetensors
  diffusion_models/zimage/    my-finetune.safetensors
  text_encoders/              qwen_3_4b.safetensors
  vae/                        ae.safetensors
  loras/                      my-style.safetensors
  loras/zimage/               z-style.safetensors
```

- `"model": "my-anime"` (or `"my-anime.safetensors"`) uses
  `checkpoints/my-anime.safetensors` with the `illustrious` engine (any
  SDXL-architecture checkpoint). Sub-folders are searched too; if two files
  have the same name, the error lists both — write the sub-folder path
  (`"sdxl/my-anime"`) or rename one.
- `"model": "my-finetune"` uses `diffusion_models/zimage/my-finetune.safetensors`
  as a fine-tune of `zimage`: the first sub-folder names the engine (`zimage`,
  `krea2`, `qwen` or `klein`), and the text encoder and VAE are those of the
  engine's built-in model (`zimage`, `krea2`, `qwen`, `klein-4b`). A file
  directly in `diffusion_models/`, or in a folder that is not an engine, is an
  error when used by name: move it into an engine folder, or write the model as
  an [object](#a-model-object) with its `engine`. A name found both in
  `checkpoints/` and `diffusion_models/` is an error that lists both.
- `"loras": ["my-style"]` uses a file in `loras/`. When its first sub-folder is
  an engine name (`loras/zimage/…`), using it with another engine is an error;
  anywhere else it is accepted with any model. A folder LoRA knows no
  trigger word: give it with the LoRA,
  `{ "name": "my-style", "scale": 0.8, "trigger": "mystyle" }`.
- Text encoders and VAEs of your own are used through a model object
  (`"vae": "my-vae"`).
- The folder's files keep their path on the pod (`models/loras/my-style.safetensors`
  there too), so local ComfyUI and RunPod load them by the same name.

Only weight files (`.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`,
`.sft`) are considered; a `README.md` or `.part` download is ignored. Copies
of built-in files (what `pull` downloads) are recognized in any sub-folder of
their role folder. Weights in `models/` are git-ignored.

## Addresses

A file that is online needs no copy and no setup: write its address where a
name would go.

```json
"model": "hf://OnomaAIResearch/Illustrious-XL-v1.0/Illustrious-XL-v1.0.safetensors",
"loras": ["hf://someone/their-lora/their-lora.safetensors",
          { "name": "https://my-storage.example.com/x.safetensors", "scale": 0.8 }]
```

- `hf://owner/repo/path/in/repo` reads the repo's `main` branch;
  `hf://owner/repo@<revision>/path` pins a commit, so the file cannot change
  under you. `HF_TOKEN` is sent for private or gated repos (to
  `huggingface.co` only).
- `https://…` is any URL the pod can download: an S3/R2 pre-signed URL, your
  own server.
- A `model` address is an SDXL-type checkpoint (`illustrious`); its outputs
  go in a folder named after the file's stem. Other engines take an
  [object](#a-model-object).
- A LoRA address fits any engine and knows no trigger word; give one with
  `"trigger"` in the same way.
- No SHA-256 or size is known, so the pod accepts the download as it comes.
- Each file is stored as `<folder>/_url/<hash>/<file name>` (the hash is of
  the address), so two addresses ending in the same file name never collide.
  A local ComfyUI finds it by file name (and gets it downloaded there).

[`configs/basics/08-from-url.json`](../configs/basics/08-from-url.json) shows
all three forms.

### A model object

For a fine-tune of a three-file engine that is online, or to swap any of an
engine's files:

```json
"model": { "engine": "zimage", "diffusion": "hf://someone/finetune/model.safetensors", "name": "my-z" }
```

- `engine` is required. The file roles are `diffusion`, `text_encoder` and
  `vae`, or `checkpoint` for `illustrious` (which needs it).
- Each role is an address or a file name in that role's folder
  (`diffusion_models/`, `text_encoders/`, `vae/`, `checkpoints/`). Roles you
  leave out come from the engine's built-in model.
- `name` is the output folder; by default the diffusion or checkpoint file's
  stem. It must not be a built-in alias, and two different models may not
  share one.
- A `model` list may mix names, addresses and objects.

## The built-in catalog

Every entry is a public Hugging Face file pinned to a revision, with its
SHA-256 and size, so downloads are verified. On RunPod the pod downloads them
itself; a ComfyUI on this computer gets them downloaded into its folders on
the first run (or ahead of time with `jenair pull <alias>`).

| alias | engine | what | license |
|---|---|---|---|
| `zimage` | zimage | Z-Image Turbo ([Comfy-Org/Z-Image-Turbo](https://huggingface.co/Comfy-Org/Z-Image-Turbo)) | Apache-2.0 |
| `krea2` | krea2 | Krea 2 Turbo ([Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)) | Krea 2 Community |
| `qwen` | qwen | Qwen-Image 2.1 ([Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)) | see model card |
| `klein-4b` | klein | FLUX.2 klein 4B ([Comfy-Org/Flux-2-Klein-4B](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B)) | Apache-2.0 |
| `illustrious` | illustrious | Illustrious XL v0.1 ([OnomaAIResearch/Illustrious-xl-early-release-v0](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)) | Fair AI Public 1.0-SD |

Example LoRAs, one or two per engine. Each is used by a job in `configs/`:

| alias | what | trigger (added for you) | notes | license |
|---|---|---|---|---|
| `zimage-pixel-art` | [tarn59/pixel_art_style_lora_z_image_turbo](https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo) | `Pixel art style` | | Apache-2.0 |
| `zimage-technicolor` | [renderartist/Technically-Color-Z-Image-Turbo](https://huggingface.co/renderartist/Technically-Color-Z-Image-Turbo) | `t3chnic4lly` | classic film color | Apache-2.0 |
| `krea2-retro-anime` | [krea/Krea-2-LoRA-retroanime](https://huggingface.co/krea/Krea-2-LoRA-retroanime) | `Purple retro anime style` | official Krea LoRA | Krea 2 Community |
| `krea2-watercolor` | [krea/Krea-2-LoRA-softwatercolor](https://huggingface.co/krea/Krea-2-LoRA-softwatercolor) | `Art Deco watercolor style` | official Krea LoRA | Krea 2 Community |
| `qwen-lightning` | [lightx2v/Qwen-Image-2.1-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning) | | speed LoRA: set `"steps": 4` | see model card |
| `qwen-anime-fusal` | [MicksHF/Qwen-Image-2.1-Anime-Fusal](https://huggingface.co/MicksHF/Qwen-Image-2.1-Anime-Fusal) | `fusal style.` | wants the sectioned prompt shown in `configs/engines/qwen.json`; scale 0.7–1.0 | Qwen research |
| `klein-anime` | [Sawata97/flux2_4b_koni_animestyle](https://huggingface.co/Sawata97/flux2_4b_koni_animestyle) | `anime_style` | | Apache-2.0 |
| `klein-mikkoani` | [mikkoph/mikkoani-klein4b](https://huggingface.co/mikkoph/mikkoani-klein4b) | | style follows the prompt, e.g. "mature anime illustration" | Apache-2.0 |
| `illustrious-niji` | [LyliaEngine/Midjourney_Niji_Anime](https://huggingface.co/LyliaEngine/Midjourney_Niji_Anime) | `midniji` | | CDLA-Permissive-2.0 |
| `illustrious-the-artist` | [robb-0/TheArtist-Style-IllustriousXL](https://huggingface.co/robb-0/TheArtist-Style-IllustriousXL) | | | CC-BY-4.0 |

These are third-party files chosen as small, public examples; this project
does not host or endorse them. Licenses are as stated on each model card at
the pinned revision — read the card before using results commercially.

## Where each file comes from

| file | local ComfyUI (`--local`) | RunPod (`--runpod`) |
|---|---|---|
| has a URL (catalog, an address in the job, or found in your storage) | used by name if ComfyUI lists it; else downloaded into its folder (ComfyUI on this computer) | the pod downloads it (even if you have a copy) |
| only on this computer (models folder) | used by name if ComfyUI lists it; else linked into its folder (ComfyUI on this computer) | with `JENA_STORAGE`: uploaded once to your storage, then fetched by the pod. Without: uploaded to the pod in 32 MB chunks, SHA-256 verified, skipped when the pod already has it |
| neither | error | error |

`--dry-run` together with a target shows this per file:

```
$ jenair my-job.json --runpod --dry-run
files (RunPod):
  fetch   https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/… → models/vae/ae.safetensors
  upload  models/loras/my-style.safetensors → models/loras/my-style.safetensors
```

`fetch` (the pod downloads a URL), `storage` (through your storage), `upload`
(straight to the pod), `local` (your ComfyUI, when it did not answer) or
`missing`. With a ComfyUI on this computer answering, the local plan shows
`have`, `download → <folder>`, `link → <folder>` or `missing`.

## `pull`: downloading ahead of time

```bash
jenair pull zimage zimage-pixel-art   # aliases
jenair pull configs/engines/zimage.json       # everything a job file needs
```

With ComfyUI on this computer answering, files go into its own folders and
those it already lists are skipped; otherwise into the models folder. Each
file keeps its path (catalog LoRAs to
`loras/<engine>/`, addresses to `<folder>/_url/<hash>/`). Downloads resume from a `.part` file, are checked against
their SHA-256, and are skipped when the file is already there with the right
hash. `HF_TOKEN` is sent to `huggingface.co` only.

A normal run does the same by itself, so `pull` is for downloading before you
need it. For a ComfyUI that cannot be filled automatically (on another
computer, or without `/internal/folder_paths`), see
[docs/local.md](local.md#when-it-is-not-automatic).

## Your own storage (RunPod)

Uploading a 7 GB checkpoint from home to every new pod is slow. With a storage
set, a file that is only on your computer is uploaded **once** and every pod
downloads it at datacenter speed:

```bash
# .env
JENA_STORAGE=hf://your-name/your-models   # created as a private repo if missing
HF_TOKEN=hf_...                            # needs write access
```

Each file is stored at its models-folder path (`loras/my-style.safetensors`,
`diffusion_models/zimage/my-finetune.safetensors`), and the pod receives a URL
pinned to a commit. `jenair push my-job.json` uploads a job's files ahead of
time, without a pod. Only Hugging Face repos are supported for now.

- **No second upload.** `.jena/storage.json` remembers what this computer
  uploaded, by SHA-256. When it has no record (another computer, a fresh
  clone), the repo is asked first: a file already there at the same path with
  the same SHA-256 is used as it is.
- **Names from any computer.** A `model` or LoRA name that is neither built in
  nor in the models folder is looked up in the repo, with the folder rules
  above (`checkpoints/`, `diffusion_models/<engine>/`, `loras/[<engine>/]`).
  So a file pushed from one computer renders from another by the same name.
- The repo's file list is asked at most once per run, and only when needed.
  If that fails (network, token, `huggingface_hub` not installed), you get one
  warning and the name is reported as not found.

### Civitai

Civitai blocks most datacenter IPs, so a RunPod pod usually **cannot**
download Civitai URLs. Download the file once into the models folder; on RunPod it is
uploaded (or goes through your storage). `CIVITAI_TOKEN` in `.env` is still
sent for the cases where it works.

## Gated models

FLUX.2 klein 9B and some others need you to accept the license on Hugging Face
and set `HF_TOKEN`. Check each model's license; several are non-commercial.

## Checking

```bash
jenair configs/basics/03-compare-models.json --dry-run
```

lists every file with where it comes from, and fails on unknown names,
ambiguous file names, diffusion models outside an engine folder, model objects
with wrong roles, and LoRAs used with the wrong engine.
