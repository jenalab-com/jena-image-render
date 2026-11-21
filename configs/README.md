# Examples

Copy one, change the values, run it. Every key starting with `_` (like
`_note`) is a comment and is ignored.

```bash
jenair configs/full.json --dry-run   # check, render nothing
jenair configs/full.json             # render on your own ComfyUI (start it first)
jenair configs/full.json --runpod    # or on a rented GPU
```

## [full.json](full.json) — the dictionary

Every key and every form this tool understands, in one file. Do not read it
top to bottom: look up what you need and copy that job.

| group | what is there |
|---|---|
| top level | `target`, `output`, `defaults`, `jobs` |
| `defaults` | model, LoRAs, negative prompt, size, seeds, `steps`, `cfg`, `sampler`, `scheduler` — written once for all jobs |
| size | another `aspect`; exact `width`/`height` |
| seeds | one `seed`; a `seeds` list |
| sampling | `steps`, `cfg`, ComfyUI sampler + `scheduler`; a model-card sampler name |
| negative | another negative prompt; none (`""`) |
| LoRAs | replace and add; `{name, scale, trigger}`; no trigger (`""`); no LoRA (`[]`); a LoRA by address |
| models | another engine (`null` = back to the engine default); several models; a checkpoint by address; a model as an object |
| your files | a checkpoint, a LoRA and a fine-tune from `models/` (skipped until the file is there) |
| `skip` | keep a job without rendering it |

The rule it shows: **a job starts from `defaults` and repeats a key only to
change it; the job's value replaces the default as a whole. `null` hands a
setting back to the engine's own default.**

## basics/ — one idea per file

Run on a fresh clone: they use only the built-in models and LoRAs, and
public files by address. With ComfyUI on this computer, the first run
downloads them into it.

| file | shows |
|---|---|
| [01-one-image](basics/01-one-image.json) | the smallest job: `model` + `prompt` |
| [02-batch](basics/02-batch.json) | `defaults`, `jobs`, `seeds`, per-job overrides, `skip`, `output` |
| [03-compare-models](basics/03-compare-models.json) | a list of models: prompt × model × seed |
| [04-loras](basics/04-loras.json) | LoRAs: short form, `scale`, several at once |
| [05-sizes](basics/05-sizes.json) | `aspect`, `width`/`height`, engine default |
| [06-sampling](basics/06-sampling.json) | `steps`, `cfg`, `sampler`, `scheduler`, `negative`, a speed LoRA |
| [07-target](basics/07-target.json) | `target`: your ComfyUI (default) or a rented RunPod GPU |
| [08-from-url](basics/08-from-url.json) | files that are online, by address: a LoRA, a checkpoint, a model object |

## engines/ — one file per engine

A plain job and the same scene with a built-in LoRA, with prompts written the
way each model likes them.

[zimage](engines/zimage.json) · [krea2](engines/krea2.json) ·
[qwen](engines/qwen.json) · [klein](engines/klein.json) ·
[illustrious](engines/illustrious.json)

## your-files/ — your own checkpoints and LoRAs

These need a file of yours in [`models/`](../models); the `_note` in each says
which file goes where. File names are just examples: rename them in the JSON
to match your files. A file that is online needs no copy here: write its
address in the job, as in [08-from-url](basics/08-from-url.json).

| file | you put | |
|---|---|---|
| [checkpoint](your-files/checkpoint.json) | `models/checkpoints/my-checkpoint.safetensors` | an SDXL-type checkpoint, e.g. from Civitai |
| [lora](your-files/lora.json) | `models/loras/my-lora.safetensors` | your LoRA on a built-in model |
| [checkpoint-and-lora](your-files/checkpoint-and-lora.json) | both of the above | LoRAs shared through `defaults`, plus a built-in LoRA, on RunPod |
| [finetune](your-files/finetune.json) | `models/diffusion_models/zimage/my-zimage-finetune.safetensors` | a fine-tune of a three-file engine: the folder names the engine |

Where the files end up:

| `target` | your file in `models/` |
|---|---|
| `local` (default) | linked into the ComfyUI on this computer when it lacks it; a ComfyUI elsewhere must already have it ([docs/local.md](../docs/local.md)) |
| `runpod` | uploaded to the pod; or, with `JENA_STORAGE` in `.env`, uploaded once to your storage and downloaded by every pod ([docs/runpod.md](../docs/runpod.md)) |

All keys: [docs/config.md](../docs/config.md).
