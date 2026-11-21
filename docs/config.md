# Job files (schema v1)

A job file says *what* to render. *Where* (local ComfyUI or RunPod) is usually
chosen on the command line, so the same file works everywhere and can be
shared; a job file may still name a default `target`.

## Shapes

One image:

```json
{ "model": "zimage", "prompt": "a quiet cafe at dawn" }
```

A batch: `defaults` apply to every job, a job's own value wins.

```json
{
  "output": "cafe-test",
  "defaults": { "model": "zimage", "seeds": [1, 2, 3, 4] },
  "jobs": [
    { "name": "cafe", "prompt": "a quiet cafe at dawn" },
    { "name": "street", "prompt": "a rainy neon street", "seed": 7 }
  ]
}
```

**Combinations:** `model` and `seeds` accept lists. Every job renders once per
model × seed. A job's `seed`/`seeds` replaces the defaults' seeds as a whole.

## Fields

| field | where | | meaning |
|---|---|---|---|
| `version` | top | | schema version. `1`, which is also what a file without it means |
| `target` | top | | `"local"` or `"runpod"`: where to render when the command line does not say |
| `output` | top | | output sub-folder. Default: the job file's name |
| `defaults` | top | | values every job inherits |
| `jobs` | top | | list of jobs. Without it, the top level is one job |
| `name` | job | | file name prefix. Default `job1`, `job2`, … |
| `model` | job / defaults | required | a name, an address or a model object — or a list of them; see [Models and LoRAs](#models-and-loras) |
| `prompt` | job / defaults | required | the prompt |
| `negative` | job / defaults | | used by `illustrious`, and by `krea2` when `cfg` > 1. Ignored (with a warning) elsewhere |
| `seed` / `seeds` | job / defaults | | integer or list. Default `0` |
| `aspect` | job / defaults | | size shorthand, see below |
| `width`, `height` | job / defaults | | explicit size, rounded to the engine's multiple (with a warning) |
| `loras` | job / defaults | | `["name"]` or `[{"name": "name", "scale": 0.8, "trigger": "words"}]`, where the name may also be an address; see [LoRAs](#loras). Must match the model's engine; see [Models and LoRAs](#models-and-loras) |
| `steps` | job / defaults | | sampling steps. Default per engine |
| `cfg` | job / defaults | | CFG scale. Default per engine; fixed at 1 for `zimage`, `qwen`, `klein` |
| `sampler`, `scheduler` | job / defaults | | default per engine; see below |
| `skip` | job | | `true` skips the job |

Keys starting with `_` (such as `"_note"`) are comments, at the top level and
in any job. Other unknown keys are ignored with a warning. `hires`, `upscale`, `face_detail`,
`soft`, `source` and video keys are planned for a later version.

### Models and LoRAs

A `model` value is one of:

| form | example | means |
|---|---|---|
| built-in alias | `"zimage"` | the official release ([models.md](models.md#the-built-in-catalog)) |
| file name | `"my-anime"`, `"my-finetune.safetensors"` | `models/checkpoints/my-anime.safetensors` (an SDXL-type checkpoint, `illustrious`), or `models/diffusion_models/<engine>/my-finetune.safetensors` (a fine-tune of that engine) |
| address | `"hf://owner/repo/file.safetensors"`, `"https://…"` | an SDXL-type checkpoint that is online |
| object | `{ "engine": "zimage", "diffusion": "hf://…", "name": "my-z" }` | an engine and the files you change; the rest from the engine's built-in model |

The object takes `engine` (required), any of the engine's file roles
(`diffusion`, `text_encoder`, `vae`; `checkpoint` for `illustrious`, which
needs it) as an address or a file name in that role's folder, and `name`: the
output folder, by default the diffusion or checkpoint file's stem. It must not
be a built-in alias. A list may mix all four forms.

A `loras` name is a built-in alias, a file in `models/loras/` (checked against
the engine when it is in `loras/<engine>/`), or an address (used with any
engine). Names that are none of these are looked up in your storage when
`JENA_STORAGE` is set.

An address is `hf://owner/repo[@revision]/path/in/repo` (the `main` branch
unless a revision pins it) or an `https://` URL. Details: [models.md](models.md).

### LoRAs

Each entry is a name (or an address), or an object:

```json
"loras": [
  "zimage-pixel-art",
  { "name": "my-lora", "scale": 0.8, "trigger": "mylora" }
]
```

| key | | meaning |
|---|---|---|
| `name` | required | a built-in LoRA, a file in `models/loras/`, or an address |
| `scale` | | strength. Default `1.0` |
| `trigger` | | words put in front of the prompt (skipped when the prompt already has them) |

Built-in LoRAs already know their trigger; `trigger` replaces it, and `""`
adds none. Your own LoRAs (folder files and addresses) have no trigger until
you give one.

### Sampler names

ComfyUI names (`euler`, `dpmpp_2m`, …) work everywhere. For SDXL model cards,
A1111 names are accepted too: `Euler a`, `Euler`, `DPM++ 2M`, `DPM++ 2M Karras`,
`DPM++ SDE Karras`, `UniPC`.

### `cfg`, and engine defaults

| engine | steps | cfg | sampler / scheduler | negative prompt |
|---|---|---|---|---|
| `zimage` | 9 | 1 (fixed) | `res_multistep` / `simple` | no |
| `krea2` | 8 | 1 | `euler` / `simple` | only when `cfg` > 1 |
| `qwen` | 25 | 1 (fixed) | `euler` / `simple` | no |
| `klein` | 4 | 1 (fixed) | `euler` / `simple` | no |
| `illustrious` | 28 | 6 | `Euler a` / `normal` | yes |

`jenair engines` prints the same. `zimage`, `qwen` and `klein` are
distilled models that always sample at CFG 1: a `cfg` given for them is ignored
with a warning. `krea2` at `cfg` 1 is its turbo mode; above 1 it runs a
negative pass and takes about twice as long. (`guidance` is still read as an
older name; for `krea2` it meant CFG − 1.)

### `null`: back to the engine default

A job value replaces the default. To *unset* one — for example sampling
settings in `defaults` that suit one engine, in a job that uses another — give
`null`:

```json
{ "name": "on-zimage", "model": "zimage", "steps": null, "cfg": null, "sampler": null, "scheduler": null }
```

## Sizes

| `aspect` | size | | `aspect` | size |
|---|---|---|---|---|
| `1:1` | 1024 × 1024 | | | |
| `2:3` | 832 × 1248 | | `3:2` | 1248 × 832 |
| `3:4` | 896 × 1152 | | `4:3` | 1152 × 896 |
| `9:16` | 768 × 1344 | | `16:9` | 1344 × 768 |

All values are multiples of 32 (valid for every engine) and close to one
megapixel. Only `2:3` is exact; the others are the nearest multiples of 32.

Precedence: the job's `width`/`height` or `aspect` → the defaults' → the
engine default. Using `aspect` and `width`/`height` in the same place is an
error.

## Outputs

```
outputs/<output>/<model>/<name>_<seed>.png
outputs/<output>/<model>/<name>_<seed>.json   ← prompt, model, LoRAs, seed, all settings, time
```

Existing PNGs are skipped, so re-running a job file resumes it.
`--overwrite` renders again. Two jobs writing the same path is an error
before anything starts.

## Command line

```
jenair JOB.json [JOB.json ...] [--local [URL] | --runpod] [--dry-run]
    --dry-run             check and show the plan for the target, render nothing
    --no-download         local: do not download missing files into ComfyUI
    --outputs DIR         default ./outputs
    --only PREFIX         only jobs whose name starts with PREFIX
    --limit N             render at most N images
    --overwrite           render images that already exist
    --continue-on-error   keep going after a failed image
    --secure              RunPod: skip Community Cloud
    --max-hours H         RunPod: hard pod lifetime (default 3)

jenair pull NAME|JOB.json ...   download URL files into ComfyUI's folders
                                (a ComfyUI on this computer), else the models folder
jenair push JOB.json ...        upload local-only files to JENA_STORAGE
jenair runpod status | down | up
jenair engines
```

### Target

Where to render, first match wins:

1. `--local [URL]` or `--runpod` on the command line,
2. `"target"` in the job files (all files given at once must agree),
3. `JENA_TARGET` in `.env`,
4. otherwise `local`: your own ComfyUI. A GPU is never rented unless one of
   the above says `runpod`.

`--dry-run` works with any of them and shows, per file, how it would reach that
target (`fetch`, `storage`, `upload`, `local`; with a ComfyUI on this
computer answering: `have`, `download`, `link`, `missing`).

## Settings

Everything is optional. Settings are read from the environment, or from a
`.env` file in the project folder (real environment variables win).
`.env.example` lists the first four; the rest are rarely needed.

| setting | | meaning |
|---|---|---|
| `RUNPOD_API_KEY` | for RunPod | your RunPod API key |
| `HF_TOKEN` | | private or gated Hugging Face files; write access when `JENA_STORAGE` is set. Sent only to huggingface.co |
| `JENA_TARGET` | default `local` | `runpod` renders on RunPod without `--runpod` |
| `JENA_STORAGE` | | `hf://owner/repo`: your own files are uploaded there once ([runpod.md](runpod.md)) |
| `JENA_COMFY_URL` | default `http://127.0.0.1:8188` | your ComfyUI's address, when `--local URL` is not given |
| `JENA_MODELS_DIR` | default `./models` | the models folder; usually left as it is ([models.md](models.md), [local.md](local.md)) |
| `CIVITAI_TOKEN` | | sent only to civitai.com by a pod. Civitai blocks most datacenter IPs, so on RunPod prefer a file in `models/` or your storage |
| `JENA_POD_IMAGE` | | your own build of the pod image ([runpod.md](runpod.md)) |
