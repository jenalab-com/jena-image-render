# Rendering on your own ComfyUI

Local is the default target: the graphs go to a ComfyUI you run yourself. With
ComfyUI on this computer there is nothing to set up — start it and run a job;
`jenair` fills ComfyUI's model folders with what the job needs.

## 1. Install ComfyUI

The graphs follow the official templates of **ComfyUI v0.38.0** (the Qwen-Image
2.1 and FLUX.2 klein nodes need at least that version). Newer versions should
work; v0.38.0 is what this release was verified against.

```bash
git clone --branch v0.38.0 https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv .venv && . .venv/bin/activate
# install PyTorch for your GPU first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python main.py            # http://127.0.0.1:8188
```

ComfyUI Desktop and portable builds work as well, as long as the version is
recent enough. No custom nodes are needed.

## 2. Render

```bash
jenair configs/engines/zimage.json
jenair configs/engines/zimage.json --local http://192.168.0.10:8188   # another machine
```

The default address is `http://127.0.0.1:8188`; `--local URL` or
`JENA_COMFY_URL` in `.env` changes it.

## What happens with ComfyUI on this computer

When the address is a loopback one (`127.0.0.1`, `localhost`, `::1`) and the
folders ComfyUI reports exist here, ComfyUI is on this computer. `jenair` asks
it for its model folders (`GET /internal/folder_paths`, which also lists the
folders of `extra_model_paths.yaml`) and then:

1. **Finds names there too.** A `model` or LoRA name is looked up in `./models`
   first, then in every ComfyUI model folder, with the same rules
   (`checkpoints/`, `diffusion_models/<engine>/`, `loras/[<engine>/]`, by file
   name or stem). A model you already keep in ComfyUI is used by its name;
   nothing is moved. When both have the name, `./models` wins.
2. **Fills in what ComfyUI lacks**, before the first image. Each file is looked
   up in ComfyUI's lists by file name; one that is not there is
   - **downloaded** when it has a URL (the built-in models and LoRAs, files a
     job names by address): into ComfyUI's own folder for its role
     (`models/checkpoints`, `models/diffusion_models`, `models/text_encoders`,
     `models/vae`, `models/loras`), keeping its sub-folder
     (`loras/<engine>/`, `<folder>/_url/<hash>/`). A summary line shows the
     number of files, their size and the folders first. Downloads resume from
     a `.part` file, are checked by SHA-256 when it is known, and send
     `HF_TOKEN` to `huggingface.co` only.
   - **linked** when it is only in `./models`: a symbolic link in ComfyUI's
     folder, or a hard link where symbolic links are not allowed (Windows
     without the right), or else a copy. Each one is logged.

   Then ComfyUI's lists are read again; a file still missing stops the run
   with the folder it should go in.

The first run of a built-in model downloads all its files: Z-Image Turbo is
about 20 GB, Qwen-Image about 32 GB. Later runs find them and start at once.

- `--dry-run` asks ComfyUI the same questions but writes nothing; it shows,
  per file, `have`, `download → <folder>`, `link → <folder>` or `missing`. When
  ComfyUI does not answer it shows the plan without it.
- `--no-download` turns downloads off: a file ComfyUI lacks is an error (links
  are still made).
- `jenair pull JOB.json` (or `pull zimage`) downloads ahead of time, into
  ComfyUI's folders when it answers, else into `./models`.

## When it is not automatic

A ComfyUI **on another computer** (or in a container whose folders are not
visible here) is used by name only: nothing can be put into it from here. Put
the files in that computer's ComfyUI `models/` folders yourself, for example by
running `jenair pull` there. The same goes for a ComfyUI version without
`/internal/folder_paths`.

ComfyUI matches files by **file name** in any sub-folder
(`models/loras/sdxl/x.safetensors` matches a LoRA whose file is
`x.safetensors`). This tool's models folder has the same layout:

| role | folder |
|---|---|
| `checkpoint` | `checkpoints/` |
| `diffusion` | `diffusion_models/` |
| `text_encoder` | `text_encoders/` |
| `vae` | `vae/` |
| LoRAs | `loras/` |

To share one folder by hand with a ComfyUI on this computer that does not
report its folders, either set `JENA_MODELS_DIR=/path/to/ComfyUI/models` in
`.env` (then `pull` downloads straight into it), or add this folder to
`ComfyUI/extra_model_paths.yaml`:

```yaml
jena:
  base_path: /path/to/jena-image-render/models
  checkpoints: checkpoints
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  loras: loras
```

If ComfyUI was already running, restart it (or refresh its model list) so it
sees new files.

## VRAM

Rough guide for the default sizes: `klein` 4B and `zimage` run on 16 GB cards,
`krea2` and `qwen` want 24 GB or more (ComfyUI offloads to system RAM when
VRAM is short, which is slower). If your card is small, RunPod is the easier
path.

### Apple Silicon

ComfyUI runs on Apple Silicon without an NVIDIA card (it uses the Mac's GPU
through MPS), and so does `jenair`. Measured on a MacBook Pro with 48 GB of
memory, ComfyUI v0.38.0, default sizes, one image:

| engine | time per image |
|---|---|
| `klein` 4B | about 30–60 s |
| `illustrious` | about 80–95 s |
| `zimage` | about 100–135 s |
| `krea2` | about 220–280 s |
| `qwen` | about 290–320 s (25 steps); about 130 s with the 4-step `qwen-lightning` LoRA |
