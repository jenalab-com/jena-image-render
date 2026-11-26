# The web page (`jenair ui`)

A page for writing job files and rendering them, instead of editing JSON and
running commands in a terminal. It runs on your computer only: a small server
that `jenair` starts, and a page in your browser that talks to it.

```bash
jenair ui                    # opens http://127.0.0.1:8765/ in your browser
jenair ui --port 8800        # another port
jenair ui --no-browser       # print the address, open it yourself
jenair ui --outputs DIR      # where images are saved (default ./outputs)
./init.sh ui                 # the same, without activating .venv
```

Start it in the project folder: the page saves your job files in `jobs/`, lists
the examples in `configs/`, uses `./models`, `.env` and `runpod.json` like the
command line, and saves images in `outputs/`. Ctrl+C in the terminal stops it.
Nothing is added: the server uses only Python's standard library, and the page
is one file with no external scripts or fonts (it works offline).

## The screen

- **Tabs = batches.** Each tab is one set of settings with its prompts — for
  example one tab per model. The tab name is the batch name and its output
  folder. Double-click a tab to rename it, drag tabs to reorder them, and use
  ⧉ to duplicate or × to close one. Unticking a tab's box skips it (the batch
  gets `"skip": true`). The number on a tab is how many images it makes.
- **Model.** Choose from the built-in models, your `models/` folder, the model
  folders of a ComfyUI on this computer (target Local, while it answers) and
  your storage (`JENA_STORAGE`), or pick *Other* and type a file name or an
  address (`hf://…`, `https://…`). A list of models or a model object can be
  typed as JSON. The model decides the engine, and the engine its defaults.
- **Settings.** Steps, CFG, sampler, scheduler and size show the engine's
  default in grey: leave a field empty to use it (an empty field is not written
  to the file). CFG is locked for engines that always run at CFG 1, and
  Negative for engines that do not use one (for `krea2`, until CFG is above 1);
  the tooltip says why.
- **LoRAs.** Any number, each with a scale and its trigger words (a built-in
  LoRA shows its own trigger in grey; untick the box to add none). The list
  offers LoRAs for the model's engine and LoRAs of no known engine; one that
  does not fit the model is marked red.
- **A file on this computer.** *Browse…* beside the Model picker and beside
  each LoRA (or *+ LoRA from a file…*) opens a file browser that the server
  runs, so nothing is uploaded through the browser: your home folder, the
  matching folder in `models/`, the folders of a connected local ComfyUI and
  your drives (on macOS `/Volumes/*`), with a breadcrumb and a filter. Only
  folders and weight files (`.safetensors`, `.sft`, `.ckpt`, `.gguf`) are
  shown. Double-click a file, or select it and press *Select*. The file is
  put into `models/` — as a symbolic link, else a hard link, else a copy — so
  the usual rule "a file in `models/` is used by its name" covers it on every
  target: `models/loras/<engine>/` when *only for <engine>* is ticked (it is,
  when the tab has a model), else `models/loras/`; for a model, choose
  *checkpoint (SDXL type)* (`models/checkpoints/`) or *diffusion model for*
  an engine (`models/diffusion_models/<engine>/`). A file that is already in
  `models/` or in the local ComfyUI's folders is used where it is. The field
  gets the name a job writes, and one line says what was done. A different
  file with the same name is replaced only when you confirm.
- **Seeds.** A comma-separated list; *+ random* adds a random seed. Each prompt
  renders once per seed (and per model).
- **Prompts.** As many as you like: *+ Add*, or *Paste many* (blocks separated
  by an empty line become one prompt each). Each prompt has a name (its file
  name; `p1`, `p2`, … by default), a box to skip it, ↑ ↓ to move it, ⧉ to
  duplicate it and × to remove it. **⋮** opens the prompt's own settings as
  JSON — another seed, size, LoRA list, negative or anything else for that
  prompt only (`{"seed": 7, "aspect": "16:9"}`).
- **Other keys.** Keys of a batch that the page has no field for are kept as
  they are, in a JSON box under the settings, so opening and saving a file
  never drops them.

### The Render side

Always on the right, for **all tabs at once**:

1. **Target**: Local (your ComfyUI; its address can be changed) or RunPod.
2. **Connect**. Local: asks ComfyUI for its version and device, and whether it
   is on this computer (then missing files are downloaded or linked into it).
   RunPod: asks first, then reuses a running pod of yours or rents one. While it
   starts, a bar shows the step (renting → pulling image → starting ComfyUI →
   ready) and a clock; the image pull reports no percentage, so the bar is an
   estimate against the usual time (about 5 minutes). When the pod is ready the
   side shows the GPU, the **time until it stops itself** (counted from this
   page's last request to it), the hard limit, your balance and the spend per
   hour. **Stop pod** terminates it (after asking). Without a RunPod key the
   side says so, with a button that opens Settings.
3. **Check** shows what would happen without rendering: the number of images
   (per tab), the files to download, link, upload or fetch, and any errors or
   warnings.
4. **Render** renders every tab, one image after another (it connects first
   if needed; renting a pod and re-rendering saved images are confirmed). The
   side shows the overall progress with an estimate of the time left, the step
   of the current image, file transfers, failures and a thumbnail of each new
   image (click it to open the full image). **Cancel** stops after the current
   image (and asks ComfyUI to interrupt it). Tabs can be edited while it runs.
5. **Log**: the same lines the command line prints, folded away.

Keyboard: Ctrl/Cmd+S saves, Ctrl/Cmd+Enter renders.

## Saving and opening

- **Save** / **Save as…** write `jobs/<name>.json` in the project folder
  (`jobs/` is git-ignored). Saving over a file asks first.
- **Open** lists your `jobs/` files and the examples in `configs/`.
- **Import** / **Export** read a JSON file from your computer or download one.
- A • after the file name means unsaved changes; closing the page then asks.
  The page also keeps your last work in the browser, so a reload does not lose
  it.

A saved file is an ordinary job file: `jenair jobs/<name>.json` renders it
the same way. The page always saves the `batches` form (even with one tab);
see [config.md](config.md#batches). Any job file opens: a file with
`batches` becomes one tab per batch; a file with `defaults` + `jobs`, or a
single job, becomes one tab named `main`. Saved again, such a file renders the
same images, now under `outputs/<output>/main/`.

## Settings

**⚙ Settings** in the header holds what `.env` holds, so you never have to
edit it by hand:

- **RunPod API key** (RunPod console → Settings → API Keys), **Hugging Face
  token** (huggingface.co → Settings → Access Tokens; write access for your
  storage) and **Civitai token** (civitai.com → Account → API Keys). Each says
  only *set* or *not set*: the value is never shown again. Leave the box
  empty to keep a key, type a new one to change it, or *Clear* it. *Test*
  checks the RunPod key (your balance) or the Hugging Face token (your account
  name), the typed one or else the saved one.
- **Storage** (`hf://owner/repo`), **Default target** and **ComfyUI address**,
  and under *Advanced* the **Pod image** and the **Models folder**.

*Save* writes `.env` in the project folder (created if missing, readable only
by you; comments and other lines are kept) and takes effect at once: no
restart. A key also set by a real environment variable keeps that value until
you change it there; the screen says so.

## Safety

The server holds your keys and can rent a GPU, so:

- it listens on `127.0.0.1` only, so other computers cannot reach it;
- it answers only requests addressed to `127.0.0.1:<port>` or
  `localhost:<port>` (this blocks web pages that try to reach it through
  their own domain names);
- each start makes a new random token that only the page it serves knows;
  every request to `/api/` must carry it, so other web pages you have open
  cannot use the server;
- it reads and writes files only in the outputs folder, `jobs/` and `configs/`;
- no answer contains a key: the page is only told whether one is set, and
  error messages from the key tests have the key blanked out;
- the file browser lists folder and weight-file names and sizes, never file
  contents.

Checking the pod's state never sends a request to the pod (that would keep it
from stopping itself when idle): the page asks this server, which asks only
the RunPod API, at most every 10 seconds.

Reaching the page from another computer is not supported. To use a GPU
elsewhere, render on RunPod, or give the address of a ComfyUI on another
machine under Local.
