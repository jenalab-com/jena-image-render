# models/

Your model files, in ComfyUI's layout. Everything here except this README is
git-ignored. With ComfyUI on this computer, what a job uses from here is linked
into ComfyUI's own folders on the run; nothing needs to be set up. (Usually not
needed: `JENA_MODELS_DIR` uses another folder instead of this one.)

```
checkpoints/        SDXL-type checkpoints: "model": "<file name or stem>"
diffusion_models/<engine>/
                    fine-tunes of zimage, krea2, qwen or klein: "model": "<file name>"
                    (text encoder and VAE: the engine's built-in ones)
text_encoders/      text encoders
vae/                VAEs
loras/              LoRAs for any engine: "loras": ["<file name or stem>"]
loras/<engine>/     LoRAs checked against that engine (zimage, krea2, ...)
```

The built-in models and LoRAs, and files a job names by address, need no
place here: with ComfyUI on this computer they are downloaded straight into
its folders on the first run. `jenair pull <alias | JOB.json>` downloads them
ahead of time — into ComfyUI when it answers, else here (under
`<folder>/_url/` for addresses). See [docs/models.md](../docs/models.md).
