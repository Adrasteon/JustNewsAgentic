# Bootstrap local model weights

Use `scripts/bootstrap_models.py` to prefetch and cache all model weights so services can run fully offline (local-first policy).

Quick start

1) Ensure dependencies (if not already installed by your agents):
   - `huggingface_hub`
2) Optional: export `HF_TOKEN` if you pull gated/large repos.
3) Run the bootstrap:

```bash
python scripts/bootstrap_models.py --manifest scripts/model_manifest.example.json
```

Options

- `--subset analyst synthesizer` to limit components
- `--include-vision` to add large vision models
- `--cache-dir /path/to/cache` to override `TRANSFORMERS_CACHE`
- `--force` to redownload even if cached
- `--ignore-errors` to continue on individual failures

Outputs

- Report: `training_system/bootstrap_report.json` contains resolved revisions and paths
- Cached files under `TRANSFORMERS_CACHE` (or the provided `--cache-dir`)

After bootstrap

- Enforce offline operation:

```bash
export TRANSFORMERS_OFFLINE=1
```

Notes

- Manifest is an example; adjust to your production model choices.
- This script avoids running any models; it only downloads repository snapshots.
