# GitHub, privacy, and release checklist

## Commit to GitHub

- Python source: `app.py`, `vyndra/`, `scripts/`, `tests/`.
- `requirements.txt`, `.gitignore`, `README.md`, `DATASET_INVENTORY.txt`, `docs/`, notebooks.
- Small derived/reviewable data: class lists, mapping review CSVs, synthetic Part 4/5 demo inputs.
- Evaluation tables/figures that are reproducible and appropriate to share, with a note about dataset limitations.
- A `.env.example` containing placeholder values only, if environment variables are needed.

## Do not commit

- `env/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.DS_Store`, editor temp files.
- `models/*.pt`, downloaded model weights, large raw image datasets, dataset archives/caches, and large training artifacts. Document where to obtain them and their licenses instead.
- `uploads/*`, personal photos, `data/test_images_own/`, user-provided images, private identifiers, or consent-sensitive data. Keep only `.gitkeep` if a folder must exist.
- `.env`, API keys, passwords, tokens, database files (`*.sqlite3`, `*.db`), logs, and generated personal user data.
- Any dataset whose license or redistribution permission has not been checked. `data/raw/` may contain files with different licenses; do not commit wholesale without checking each source's terms.

## Before first push

```bash
git status --short
git check-ignore -v env models/best.pt uploads/20250404_093458_captured_image.jpg data/test_images_own/example.jpg data/vyndra.sqlite3 .env
 git ls-files | grep -E '(^env/|\.pt$|^uploads/|test_images_own|\.sqlite3$|(^|/)\.env$)' || true
```

If sensitive or large files are already tracked, `.gitignore` does not remove them from Git history. Untrack them with `git rm --cached <path>`; if secrets were committed, rotate them and consider history cleanup before publishing.
