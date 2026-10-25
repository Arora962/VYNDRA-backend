# VYNDRA remaining analytics: runbook

This is an overlay patch for the existing Part 1 project. It does not replace the repository, model weights, approved `data/processed/class_to_food.csv`, or original datasets.

## Order

1. Generate text and social data, then run their analysis scripts.
2. Run nutrition clustering and matching comparison.
3. Configure local image dataset roots and reviewed class maps, then prepare the vision manifest and evaluate the model.
4. Run acceptance checks and the cached-results notebook.

Run commands from the backend project root with the existing `env` activated.

```bash
python -m compileall -q app.py config.py vyndra scripts tests notebooks
python scripts/make_chat_dataset.py
python scripts/run_text_analytics.py
python scripts/evaluate_matching.py
python scripts/make_social_data.py --events 1800
python scripts/run_social_analytics.py
python scripts/cluster_foods.py
python -m pytest -q
python scripts/acceptance_check.py --part 2
python scripts/acceptance_check.py --part 3
```

`run_text_analytics.py` needs `vaderSentiment` for its sentiment output. If absent, install the pinned requirement in the active environment with `python -m pip install vaderSentiment==3.3.2`.

## Vision dataset setup

Images for Khana, Food-101 and Kaggle are external to the original ZIP. Do not move or duplicate the full datasets unless you want to. Pass their local roots to the manifest builder:

```bash
python scripts/prepare_vision_manifest.py \
  --kaggle-root "/absolute/path/to/Indian Food Images" \
  --khana-root "/absolute/path/to/khana/images" \
  --food101-root "/absolute/path/to/food-101/images"
```

The current repository has a reviewed dataset mapping for only nine Kaggle classes. For Khana and Food-101, the script creates a `*_class_map_TEMPLATE.csv` beside each supplied root if it cannot find a reviewed mapping file in `data/processed/`. Review and save those mapping files as `data/processed/khana_class_map.csv` and `data/processed/food101_class_map.csv`, with columns `dataset,dataset_class,best_pt_class,match_type,status`; only rows with `status=ok` and `match_type=exact` or `family` are eligible. Do not treat fuzzy suggestions as approved mappings. The script skips files named `._*` and writes relative paths.

Then run:

```bash
python scripts/prepare_vision_manifest.py \
  --kaggle-root "/absolute/path/to/Indian Food Images" \
  --khana-root "/absolute/path/to/khana/images" \
  --food101-root "/absolute/path/to/food-101/images"
python scripts/evaluate_vision_suite.py \
  --kaggle-root "/absolute/path/to/Indian Food Images" \
  --khana-root "/absolute/path/to/khana/images" \
  --food101-root "/absolute/path/to/food-101/images"
python scripts/acceptance_check.py --part 4
```

The HSV baseline runs on available mapped manifest classes, up to 100 images per class. If fewer images/classes are available, the output will reflect the available sample and must not be presented as a 20-class x 100-image experiment. To meet that target, ensure at least 100 usable images for each of 20 reviewed classes. Keep own photos in `data/test_images_own/<model class name>/`; they are included as a separate `own_photos` dataset in the manifest.

## Final checks

```bash
python -m pytest -q
python scripts/acceptance_check.py --part 1
python scripts/acceptance_check.py --part 2
python scripts/acceptance_check.py --part 3
python scripts/acceptance_check.py --part 4
python scripts/acceptance_check.py --part 5
python scripts/acceptance_check.py --all
python notebooks/VYNDRA_analytics.py
```

The acceptance check intentionally fails when required result files do not yet exist. That is a signal to run the corresponding analysis, not a reason to fabricate a result.

## Interpretation warnings

- The generated chat and social datasets are synthetic; they are reproducible test material, not real user behaviour.
- The “handwritten review” messages are a review set with human-authored seeds and natural variations. A human must review and correct the labels before reporting it as a manually annotated independent test set.
- Sentiment labels in the generated chat CSV are heuristic and are not independent gold-standard labels; review them before interpreting VADER scores.
- Matching agreement against `class_to_food.csv` measures agreement with the reviewed mapping, not independent semantic correctness.
- Diet/mood tags in `Selective_Diet_Mapping.csv` are rule-generated.
- Vision evaluation cannot include absent external datasets. Keep the actual dataset roots and mapping review files available locally and report which datasets produced valid rows.
- The 0.59 vision reject threshold inherited from the existing evaluation is provisional. The chat nudge threshold is selected on validation data and capped at 0.90 because synthetic templates can saturate confidence; both thresholds require real held-out calibration. Calibrate it on held-out correct/incorrect predictions before product use.

For a dataset class with more than one acceptable `best.pt` label, put the pipe-separated labels in `accepted_model_classes` (for example `label one|label two`) in the reviewed class-map CSV. The evaluation records both accepted-class top-1/top-3 and primary-label metrics. The known Kaggle `aloo_gobi` row is mapped to the classifier label `allo gobi`; this changes `image_class_map.csv` at runtime only in the generated manifest and does not change the approved `class_to_food.csv`.


## Final code-quality pass

The patch adds a registered `slow` marker and `tests/test_vision_real_model_slow.py`. To execute the real-model test, put a sample photo under `data/test_images_own/<approved model class>/` and run `python -m pytest -m slow -q`. Without a local sample photo, this test skips rather than fabricating an image.

If Black is not installed, install it as a development tool (do not blindly add it to the runtime requirements): `python -m pip install black`. Then format the code with `python -m black app.py config.py vyndra scripts tests notebooks`. Finally run `python -m pytest -q` again. Before changing `requirements.txt`, capture `python -m pip freeze | grep -i -E "ultralytics|torch|numpy|flask|scikit|networkx"` from your own active `env`, as required by the project brief.
