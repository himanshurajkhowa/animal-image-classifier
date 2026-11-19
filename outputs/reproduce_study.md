# Reproduce the study

Install the research dependencies and download Animals-10 as described in the [project README](../README.md). Run the four notebooks in order. The research environment includes PyTorch and torchvision for feature extraction and CNN training, scikit-learn for Logistic Regression, and Jupyter for the notebooks.

For a prediction from the command line, install the research requirements and run:

```bash
python demo.py --image "/absolute/path/to/animal.jpg" --model softmax_features
```

Other saved models include `binary_features`, `transfer_cnn`, `binary`, `softmax` and `cnn`. Run `python demo.py` to open the original local comparison interface.

For Jupyter:

```bash
python -m ipykernel install --prefix .venv --name animal-classifier --display-name "Animal classifier Python"
JUPYTER_PATH="$PWD/.venv/share/jupyter" python -m notebook notebooks
```

Choose the Animal classifier Python kernel. The notebooks reuse completed preparation and saved experiments by default. On a fresh data download, notebook 01 builds the processed arrays. Set `RETRAIN = True` in a model notebook only when deliberately retraining it. Detailed searches and logs are kept outside the main notebook output.

For a complete training reproduction after data preparation:

```bash
python train_improved.py --stage all
python run_notebooks.py
python assess_external.py --download --run-name reproduction
python verify.py --external-run reproduction
```

Feature extraction reuses the included frozen backbone. If that file is absent, it downloads official pretrained weights into the ignored `models/pretrained_cache/` folder. The feature cache is tied to the split-manifest hash. Use `--stage logistic` or `--stage transfer` for a specific part, and `--resume` to skip completed Logistic Regression experiments. MPS training can vary slightly between runs. The external assessor freezes model fingerprints and refuses to silently overwrite the original audit with changed models; `--run-name` records a retrained version separately, and `--external-run` verifies that named audit. Use a new name for each changed model. Reusing these external images is a reproducibility check, not a fresh independent test.

The original source and notebook version is preserved by Git tag `baseline-raw-pixels`. Original models and results keep their filenames. New experiment records are under `outputs/improved/`, including the plan, diagnostic review, candidate settings, CV scores, validation behaviour, historical predictions, external audit and verification record.

The `.gitattributes` rule clears outputs on Git check-in while preserving executed notebooks locally. Enable it after cloning:

```bash
git config filter.notebook-clean.clean '.venv/bin/python run_notebooks.py --strip'
git config filter.notebook-clean.required true
```
