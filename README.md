# Animal image classifier

Classify Cat, Dog, Cow, Horse and Sheep photographs with Logistic Regression. The project follows a simple sequence: prepare the data, learn binary classification with Sigmoid, extend it to five classes with Softmax, then compare against CNNs.

The original raw-pixel models are preserved. The improved models still use Logistic Regression as the final classifier, but receive image features from a frozen pretrained CNN. Those pretrained features are part of the method and explain much of the improvement.

## Results

Accuracy and macro-F1 on the **same historical Animals-10 test images**:

| Model | Previous accuracy | Improved accuracy | Previous macro-F1 | Improved macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Binary Logistic Regression | 72.73% | **98.26%** | 48.43% | **97.70%** |
| Five-class Logistic Regression | 47.48% | **94.23%** | 39.01% | **93.54%** |
| CNN comparison | 63.27% | 94.65% | 58.56% | 94.24% |

The binary task has 979 test images; the five-class tasks share 1,925. Macro-F1 averages the class F1 scores equally, so weak performance on smaller classes remains visible. Notebook summaries also show precision and recall as percentages.

The binary model's Cat recall rose from 8.03% to **95.98%**. It now predicts Dog for 74.87% of test images, close to the actual 74.57%, compared with 94.08% previously. Five-class Dog predictions fell from 55.06% to 38.03%, close to the actual 37.92%.

![Historical model comparison](outputs/improved/performance_comparison.png)

### A separate external check

The historical test results were already inspected before this improvement study. They are useful for comparison, but they are not an untouched final assessment.

After fixing the models, we also evaluated a separately collected iNaturalist sample with observation dates in 2025–2026. Selection used photo licenses, source IDs and duplicate checks, without model predictions. Visual review excluded 14 observations showing only bones, teeth, tracks or dung, before inference. Difficult animal photographs were retained, and excluded observations were not replaced.

| Model | External animal photos | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Binary Logistic Regression | 37 | **64.86%** | 64.76% |
| Five-class Logistic Regression | 86 | **58.14%** | 58.35% |
| CNN comparison | 86 | 53.49% | 52.46% |

This sample contains distant, dark and partly hidden animals. It shows that the strong Animals-10 scores do **not** establish 85%+ accuracy on arbitrary new photos. The external sample is small and not representative of every use case. Contributor dates and imperfect hashing cannot prove complete source independence from either dataset or pretraining. Results for all 100 observations, including non-animal remains, and Wilson accuracy intervals are also saved in `outputs/improved/`. No model was tuned after seeing these results.

## Data and preparation

[Animals-10 by Alessio Corrado](https://www.kaggle.com/datasets/alessiocorrado99/animals10), version 2, contains 26,179 images in ten Italian-labelled folders. We select:

| Folder | Class | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `gatto` | Cat | 1,168 | 251 | 249 |
| `cane` | Dog | 3,402 | 731 | 730 |
| `mucca` | Cow | 1,305 | 280 | 280 |
| `cavallo` | Horse | 1,837 | 393 | 393 |
| `pecora` | Sheep | 1,274 | 273 | 273 |

The selected folders contain 12,840 files. Original preparation found no corrupt images or exact pixel duplicates and excluded one visually confirmed label error, leaving **12,839 images**. Perceptual hashing found 1,108 candidate related pairs. Detected connected groups stay in one split, including uncertain matches. The target 70/15/15 proportions produce 8,986 / 1,928 / 1,925 images because groups remain intact. Seed 42 and the original image IDs are unchanged in the improvement study.

Kaggle lists GPL-2.0, but the collection came from Google Images without individual photo credits. Dataset photographs and image-bearing notebook outputs are therefore kept out of Git. See the [dataset license record](outputs/dataset_license.md). The external audit uses only CC0, CC BY or CC BY-SA photographs, with source links, credits, licenses and hashes recorded in its [fixed manifest](outputs/improved/external_manifest.csv). External images also stay local.

## How the models work

- Binary Logistic Regression computes a linear score and uses Sigmoid to estimate Dog probability. The threshold is 0.5; Cat is 0 and Dog is 1.
- Multinomial Logistic Regression learns one score per animal. Softmax converts the scores into probabilities that sum to one, and the largest determines the label.
- The improved input is a 576-value feature vector from frozen ImageNet-pretrained MobileNetV3-Small. Its documented preprocessing resizes the short edge to 256, centre-crops to 224 × 224, and applies ImageNet normalization. Logistic Regression then learns the classification weights. This is **pretrained-feature Logistic Regression**, not a purely raw-pixel model.
- The CNN comparison freezes the same backbone and trains a small five-class neural head. The original CNN trained from scratch is retained.

Three stratified group folds inside the training set select C from 0.01, 0.1 and 1, with or without balanced class weights. StandardScaler is inside the Pipeline, so each fold fits scaling only on its training images. Both feature classifiers selected C = 0.01 without class weighting by CV macro-F1. Validation is reported separately; no test score selected settings.

A small raw-pixel experiment also compared balanced class weights, 24/32-pixel inputs and two regularization strengths. Weighting improved macro-F1 but lowered accuracy: the selected binary model scored 60.88% accuracy / 55.65% macro-F1, and the five-class model scored 39.84% / 39.40%. It did not solve the weak representation. All candidates and CV evidence remain in supporting outputs.

The CNN head uses training-only crop/flip augmentation, inverse-frequency class weights and Adam. Validation loss chooses a checkpoint within eight epochs, with patience three. Its backbone was not fine-tuned. Training curves and full epoch records are saved.

## Setup

Tested on an Apple M3 MacBook Air with 8 GB RAM and **Python 3.12.13**. Use Python 3.12. CPU inference works; training uses Apple MPS when available. Reserve about 3 GB for dependencies, data and generated files. No paid service or AI API key is needed.

```bash
cd "$HOME/Desktop/animal classifier"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `.venv` already exists, activate it instead of recreating it. A fresh clone uses the same commands from its own folder.

| Library | Purpose |
| --- | --- |
| Pillow, torchvision | Read images and apply the correct transforms |
| NumPy, pandas | Store features, labels, splits and result tables |
| Matplotlib | Plot distributions, curves, confusion matrices and predictions |
| scikit-learn | Logistic Regression, Pipelines, group CV and metrics |
| PyTorch | Frozen feature extraction and CNN training |
| ImageHash | Identify candidate related images |
| SciPy, joblib | Sigmoid/Softmax calculations and saved classifiers |
| Kaggle, requests | Download the original data and fixed external images |
| Jupyter, nbclient | Read and execute notebooks |

## Prediction demo

The saved models and preprocessing are included. Prediction does not require the training dataset or a new pretrained-weight download.

```bash
python demo.py
```

Open [the local demo](http://127.0.0.1:8765), select a model, choose an image and click Predict animal. Five-class Logistic Regression is the default. Binary mode supports only Cat and Dog. The page shows all supported-class probabilities. Images are processed in memory and are not saved. Stop the server with Ctrl+C.

For a single file:

```bash
python demo.py --image "/absolute/path/to/animal.jpg" --model softmax_features
```

Other choices include `binary_features`, `transfer_cnn`, and the original `binary`, `softmax` and `cnn`. Each saved model uses its own preprocessing. These are uncalibrated estimates: an unrelated animal or object will still receive a supported label.

## Reproduce the study

Download the original dataset with its official client:

```bash
mkdir -p data/raw data/processed
kaggle datasets download -d alessiocorrado99/animals10 -p data/raw --unzip
```

The public download worked without credentials during this run. If authentication is required, use `kaggle auth login`. Keep credentials outside the repository.

Read or execute the four numbered notebooks in order:

1. Explore and prepare animal images.
2. Cats vs dogs with Logistic Regression and Sigmoid.
3. Five animals with multinomial Logistic Regression and Softmax.
4. CNN comparison and evaluation.

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

The useful `.gitattributes` rule clears outputs on Git check-in while preserving executed notebooks locally. Enable it after cloning:

```bash
git config filter.notebook-clean.clean '.venv/bin/python run_notebooks.py --strip'
git config filter.notebook-clean.required true
```

## Limitations and future work

Training-label review found clear errors and mixed-animal scenes. Candidate flags are documented without automatic relabelling or changing the frozen comparison. The pretrained extractor's centre crop can miss small or off-centre animals. The external audit demonstrates this domain gap; label quality, composition and new data need further work. The classifier cannot detect unknown animals, locate animals in a scene or guarantee confidence calibration.

Streamlit Community Cloud is the preferred future host. A later interface can add phone-camera input, upload, class probabilities and at least ten individually licensed example images with credits. Deployment, that example gallery and unknown-animal detection are future work. Nothing has been deployed, and Vercel remains untouched. See the [deployment note](outputs/deployment_feasibility.md).

## References

- [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- [Official Logistic Regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Stratified group cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Pretrained feature extractor and documented transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
- [iNaturalist API](https://www.inaturalist.org/pages/api+reference) and [photo licensing](https://help.inaturalist.org/en/support/solutions/articles/151000169918-can-i-use-the-photos-and-sounds-that-are-posted-on-inaturalist-)
