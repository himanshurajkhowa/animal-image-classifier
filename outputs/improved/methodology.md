# Improvement study

Logistic Regression is the main classifier. The learning sequence remains data preparation, binary Sigmoid classification, five-class Softmax classification and a CNN comparison. Frozen pretrained image features replace weak raw-pixel features in the improved Logistic Regression models. Feature extraction is a neural-network step; the final decision model is still Logistic Regression.

## What was fixed before experimentation

The experiment plan is saved in `experiment_plan.json`. The class list, labels, raw files and 70/15/15 split IDs are unchanged. Original models and records retain their filenames. The Git tag `baseline-raw-pixels` preserves the earlier source and notebooks. New experiment records are under `outputs/improved/`.

The raw-pixel ablation compares balanced Logistic Regression at 24 and 32 pixels with C of 0.001 and 0.01, selecting by validation macro-F1. A three-fold training-only group CV score is recorded for the selected setting. The pretrained-feature classifiers use a Pipeline so scaling is fitted separately inside each fold; three group-aware folds compare C of 0.01, 0.1 and 1, and optional balanced class weights. CV macro-F1 selects each pipeline. Validation is checked separately; no refit includes validation or test examples.

The frozen pretrained extractor uses official ImageNet1K MobileNetV3-Small weights, 256-pixel short-edge bilinear resizing, a 224-pixel centre crop and the documented channel normalization. Its pooled 576 features are cached in manifest order. Learned scaling is never fitted on the feature cache as a whole.

The comparison CNN freezes the same backbone and trains a 576-128-5 neural head with training-only random resized crops and horizontal flips. Batch normalization in the frozen backbone stays in evaluation mode. Class weights come from training counts. Validation loss chooses the checkpoint within eight epochs, with patience three. Fine-tuning the backbone is deferred to keep this a focused comparison; it is not needed to explain Logistic Regression.

## Evaluation limits

The original Animals-10 test results were already inspected before this improvement study. New scores on those IDs are historical benchmark results, not a genuinely untouched final assessment. Model choices use training and validation only, and we do not alter test labels or balance the test set. At planning time no independent collection was available locally. A later independent-source audit is documented below; its limited sample does not establish broad real-world accuracy. We do not relabel a new split as independent. ImageNet pretraining also makes complete source-overlap guarantees impossible with this dataset's missing image provenance.

## References reviewed

- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Official pretrained model and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Example animal-classification repository](https://github.com/techno-ankitt/animal-image-classifier): reviewed for approach only. It uses different classes, MobileNetV2 and a different split; its reported accuracy is not directly comparable. Our implementation was written for this project.

## Separately collected external audit

The initial plan noted that no independent collection was available locally. A subsequent source search found suitable licensed iNaturalist photographs. A separate selection plan was fixed before inference in `external_plan.json`: 20 observations per class, dated 2025–2026, one per observer per class, and only CC0/CC BY/CC BY-SA photos. The manifest records photo-page links, credits, hashes and pHash distances from the original data and prior external selections.

Visual review covered all 100 observations before model predictions. Fourteen consisted of isolated remains, teeth, dung or footprints. Their exclusion from the animal-photo scope was recorded before inference; they were not replaced. All dark, distant, occluded and otherwise difficult animal photographs were retained. Both the full 100-observation audit and the eligible 86-photo scope are reported. The binary subset has 37 eligible photographs.

Model and manifest fingerprints were frozen before external scoring. None of the new results led to retuning, category changes, relabelling or sample replacement. This is an independent-source audit of a small convenience sample, with contributor-supplied observation dates and labels; it is not proof of zero pretraining overlap or a representative population benchmark. The much weaker external results are reported prominently alongside the historical test scores.

A separate review of 15 high-confidence training disagreements found clear wrong labels, mixed-animal images and some model errors. These remain flags only in `training_review_candidates.csv`; no training labels or comparison artifacts were changed after the external assessment.
