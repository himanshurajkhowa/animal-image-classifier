"""Shared image preparation, CNN architecture and evaluation used by notebooks and demo."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
LABELS = ['Cat', 'Dog', 'Cow', 'Horse', 'Sheep']
SEED = 42

def pixels(source, size):
    # Use the same image conversion during training and prediction.
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)

def small_cnn():
    # Learn local patterns, then combine them into five class scores.
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 5))

def evaluate(name, truth, probability, ids, labels, model_path, elapsed, extra=None):
    prediction = probability.argmax(axis=1)
    report = classification_report(truth, prediction, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    metrics = dict(model=name, accuracy=report['accuracy'],
                   precision_macro=report['macro avg']['precision'], recall_macro=report['macro avg']['recall'],
                   f1_macro=report['macro avg']['f1-score'], test_images=len(truth),
                   file_bytes=model_path.stat().st_size, training_seconds=elapsed)
    if extra: metrics.update(extra)
    (ROOT / f'outputs/{name}_metrics.json').write_text(json.dumps({'summary':metrics,'class_report':report}, indent=2))
    predictions = pd.DataFrame({'image_id': ids, 'truth':truth,'prediction':prediction})
    for i, label in enumerate(labels): predictions[f'p_{label}'] = probability[:,i]
    predictions.to_csv(ROOT / f'outputs/{name}_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(truth, prediction, display_labels=labels, ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name.replace('_',' ').capitalize() + ' test predictions')
    fig.tight_layout(); fig.savefig(ROOT / f'outputs/{name}_confusion.png', dpi=180); plt.show()
    return metrics, pd.DataFrame(report).T

def show_predictions(name, manifest, count=8):
    predictions = pd.read_csv(ROOT / f'outputs/{name}_predictions.csv')
    rows = predictions.merge(manifest[['image_id','path']], on='image_id', validate='one_to_one')
    labels = LABELS[:2] if name == 'binary' else LABELS
    per_class = 2 if len(labels) == 2 else 1
    samples = []
    for correct in [True, False]:
        for label_id in range(len(labels)):
            matches = rows[(rows.truth == label_id) & ((rows.truth == rows.prediction) == correct)]
            samples.append(matches.head(per_class))
    rows = pd.concat(samples)
    columns = len(labels) * per_class
    fig, axes = plt.subplots(2, columns, figsize=(13, 6.5))
    for ax in axes.flat: ax.axis('off')
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        ax.imshow(pixels(ROOT / row.path, 160))
        label = labels[int(row.prediction)]
        ax.set_title(f'Label: {labels[int(row.truth)]}\nPredicted: {label} ({row[f"p_{label}"]:.1%})', fontsize=9, pad=9)
    fig.subplots_adjust(hspace=0.5, wspace=0.12, top=0.90, bottom=0.04)
    (ROOT / 'outputs/private').mkdir(exist_ok=True)
    fig.savefig(ROOT / f'outputs/private/{name}_examples.png', dpi=160); plt.show()
