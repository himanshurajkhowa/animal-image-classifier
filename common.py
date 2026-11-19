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

# Notebook summaries keep the main results readable; detailed records stay in outputs.
def result_summary(name, labels, improved=True):
    from IPython.display import display
    folder = ROOT / ('outputs/improved' if improved else 'outputs')
    result = json.loads((folder / f'{name}_metrics.json').read_text())
    summary, report = result['summary'], result['class_report']
    if len(labels) == 2:
        values = {'Accuracy':summary['accuracy'], 'Precision (Dog)':report['Dog']['precision'],
                  'Recall (Dog)':report['Dog']['recall'], 'F1 (Dog)':report['Dog']['f1-score'],
                  'Macro-F1':summary['f1_macro']}
    else:
        values = {'Accuracy':summary['accuracy'], 'Precision (macro)':summary['precision_macro'],
                  'Recall (macro)':summary['recall_macro'], 'F1 (macro)':summary['f1_macro']}
    display(pd.DataFrame({'Score':[f'{value:.2%}' for value in values.values()]},index=list(values)))
    predictions = pd.read_csv(folder / f'{name}_predictions.csv')
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    ConfusionMatrixDisplay.from_predictions(predictions.truth, predictions.prediction,
        display_labels=labels,ax=ax,colorbar=False,cmap='Blues')
    ax.set_title('Predictions on the historical test split');fig.tight_layout()
    fig.savefig(folder / f'{name}_confusion.png',dpi=160);plt.show()

def prediction_examples(name, labels, improved=True):
    folder = ROOT / ('outputs/improved' if improved else 'outputs')
    manifest = pd.read_csv(ROOT / 'data/processed/split_manifest.csv')
    predictions = pd.read_csv(folder / f'{name}_predictions.csv').merge(
        manifest[['image_id','path']],on='image_id',validate='one_to_one')
    rows=[]
    for correct in [True,False]:
        for label_id in range(len(labels)):
            matches=predictions[(predictions.truth==label_id)&((predictions.truth==predictions.prediction)==correct)]
            rows.append(matches.head(1))
    sample=pd.concat(rows)
    fig,axes=plt.subplots(2,len(labels),figsize=(11,5.6),squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax,(_,row) in zip(axes.flat,sample.iterrows()):
        ax.imshow(pixels(ROOT / row.path,160));predicted=labels[int(row.prediction)]
        ax.set_title(f'Label: {labels[int(row.truth)]}\nPrediction: {predicted}\n{row[f"p_{predicted}"]:.1%}',fontsize=9)
    fig.subplots_adjust(hspace=0.6,wspace=0.15,top=0.84,bottom=0.04)
    fig.savefig(ROOT / f'outputs/private/{name}_improved_examples.png',dpi=150);plt.show()

def comparison_table():
    rows=[]
    names=[('Binary LR · pixels','binary',False),('Binary LR · pretrained features','binary_features',True),
           ('Five-class LR · pixels','softmax',False),('Five-class LR · pretrained features','softmax_features',True),
           ('Small CNN','cnn',False),('Transfer CNN comparison','transfer_cnn',True)]
    for label,name,improved in names:
        folder=ROOT / ('outputs/improved' if improved else 'outputs')
        summary=json.loads((folder / f'{name}_metrics.json').read_text())['summary']
        rows.append({'Model':label,'Accuracy':summary['accuracy'],'Precision (macro)':summary['precision_macro'],
                     'Recall (macro)':summary['recall_macro'],'Macro-F1':summary['f1_macro']})
    table=pd.DataFrame(rows).set_index('Model')
    table.to_csv(ROOT / 'outputs/improved/model_comparison.csv')
    return table.map(lambda value:f'{value:.2%}')
