"""Evaluate fixed models on the frozen, separately collected public-photo audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from common import ROOT,LABELS
from demo import Predictor
from train_improved import OUT,save_result

def wilson(correct,total):
    z=1.96;p=correct/total;denominator=1+z*z/total
    centre=(p+z*z/(2*total))/denominator
    radius=z*((p*(1-p)/total+z*z/(4*total*total))**0.5)/denominator
    return [centre-radius,centre+radius]

def assess(download=False, run_name=None):
    destination=OUT
    if run_name:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run_name):raise ValueError('Use a simple run name.')
        destination=OUT/'external_runs'/run_name;destination.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'external_manifest.csv'
    plan=json.loads((OUT/'external_plan.json').read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==plan['frozen_manifest_sha256']
    rows=pd.read_csv(manifest_path)
    for _,row in rows.iterrows():
        path=ROOT/row.path
        if not path.exists():
            if not download: raise FileNotFoundError('Run python assess_external.py --download first.')
            response=requests.get(row.photo_url,timeout=40);response.raise_for_status()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row.sha256
    models=['binary_features','softmax_features','transfer_cnn']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/f'models/{name}.joblib' for name in models[:2]]+
                     [ROOT/'models/transfer_cnn.pt',ROOT/'models/feature_extractor.pt']}
    freeze=destination/'external_model_freeze.json'
    if freeze.exists(): assert json.loads(freeze.read_text())==hashes, 'Models changed after external assessment.'
    else: freeze.write_text(json.dumps(hashes,indent=2))
    predictor=Predictor();summaries=[]
    for model in models:
        selected=rows[rows.label.isin(LABELS[:2])] if model.startswith('binary') else rows
        labels=LABELS[:2] if model.startswith('binary') else LABELS
        probability=np.array([list(predictor.predict(ROOT/row.path,model)['probabilities'].values())
                              for _,row in selected.iterrows()])
        for scope,mask in [('all_observations',np.ones(len(selected),dtype=bool)),
                           ('animal_photos',selected.eligible_animal_photo.to_numpy())]:
            truth=selected.label_id.to_numpy()[mask];predicted=probability[mask].argmax(1)
            summary=save_result(f'external_{model}_{scope}',truth,probability[mask],
                selected.loc[mask,'image_id'].to_numpy(),labels,
                {'assessment':'Small external audit, not used for training or tuning',
                 'sample_scope':scope,'accuracy_wilson_95':wilson(int((truth==predicted).sum()),len(truth))},folder=destination)
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(destination/'external_comparison.csv',index=False)
    print('External audit completed with model and manifest fingerprints fixed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Download only the fixed manifest images if missing.')
    parser.add_argument('--run-name',help='Record a retrained model audit separately; this sample is already inspected.')
    args=parser.parse_args()
    assess(args.download,args.run_name)
