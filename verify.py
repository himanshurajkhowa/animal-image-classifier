"""Verify saved experiment records, split reproducibility and model inference."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import joblib,nbformat,torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedGroupKFold,cross_val_score
from threadpoolctl import threadpool_limits
from common import ROOT,LABELS,SEED,pixels,small_cnn

checks=[]
def record(name,**details):
    checks.append({'check':name,'passed':True,**details}); print('PASS',name,flush=True)

manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv')
arrays=np.load(ROOT/'data/processed/images.npz')
assert manifest.image_id.is_unique
assert np.array_equal(manifest.image_id,arrays['image_ids'])
assert set(manifest.label)==set(LABELS)
assert not set(pd.read_csv(ROOT/'outputs/manual_review_decisions.csv').image_id)&set(manifest.image_id)
for key in ['group_id','sha256','pixel_sha256']:
    assert manifest.groupby(key).split.nunique().max()==1
record('Saved array IDs, labels, exclusion and duplicate isolation',retained=len(manifest))

# Reconstruct the exact final fold assignment from a fixed seed and saved groups.
fold=np.empty(len(manifest),dtype=int)
cv=StratifiedGroupKFold(n_splits=20,shuffle=True,random_state=SEED)
for i,(_,index) in enumerate(cv.split(manifest,manifest.label,manifest.group_id)): fold[index]=i
expected=np.where(fold<14,'train',np.where(fold<17,'validation','test'))
assert np.array_equal(expected,manifest.split)
record('70/15/15 group split reproduced',counts=manifest.split.value_counts().to_dict())

for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    n=nbformat.read(path,as_version=4)
    nbformat.validate(n)
    for cell in n.cells:
        if cell.cell_type!='code': continue
        assert cell.execution_count is not None, f'Unexecuted cell: {path.name}'
        for out in cell.outputs:
            assert out.output_type!='error',f'Error output: {path.name}'
            if out.output_type=='stream':
                assert 'ConvergenceWarning' not in out.text
                assert 'Traceback' not in out.text
record('All four notebooks fully executed with no error outputs or convergence warnings')

x=arrays['pixels24'].reshape(len(manifest),-1).astype(np.float32)/255
five_test=manifest.split.eq('test').to_numpy()
for name in ['binary','softmax','cnn']:
    result=pd.read_csv(ROOT/f'outputs/{name}_predictions.csv')
    saved=json.loads((ROOT/f'outputs/{name}_metrics.json').read_text())['summary']
    probability=result[[f'p_{label}' for label in (LABELS[:2] if name=='binary' else LABELS)]].to_numpy()
    assert np.isfinite(probability).all() and (probability>=0).all() and (probability<=1).all()
    np.testing.assert_allclose(probability.sum(1),1,atol=1e-5)
    assert np.array_equal(probability.argmax(1),result.prediction)
    for metric,function in [('accuracy',accuracy_score),('precision_macro',precision_score),
                            ('recall_macro',recall_score),('f1_macro',f1_score)]:
        kwargs={} if metric=='accuracy' else {'average':'macro','zero_division':0}
        measured=function(result.truth,result.prediction,**kwargs)
        np.testing.assert_allclose(measured,saved[metric],atol=1e-12)
    indices=manifest.set_index('image_id').loc[result.image_id].reset_index()
    assert indices.split.eq('test').all()
    assert np.array_equal(indices.label_id,result.truth)
    if name!='cnn':
        model=joblib.load(ROOT/f'models/{name}.joblib')
        positions=manifest.index[manifest.image_id.isin(result.image_id)].to_numpy()
        np.testing.assert_allclose(model.predict_proba(x[positions]),probability,rtol=1e-5,atol=1e-7)
        train=manifest.split.eq('train').to_numpy()
        if name=='binary': train = train & manifest.label.isin(LABELS[:2]).to_numpy()
        np.testing.assert_allclose(model['scale'].mean_,x[train].mean(0,dtype=np.float64),rtol=1e-6,atol=1e-7)
        # Repeat only the selected regularization setting to check the recorded CV mean.
        fold_cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
        for fit,score in fold_cv.split(x[train],manifest.label_id[train],manifest.group_id[train]):
            groups=manifest.group_id[train].to_numpy()
            assert set(groups[fit]).isdisjoint(groups[score])
        with threadpool_limits(limits=4):
            scores=cross_val_score(model,x[train],manifest.label_id[train],groups=manifest.group_id[train],
                                   cv=fold_cv,scoring='f1_macro',n_jobs=1)
        cv_results=pd.read_csv(ROOT/f'outputs/{name}_cross_validation.csv')
        expected=cv_results.loc[np.isclose(cv_results.param_classifier__C,saved['selected_C']),'mean_test_score'].iloc[0]
        np.testing.assert_allclose(scores.mean(),expected,atol=1e-5)
        record(name+' training-only scaler and selected-C group cross-validation reproduced')
    record(name+' saved predictions and test metrics verified')

a=pd.read_csv(ROOT/'outputs/softmax_predictions.csv');b=pd.read_csv(ROOT/'outputs/cnn_predictions.csv')
assert a[['image_id','truth']].equals(b[['image_id','truth']])
record('Softmax and CNN have identical held-out image IDs and labels',test_images=len(a))

# Independently decode source images and use the same function as the upload demo.
from demo import Predictor
predictor=Predictor()
example_ids=[]
for label in LABELS:
    example_ids.extend(manifest[manifest.label.eq(label)&manifest.split.eq('test')].image_id.head(3))
lookup=manifest.set_index('image_id')
for kind,result in [('softmax',a),('cnn',b)]:
    baseline=result.set_index('image_id')
    timing=[]
    for image_id in example_ids:
        source=ROOT/lookup.loc[image_id,'path']
        start=time.perf_counter(); prediction=predictor.predict(source,kind); timing.append(time.perf_counter()-start)
        expected=baseline.loc[image_id,[f'p_{label}' for label in LABELS]].to_numpy(dtype=float)
        np.testing.assert_allclose(list(prediction['probabilities'].values()),expected,rtol=1e-3,atol=1e-4)
    record(kind+' raw-image demo parity',images=len(example_ids),
           mean_milliseconds_including_decode=1000*float(np.mean(timing)),
           median_milliseconds_including_decode=1000*float(np.median(timing)))

# Record actual artifacts and installed package consistency in the final evidence file.
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/processed/split_manifest.csv',
        ROOT/'models/preprocessing.json',ROOT/'models/binary.joblib',ROOT/'models/softmax.joblib',ROOT/'models/cnn.pt']}
summary={'status':'passed','checks':checks,'artifact_sha256':hashes,
         'notes':['MPS training may vary slightly across runs or devices.',
                  'Perceptual hashing cannot detect every related source image.',
                  'CPU inference latency is a local measurement, not a hosting benchmark.']}
(ROOT/'outputs/verification.json').write_text(json.dumps(summary,indent=2))
print('Verification saved to outputs/verification.json')
