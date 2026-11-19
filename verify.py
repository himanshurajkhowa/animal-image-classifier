"""Verify historical preservation, current notebooks, model metrics and saved inference."""
import hashlib,json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,nbformat
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedGroupKFold,cross_val_score
from threadpoolctl import threadpool_limits
from common import ROOT,LABELS,SEED
from demo import Predictor

OUT=ROOT/'outputs/improved'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--external-run',help='Verify a separately named external audit after deliberate retraining.')
args=parser.parse_args()
EXTERNAL=OUT if not args.external_run else OUT/'external_runs'/args.external_run
checks=[]
def record(name,**details):
    checks.append({'check':name,'passed':True,**details});print('PASS',name,flush=True)

plan=json.loads((OUT/'experiment_plan.json').read_text())
for name,digest in plan['historical_artifact_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((ROOT/'data/processed/split_manifest.csv').read_bytes()).hexdigest()==plan['manifest_sha256']
record('Original models, metrics, predictions and split preserved byte for byte')

manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv')
assert manifest.image_id.is_unique
for column in ['group_id','sha256','pixel_sha256']:
    assert manifest.groupby(column).split.nunique().max()==1
fold=np.empty(len(manifest),dtype=int)
cv=StratifiedGroupKFold(n_splits=20,shuffle=True,random_state=SEED)
for i,(_,index) in enumerate(cv.split(manifest,manifest.label,manifest.group_id)):fold[index]=i
assert np.array_equal(np.where(fold<14,'train',np.where(fold<17,'validation','test')),manifest.split)
record('Original 70/15/15 stratified group split reproduced without leakage')

for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    n=nbformat.read(path,4);nbformat.validate(n)
    for cell in n.cells:
        if cell.cell_type!='code':continue
        assert cell.execution_count is not None,path.name
        for out in cell.outputs:
            assert out.output_type!='error',path.name
            if out.output_type=='stream':
                assert not any(word in out.text for word in ['Traceback','ConvergenceWarning','Epoch 01','file_bytes','training_seconds'])
record('Four cleaned notebooks executed with no errors, convergence warnings or excessive diagnostics')

cache=np.load(ROOT/'data/processed/pretrained_features.npz')
assert np.array_equal(cache['image_ids'],manifest.image_id)
assert str(cache['manifest_sha256'])==plan['manifest_sha256']
features=cache['features'];assert features.shape==(len(manifest),576) and np.isfinite(features).all()
record('Frozen pretrained feature cache matches every manifest ID')

names=['binary_features','softmax_features','transfer_cnn','binary_weighted_pixels','softmax_weighted_pixels']
for name in names:
    result=pd.read_csv(OUT/f'{name}_predictions.csv')
    summary=json.loads((OUT/f'{name}_metrics.json').read_text())['summary']
    labels=LABELS[:2] if name.startswith('binary') else LABELS
    expected=manifest[manifest.split.eq('test')&manifest.label.isin(labels)]
    assert np.array_equal(result.image_id,expected.image_id)
    assert np.array_equal(result.truth,expected.label_id)
    probability=result[[f'p_{label}' for label in labels]].to_numpy()
    assert np.isfinite(probability).all() and (probability>=0).all() and (probability<=1).all()
    np.testing.assert_allclose(probability.sum(1),1,atol=1e-5)
    assert np.array_equal(probability.argmax(1),result.prediction)
    for key,fn in [('accuracy',accuracy_score),('precision_macro',precision_score),('recall_macro',recall_score),('f1_macro',f1_score)]:
        kwargs={} if key=='accuracy' else {'average':'macro','zero_division':0}
        np.testing.assert_allclose(fn(result.truth,result.prediction,**kwargs),summary[key],atol=1e-12)
    record(name+' metrics and identical historical test IDs verified')

for name in ['binary_features','softmax_features']:
    labels=LABELS[:2] if name.startswith('binary') else LABELS
    keep=manifest.label.isin(labels).to_numpy();m=manifest[keep];x=features[keep]
    train=m.split.eq('train').to_numpy();test=m.split.eq('test').to_numpy()
    model=joblib.load(ROOT/f'models/{name}.joblib')
    np.testing.assert_allclose(model['scale'].mean_,x[train].mean(0,dtype=np.float64),rtol=1e-6,atol=1e-7)
    expected=pd.read_csv(OUT/f'{name}_predictions.csv')[[f'p_{label}' for label in labels]].to_numpy()
    np.testing.assert_allclose(model.predict_proba(x[test]),expected,rtol=1e-5,atol=1e-7)
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    for fit,score in cv.split(x[train],m.label_id.to_numpy()[train],m.group_id.to_numpy()[train]):
        groups=m.group_id.to_numpy()[train];assert set(groups[fit]).isdisjoint(groups[score])
    with threadpool_limits(limits=4):
        scores=cross_val_score(model,x[train],m.label_id.to_numpy()[train],groups=m.group_id.to_numpy()[train],
                               cv=cv,scoring='f1_macro',n_jobs=1)
    selected=json.loads((OUT/f'{name}_selection.json').read_text())
    np.testing.assert_allclose(scores.mean(),selected['cv_macro_f1'],atol=1e-5)
    record(name+' fold-only scaling and selected-setting group CV reproduced')

predictor=Predictor()
lookup=manifest.set_index('image_id')
for name,folder in [(n,OUT) for n in names]+[(n,ROOT/'outputs') for n in ['binary','softmax','cnn']]:
    prediction=pd.read_csv(folder/f'{name}_predictions.csv')
    selected=pd.concat([prediction[prediction.truth.eq(i)].head(3) for i in sorted(prediction.truth.unique())])
    labels=LABELS[:2] if name.startswith('binary') else LABELS
    for _,row in selected.iterrows():
        actual=predictor.predict(ROOT/lookup.loc[row.image_id,'path'],name)
        expected=row[[f'p_{label}' for label in labels]].to_numpy(dtype=float)
        np.testing.assert_allclose(list(actual['probabilities'].values()),expected,rtol=0.003,atol=0.0003)
        assert actual['animal']==labels[int(row.prediction)]
    record(name+' source-image saved inference verified',sample_images=len(selected))

external=pd.read_csv(OUT/'external_manifest.csv')
external_plan=json.loads((OUT/'external_plan.json').read_text())
assert hashlib.sha256((OUT/'external_manifest.csv').read_bytes()).hexdigest()==external_plan['frozen_manifest_sha256']
assert external.image_id.is_unique
assert set(external.license)<=set(['cc0','cc-by','cc-by-sa'])
assert external.nearest_original_or_prior_external_phash.gt(6).all()
for name,digest in json.loads((EXTERNAL/'external_model_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
for _,row in external.iterrows():
    assert hashlib.sha256((ROOT/row.path).read_bytes()).hexdigest()==row.sha256
for name in ['binary_features','softmax_features','transfer_cnn']:
    for scope in ['all_observations','animal_photos']:
        record_name=f'external_{name}_{scope}'
        p=pd.read_csv(EXTERNAL/f'{record_name}_predictions.csv')
        s=json.loads((EXTERNAL/f'{record_name}_metrics.json').read_text())['summary']
        expected=external[external.label.isin(LABELS[:2])] if name.startswith('binary') else external
        if scope=='animal_photos':expected=expected[expected.eligible_animal_photo]
        assert np.array_equal(p.image_id,expected.image_id)
        np.testing.assert_allclose(accuracy_score(p.truth,p.prediction),s['accuracy'])
        np.testing.assert_allclose(f1_score(p.truth,p.prediction,average='macro'),s['f1_macro'])
record('External source licenses, frozen selection, model fingerprints and both audit scopes verified',
       observations=len(external),eligible_animal_photos=int(external.eligible_animal_photo.sum()))

(OUT/'verification.json').write_text(json.dumps({'status':'passed','checks':checks,
 'limitations':['Historical test already inspected.','External audit is small, dated by contributors and not representative.',
 'Hashing cannot prove complete source independence.','ImageNet pretraining is part of improved features.']},indent=2))
print('Verification saved to outputs/improved/verification.json')
