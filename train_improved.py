"""Reproduce the improved Logistic Regression models and the CNN comparison."""
from pathlib import Path
import argparse, hashlib, json, random, time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from common import ROOT, LABELS, SEED

OUT = ROOT / 'outputs/improved'
OUT.mkdir(exist_ok=True)
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
FEATURE_PATH = ROOT / 'data/processed/pretrained_features.npz'

def load_rgb(source):
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert('RGB')

def feature_network(pretrained=True):
    network = mobilenet_v3_small(weights=WEIGHTS if pretrained else None)
    network.classifier = nn.Identity()
    return network

class AnimalImages(Dataset):
    def __init__(self, manifest, transform):
        self.rows = manifest.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return self.transform(load_rgb(ROOT / row.path)), int(row.label_id)

def save_result(name, truth, probability, ids, labels, details, folder=OUT):
    predicted = probability.argmax(1)
    report = classification_report(truth, predicted, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    summary = dict(accuracy=report['accuracy'],
        precision_macro=report['macro avg']['precision'], recall_macro=report['macro avg']['recall'],
        f1_macro=report['macro avg']['f1-score'], test_images=len(truth), **details)
    summary['model'] = name
    (folder / f'{name}_metrics.json').write_text(json.dumps({'summary':summary,'class_report':report}, indent=2))
    table = pd.DataFrame({'image_id':ids,'truth':truth,'prediction':predicted})
    for i,label in enumerate(labels): table[f'p_{label}'] = probability[:,i]
    table.to_csv(folder / f'{name}_predictions.csv', index=False)
    return summary

def extract_features():
    manifest = pd.read_csv(ROOT / 'data/processed/split_manifest.csv')
    manifest_hash = hashlib.sha256((ROOT / 'data/processed/split_manifest.csv').read_bytes()).hexdigest()
    if FEATURE_PATH.exists():
        cached = np.load(FEATURE_PATH)
        assert np.array_equal(cached['image_ids'],manifest.image_id)
        assert str(cached['manifest_sha256']) == manifest_hash
        return cached['features']
    torch.manual_seed(SEED); torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / 'models/pretrained_cache'))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    backbone_path = ROOT / 'models/feature_extractor.pt'
    if backbone_path.exists():
        saved = torch.load(backbone_path,map_location='cpu',weights_only=True)
        network = feature_network(pretrained=False)
        network.load_state_dict(saved['state_dict'])
    else:
        network = feature_network()
        torch.save({'state_dict':network.state_dict(),'weights':'IMAGENET1K_V1',
                    'feature_dimension':576},backbone_path)
    network = network.eval().to(device)
    loader = DataLoader(AnimalImages(manifest,WEIGHTS.transforms()), batch_size=32, shuffle=False, num_workers=0)
    parts=[]; started=time.perf_counter()
    with torch.inference_mode():
        for step,(images,_) in enumerate(loader):
            parts.append(network(images.to(device)).cpu().numpy())
            if step % 50 == 0: print(f'Feature batches {step+1}/{len(loader)}',flush=True)
    features=np.concatenate(parts)
    np.savez_compressed(FEATURE_PATH,features=features,image_ids=manifest.image_id.to_numpy(dtype=str),
                        manifest_sha256=manifest_hash)
    config={'architecture':'MobileNetV3-Small','weights':'IMAGENET1K_V1','role':'frozen image feature extractor',
            'features':576,'orientation':'EXIF transpose','color':'RGB','resize_short_edge':256,
            'center_crop':224,'interpolation':'bilinear','antialias':True,'pixel_scale':'divide by 255',
            'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225],
            'manifest_sha256':manifest_hash,'device':str(device),'extraction_seconds':time.perf_counter()-started}
    (ROOT / 'models/improved_preprocessing.json').write_text(json.dumps(config,indent=2))
    return features

def raw_experiments(binary):
    manifest=pd.read_csv(ROOT / 'data/processed/split_manifest.csv')
    labels=LABELS[:2] if binary else LABELS
    keep=manifest.label.isin(labels).to_numpy(); manifest=manifest[keep].reset_index(drop=True)
    arrays=np.load(ROOT / 'data/processed/images.npz')
    # Compare two modest resolutions; every setting is chosen without the test split.
    resolutions={24:arrays['pixels24'][keep],32:np.stack([np.asarray(
        load_rgb(ROOT / p).resize((32,32),Image.Resampling.BILINEAR)) for p in manifest.path])}
    train=manifest.split.eq('train').to_numpy(); validation=manifest.split.eq('validation').to_numpy()
    y=manifest.label_id.to_numpy(); candidates=[]; fitted=[]
    for size,images in resolutions.items():
        x=images.reshape(len(manifest),-1).astype(np.float32)/255
        for c in [0.001,0.01]:
            model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(
                C=c,class_weight='balanced',max_iter=700,random_state=SEED))])
            with threadpool_limits(limits=4): model.fit(x[train],y[train])
            predicted=model.predict(x[validation])
            candidates.append({'size':size,'C':c,'class_weight':'balanced',
                'validation_accuracy':accuracy_score(y[validation],predicted),
                'validation_macro_f1':f1_score(y[validation],predicted,average='macro'),
                'train_accuracy':accuracy_score(y[train],model.predict(x[train]))})
            fitted.append(model)
    task='binary' if binary else 'softmax'
    table=pd.DataFrame(candidates); table.to_csv(OUT / f'{task}_raw_candidates.csv',index=False)
    index=int(table.validation_macro_f1.idxmax()); best=fitted[index]; setting=candidates[index]
    images=resolutions[setting['size']];x=images.reshape(len(manifest),-1).astype(np.float32)/255
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    with threadpool_limits(limits=4):
        scores=cross_val_score(best,x[train],y[train],groups=manifest.loc[train,'group_id'],cv=cv,scoring='f1_macro')
    name=task+'_weighted_pixels'
    joblib.dump({'pipeline':best,'size':setting['size'],'labels':labels},ROOT / f'models/{name}.joblib',compress=3)
    test=manifest.split.eq('test').to_numpy()
    return save_result(name,y[test],best.predict_proba(x[test]),manifest.loc[test,'image_id'].to_numpy(),labels,
                       {**setting,'cv_macro_f1':float(scores.mean()),'features':'flattened RGB pixels',
                        'assessment':'Previously inspected Animals-10 test split'})

def train_logistic(binary):
    features=extract_features()
    manifest=pd.read_csv(ROOT / 'data/processed/split_manifest.csv')
    labels=LABELS[:2] if binary else LABELS
    keep=manifest.label.isin(labels).to_numpy(); manifest=manifest[keep].reset_index(drop=True);x=features[keep]
    y=manifest.label_id.to_numpy();train=manifest.split.eq('train').to_numpy()
    validation=manifest.split.eq('validation').to_numpy();test=manifest.split.eq('test').to_numpy()
    pipeline=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(
        solver='lbfgs',max_iter=1000,random_state=SEED))])
    # Cross-validation is entirely within the original training split.
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    for fit,score in cv.split(x[train],y[train],manifest.loc[train,'group_id']):
        groups=manifest.loc[train,'group_id'].to_numpy()
        assert set(groups[fit]).isdisjoint(groups[score])
    search=GridSearchCV(pipeline,{'classifier__C':[0.01,0.1,1.0],
        'classifier__class_weight':[None,'balanced']},cv=cv,scoring='f1_macro',n_jobs=1,return_train_score=True)
    started=time.perf_counter()
    with threadpool_limits(limits=4):
        search.fit(x[train],y[train],groups=manifest.loc[train,'group_id'])
    name='binary_features' if binary else 'softmax_features'
    pd.DataFrame(search.cv_results_).to_csv(OUT / f'{name}_cross_validation.csv',index=False)
    model=search.best_estimator_
    assert model['classifier'].n_iter_.max()<1000
    behaviour=[]
    for split,mask in [('train',train),('validation',validation)]:
        predicted=model.predict(x[mask])
        behaviour.append({'split':split,'accuracy':accuracy_score(y[mask],predicted),
                          'macro_f1':f1_score(y[mask],predicted,average='macro')})
    pd.DataFrame(behaviour).to_csv(OUT / f'{name}_behaviour.csv',index=False)
    model_path=ROOT / f'models/{name}.joblib';joblib.dump(model,model_path,compress=3)
    val=behaviour[1]
    selection={'model':name,'features':'576 frozen pretrained CNN features; Logistic Regression classifier',
               'C':model['classifier'].C,'class_weight':model['classifier'].class_weight,
               'cv_macro_f1':float(search.best_score_),'validation_accuracy':val['accuracy'],
               'validation_macro_f1':val['macro_f1'],'training_seconds':time.perf_counter()-started}
    (OUT / f'{name}_selection.json').write_text(json.dumps(selection,indent=2))
    # The selected model is now fixed. Test results do not trigger any new tuning.
    summary=save_result(name,y[test],model.predict_proba(x[test]),manifest.loc[test,'image_id'].to_numpy(),labels,
                       {**selection,'assessment':'Previously inspected Animals-10 test split'})
    print(name, 'validation accuracy',f"{val['accuracy']:.1%}",flush=True)
    return summary

def train_transfer():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);torch.set_num_threads(4)
    manifest=pd.read_csv(ROOT / 'data/processed/split_manifest.csv')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Reuse the same frozen features; apply training-only image augmentation below.
    network=feature_network(pretrained=False)
    saved=torch.load(ROOT / 'models/feature_extractor.pt',map_location='cpu',weights_only=True)
    network.load_state_dict(saved['state_dict'])
    for p in network.parameters(): p.requires_grad=False
    head=nn.Sequential(nn.Linear(576,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,5))
    model=nn.Sequential(network,head).to(device)
    augmentation=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train=manifest.split.eq('train');validation=manifest.split.eq('validation');test=manifest.split.eq('test')
    generator=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(AnimalImages(manifest[train],augmentation),batch_size=32,shuffle=True,
                            generator=generator,num_workers=0)
    val_loader=DataLoader(AnimalImages(manifest[validation],WEIGHTS.transforms()),batch_size=64,num_workers=0)
    counts=np.bincount(manifest.loc[train,'label_id'],minlength=5)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(5*counts),dtype=torch.float32,device=device))
    optimizer=torch.optim.Adam(head.parameters(),lr=0.001,weight_decay=0.0001)
    best=float('inf');waiting=0;history=[];started=time.perf_counter()
    for epoch in range(1,9):
        head.train();network.eval();total=correct=0;loss_total=0.0
        for images,targets in train_loader:
            images,targets=images.to(device),targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores=model(images);loss=criterion(scores,targets);loss.backward();optimizer.step()
            loss_total+=loss.item()*len(targets);correct+=(scores.argmax(1)==targets).sum().item();total+=len(targets)
        model.eval();val_loss=0.0;val_total=val_correct=0
        with torch.inference_mode():
            for images,targets in val_loader:
                images,targets=images.to(device),targets.to(device);scores=model(images)
                val_loss+=criterion(scores,targets).item()*len(targets)
                val_correct+=(scores.argmax(1)==targets).sum().item();val_total+=len(targets)
        row={'epoch':epoch,'train_loss':loss_total/total,'train_accuracy':correct/total,
             'validation_loss':val_loss/val_total,'validation_accuracy':val_correct/val_total}
        history.append(row);pd.DataFrame(history).to_csv(OUT / 'transfer_history.csv',index=False)
        print('Comparison epoch',epoch,'validation accuracy',f"{row['validation_accuracy']:.1%}",flush=True)
        if row['validation_loss']<best-0.001:
            best=row['validation_loss'];waiting=0;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: waiting+=1
        if waiting>=3: break
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':best_state,'labels':LABELS,'best_epoch':best_epoch,
                'architecture':'frozen MobileNetV3-Small backbone plus 576-128-5 head'},ROOT / 'models/transfer_cnn.pt')
    behaviour=[]
    for split,mask in [('train',train),('validation',validation)]:
        loader=DataLoader(AnimalImages(manifest[mask],WEIGHTS.transforms()),batch_size=64,num_workers=0)
        truth=[];pred=[]
        with torch.inference_mode():
            for images,targets in loader:
                pred.extend(model(images.to(device)).argmax(1).cpu().numpy());truth.extend(targets.numpy())
        behaviour.append({'split':split,'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,average='macro')})
    pd.DataFrame(behaviour).to_csv(OUT / 'transfer_cnn_behaviour.csv',index=False)
    # Only evaluate the fixed validation-selected checkpoint on the historical test split.
    parts=[]
    with torch.inference_mode():
        for images,_ in DataLoader(AnimalImages(manifest[test],WEIGHTS.transforms()),batch_size=64,num_workers=0):
            parts.append(torch.softmax(model(images.to(device)),1).cpu().numpy())
    return save_result('transfer_cnn',manifest.loc[test,'label_id'].to_numpy(),np.concatenate(parts),
        manifest.loc[test,'image_id'].to_numpy(),LABELS,{'best_epoch':best_epoch,'device':str(device),
        'features':'pretrained CNN with frozen backbone and trained five-class neural head',
        'training_seconds':time.perf_counter()-started,'fine_tuning':'none; frozen backbone is sufficient comparison',
        'assessment':'Previously inspected Animals-10 test split'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['features','logistic','transfer','all'],default='all')
    parser.add_argument('--resume',action='store_true',help='Skip experiment stages whose result files already exist.')
    args=parser.parse_args()
    if args.stage in ['features','all']: extract_features()
    if args.stage in ['logistic','all']:
        for binary in [True,False]:
            task='binary' if binary else 'softmax'
            if not args.resume or not (OUT / f'{task}_weighted_pixels_metrics.json').exists(): raw_experiments(binary)
            if not args.resume or not (OUT / f'{task}_features_metrics.json').exists(): train_logistic(binary)
    if args.stage in ['transfer','all']: train_transfer()
