"""Validate web inference, licensed assets, upload limits and CPU resource use."""
from pathlib import Path
import hashlib, io, json, resource, statistics, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def runtime_profile():
    started=time.perf_counter()
    from inference import AnimalPredictor
    from PIL import Image
    model=AnimalPredictor()
    startup=time.perf_counter()-started
    peak_after_load=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows=json.loads((ROOT/'web/assets/gallery/credits.json').read_text())
    times=[]
    for row in rows:
        payload=(ROOT/'web/assets/gallery'/row['file']).read_bytes()
        begin=time.perf_counter();model.predict(payload);times.append(time.perf_counter()-begin)
    peak_after_gallery=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # A large phone-photo workload, not an accuracy experiment.
    buffer=io.BytesIO();Image.new('RGB',(6000,4000),(120,110,90)).save(buffer,format='JPEG')
    model.predict(buffer.getvalue())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=int(peak if sys.platform=='darwin' else peak*1024)
    assert all(name not in sys.modules for name in ['torch','torchvision','train_improved','common','matplotlib.pyplot'])
    import onnxruntime
    return {'platform':sys.platform,'python':sys.version.split()[0],'onnxruntime':onnxruntime.__version__,
            'device':'CPU','cold_import_and_model_load_seconds':startup,
            'gallery_mean_prediction_ms':statistics.mean(times)*1000,
            'gallery_median_prediction_ms':statistics.median(times)*1000,
            'peak_rss_bytes_including_24mp_photo':peak_bytes,
            'peak_rss_after_model_load_bytes':int(peak_after_load if sys.platform=='darwin' else peak_after_load*1024),
            'peak_rss_after_gallery_bytes':int(peak_after_gallery if sys.platform=='darwin' else peak_after_gallery*1024),
            'training_and_plotting_modules_loaded':False,'pytorch_loaded':False}


def run_checks(full_parity=False):
    from inference import AnimalPredictor, read_photo, MAX_UPLOAD_BYTES
    from demo import Predictor
    from PIL import Image
    import numpy as np
    rows=json.loads((ROOT/'web/assets/gallery/credits.json').read_text())
    assert len(rows)==10
    assert {label:sum(row['animal']==label for row in rows) for label in ['Cat','Dog','Cow','Horse','Sheep']}==dict.fromkeys(['Cat','Dog','Cow','Horse','Sheep'],2)
    import csv
    external_ids={str(row['photo_id']) for row in csv.DictReader((ROOT/'outputs/improved/external_manifest.csv').open())}
    for row in rows:
        assert row['license'] in ['CC-BY-4.0','CC0-1.0','cc-by','cc0']
        assert str(row['photo_id']) not in external_ids
        path=ROOT/'web/assets/gallery'/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    model=AnimalPredictor();old=Predictor();largest_difference=0.0
    from inference import prepare_photo
    for width,height in [(224,224),(317,201),(201,317),(99,102),(4000,3000)]:
        image=Image.new('RGB',(width,height),(90,120,150))
        np.testing.assert_array_equal(prepare_photo(image),old.transform(image).unsqueeze(0).numpy())
    for row in rows:
        payload=(ROOT/'web/assets/gallery'/row['file']).read_bytes()
        for mode in ['softmax','binary']:
            actual=model.predict(payload,mode);expected=old.predict(io.BytesIO(payload),mode+'_features')
            values=np.array(list(actual['probabilities'].values()));reference=np.array(list(expected['probabilities'].values()))
            np.testing.assert_allclose(values,reference,rtol=1e-4,atol=1e-6)
            assert actual['animal']==expected['animal'];assert abs(values.sum()-1)<1e-5
            largest_difference=max(largest_difference,float(abs(values-reference).max()))
    historical_parity = None
    if full_parity:
        import pandas as pd
        manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv').set_index('image_id')
        biggest=0.0;counts={}
        for mode,name in [('softmax','softmax_features'),('binary','binary_features')]:
            saved=pd.read_csv(ROOT/f'outputs/improved/{name}_predictions.csv')
            for _,row in saved.iterrows():
                payload=(ROOT/manifest.loc[row.image_id,'path']).read_bytes()
                actual=model.predict(payload,mode)
                expected=np.array([row[f'p_{label}'] for label in actual['probabilities']])
                values=np.array(list(actual['probabilities'].values()))
                biggest=max(biggest,float(abs(values-expected).max()))
                np.testing.assert_allclose(values,expected,rtol=.003,atol=.0003)
                assert int(np.argmax(values))==int(row.prediction)
            counts[mode]=len(saved)
        historical_parity={'images':counts,'all_predicted_classes_unchanged':True,'maximum_probability_difference':biggest}
    # Magic-byte validation, size checks, animation rejection and common photo formats.
    invalid=[b'',b'not an image',b'0'*(MAX_UPLOAD_BYTES+1)]
    narrow = io.BytesIO()
    Image.new('RGB', (1, 1000)).save(narrow, format='PNG')
    invalid.append(narrow.getvalue())
    for payload in invalid:
        try:read_photo(payload)
        except ValueError:pass
        else:raise AssertionError('Invalid upload accepted')
    for image_format in ['JPEG','PNG','WEBP']:
        image=Image.new('RGB',(100,80),(200,80,60));buffer=io.BytesIO();image.save(buffer,format=image_format)
        assert read_photo(buffer.getvalue()).size==(100,80)
    animated=io.BytesIO();frames=[Image.new('RGB',(40,40),color) for color in ['red','blue']]
    frames[0].save(animated,format='WEBP',save_all=True,append_images=frames[1:])
    try:read_photo(animated.getvalue())
    except ValueError:pass
    else:raise AssertionError('Animated image accepted')
    exif=Image.Exif();exif[274]=6;buffer=io.BytesIO();Image.new('RGB',(120,80)).save(buffer,format='JPEG',exif=exif)
    assert read_photo(buffer.getvalue()).size==(80,120)
    model_files=['feature_extractor.onnx','softmax_features.joblib','binary_features.joblib','improved_preprocessing.json']
    sizes={name:(ROOT/'models'/name).stat().st_size for name in model_files}
    historical=json.loads((ROOT/'outputs/improved/experiment_plan.json').read_text())['historical_artifact_sha256']
    improved=json.loads((ROOT/'outputs/improved/external_model_freeze.json').read_text())
    for filename,digest in {**historical,**improved}.items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest
    result={'status':'passed','gallery_images':10,'gallery_modes_compared':2,
        'maximum_probability_difference_from_demo':largest_difference,
        'input_validation':['Empty input','Invalid file content','Over 8 MB','Extreme aspect ratio','Animated image','JPEG / PNG / WEBP decoding','EXIF orientation'],
        'model_package_bytes':sizes,'default_five_class_pipeline_bytes':sizes['feature_extractor.onnx']+sizes['softmax_features.joblib']+sizes['improved_preprocessing.json'],
        'both_lr_modes_package_bytes':sum(sizes.values()),
        'historical_models_unchanged':True,'historical_test_export_parity':historical_parity}
    (ROOT/'outputs/web_app_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if '--profile' in sys.argv:print(json.dumps(runtime_profile()))
    else:run_checks(full_parity="--full-parity" in sys.argv)
