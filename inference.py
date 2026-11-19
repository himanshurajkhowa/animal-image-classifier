"""CPU inference shared by the web app, without notebook or training imports."""
from io import BytesIO
from pathlib import Path
from threading import Lock
import warnings
import json
import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
LABELS = ['Cat', 'Dog', 'Cow', 'Horse', 'Sheep']
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_ASPECT_RATIO = 32
SUPPORTED_FORMATS = ['JPEG', 'PNG', 'WEBP']
PREPROCESSING = json.loads((ROOT / 'models/improved_preprocessing.json').read_text())


def read_photo(payload):
    """Validate image content and decode it in memory, regardless of its filename."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError('Choose an image first.')
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError('Choose a photo smaller than 8 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload), formats=SUPPORTED_FORMATS) as photo:
                if photo.width * photo.height > MAX_IMAGE_PIXELS:
                    raise ValueError('This photo is too large to process. Resize it below 24 megapixels.')
                if max(photo.size) > MAX_ASPECT_RATIO * min(photo.size):
                    raise ValueError('This photo is too narrow to process. Crop it around the animal and try again.')
                if getattr(photo, 'n_frames', 1) != 1:
                    raise ValueError('Choose a still photograph rather than an animated image.')
                photo.verify()
            with Image.open(BytesIO(payload), formats=SUPPORTED_FORMATS) as photo:
                photo.load()
                return ImageOps.exif_transpose(photo).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError('We could not read that photo. Choose a valid JPG, PNG or WEBP image.') from None


def prepare_photo(photo):
    """Match the original torchvision resize, centre crop and normalization."""
    width, height = photo.size
    edge = PREPROCESSING['resize_short_edge']
    if width <= height:
        size = (edge, int(edge * height / width))
    else:
        size = (int(edge * width / height), edge)
    photo = photo.resize(size, Image.Resampling.BILINEAR)
    crop = PREPROCESSING['center_crop']
    left = int(round((photo.width - crop) / 2.0))
    top = int(round((photo.height - crop) / 2.0))
    photo = photo.crop((left, top, left + crop, top + crop))
    array = np.asarray(photo, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    mean = np.array(PREPROCESSING['mean'], dtype=np.float32)[:, None, None]
    std = np.array(PREPROCESSING['std'], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((array - mean) / std)[None])


class AnimalPredictor:
    """Reuse the exported frozen CNN and the unchanged Logistic Regression files."""
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.extractor = ort.InferenceSession(str(ROOT / 'models/feature_extractor.onnx'),
            sess_options=options, providers=['CPUExecutionProvider'])
        self.classifiers = {name: joblib.load(ROOT / f'models/{name}_features.joblib')
                            for name in ['softmax', 'binary']}
        self.lock = Lock()

    def predict(self, payload, mode='softmax'):
        if mode not in self.classifiers:
            raise ValueError('Choose the five-animal or Cat and Dog model.')
        # Limit simultaneous photo decoding as well as inference on shared CPU instances.
        with self.lock:
            tensor = prepare_photo(read_photo(payload))
            features = self.extractor.run(['features'], {'image': tensor})[0]
            probability = self.classifiers[mode].predict_proba(features)[0]
        labels = LABELS[:2] if mode == 'binary' else LABELS
        winner = int(np.argmax(probability))
        return {'animal': labels[winner], 'probability': float(probability[winner]),
                'probabilities': {label: float(p) for label, p in zip(labels, probability)},
                'model': f'{mode}_features'}
