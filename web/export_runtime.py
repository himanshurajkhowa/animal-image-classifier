"""Export the existing frozen extractor for CPU inference; no fitting or training."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
import onnx

if __name__=='__main__':
    source=ROOT/'models/feature_extractor.pt'
    destination=ROOT/'models/feature_extractor.onnx'
    model=mobilenet_v3_small(weights=None);model.classifier=nn.Identity()
    model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True)['state_dict']);model.eval()
    torch.onnx.export(model,torch.zeros(1,3,224,224),str(destination),
        input_names=['image'],output_names=['features'],opset_version=17,dynamo=False,
        do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(destination)))
    record={'source':'models/feature_extractor.pt','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'export':'models/feature_extractor.onnx','export_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'training_performed':False,'opset':17,'input_shape':[1,3,224,224],'output_shape':[1,576]}
    (ROOT/'models/web_export.json').write_text(json.dumps(record,indent=2))
    print('Exported the frozen feature extractor; original model unchanged.')
