"""Predict one local image from a file path or the browser upload demo."""
from pathlib import Path
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse, json
import numpy as np
from PIL import Image, UnidentifiedImageError
import joblib, torch
from common import ROOT, LABELS, pixels, small_cnn

class Predictor:
    def __init__(self):
        self.softmax=joblib.load(ROOT/'models/softmax.joblib')
        checkpoint=torch.load(ROOT/'models/cnn.pt',map_location='cpu',weights_only=True)
        self.cnn=small_cnn(); self.cnn.load_state_dict(checkpoint['state_dict']); self.cnn.eval()
        torch.set_num_threads(4)

    def predict(self,source,kind='cnn'):
        if kind not in ('cnn','softmax'): raise ValueError('Choose cnn or softmax.')
        if kind=='softmax':
            image=pixels(source,24).reshape(1,-1).astype(np.float32)/255.0
            probability=self.softmax.predict_proba(image)[0]
        else:
            image=pixels(source,64).transpose(2,0,1).copy()
            with torch.no_grad():
                tensor=torch.from_numpy(image).float().unsqueeze(0)/255.0
                probability=torch.softmax(self.cnn(tensor),dim=1)[0].numpy()
        index=int(np.argmax(probability))
        return {'animal':LABELS[index],'probability':float(probability[index]),'model':kind,
                'probabilities':{label:float(p) for label,p in zip(LABELS,probability)}}

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Animal image classifier</title><style>
body{font:17px system-ui,sans-serif;max-width:760px;margin:50px auto;padding:0 22px;color:#182b3a;background:#f6f8fa}
main{background:white;padding:30px;border:1px solid #d5dee5;border-radius:12px}h1{font-size:29px;margin-top:0}p{line-height:1.55}
label{display:block;margin:20px 0 8px}select,input,button{font:inherit;padding:10px;max-width:100%}button{display:block;background:#205a7b;color:white;border:0;border-radius:5px;margin:22px 0;cursor:pointer}
img{max-width:100%;max-height:300px;border-radius:5px}#result{font-size:21px;font-weight:600;margin:20px 0}table{width:100%;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #e0e5ea}small{color:#526372}button:disabled{opacity:.6}
</style><main><h1>Animal image classifier</h1><p>Choose a photograph of a cat, dog, cow, horse or sheep. Prediction runs on this Mac.</p>
<label for="model">Model</label><select id="model"><option value="cnn">Small CNN</option><option value="softmax">Multinomial Logistic Regression</option></select>
<label for="image">Animal photograph</label><input id="image" type="file" accept="image/*"><p><img id="preview" alt="" hidden></p>
<button id="predict">Predict animal</button><div id="result" role="status" aria-live="polite"></div><table id="probabilities"></table>
<p><small>Estimated probabilities can be wrong. This model always chooses one of five labels, even for unsupported animals or unrelated images. Images are processed in memory and are not saved.</small></p></main>
<script>
const file=document.querySelector('#image'),button=document.querySelector('#predict');let previewUrl;
file.addEventListener('change',()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);if(!file.files[0])return;previewUrl=URL.createObjectURL(file.files[0]);const p=document.querySelector('#preview');p.src=previewUrl;p.hidden=false;document.querySelector('#result').textContent='';document.querySelector('#probabilities').replaceChildren();});
button.addEventListener('click',async()=>{const result=document.querySelector('#result');if(!file.files[0]){result.textContent='Select an image first.';return;}
button.disabled=true;result.textContent='Checking image…';try{const response=await fetch('/predict?model='+document.querySelector('#model').value,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file.files[0]});const data=await response.json();if(!response.ok)throw Error(data.error);result.textContent='Predicted animal: '+data.animal+' · Estimated probability: '+(data.probability*100).toFixed(1)+'%';const table=document.querySelector('#probabilities');table.replaceChildren();for(const [animal,p] of Object.entries(data.probabilities)){const row=table.insertRow();row.insertCell().textContent=animal;row.insertCell().textContent=(p*100).toFixed(1)+'%';}}catch(error){result.textContent=error.message;}finally{button.disabled=false;}});
</script></html>'''

def serve(port):
    predictor=Predictor()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/': self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(PAGE.encode())
        def do_POST(self):
            from urllib.parse import urlparse,parse_qs
            parsed=urlparse(self.path)
            if parsed.path!='/predict': self.send_error(404); return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=20*1024*1024: raise ValueError('Choose an image smaller than 20 MB.')
                body=self.rfile.read(length)
                result=predictor.predict(BytesIO(body),parse_qs(parsed.query).get('model',['cnn'])[0]); status=200
            except (ValueError,UnidentifiedImageError,OSError,Image.DecompressionBombError):
                result={'error':'Could not read this image. Use a valid JPEG or PNG under 20 MB.'}; status=400
            self.send_response(status); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(json.dumps(result).encode())
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Open http://127.0.0.1:{port} in your browser. Press Ctrl+C to stop.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,help='Predict this image without opening the browser demo.')
    parser.add_argument('--model',choices=['cnn','softmax'],default='cnn')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if args.image: print(json.dumps(Predictor().predict(args.image,args.model),indent=2))
    else: serve(args.port)
