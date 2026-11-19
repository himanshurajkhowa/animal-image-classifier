# Streamlit deployment preparation

The app is live on [Streamlit Community Cloud](https://animalxclassifier.streamlit.app/). Its build logs confirmed Python 3.12.16 and the production requirements in `web/requirements.txt`.

## Run locally

Use Python 3.12 (tested: 3.12.13), activate a virtual environment, then run from the repository root:

```bash
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py
```

The application uses Streamlit's supported Components v2 API for its isolated HTML/CSS/JavaScript interface. It does not target Streamlit's internal CSS selectors. The navy, ivory and coral interface includes upload, camera capture, a swipeable ten-photo gallery, both LR modes, probability bars and photo credits. Fonts are served locally under their OFL licenses; gallery attribution and modification records are in `web/assets/gallery/credits.json`.

## Community Cloud settings after review

Use the existing `himanshurajkhowa/animal-image-classifier` repository, branch `main`, entry point `web/app.py`, and select Python 3.12 in Advanced settings. Community Cloud searches the entry-point folder for dependencies, so `web/requirements.txt` supplies the inference-only environment. The root `requirements.txt` remains the full research environment. The root `.streamlit/config.toml` supplies the theme and upload limit. No app secret, dataset download, pretrained-weight download or training step is required.

The web app uses its inference environment and the models already committed to the repository.

## Inference package

The default remains multinomial Logistic Regression. A frozen pretrained CNN creates the features; ONNX Runtime runs that extractor on CPU. Binary LR is available for cats and dogs. Model loading is cached once per server process, while uploaded photos are not cached or written to disk. Processing occurs on the server, so images travel to the hosting service when the app is deployed.

| Component | Bytes |
| --- | ---: |
| Five-class LR classifier and scaler | 24,981 |
| Frozen extractor, ONNX | 3,717,004 |
| Preprocessing record | 572 |
| Complete five-class model pipeline | 3,742,557 |
| Binary classifier and scaler, optional additional mode | 16,351 |

The complete package for both modes is 3.58 MiB. Dependencies, fonts, gallery photos, the interface and the interpreter are additional. The original small CNN weighs 361,413 bytes, smaller than the full LR pipeline. The transfer CNN weighs 4,129,209 bytes.

The export comes from the existing frozen `.pt` file without retraining. To recreate it in the full research environment, install `onnx==1.23.2` and run `python web/export_runtime.py`. The exported model is already included; production needs neither ONNX's export package nor PyTorch. Hashes and export settings are in `models/web_export.json`.

## Measured local resource use

Measurements used a clean production-only environment on the Apple M3 MacBook Air with 8 GB RAM, Python 3.12.13 and CPU inference. See `outputs/web_runtime_profile.json`.

- Installed production libraries: 444.6 MiB on macOS.
- Import and model startup: 33.0 seconds in the first process in a new environment; 3.09 seconds in a later fresh process.
- Ten gallery photos: median inference 7.8 ms, including decoding and preprocessing, excluding browser/transport latency.
- Peak resident memory: about 216 MiB after model loading, 231 MiB after gallery predictions and 503 MiB after a synthetic 24-megapixel JPEG workload.

These are standalone inference measurements. Streamlit, browser payloads and simultaneous sessions add memory and latency. Linux dependency installation and cloud predictions passed. Hosted concurrency has not been measured. Community Cloud's published approximate memory range is 690 MB to 2.7 GB, subject to change; its actual allocation and cold-start behaviour must be checked after a reviewed deployment. The local measurements are evidence of feasibility, not a capacity guarantee.

## Verification and remaining checks

`python web/check_app.py --full-parity` compares the web runtime with recorded historical predictions and checks original artifact hashes, gallery licenses and input decoding. All 1,925 five-class and 979 binary historical predictions retained their labels; maximum probability difference was 0.0000119. Tests also cover JPEG/PNG/WEBP, EXIF rotation, malformed or empty input, oversize payloads, extreme aspect ratios and animated images. The normal check needs the full research environment for comparison with the original inference code; `--profile` needs only production dependencies.

Local browser checks cover welcome/main transitions, upload, both LR modes, prediction bars, reset, gallery navigation, photo credits, narrow layouts, reduced motion, simulated live-camera capture with camera-track cleanup, and the native camera-file fallback. The app accepts still images up to 8 MB and 24 megapixels. HEIC must be exported as JPEG. Camera access requires permission and HTTPS or localhost. A phone's localhost address refers to the phone itself, so a real phone test needs a suitable secure served address after deployment review.

Physical iPhone/Android camera behaviour and Safari remain untested. The public Cloud app loaded and completed gallery and upload predictions in a separate browser session. The ten example photos are demonstration assets, not an accuracy benchmark. They were selected without classifier predictions and checked against training/external photo IDs and perceptual hashes; this cannot prove absolute independence from all pretraining sources.

The five-class historical accuracy is 94.23%; the separate 86-photo external evaluation is 58.14%. Both appear in the app's About dialog. No probability calibration or unknown-animal detection is claimed.

## Sources

Checked 19 November 2026:

- [Supported Components v2 API](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component)
- [Community Cloud deployment and Python settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Community Cloud resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [ONNX Runtime Python inference](https://onnxruntime.ai/docs/api/python/api_summary.html)
