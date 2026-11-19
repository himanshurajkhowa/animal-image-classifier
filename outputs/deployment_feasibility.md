# Future deployment feasibility

Research only, checked 19 November 2026. No Vercel account, project, build, deployment or settings were accessed or changed.

## Recommended first step

Use a small Python backend that reuses `common.pixels` and the saved CPU CNN checkpoint, with a simple upload page. This is the least work to preserve the local model's behaviour because it uses the same Pillow EXIF orientation, RGB conversion, bilinear square resizing, division by 255, class order and PyTorch inference. Train offline; load weights once when the backend starts. A later static frontend could be hosted separately, but any Vercel usage shares the account's allowances with the hackathon.

The local demo already implements this flow on `127.0.0.1`, independently of any hosting provider. It accepts a file, displays the predicted class and all five estimated probabilities, processes the bytes in memory, and does not save uploads. The 20 MB local upload allowance would need reducing or bypassing through a separately designed upload path on Vercel.

## Browser inference and ONNX

ONNX Runtime Web can run supported models in the browser through WebAssembly; WebGPU depends on the browser and operators. A small CNN of convolutions, ReLUs, pooling and linear layers is a plausible candidate. The user would download the runtime and model, then photos could stay on the device. This reduces inference-server work but does not remove model bandwidth or other hosting quotas.

No ONNX export or browser parity test has been performed. Before choosing this route, export the checkpoint, verify operator support and compare probabilities against PyTorch on a fixed set of images. Browser canvas resizing and EXIF handling can differ from Pillow; implement and validate preprocessing parity instead of assuming it. Browser model downloads also expose the weights. The recommendation remains Python until these checks pass.

## Hosting requirements and Vercel Hobby

A Python backend needs a compatible CPU build of PyTorch, Pillow and NumPy, sufficient RAM, writable temporary memory only if needed, and an HTTPS endpoint. The full training environment is inappropriate for production. Before packaging, separate image conversion and network inference from the shared plotting helpers, then exclude Kaggle, Jupyter, training data and plotting libraries. macOS MPS is not available in a Linux hosting environment. Cold-start time, Linux dependency size and deployed memory have not been measured.

Current Vercel documentation gives Hobby Functions 2 GB memory and one vCPU, with Fluid Compute duration up to 300 seconds. The standard Python uncompressed bundle limit is 500 MB. A June 2026 announcement adds a public beta for Node.js and Python bundles up to 5 GB on Fluid Compute; availability and opt-in conditions must be checked before relying on it. Request and response bodies are capped at 4.5 MB. The Hobby plan includes shared usage such as 4 active CPU-hours and 360 GB-hours provisioned memory per month, and its general limits include 100 deployments per day and one concurrent build. A separate project does not provide a separate account allowance. These constraints make dependency size, upload size and shared hackathon usage more relevant than weight-file size alone.

These are published plan figures, not a deployment guarantee. No account eligibility or remaining quota was inspected. Keep this project local until after the hackathon and a separate deployment decision.

## Phone camera and privacy

An eventual file input can offer `accept="image/*"` with `capture="environment"` as a mobile hint, while preserving file selection. Direct camera access through `getUserMedia()` needs HTTPS or localhost and explicit permission. Handle denial, stop media tracks after capture, and explain whether bytes stay in the browser or are uploaded. Avoid storing photographs or sending them to analytics. The current demo uses file selection only; phone-camera capture is future work.

## Sources

- [ONNX Runtime Web documentation](https://onnxruntime.ai/docs/tutorials/web/)
- [Vercel Function limits](https://vercel.com/docs/functions/limitations)
- [Vercel Functions up to 5 GB announcement](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size)
- [Vercel Hobby plan](https://vercel.com/docs/plans/hobby)
- [Vercel general limits](https://vercel.com/docs/limits)
- [MDN camera permission and secure contexts](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
