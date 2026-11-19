# Future deployment

Research only. No deployment or account changes were made. Vercel remains untouched.

Streamlit Community Cloud is the preferred free host. Reuse the verified Python preprocessing and Logistic Regression pipelines, loading saved weights once. First measure hosted CPU memory, startup time and latency with a minimal inference dependency set. Local inference has been tested; hosted performance has not.

The planned mobile interface includes photo upload, phone-camera capture, a scrollable gallery of at least ten individually licensed example images with credits, one-click prediction and class probabilities. Streamlit supports file upload and camera input, but those controls and the licensed example gallery have not been built for hosting. Camera access needs a secure browser context and permission. Explain that uploaded photos are processed on the server, avoid retaining them, and handle denied permissions.

Preserve the exact preprocessing for each model: improved Logistic Regression uses the frozen pretrained extractor's resize/crop/normalization, while the original raw-pixel models use their recorded square resize and scaling. A browser ONNX version would require separate export, operator and preprocessing-parity tests; none has been performed. Unknown-animal detection is future scope.

The historical benchmark is strong, but the small external audit shows a substantial gap on distant or difficult photographs. Do not present the deployed app as having verified 94% real-world accuracy. Finish the model review and make a separate deployment decision before publishing an app.

Sources checked on 19 November 2026: [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud), [camera input](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input), [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
