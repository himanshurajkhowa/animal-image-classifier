# Web app

From the repository root, with a Python 3.12 virtual environment active:

```bash
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py
```

`app.py` connects the supported Streamlit v2 component to cached CPU inference in `../inference.py`. `interface.html`, `style.css` and `interface.js` define the isolated interface. `assets/` contains licensed example photos, credits and local fonts. The app uses the saved Logistic Regression models and an export of the existing frozen feature extractor. It does not train models or retain uploaded files.

[Deployment preparation and measured resource use](../outputs/deployment_feasibility.md) covers Community Cloud settings, export verification and remaining physical-device checks. The app is live on [Streamlit Community Cloud](https://animalxclassifier.streamlit.app/).
