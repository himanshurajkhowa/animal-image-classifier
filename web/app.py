"""Animal Classifier: one supported Streamlit v2 component and cached CPU inference."""
from pathlib import Path
import base64
import json
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from inference import AnimalPredictor, MAX_UPLOAD_BYTES

WEB = Path(__file__).resolve().parent
st.set_page_config(page_title='Animal Classifier', page_icon='◉', layout='wide')


@st.cache_resource(show_spinner=False)
def load_models():
    return AnimalPredictor()


@st.cache_data(show_spinner=False)
def visual_assets():
    gallery = json.loads((WEB / 'assets/gallery/credits.json').read_text())
    for item in gallery:
        path = WEB / 'assets/gallery' / Path(item['file']).name
        item['src'] = 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()
    return gallery


@st.cache_resource(show_spinner=False)
def interface():
    # These are trusted local files. Uploaded names and other user text are never interpolated into HTML.
    css = (WEB / 'style.css').read_text()
    for family in ['Manrope', 'Caveat']:
        font = base64.b64encode((WEB / f'assets/fonts/{family}.ttf').read_bytes()).decode()
        css = css.replace(f'__{family.upper()}_FONT__', f'data:font/ttf;base64,{font}')
    return st.components.v2.component('animal_classifier_experience',
        html=(WEB / 'interface.html').read_text(), css=css,
        js=(WEB / 'interface.js').read_text(), isolate_styles=True)


def answer_request():
    request = st.session_state.get('animal_experience', {}).get('predict')
    if not isinstance(request, dict):
        return
    request_id = str(request.get('id', ''))[:80]
    try:
        mode = request.get('mode', 'softmax')
        example = request.get('example')
        if example is not None:
            items = {item['id']: item for item in visual_assets()}
            if example not in items:
                raise ValueError('Choose an example from the gallery.')
            path = WEB / 'assets/gallery' / Path(items[example]['file']).name
            payload = path.read_bytes()
        else:
            encoded = request.get('photo', '')
            if not isinstance(encoded, str) or len(encoded) > (MAX_UPLOAD_BYTES * 4 // 3 + 8):
                raise ValueError('Choose a photo smaller than 8 MB.')
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                raise ValueError('We could not read that upload. Please choose the photo again.') from None
        result = load_models().predict(payload, mode)
        st.session_state['animal_result'] = {'id': request_id, **result}
    except ValueError as error:
        st.session_state['animal_result'] = {'id': request_id, 'error': str(error)}
    except Exception:
        st.session_state['animal_result'] = {'id': request_id, 'error': 'The model could not finish this prediction. Please try again.'}
    finally:
        # Trigger values are transient. Clear our request reference; retain only the numerical result.
        request.clear()


interface()(key='animal_experience', data={'gallery': visual_assets(),
    'result': st.session_state.get('animal_result')}, on_predict_change=answer_request,
    width='stretch', height='content')
