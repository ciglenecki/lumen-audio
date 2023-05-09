import os
import re

from src.enums.enums import SupportedDatasetDirType
from src.server.server_store import server_store

RESOURCES_TAG = "Resources"
MODELS_INFERENCE_TAG = "Model Inference"


help_text = re.sub(r"^", r"\t", server_store.parser_help, flags=re.MULTILINE).replace(
    "main.py", "python3 src/server/main.py"
)

api_description = f"""API used for testing and predicting audio data on trained models.\n\n{help_text}

### Quick instructions:
0. Make sure you have a `.ckpt` model inside of the `--model-dir` directory.
1. Click on `GET /models`
2. Send a request to get all available models.
2. Copy the model path you want to use for prediction.
3. Click on `POST /model/predict-files`
4. Paste the model path into the `model_ckpt_path` field.
5. Choose the audio files you want to predict.
6. Predict labels for the audio files.
"""
# api_description = """API used for testing and predicting audio data on trained models.\n\n\testa\n\ttestb"""
PREDICT_DESC = """Endpoints used for predicting audio data on trained models. Curl example for multiple audios:

    curl -X 'POST' \\
        'http://localhost:8090/model/predict-files?model_ckpt_path=models%2Fmy_model.ckpt' \\
        -H 'accept: application/json' \\
        -H 'Content-Type: multipart/form-data' \\
        -F 'audio_files=@track_23.wav;type=audio/wav'
"""

GET_DATASET_DESC = f"""Returns dataset types, each dataset type has a strategy for loading labels from a given directory. If you don't have labels or don't care about them, you can use the empty string `""` which only loads the audio. If you want to load labels use the following dataset types: {[e.value for e in SupportedDatasetDirType]}"""

GET_MODELS_DESC = """Returns paths to all available models on the server. Use these model paths for all `POST` request predictions. Models are fetched from the directory `--model-dir`. Only models with the extension `*.ckpt` are fetched.
"""

TEST_DIR_STREAM_DESC = """Streams classes of audios from a directory and returns metrics for multiclass classification. Streaming is useful if your directory is large and you don't want to wait for the whole directory to be processed. If you want to process the whole directory at once, use the endpoint `test-directory`"""

TEST_DIR_DESC = """Infers classes of audios from a directory and returns metrics for multiclass classification."""

PREDICT_SUFFIX = (
    "**You don't have to specify the dataset structure type is always `inference`**"
)
PREDICT_DIR_STREAM_DESC = (
    """Streams classes of audios from a directory. Streaming is useful if your directory is large and you don't want to wait for the whole directory to be processed. If you want to process the whole directory at once, use the endpoint `predict-directory`"""
    + PREDICT_SUFFIX
)

PREDICT_DIR_DESC = """Infers classes of audios from a directory.""" + PREDICT_SUFFIX

PREDICT_FILES_DESC = (
    """Infers classes of audios from a list of files.""" + PREDICT_SUFFIX
)
