import os

from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/"
HUGGINGFACE_INFERENCE_API_STATUS_URL = f"https://api-inference.huggingface.co/status/"


def get_hf_headers(model_id):
    model_org, model_name = model_id.split("/")
    if model_org == "BISPL-KAIST":
        HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN_BISPL")
    else:
        HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN_DXD")
    return {
        "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}",
    }
