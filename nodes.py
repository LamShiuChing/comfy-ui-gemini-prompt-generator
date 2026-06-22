"""ComfyUI node: caption a reference image with Gemini into editable element boxes,
then merge the enabled boxes with the user's master prompt into one positive string.

Flow: pick/upload a reference image -> click "Caption image" (JS button hits the
route below) -> edit/toggle the element boxes -> queue. At run time the node only
assembles its widget values; the Gemini call already happened at button time.
"""

import os
import asyncio
import mimetypes

import torch
import numpy as np
from PIL import Image, ImageOps

import folder_paths
from server import PromptServer
from aiohttp import web

from .gemini import caption_image, ELEMENTS, ELEMENT_KEYS, MODEL_DEFAULT, DETAIL_LEVELS, DETAIL_DEFAULT


_KEY_FILE = os.path.join(os.path.dirname(__file__), ".gemini_api_key")


def _resolve_key(api_key: str) -> str:
    """Field, else GEMINI_API_KEY env, else the key build() last remembered.
    Read-only: never persists, so the unauthenticated caption route can't poison it."""
    return api_key.strip() or os.environ.get("GEMINI_API_KEY", "") or (
        open(_KEY_FILE).read().strip() if os.path.isfile(_KEY_FILE) else ""
    )


def _remember_key(key: str) -> None:
    """Persist the key owner-only so later sessions default to it. Delete
    `.gemini_api_key` to clear. Called from build() (graph execution), not the route."""
    with open(os.open(_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(key)


# ── Backend route: the JS "Caption image" button posts here ──────────────────
@PromptServer.instance.routes.post("/gemini_prompt_builder/caption")
async def _caption_route(request):
    data = await request.post()
    image = data.get("image", "")
    api_key = _resolve_key(data.get("api_key", ""))
    model = data.get("model", "") or MODEL_DEFAULT
    detail = data.get("detail", "") or DETAIL_DEFAULT
    if not api_key:
        return web.json_response({"error": "No API key (set GEMINI_API_KEY or the api_key field)."}, status=400)

    path = folder_paths.get_annotated_filepath(image)
    if not path or not os.path.isfile(path):
        return web.json_response({"error": f"Image not found: {image}"}, status=400)

    with open(path, "rb") as f:
        raw = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"

    try:
        # genai client is sync; run off the event loop.
        result = await asyncio.get_event_loop().run_in_executor(
            None, caption_image, raw, mime, api_key, model, detail
        )
    except Exception as e:  # surface the real error to the UI
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(result)


def _load_image_tensor(image: str):
    """Same load path as core LoadImage: file -> RGB float tensor [1,H,W,3]."""
    path = folder_paths.get_annotated_filepath(image)
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class GeminiPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        files = sorted(
            f for f in os.listdir(folder_paths.get_input_directory())
            if os.path.isfile(os.path.join(folder_paths.get_input_directory(), f))
        )
        box = lambda: ("STRING", {"multiline": True, "default": ""})
        required = {
            "image": (files, {"image_upload": True}),
            "master_prepend": ("STRING", {"multiline": True, "default": ""}),
        }
        for key, on in ELEMENTS:  # tags box + NL box + one toggle (drives both)
            required[key] = box()
            required[f"{key}_nl"] = box()
            required[f"{key}_enabled"] = ("BOOLEAN", {"default": on})
        required["describe_nl"] = ("BOOLEAN", {"default": True})
        required["detail"] = (list(DETAIL_LEVELS), {"default": DETAIL_DEFAULT})
        required["master_append"] = ("STRING", {"multiline": True, "default": ""})
        required["model"] = ("STRING", {"default": MODEL_DEFAULT})
        required["api_key"] = ("STRING", {"default": ""})
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("positive", "nl", "image")
    FUNCTION = "build"
    CATEGORY = "gemini"

    def build(self, image, master_prepend, master_append, model, api_key, describe_nl, **kw):
        if key := api_key.strip():
            _remember_key(key)
        enabled = [k for k in ELEMENT_KEYS if kw.get(f"{k}_enabled", False)]
        parts = [master_prepend, *(kw.get(k, "") for k in enabled), master_append]
        positive = ", ".join(t.strip().strip(",") for t in parts if t and t.strip())
        nl = ""
        if describe_nl:
            nl = " ".join(s.strip() for k in enabled if (s := kw.get(f"{k}_nl", "")).strip())
        return (positive, nl, _load_image_tensor(image))


NODE_CLASS_MAPPINGS = {"GeminiPromptBuilder": GeminiPromptBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPromptBuilder": "Gemini Prompt Builder"}
