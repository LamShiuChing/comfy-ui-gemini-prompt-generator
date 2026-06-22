"""Gemini-backed image captioner for the Gemini Prompt Builder ComfyUI node.

Mirrors the Anima v10 tag rubric used by ../portrait-prompt-extractor, but groups
the tokens into the editable element boxes the node exposes (quality / lighting /
pose / clothes / background / character / extra). Gemini fills each box with
comma-joined trained tokens; the node assembles the enabled boxes with the user's
master prompt at run time.
"""

from pydantic import BaseModel
from google import genai
from google.genai import types

MODEL_DEFAULT = "gemini-3-flash-preview"  # same model the web extractor uses

# Element boxes the node shows, in final assembly order. `default_on=False` for
# `character` because the subject identity comes from the master prompt.
ELEMENTS = [
    ("quality", True),
    ("lighting", True),
    ("pose", True),
    ("clothes", True),
    ("background", True),
    ("character", False),
    ("extra", True),
]
ELEMENT_KEYS = [k for k, _ in ELEMENTS]


class CaptionElements(BaseModel):
    """Per element box: a comma-joined tag string + one natural-language sentence
    (`{key}_nl`). Empty strings when nothing fits."""

    quality: str
    quality_nl: str
    lighting: str
    lighting_nl: str
    pose: str
    pose_nl: str
    clothes: str
    clothes_nl: str
    background: str
    background_nl: str
    character: str
    character_nl: str
    extra: str
    extra_nl: str


# Allowed vocab per box — the exact enum tokens the Anima/SDXL realism finetune was
# trained on (mirrors portrait-prompt-extractor VOCAB), inlined as guidance.
PROMPT = """You are labeling a reference photograph to build an SDXL/Anima-realism prompt. Break the image into the element boxes below. For EACH box produce two fields:
- `<box>`: a DETAILED comma-separated list of trained tokens (lowercase) — be thorough, 6-15 precise tokens where the image supports it (materials, colors, patterns, props, framing), not just 1-2. Use the suggested vocabulary where it fits and add concrete extra tokens.
- `<box>_nl`: ONE vivid natural-language sentence describing that element. Leave it "" if the box doesn't apply.
Leave any field "" when nothing applies.

CRITICAL — the `character` box (both fields) is the ONLY place you may describe the subject's face, hair, body, age, or skin. EVERY OTHER box (tags AND sentence) must be subject-agnostic (no body/face words) so the prompt is reusable. Pronouns are fine.

Boxes:
- quality: overall technical quality + camera feel. Vocab: masterpiece, best quality, high quality, normal quality, low quality, worst quality | safe, suggestive, explicit | amateur snapshot, casual phone photo, social media selfie, candid photo, semi-professional, professional photograph, editorial photography, studio portrait | sharp focus, soft focus, grainy / high ISO, motion blur, overexposed, underexposed, lens flare, chromatic aberration, vignette | natural color, warm tones, cool tones, muted, vibrant, high contrast, film grain, film look, black and white, sepia, faded, teal and orange | phone camera, compact camera, DSLR, 85mm bokeh, 50mm, 35mm, wide-angle, fisheye, macro, film camera | shallow depth of field, deep focus. Judge honestly — do NOT default to high quality; an unedited phone snapshot is "normal quality"/"low quality".
- lighting: 1-2 tokens. Vocab: direct flash, natural daylight, golden hour, blue hour, overcast flat light, indoor artificial light, low light, soft window light, studio lighting, backlit, rim light, neon lighting, harsh sunlight, ring light, candlelight.
- pose: framing + camera angle + body pose (arms/stance/head), NOT body shape. Vocab for crop/angle: extreme close-up, close-up, portrait, upper body, cowboy shot, full body, wide shot | front view, three-quarter view, profile view, back view, looking over shoulder, looking at viewer, looking away | eye level, from above, from below, overhead, dutch angle. Add pose tags like "hand on hip", "arms crossed", "leaning", "standing".
- clothes: clothing + accessories tokens (garments, materials, colors, jewelry, eyewear, shoes). No body description.
- background: setting + environment. Vocab: bedroom, living room, kitchen, bathroom, studio, office, city street, nature, beach, pool, cafe, restaurant, bar, gym, car, party. Add scene/prop tokens.
- character: the subject's visible appearance — face, hair, body type, skin, age, expression. This box (and only this box) describes the person.
- extra: any other salient tokens not covered above (objects held, text/watermark, notable details)."""


def caption_image(image_bytes: bytes, mime_type: str, api_key: str, model: str) -> dict:
    """Call Gemini and return {element_key: comma_token_string} for every box."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model or MODEL_DEFAULT,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=PROMPT),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CaptionElements,
            # Gemini 3 is a thinking model; thinking tokens can truncate JSON. Disable.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    parsed = CaptionElements.model_validate_json(response.text)
    return parsed.model_dump()
