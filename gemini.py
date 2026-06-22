"""Gemini-backed image captioner for the Gemini Prompt Builder ComfyUI node.

Mirrors the Anima v10 tag rubric used by ../portrait-prompt-extractor, but groups
the tokens into the editable element boxes the node exposes (quality / lighting /
pose / action / interaction / body / clothes / accessories / tattoos / background /
character / extra). Gemini fills each box with
comma-joined trained tokens; the node assembles the enabled boxes with the user's
master prompt at run time.
"""

import re

from pydantic import create_model
from google import genai
from google.genai import types

MODEL_DEFAULT = "gemini-3-flash-preview"  # same model the web extractor uses

# Element boxes the node shows, in final assembly order. `default_on=False` for the
# identity boxes (character/body/tattoos) — that detail comes from the master prompt,
# so they stay off until the user opts to pull it from the reference.
ELEMENTS = [
    ("quality", True),
    ("lighting", True),
    ("pose", True),
    ("action", True),
    ("interaction", True),
    ("body", False),
    ("clothes", True),
    ("accessories", True),
    ("tattoos", False),
    ("background", True),
    ("character", False),
    ("extra", True),
]
ELEMENT_KEYS = [k for k, _ in ELEMENTS]

# Per box: a comma-joined tag string + one natural-language sentence (`{key}_nl`).
# Built from ELEMENT_KEYS so splitting a box only needs the list above + the JS copy.
CaptionElements = create_model(
    "CaptionElements",
    **{f: (str, ...) for k in ELEMENT_KEYS for f in (k, f"{k}_nl")},
)


# Allowed vocab per box — the exact enum tokens the Anima/SDXL realism finetune was
# trained on (mirrors portrait-prompt-extractor VOCAB), inlined as guidance.
PROMPT = """You are labeling a reference photograph to build an SDXL/Anima-realism prompt. Break the image into the element boxes below. For EACH box produce two fields:
- `<box>`: a DETAILED comma-separated list of trained tokens (lowercase) — be thorough, 6-15 precise tokens where the image supports it (materials, colors, patterns, props, framing), not just 1-2. Use the suggested vocabulary where it fits and add concrete extra tokens.
- `<box>_nl`: ONE vivid natural-language sentence describing that element. Leave it "" if the box doesn't apply.
Leave any field "" when nothing applies.

NEVER output the word "camera" (the model would draw a literal camera) — describe lens/photo style with the mm/format tokens instead.

CRITICAL — only the `character`, `body`, and `tattoos` boxes may describe the subject's face, hair, body shape, skin, age, or markings, each ONLY its own slice (face/hair/skin in `character`, build/proportions in `body`, ink/piercings/scars in `tattoos`). EVERY OTHER box (tags AND sentence) must be subject-agnostic (no body/face words) so the prompt is reusable. Pronouns are fine.

Boxes:
- quality: overall technical quality + photo feel. Vocab: masterpiece, best quality, high quality, normal quality, low quality, worst quality | safe, suggestive, explicit | amateur snapshot, candid photo, semi-professional, professional photograph, editorial photography, studio portrait | sharp focus, soft focus, grainy / high ISO, motion blur, overexposed, underexposed, lens flare, chromatic aberration, vignette | natural color, warm tones, cool tones, muted, vibrant, high contrast, film grain, film look, black and white, sepia, faded, teal and orange | 85mm bokeh, 50mm, 35mm, wide-angle, fisheye, macro | shallow depth of field, deep focus. Judge honestly — do NOT default to high quality; an unedited snapshot is "normal quality"/"low quality".
- lighting: 1-2 tokens. Vocab: direct flash, natural daylight, golden hour, blue hour, overcast flat light, indoor artificial light, low light, soft window light, studio lighting, backlit, rim light, neon lighting, harsh sunlight, ring light, candlelight.
- pose: framing + view angle + STATIC body pose (stance/arms/head position), NOT body shape, NOT what they're doing. Vocab for crop/angle: extreme close-up, close-up, portrait, upper body, cowboy shot, full body, wide shot | front view, three-quarter view, profile view, back view, looking over shoulder, looking at viewer, looking away | eye level, from above, from below, overhead, dutch angle. Add pose tags like "hand on hip", "arms crossed", "standing", "sitting", "kneeling", "lying down".
- action: what the subject is actively DOING — verbs/motion (walking, running, dancing, jumping, stretching, eating, drinking, reading, applying makeup, undressing). Leave "" if static/just posing.
- interaction: contact with objects, people, or the environment — holding/using a prop, leaning on a wall, sitting on a chair, hand on a table, hugging/touching someone. Name the object/surface. Leave "" if none.
- body: the subject's body shape ONLY — build/frame, proportions, height impression, weight class, bust/waist/hip impression, muscle tone. NO clothes, NO face/hair, NO tattoos. Leave "" to keep the body from the master prompt.
- clothes: worn GARMENTS only — tops, bottoms, dresses, outerwear, lingerie, swimwear, footwear, with materials, colors, patterns, fit. NO jewelry/accessories, NO body description.
- accessories: non-garment worn items — jewelry (rings, necklace, earrings, bracelet, anklet, watch), eyewear, hats, belts, bags, gloves, hair accessories. NO clothes, NO body.
- tattoos: body modification ONLY — tattoos (placement + motif), piercings, scars, body markings, nail art. Leave "" if none visible.
- background: setting + environment. Vocab: bedroom, living room, kitchen, bathroom, studio, office, city street, nature, beach, pool, cafe, restaurant, bar, gym, car, party. Add scene/prop tokens.
- character: the subject's face, hair, skin, age, expression — NOT body shape (use `body`), NOT tattoos (use `tattoos`).
- extra: any other salient tokens not covered above (objects held, text/watermark, notable details)."""


_CAMERA = re.compile(r"ca(?:mera|mear)", re.IGNORECASE)


def _strip_camera(elements: dict) -> dict:
    """Drop any 'camera'/'camear' wording so SDXL doesn't render a literal camera.
    Token boxes lose the whole offending token; NL sentences lose just the word."""
    out = {}
    for key, val in elements.items():
        if key.endswith("_nl"):
            val = _CAMERA.sub("", val)
            val = re.sub(r"\s+([,.])", r"\1", val)
            val = re.sub(r"\s{2,}", " ", val).strip()
        else:
            val = ", ".join(t for t in (p.strip() for p in val.split(",")) if t and not _CAMERA.search(t))
        out[key] = val
    return out


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
    return _strip_camera(parsed.model_dump())
