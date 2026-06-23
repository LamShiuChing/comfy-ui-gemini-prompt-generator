# Gemini Prompt Builder — ComfyUI custom node

Captions a reference image with the Gemini API, broken into editable element
boxes (quality / lighting / pose / clothes / background / character / extra), lets
you edit and toggle each, then merges the enabled boxes with your master prompt
into one positive string. Same Anima/SDXL tag rubric as `../portrait-prompt-extractor`.

## Install

1. Copy this folder into `ComfyUI/custom_nodes/`.
2. Install the dep into ComfyUI's Python:
   ```
   <ComfyUI python> -m pip install -r requirements.txt
   ```
   (portable build: `python_embeded\python.exe -m pip install -r requirements.txt`)
3. Set your key — either an env var before launching ComfyUI:
   ```
   set GEMINI_API_KEY=your_key_here
   ```
   or paste it into the node's `api_key` field.
4. Restart ComfyUI.

## Use

1. Add **Gemini Prompt Builder** (category `gemini`).
2. Upload / pick a reference image on the node.
3. Click **Caption image** — Gemini fills the element boxes.
4. Edit any box; untick the toggle to drop a box. `character` is **off** by default
   (your master prompt owns the identity — turn it on per-image if you want it).
5. Put your trigger/character master prompt in `master_prepend` (e.g.
   `score_9, <your_character>, …`) and any trailing boosters in `master_append`.
6. Wire `positive` into your CLIPTextEncode. `image` is an optional passthrough so
   the same reference can feed your ControlNet/loader.

## Notes

- `model` defaults to `gemini-3-flash-preview` (the value the web extractor uses);
  change it if your account uses another.
- Captioning happens on the button click, not at queue time — so you caption once,
  edit, then run the heavy generation. Re-click to re-caption a new image.
- The subject's face/body is described **only** in the `character` box; every other
  box stays subject-agnostic so the prompt is reusable.
