import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Must match ELEMENT_KEYS in gemini.py.
const ELEMENT_KEYS = ["quality", "lighting", "pose", "action", "interaction", "body", "clothes", "accessories", "tattoos", "background", "character", "extra"];

app.registerExtension({
  name: "comfy.GeminiPromptBuilder",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "GeminiPromptBuilder") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const get = (name) => node.widgets.find((w) => w.name === name);

      let busy = false;
      const caption = async (notify) => {
        const image = get("image")?.value;
        if (!image) return notify ? alert("Pick a reference image first.") : undefined;
        if (busy) return;
        busy = true;

        const label = btn.name;
        btn.name = "Captioning…";
        node.setDirtyCanvas(true, true);
        try {
          const body = new FormData();
          body.append("image", image);
          body.append("api_key", get("api_key")?.value ?? "");
          body.append("model", get("model")?.value ?? "");
          body.append("detail", get("detail")?.value ?? "");
          const res = await api.fetchApi("/gemini_prompt_builder/caption", { method: "POST", body });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || res.statusText);
          for (const k of ELEMENT_KEYS) {
            for (const name of [k, `${k}_nl`]) {
              const w = get(name);
              if (w) w.value = data[name] ?? "";
            }
          }
          node.setDirtyCanvas(true, true);
        } catch (e) {
          if (notify) alert("Caption failed: " + e.message);
        } finally {
          btn.name = label;
          busy = false;
          node.setDirtyCanvas(true, true);
        }
      };

      const btn = node.addWidget("button", "Caption image", null, () => caption(true));

      // Auto-caption when a new image is uploaded/selected.
      const imageWidget = get("image");
      if (imageWidget) {
        const prevCb = imageWidget.callback;
        let last = imageWidget.value;
        imageWidget.callback = function () {
          const r = prevCb?.apply(this, arguments);
          if (imageWidget.value !== last) {
            last = imageWidget.value;
            caption(false);
          }
          return r;
        };
      }

      // Keep the button near the top, just under the image widget.
      node.widgets.splice(node.widgets.indexOf(btn), 1);
      node.widgets.splice(1, 0, btn);
    };
  },
});
