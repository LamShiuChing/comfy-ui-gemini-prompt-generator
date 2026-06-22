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

      const btn = node.addWidget("button", "Caption image", null, async () => {
        const image = get("image")?.value;
        if (!image) return alert("Pick a reference image first.");

        const label = btn.name;
        btn.name = "Captioning…";
        node.setDirtyCanvas(true, true);
        try {
          const body = new FormData();
          body.append("image", image);
          body.append("api_key", get("api_key")?.value ?? "");
          body.append("model", get("model")?.value ?? "");
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
          alert("Caption failed: " + e.message);
        } finally {
          btn.name = label;
          node.setDirtyCanvas(true, true);
        }
      });

      // Keep the button near the top, just under the image widget.
      node.widgets.splice(node.widgets.indexOf(btn), 1);
      node.widgets.splice(1, 0, btn);
    };
  },
});
