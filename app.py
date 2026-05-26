import os, json
import numpy as np
from PIL import Image, ImageOps

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import gradio as gr

print(f"TensorFlow : {tf.__version__}")
print(f"Gradio     : {gr.__version__}")

ICONS: dict[str, str] = {
    "airplane":    "✈️",  "apple":       "🍎",  "banana":      "🍌",
    "bear":        "🐻",  "bee":         "🐝",  "butterfly":   "🦋",
    "cactus":      "🌵",  "cake":        "🎂",  "campfire":    "🔥",
    "carrot":      "🥕",  "cat":         "🐱",  "clock":       "🕐",
    "cloud":       "☁️",  "cookie":      "🍪",  "cow":         "🐄",
    "crown":       "👑",  "diamond":     "💎",  "dog":         "🐶",
    "donut":       "🍩",  "eye":         "👁️",  "face":        "😐",
    "fish":        "🐟",  "flower":      "🌸",  "hand":        "✋",
    "house":       "🏠",  "key":         "🔑",  "leaf":        "🍃",
    "lightning":   "⚡",  "moon":        "🌙",  "mouse":       "🐭",
    "mouth":       "👄",  "mushroom":    "🍄",  "nose":        "👃",
    "pig":         "🐷",  "pizza":       "🍕",  "rabbit":      "🐰",
    "rainbow":     "🌈",  "skull":       "💀",  "smiley face": "😊",
    "snake":       "🐍",  "snowflake":   "❄️",  "snowman":     "⛄",
    "soccer ball": "⚽",  "star":        "⭐",  "sun":         "☀️",
    "tree":        "🌳",  "umbrella":    "☂️",  "watermelon":  "🍉",
}

def get_icon(name: str) -> str:
    return ICONS.get(name.lower().strip(), "🎨")


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL & CLASSES
# ══════════════════════════════════════════════════════════════
def load_classes(path="classes.json") -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):  return [str(c) for c in raw]
        if "classes" in raw:       return raw["classes"]
        if "labels"  in raw:       return raw["labels"]
        return [raw[str(k)] for k in sorted(int(k) for k in raw)]
    except:
        return [f"Class_{i:02d}" for i in range(50)]

CLASS_NAMES = load_classes("classes.json")
model       = tf.keras.models.load_model("emoji_model.h5")

print(f"[OK] {len(CLASS_NAMES)} classes  |  Model output: {model.output_shape[-1]}")


# ══════════════════════════════════════════════════════════════
#  PIPELINE XỬ LÝ ẢNH & DỰ ĐOÁN
# ══════════════════════════════════════════════════════════════
def predict(sketch_input):
    """
    B1  Lấy PIL Image từ Gradio Sketchpad
    B2  Lấy kênh Alpha (nét vẽ = 255, nền = 0)
    B3  Bounding box → crop → padding
    B4  Resize 28×28 LANCZOS
    B5  Normalize → reshape(1,784) → model.predict()
    """
    if sketch_input is None:
        return _html_empty()

    # B1: Lấy ảnh
    img = sketch_input.get("composite") if isinstance(sketch_input, dict) else sketch_input
    if img is None: return _html_empty()
    if isinstance(img, np.ndarray): img = Image.fromarray(img)

    # B2: Grayscale + Invert
    img_gray = img.convert("L")
    if np.array(img_gray).min() >= 250:
        return _html_warn("⚠️  Vui lòng vẽ hình trước!")
    img_inv = ImageOps.invert(img_gray)

    # B3: Crop + Padding
    bbox = img_inv.getbbox()
    if not bbox: return _html_warn("⚠️  Không tìm thấy nét vẽ!")
    crop   = img_inv.crop(bbox)
    pad    = max(crop.width, crop.height) // 8
    padded = ImageOps.expand(crop, border=pad, fill=0)

    # B4: Resize 28×28
    img_28 = padded.resize((28, 28), Image.LANCZOS)

    # B5: Normalize + Reshape + Predict
    arr   = np.array(img_28, dtype=np.float32) / 255.0
    probs = model.predict(arr.reshape(1, 784), verbose=0)[0]
    top3  = np.argsort(probs)[::-1][:3]

    print(f"[PREDICT] {CLASS_NAMES[top3[0]]} ({probs[top3[0]]*100:.1f}%)")
    return _html_result(probs, top3)


# ══════════════════════════════════════════════════════════════
#  HTML KẾT QUẢ — Violet Dream Style
# ══════════════════════════════════════════════════════════════
RANK_STYLES = [
    # (badge_bg, bar_from, bar_to, glow_rgba, label)
    ("#9333ea", "#9333ea", "#c084fc", "rgba(147,51,234,.5)", "1ST"),
    ("#ec4899", "#ec4899", "#f9a8d4", "rgba(236,72,153,.45)", "2ND"),
    ("#3b82f6", "#3b82f6", "#93c5fd", "rgba(59,130,246,.4)", "3RD"),
]


def _html_result(probs: np.ndarray, top3: np.ndarray) -> str:
    top_name = CLASS_NAMES[top3[0]]
    top_conf = float(probs[top3[0]]) * 100
    top_icon = get_icon(top_name)
    top_disp = top_name.replace("_", " ").replace("-", " ").title()

    # Màu confidence tổng
    if   top_conf >= 70: conf_color = "#10b981"
    elif top_conf >= 40: conf_color = "#9333ea"
    else:                conf_color = "#f59e0b"

    # ── Showcase box (kết quả chính) ──────────────────────
    showcase = f"""
    <div style="
      background: linear-gradient(135deg,rgba(147,51,234,.12),rgba(236,72,153,.08));
      border: 1px solid rgba(147,51,234,.35);
      border-radius: 20px; padding: 28px 20px;
      text-align: center; margin-bottom: 18px;
      box-shadow: 0 0 40px rgba(147,51,234,.15);
    ">
      <div style="font-size:76px;line-height:1;margin-bottom:10px;
                  filter:drop-shadow(0 0 18px rgba(147,51,234,.7));">{top_icon}</div>
      <div style="font-size:20px;font-weight:800;color:#f0f4ff;margin-bottom:6px;">
        {top_disp}
      </div>
      <div style="
        font-size:40px;font-weight:900;font-family:Consolas,monospace;
        background:linear-gradient(135deg,{conf_color},{conf_color}99);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        letter-spacing:-1px;
      ">{top_conf:.1f}%</div>
      <div style="font-size:11px;color:#3d3060;margin-top:2px;">Độ tin cậy</div>
    </div>"""

    # ── 3 detail cards ────────────────────────────────────
    cards = ""
    for rank, idx in enumerate(top3):
        name = CLASS_NAMES[idx]
        pct  = float(probs[idx]) * 100
        icon = get_icon(name)
        disp = name.replace("_", " ").replace("-", " ").title()
        badge_bg, bar_from, bar_to, glow, label = RANK_STYLES[rank]
        bold = "font-weight:800;color:#f0f4ff;" if rank == 0 else "color:#8b8dbf;"

        cards += f"""
        <div style="
          background:rgba(255,255,255,{0.04 - rank*0.01});
          border:1px solid rgba(255,255,255,{0.08 - rank*0.02});
          border-radius:14px; padding:14px 16px; margin-bottom:10px;
          display:flex; align-items:center; gap:14px;
          flex-direction:column;
        ">
          <div style="display:flex;align-items:center;gap:12px;width:100%;">
            <!-- Emoji -->
            <span style="font-size:36px;line-height:1;flex-shrink:0;">{icon}</span>

            <!-- Info -->
            <div style="flex:1;min-width:0;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="
                  background:{badge_bg};color:#fff;font-size:9px;
                  font-weight:700;padding:2px 8px;border-radius:5px;
                  font-family:Consolas,monospace;flex-shrink:0;
                ">{label}</span>
                <span style="font-size:14px;{bold}
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                  {disp}
                </span>
              </div>
              <!-- Bar -->
              <div style="background:rgba(255,255,255,.07);border-radius:99px;
                          height:5px;overflow:hidden;">
                <div style="
                  width:{pct:.1f}%;height:100%;border-radius:99px;
                  background:linear-gradient(90deg,{bar_from},{bar_to});
                  box-shadow:0 0 8px {glow};
                "></div>
              </div>
            </div>

            <!-- % -->
            <div style="
              font-size:{20-rank*2}px;font-weight:900;
              font-family:Consolas,monospace;
              color:{badge_bg};flex-shrink:0;
            ">{pct:.1f}%</div>
          </div>
        </div>"""

    # Tổng hợp
    others_pct = (1 - sum(float(probs[i]) for i in top3)) * 100
    others_txt = f"""
    <div style="text-align:center;font-size:12px;color:#1e1840;margin-top:4px;">
      {len(CLASS_NAMES)-3} nhãn còn lại · {others_pct:.1f}%
    </div>"""

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;color:#e2e8f0;">
      {showcase}
      <div style="font-size:10px;color:#2d2b55;text-transform:uppercase;
                  letter-spacing:1px;font-weight:700;margin-bottom:10px;">
        ◈  Top 3 chi tiết
      </div>
      {cards}
      {others_txt}
    </div>"""


def _html_empty() -> str:
    return """
    <div style="
      font-family:'Segoe UI',system-ui,sans-serif;
      background:rgba(147,51,234,.06);
      border:1px dashed rgba(147,51,234,.25);
      border-radius:20px;padding:52px 30px;text-align:center;
    ">
      <div style="font-size:52px;margin-bottom:14px;opacity:.4;">✏️</div>
      <div style="color:#2d2560;font-size:15px;">
        Vẽ hình vào khung rồi bấm
        <b style="color:#9333ea;">Dự Đoán</b>
      </div>
    </div>"""


def _html_warn(msg: str) -> str:
    return f"""
    <div style="
      font-family:'Segoe UI',system-ui,sans-serif;
      background:rgba(245,158,11,.08);
      border:1px solid rgba(245,158,11,.3);
      border-radius:16px;padding:28px;text-align:center;
      color:#f59e0b;font-size:15px;
    ">{msg}</div>"""


# ══════════════════════════════════════════════════════════════
#  CSS TOÀN TRANG — Violet Dream
# ══════════════════════════════════════════════════════════════
CSS = """
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, #root {
  background: #04050d !important;
  min-height: 100vh;
}
.gradio-container { max-width: 1160px !important; margin: 0 auto !important; }

/* Aurora background */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 650px 450px at 8%  18%, rgba(147,51,234,.18) 0%, transparent 65%),
    radial-gradient(ellipse 500px 550px at 90% 78%, rgba(236,72,153,.13) 0%, transparent 65%),
    radial-gradient(ellipse 350px 300px at 52% 92%, rgba(59,130,246,.09) 0%, transparent 65%);
  animation: aurora 13s ease-in-out infinite alternate;
}
@keyframes aurora {
  0%   { transform: translate(0,0)      scale(1);    filter: hue-rotate(0deg); }
  50%  { transform: translate(20px,-15px) scale(1.04); filter: hue-rotate(12deg); }
  100% { transform: translate(-12px,12px) scale(.97);  filter: hue-rotate(-8deg); }
}

/* Nền panels */
.gr-block, .gr-box, .gr-group, .block, [class*="wrap"],
.gradio-row, .gradio-column, .contain, .gap {
  background: transparent !important; border: none !important;
}

/* Glass panels */
.panel-glass {
  background: rgba(255,255,255,0.035) !important;
  backdrop-filter: blur(24px) !important;
  border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 24px !important;
  box-shadow: 0 4px 28px rgba(0,0,0,.5), 0 1px 0 rgba(255,255,255,.05) inset !important;
  padding: 22px !important;
}

/* Canvas wrapper */
.draw-board {
  background: #ffffff !important;
  border-radius: 18px !important;
  border: 2px solid rgba(147,51,234,.4) !important;
  box-shadow: 0 0 36px rgba(147,51,234,.18), 0 0 0 1px rgba(236,72,153,.15) !important;
  overflow: hidden !important;
  transition: border-color .25s, box-shadow .25s !important;
}
.draw-board:hover {
  border-color: rgba(147,51,234,.7) !important;
  box-shadow: 0 0 52px rgba(147,51,234,.28), 0 0 0 1px rgba(236,72,153,.3) !important;
}

/* Canvas element */
.draw-board canvas {
  background: transparent !important;
  cursor: crosshair !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

/* Toolbar Sketchpad */
.toolbar, [class*="toolbar"] {
  background: rgba(8,6,22,.9) !important;
  border: 1px solid rgba(147,51,234,.2) !important;
  border-radius: 14px !important;
}
.toolbar button { color: #4a3a8a !important; border-radius: 8px !important; }
.toolbar button:hover { background: rgba(147,51,234,.2) !important; color: #c084fc !important; }
.toolbar button.selected, .toolbar button[aria-pressed="true"] {
  background: rgba(147,51,234,.3) !important; color: #e9d5ff !important;
}

/* Nút Dự Đoán */
button.primary, button[variant="primary"] {
  background: linear-gradient(135deg, #9333ea, #7c3aed) !important;
  border: none !important; border-radius: 14px !important;
  color: #fff !important; font-size: 16px !important; font-weight: 700 !important;
  height: 54px !important;
  box-shadow: 0 4px 24px rgba(147,51,234,.5) !important;
  transition: all .22s !important;
}
button.primary:hover, button[variant="primary"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 10px 36px rgba(147,51,234,.65) !important;
}

/* Nút Xóa */
button.secondary, button[variant="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,.09) !important;
  border-radius: 14px !important; color: #4a3a8a !important;
  height: 54px !important; transition: all .2s !important;
}
button.secondary:hover {
  background: rgba(147,51,234,.1) !important;
  border-color: rgba(147,51,234,.3) !important; color: #c084fc !important;
}

label span, p { color: #4a3a8a !important; }
footer, .footer { display: none !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(147,51,234,.4); border-radius: 99px; }
"""

# ══════════════════════════════════════════════════════════════
#  HEADER HTML
# ══════════════════════════════════════════════════════════════
HEADER_HTML = f"""
<div style="
  text-align:center;padding:44px 20px 30px;
  font-family:'Segoe UI',system-ui,sans-serif;
">
  <!-- Badge -->
  <div style="
    display:inline-flex;align-items:center;gap:8px;
    background:linear-gradient(90deg,rgba(147,51,234,.2),rgba(236,72,153,.15));
    border:1px solid rgba(147,51,234,.4);border-radius:99px;
    padding:7px 22px;font-size:11px;color:#c084fc;
    letter-spacing:1.2px;text-transform:uppercase;font-weight:700;margin-bottom:20px;
  ">
    <span style="width:7px;height:7px;border-radius:50%;background:#9333ea;
                 box-shadow:0 0 8px #9333ea;display:inline-block;
                 animation:blink 2s ease-in-out infinite;"></span>
    ANN  ·  {len(CLASS_NAMES)} Classes  ·  QuickDraw
  </div>

  <!-- Title -->
  <h1 style="
    font-size:clamp(32px,5vw,54px);font-weight:900;letter-spacing:-2px;
    line-height:1.1;margin:0 0 14px;
    background:linear-gradient(135deg,#f0e6ff 0%,#c084fc 35%,#f9a8d4 65%,#93c5fd 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  ">DrawSense AI</h1>

  <p style="color:#2d2560;font-size:16px;max-width:460px;margin:0 auto;line-height:1.6;">
    Vẽ bất kỳ hình nào — AI sẽ nhận diện và cho bạn biết đó là gì ⚡
  </p>

  <style>
    @keyframes blink {{
      0%,100%{{opacity:1;transform:scale(1);}}
      50%{{opacity:.4;transform:scale(1.5);}}
    }}
  </style>
</div>
"""

TIPS_HTML = """
<div style="
  font-family:'Segoe UI',system-ui,sans-serif;
  background:rgba(147,51,234,.07);
  border:1px solid rgba(147,51,234,.2);
  border-radius:14px;padding:14px 18px;margin-top:6px;
">
  <div style="font-size:11px;color:#9333ea;text-transform:uppercase;
              letter-spacing:.9px;font-weight:700;margin-bottom:8px;">
    ◈ Mẹo vẽ đẹp
  </div>
  <div style="font-size:13px;color:#2d2560;line-height:1.8;">
    • Vẽ <b style="color:#4a3a8a;">to</b> — chiếm ít nhất 60% canvas<br>
    • Nét <b style="color:#4a3a8a;">liên tục</b>, không đứt đoạn<br>
    • Hình <b style="color:#4a3a8a;">đơn giản</b> như phong cách vẽ nhanh tay
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════
#  BUILD GRADIO UI
# ══════════════════════════════════════════════════════════════
with gr.Blocks(
    css=CSS,
    title="✦ DrawSense AI  ·  Violet Dream",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=["Segoe UI", "system-ui", gr.themes.GoogleFont("Inter")],
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    gr.HTML('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(147,51,234,.3),rgba(236,72,153,.2),transparent);margin:0 24px 28px;"></div>')

    with gr.Row(equal_height=False):

        # ═══ CỘT TRÁI — Canvas ═══════════════════════════════════
        with gr.Column(scale=11, elem_classes="panel-glass"):
            gr.HTML("""<div style="font-size:10px;color:#9333ea;text-transform:uppercase;
                       letter-spacing:1.2px;font-weight:700;margin-bottom:12px;
                       font-family:'Segoe UI',sans-serif;">✏️  Bảng vẽ</div>""")

            canvas = gr.Image(
                label="", source="canvas", tool="sketch",
                type="pil", shape=(480, 480),
                brush_radius=13, show_label=False,
                elem_classes=["draw-board"]
            )

            with gr.Row():
                btn_predict = gr.Button("⬡  Dự Đoán", variant="primary", scale=3)
                btn_clear   = gr.Button("↺  Xóa", variant="secondary", scale=1)

            gr.HTML(TIPS_HTML)

        # ═══ CỘT PHẢI — Kết quả ══════════════════════════════════
        with gr.Column(scale=9, elem_classes="panel-glass"):
            gr.HTML("""<div style="font-size:10px;color:#9333ea;text-transform:uppercase;
                       letter-spacing:1.2px;font-weight:700;margin-bottom:12px;
                       font-family:'Segoe UI',sans-serif;">📊  Kết quả phân tích</div>""")

            out_html = gr.HTML(value=_html_empty())

            gr.HTML(f"""
            <div style="margin-top:16px;border-top:1px solid rgba(255,255,255,.05);
                        padding-top:12px;display:flex;justify-content:space-between;
                        font-family:'Segoe UI',sans-serif;">
              <div style="font-size:11px;color:#13102a;">
                ANN · input 784 · {len(CLASS_NAMES)} classes
              </div>
              <div style="font-size:11px;color:#13102a;">TF {tf.__version__}</div>
            </div>
            """)

    gr.HTML("""
    <div style="text-align:center;padding:24px;
                font-family:'Segoe UI',sans-serif;font-size:12px;color:#0d0b22;">
      DrawSense AI  ·  Violet Dream  ·  Made with ❤️ &nbsp;+&nbsp; Gradio
    </div>
    """)

    # ── Sự kiện ─────────────────────────────────────────────
    btn_predict.click(fn=predict, inputs=[canvas], outputs=[out_html])
    btn_clear.click(fn=lambda: (None, _html_empty()), outputs=[canvas, out_html])
    
    js_func = """
    function() {
        // Ép Gradio về Light Mode để nét bút mặc định là màu đen
        document.body.classList.remove('dark');
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.target.classList.contains('dark')) {
                    mutation.target.classList.remove('dark');
                }
            });
        });
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    }
    """
    demo.load(_js=js_func)

# ══════════════════════════════════════════════════════════════
#  KHỞI CHẠY
# ══════════════════════════════════════════════════════════════
demo.launch(share=True, debug=False)
