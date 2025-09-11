import os
import io
import json
import base64
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv

import google.generativeai as genai

# ------------- Setup -------------
load_dotenv()
st.set_page_config(page_title="Calorie Lens 🍽️", page_icon="🍽️", layout="wide")

# --- Keys & models from Streamlit Secrets (fallback to .env) ---
GEMINI_DEFAULT_VISION_MODEL = (
    st.secrets.get("GEMINI_VISION_MODEL", os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash"))
)
GEMINI_FALLBACK_VISION_MODEL = (
    st.secrets.get("GEMINI_FALLBACK_VISION_MODEL", os.getenv("GEMINI_FALLBACK_VISION_MODEL", "gemini-pro-vision"))
)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in your environment. Create a .env file with GOOGLE_API_KEY=your_key")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are a nutrition analyst. Given a single food photo, produce a careful, honest estimate.
Return STRICT JSON with this schema (no extra text):

{
  "items": [
    {
      "name": "string",
      "portion_grams": number,             // estimated edible grams for THIS item
      "calories_kcal": number,             // estimated kcal for THIS item
      "macros": {"protein_g": number, "carbs_g": number, "fat_g": number},
      "tags": ["vegetarian|vegan|gluten_free|dairy_free|egg_free|nut_free|halal|kosher|unknown"...],
      "allergens": ["milk","egg","fish","shellfish","tree_nuts","peanuts","wheat","soy","sesame"]
    }
  ],
  "totals": {
    "portion_grams": number,
    "calories_kcal": number,
    "macros": {"protein_g": number, "carbs_g": number, "fat_g": number}
  },
  "health_assessment": {
    "summary": "2-4 sentences about healthfulness based on visual cues; include caveats",
    "reasons_positive": ["..."],
    "reasons_negative": ["..."],
    "suggestions": ["practical swaps/portion advice"]
  },
  "confidence": 0-1
}

Rules:
- If unsure, be conservative and say so via lower confidence.
- Assume common recipes for the cuisine you detect.
- Include sauces/oils if visible.
- Avoid hallucinating nutrition for invisible fillings.
"""

# Helpers
def _image_to_part(uploaded_file) -> List[Dict[str, Any]]:
    """Convert uploaded file to Gemini inline data part."""
    if not uploaded_file:
        raise ValueError("Invalid image file")
    bytes_data = uploaded_file.getvalue()
    return [{
        "mime_type": uploaded_file.type or "image/jpeg",
        "data": bytes_data,  
    }]

def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
        return {}

def _sum_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot_g = sum(i.get("portion_grams", 0) for i in items)
    tot_cal = sum(i.get("calories_kcal", 0) for i in items)
    tot_pro = sum(i.get("macros", {}).get("protein_g", 0) for i in items)
    tot_carb = sum(i.get("macros", {}).get("carbs_g", 0) for i in items)
    tot_fat = sum(i.get("macros", {}).get("fat_g", 0) for i in items)
    return {
        "portion_grams": round(tot_g, 1),
        "calories_kcal": round(tot_cal, 0),
        "macros": {"protein_g": round(tot_pro,1), "carbs_g": round(tot_carb,1), "fat_g": round(tot_fat,1)}
    }

def _bytes_from_pil(img: Image.Image, format="JPEG", quality=90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    return buf.getvalue()

def analyze_image_with_gemini(model_name: str, uploaded_file, user_notes: str="") -> Dict[str, Any]:
    image_parts = _image_to_part(uploaded_file)

    user_prompt = SYSTEM_PROMPT
    if user_notes:
        user_prompt += f'\nUser notes/preferences/intolerances (may influence assessment): "{user_notes}".'

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content([user_prompt, image_parts[0]])
    text = resp.text or ""
    data = _safe_json_loads(text)

    # Minimal validation & backfill
    if not data.get("items"):
        data["items"] = []
    for it in data["items"]:
        it.setdefault("name", "unknown")
        it.setdefault("portion_grams", 0)
        it.setdefault("calories_kcal", 0)
        it.setdefault("macros", {"protein_g":0,"carbs_g":0,"fat_g":0})
        it.setdefault("tags", [])
        it.setdefault("allergens", [])

    if "totals" not in data or not data["totals"]:
        data["totals"] = _sum_items(data["items"])

    data.setdefault("health_assessment", {
        "summary": "No assessment available.",
        "reasons_positive": [],
        "reasons_negative": [],
        "suggestions": []
    })
    data.setdefault("confidence", 0.4)

    return data

# ------------- UI -------------
st.title("🍽️ Calorie Lens: Food Photo → Calories & Health Insights")
st.caption("Uses Google Gemini Vision. Estimates only; may be inaccurate. Not medical advice.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Vision model",
        [GEMINI_DEFAULT_VISION_MODEL, GEMINI_FALLBACK_VISION_MODEL, "gemini-1.5-pro-latest"],
        help="1.5-flash is fast & cheap, 1.5-pro is stronger, pro-vision is legacy."
    )
    enable_camera = st.toggle("Use camera input", value=False)
    st.markdown("---")
    st.write("**Optional notes** (diet goals, allergies, cuisine, etc.)")
    user_notes = st.text_area("Notes", placeholder="e.g., lactose intolerant, tracking high protein…")
    st.markdown("---")
    st.info("Tip: Good lighting and a clear angle improve recognition. Include cutlery for scale if possible.")

# Main input
col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = None
    if enable_camera:
        cam = st.camera_input("Take a photo")
        if cam: uploaded_file = cam
    else:
        uploaded_file = st.file_uploader("Upload a food image", type=["jpg","jpeg","png","webp"])

    analyze_btn = st.button("Analyze Image 🔍", type="primary", use_container_width=True, disabled=not uploaded_file)

with col2:
    st.subheader("History")
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"{h['timestamp']} · {h['totals']['calories_kcal']} kcal · {h.get('title','(image)')}"):
                st.json(h["result"])
    else:
        st.caption("No analyses yet.")

# ------------- Run analysis -------------
if analyze_btn and uploaded_file:
    try:
        # Preprocess: light EXIF fix & resize cap
        img = Image.open(uploaded_file).convert("RGB")
        img = ImageOps.exif_transpose(img)
        # limit very large images for latency
        max_dim = 1600
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))

        # Replace uploaded file bytes with processed image bytes for consistent input
        processed = _bytes_from_pil(img)
        uploaded_file = type("UploadedLike", (), {
            "getvalue": lambda self: processed,
            "type": "image/jpeg",
            "name": getattr(uploaded_file, "name", "photo.jpg")
        })()

        # Primary call
        data = analyze_image_with_gemini(model_choice, uploaded_file, user_notes)

        # UI: show image and quick metrics
        st.image(img, caption="Analyzed image", use_container_width=True)
        totals = data["totals"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Calories", f"{totals['calories_kcal']:.0f} kcal")
        m2.metric("Total Portion", f"{totals['portion_grams']:.0f} g")
        m3.metric("Protein", f"{totals['macros']['protein_g']:.1f} g")
        m4.metric("Carbs / Fat", f"{totals['macros']['carbs_g']:.1f} g / {totals['macros']['fat_g']:.1f} g")

        # Per-item editable portions
        st.subheader("Per-item breakdown (adjust portions if needed)")
        new_items = []
        for idx, item in enumerate(data["items"]):
            with st.expander(f"{idx+1}. {item['name']} — {item['calories_kcal']:.0f} kcal (est.)"):
                c1, c2 = st.columns([2,1])
                with c1:
                    st.write(f"Estimated macros: **P {item['macros']['protein_g']} g** · **C {item['macros']['carbs_g']} g** · **F {item['macros']['fat_g']} g**")
                    st.write(f"Tags: {', '.join(item.get('tags',[]) or ['—'])}")
                    st.write(f"Potential allergens: {', '.join(item.get('allergens',[]) or ['—'])}")
                with c2:
                    portion = st.number_input(
                        f"Portion (g) — {item['name']}",
                        min_value=0.0,
                        value=float(item.get("portion_grams", 0.0)),
                        step=5.0,
                        key=f"portion_{idx}"
                    )
                # Re-scale calories/macros linearly with portion change (assume model's estimate is baseline)
                baseline_g = max(item.get("portion_grams", 1e-6), 1e-6)
                scale = (portion / baseline_g) if baseline_g > 0 else 1.0
                adj_item = {
                    **item,
                    "portion_grams": round(portion,1),
                    "calories_kcal": round(item["calories_kcal"] * scale, 1),
                    "macros": {
                        "protein_g": round(item["macros"]["protein_g"] * scale, 2),
                        "carbs_g": round(item["macros"]["carbs_g"] * scale, 2),
                        "fat_g": round(item["macros"]["fat_g"] * scale, 2),
                    }
                }
                new_items.append(adj_item)

        # Recompute totals after user adjustments
        adj_totals = _sum_items(new_items)
        st.markdown("### Adjusted totals")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Calories", f"{adj_totals['calories_kcal']:.0f} kcal")
        t2.metric("Portion", f"{adj_totals['portion_grams']:.0f} g")
        t3.metric("Protein", f"{adj_totals['macros']['protein_g']:.1f} g")
        t4.metric("Carbs / Fat", f"{adj_totals['macros']['carbs_g']:.1f} g / {adj_totals['macros']['fat_g']:.1f} g")

        # Health assessment
        st.subheader("Health Assessment")
        ha = data.get("health_assessment", {})
        st.write(ha.get("summary",""))
        cpos, cneg, csug = st.columns(3)
        with cpos:
            st.markdown("**Positives**")
            st.write("\n".join([f"• {r}" for r in ha.get("reasons_positive", [])]) or "—")
        with cneg:
            st.markdown("**Watch-outs**")
            st.write("\n".join([f"• {r}" for r in ha.get("reasons_negative", [])]) or "—")
        with csug:
            st.markdown("**Suggestions**")
            st.write("\n".join([f"• {r}" for r in ha.get("suggestions", [])]) or "—")

        st.caption(f"Model confidence: **{data.get('confidence',0):.2f}** (0=low, 1=high)")

        # Downloads
        st.subheader("Export")
        export_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": user_notes,
            "items": new_items,
            "totals": adj_totals,
            "model": model_choice,
            "confidence": data.get("confidence", 0)
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(export_payload, indent=2).encode("utf-8"),
            file_name="calorie_lens_result.json",
            mime="application/json"
        )

        # CSV export
        import csv
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["name","portion_grams","calories_kcal","protein_g","carbs_g","fat_g","tags","allergens"])
        for it in new_items:
            writer.writerow([
                it["name"],
                it["portion_grams"],
                it["calories_kcal"],
                it["macros"]["protein_g"],
                it["macros"]["carbs_g"],
                it["macros"]["fat_g"],
                "|".join(it.get("tags",[])),
                "|".join(it.get("allergens",[])),
            ])
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="calorie_lens_items.csv",
            mime="text/csv"
        )

        # Save to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "title": uploaded_file.name if hasattr(uploaded_file, "name") else "photo.jpg",
            "result": export_payload,
            "totals": adj_totals
        })

        # Disclaimer
        st.info("This is an automated estimate from an image and can be wrong. For medical or dietary advice, consult a professional.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()