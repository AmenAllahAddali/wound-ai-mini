"""
Démo Wound-AI Mini (Streamlit)
1) Lancer l'API :   uvicorn api.main:app --port 8000
2) Lancer la démo : streamlit run demo/app.py
"""
import base64
import json

import requests
import streamlit as st

st.set_page_config(page_title="Wound-AI Mini", page_icon="🩹", layout="wide")

# ---------------- Barre latérale : réglages ----------------
st.sidebar.header("⚙️ Réglages")
api_url = st.sidebar.text_input("Adresse de l'API", "http://localhost:8000")
threshold = st.sidebar.slider("Seuil de détection", 0.05, 0.95, 0.50, 0.05,
                              help="Probabilité minimale pour qu'un pixel soit considéré comme plaie")
px_per_cm = st.sidebar.number_input("Pixels par cm (optionnel)", min_value=0.0, value=0.0, step=1.0,
                                    help="Si un repère de taille est connu sur la photo, donne la surface en cm²")
explain = st.sidebar.checkbox("Afficher l'explication (Grad-CAM)", value=False,
                              help="Carte de chaleur : zones sur lesquelles le modèle s'appuie (plus lent)")

# État de l'API
try:
    health = requests.get(f"{api_url}/health", timeout=3).json()
    st.sidebar.success("API connectée ✅")
    if not health.get("trained_model"):
        st.sidebar.warning("Modèle non entraîné : les résultats sont aléatoires")
except requests.exceptions.RequestException:
    health = None
    st.sidebar.error("API injoignable ❌\n\nLancez : `uvicorn api.main:app --port 8000`")

# ---------------- En-tête ----------------
st.title("🩹 Wound-AI Mini")
st.markdown("Segmentation automatique d'une plaie sur photo (**U-Net**) et mesure de sa surface (**OpenCV**).")
st.caption("⚠️ Outil de démonstration à but pédagogique — ce n'est pas un dispositif médical.")

uploaded = st.file_uploader("Déposez une photo de plaie", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("👆 Chargez une image pour lancer l'analyse.")
    st.stop()

if health is None:
    st.error("Impossible d'analyser : l'API n'est pas lancée.")
    st.stop()

# ---------------- Appel de l'API ----------------
params = {"threshold": threshold, "explain": str(explain).lower()}
if px_per_cm > 0:
    params["px_per_cm"] = px_per_cm

with st.spinner("Analyse en cours..." + (" (Grad-CAM : quelques secondes de plus)" if explain else "")):
    try:
        resp = requests.post(
            f"{api_url}/predict",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            params=params,
            timeout=120,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        st.stop()

if resp.status_code != 200:
    st.error(f"Erreur {resp.status_code} : {resp.json().get('detail', resp.text)}")
    st.stop()

data = resp.json()
m = data["measurements"]

# ---------------- Mesures ----------------
st.subheader("📏 Mesures")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Plaie détectée", "Oui" if m["wound_detected"] else "Non")
c2.metric("Surface (px)", f"{m['area_px']:,}".replace(",", " "))
c3.metric("% de l'image", f"{m['area_percent']} %")
conf = data.get("mean_confidence")
c4.metric("Confiance moyenne", f"{conf:.0%}" if conf is not None else "—")
if "area_cm2" in m:
    st.metric("Surface estimée", f"{m['area_cm2']} cm²")

# ---------------- Images ----------------
st.subheader("🖼️ Résultat")
has_cam = "gradcam_png_base64" in data
cols = st.columns(4 if has_cam else 3)
cols[0].image(uploaded.getvalue(), caption="Photo originale")
cols[1].image(base64.b64decode(data["overlay_png_base64"]), caption="Plaie détectée (rouge) + cadre")
cols[2].image(base64.b64decode(data["mask_png_base64"]), caption="Masque de segmentation")
if has_cam:
    cols[3].image(base64.b64decode(data["gradcam_png_base64"]), caption="Grad-CAM (où le modèle regarde)")
    st.caption("Grad-CAM : rouge/jaune = zones les plus utilisées par le modèle, bleu = zones ignorées. "
               "Des activations parasites peuvent apparaître sur les bandes noires de l'image.")

# ---------------- Détails ----------------
with st.expander("Détails techniques (JSON)"):
    details = {k: v for k, v in data.items() if not k.endswith("_base64")}
    st.json(details)
    st.download_button("Télécharger les mesures (JSON)",
                       json.dumps(details, indent=2, ensure_ascii=False),
                       file_name="mesures.json", mime="application/json")
