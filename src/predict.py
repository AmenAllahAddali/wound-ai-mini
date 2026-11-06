"""
Module d'inférence Wound-AI Mini.
Charge le U-Net, prédit le masque de la plaie et le renvoie à la taille d'origine.
"""
import os
import sys

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# ---------------------------------------------------------------
# Configuration : doit être IDENTIQUE à l'entraînement d'Aya
# ---------------------------------------------------------------
IMG_SIZE = 512                              # taille d'entrée du modèle
MEAN = np.array([0.485, 0.456, 0.406])      # normalisation ImageNet
STD = np.array([0.229, 0.224, 0.225])
THRESHOLD = 0.5                             # seuil probabilité -> plaie
MODEL_PATH = os.path.join("models", "best_unet.pth")
DEVICE = "cpu"


def build_model():
    """U-Net avec encodeur ResNet34, 1 classe de sortie (plaie / fond)."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # les poids viennent du fichier .pth
        in_channels=3,
        classes=1,
    )


def load_model(path=MODEL_PATH):
    """Charge le modèle entraîné. Sans fichier .pth, renvoie un modèle non entraîné (pour tester)."""
    model = build_model()
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # retire le préfixe "module." si le modèle a été entraîné avec DataParallel
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"[OK] Modèle chargé : {path}")
    else:
        print(f"[ATTENTION] {path} introuvable -> modèle NON entraîné (masques aléatoires)")
    model.to(DEVICE).eval()
    return model


def read_image(path):
    """Lit une image depuis le disque et la renvoie en RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image_bytes(data: bytes):
    """Lit une image envoyée en bytes (utile pour l'API) et la renvoie en RGB."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Le fichier envoyé n'est pas une image valide")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(image_rgb):
    """RGB (H, W, 3) uint8 -> tenseur (1, 3, 512, 512) normalisé."""
    img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)                      # HWC -> CHW
    return torch.from_numpy(img).float().unsqueeze(0)  # ajoute la dimension batch


@torch.no_grad()
def predict_mask(model, image_rgb, threshold=THRESHOLD):
    """
    Renvoie :
      - mask : uint8 (H, W), 255 = plaie, 0 = fond, à la taille d'origine
      - prob : float32 (H, W), probabilité de plaie entre 0 et 1
    """
    h, w = image_rgb.shape[:2]
    x = preprocess(image_rgb).to(DEVICE)
    logits = model(x)                                 # (1, 1, 512, 512)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > threshold).astype(np.uint8) * 255
    return mask, prob


# ---------------------------------------------------------------
# Test en ligne de commande : python -m src.predict chemin/image.png
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python -m src.predict chemin/vers/image.png")
        sys.exit(1)

    image = read_image(sys.argv[1])
    model = load_model()
    mask, prob = predict_mask(model, image)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(os.path.join("outputs", "mask.png"), mask)

    wound_px = int((mask > 0).sum())
    print(f"Image : {image.shape[1]}x{image.shape[0]} px")
    print(f"Pixels de plaie : {wound_px} ({100 * wound_px / mask.size:.2f} % de l'image)")
    print("Masque enregistré dans outputs/mask.png")