"""
Mesure de la plaie à partir du masque (OpenCV).
Surface en pixels et en %, contours, rectangle englobant, image de superposition.
"""
import os
import sys
import json

import cv2
import numpy as np

MIN_AREA = 50  # en pixels : les zones plus petites sont considérées comme du bruit


def measure_wound(mask, min_area=MIN_AREA, px_per_cm=None):
    """
    mask : uint8 (H, W), 255 = plaie, 0 = fond
    px_per_cm : optionnel, nombre de pixels pour 1 cm (si un repère de taille est connu)
    Renvoie (résultats sous forme de dict, liste des contours gardés)
    """
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # masque nettoyé (sans les petites taches) pour calculer la surface
    clean = np.zeros_like(binary)
    cv2.drawContours(clean, contours, -1, 1, thickness=cv2.FILLED)
    area_px = int(clean.sum())
    h, w = mask.shape[:2]

    result = {
        "wound_detected": area_px > 0,
        "area_px": area_px,
        "area_percent": round(100 * area_px / (h * w), 2),
        "regions": len(contours),
        "perimeter_px": round(float(sum(cv2.arcLength(c, True) for c in contours)), 1),
        "bbox": None,                 # [x, y, largeur, hauteur]
        "image_size": [w, h],
    }
    if contours:
        x, y, bw, bh = cv2.boundingRect(np.vstack(contours))
        result["bbox"] = [int(x), int(y), int(bw), int(bh)]
    if px_per_cm:
        result["area_cm2"] = round(area_px / (px_per_cm ** 2), 2)
    return result, contours


def make_overlay(image_rgb, contours, bbox=None, color=(255, 0, 0), alpha=0.4):
    """Plaie remplie en couleur semi-transparente + contour + rectangle englobant (image RGB)."""
    filled = image_rgb.copy()
    cv2.drawContours(filled, contours, -1, color, thickness=cv2.FILLED)
    out = cv2.addWeighted(filled, alpha, image_rgb, 1 - alpha, 0)
    cv2.drawContours(out, contours, -1, color, thickness=2)
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
    return out


# ---------------------------------------------------------------
# Test : python -m src.measure test.png              (masque prédit par le modèle)
#        python -m src.measure test.png test_mask.png (vrai masque du dataset)
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python -m src.measure image.png [masque.png]")
        sys.exit(1)

    from src.predict import read_image, load_model, predict_mask

    image = read_image(sys.argv[1])

    if len(sys.argv) >= 3:
        mask = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Impossible de lire le masque : {sys.argv[2]}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        print("[INFO] Masque fourni (vrai masque du dataset)")
    else:
        mask, _ = predict_mask(load_model(), image)

    result, contours = measure_wound(mask)
    overlay = make_overlay(image, contours, result["bbox"])

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(os.path.join("outputs", "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Superposition enregistrée dans outputs/overlay.png")