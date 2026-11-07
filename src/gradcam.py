"""
Grad-CAM pour le U-Net Wound-AI : montre où le modèle "regarde" pour détecter la plaie.
Test : python -m src.gradcam test.png
"""
import os
import sys

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

from src.predict import preprocess, THRESHOLD

PADDING_LEVEL = 8  # pixels plus sombres que ça sur les 3 canaux = bande noire de remplissage


def get_target_layer(model):
    """
    Couche analysée par Grad-CAM.
    v1 : dernier bloc de l'encodeur (grille 16x16) -> trop grossier pour les petites plaies.
    v2 : 3e bloc du décodeur (grille 128x128) -> carte bien plus précise.
    """
    blocks = getattr(model.decoder, "blocks", None)
    if blocks is not None and len(blocks) >= 3:
        return blocks[2]
    return model.encoder.layer4[-1]


def gradcam_overlay(model, image_rgb, threshold=THRESHOLD):
    """
    image_rgb : uint8 (H, W, 3)
    Renvoie :
      - overlay : uint8 (H, W, 3) photo + carte de chaleur (rouge = zone importante)
      - heatmap : float32 (H, W) valeurs entre 0 et 1
    """
    h, w = image_rgb.shape[:2]
    device = next(model.parameters()).device
    x = preprocess(image_rgb).to(device)              # (1, 3, 512, 512)

    # Zone prédite comme plaie : c'est elle qu'on veut expliquer
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    target_mask = (prob > threshold).astype(np.float32)
    if target_mask.sum() == 0:                         # aucune plaie prédite -> expliquer toute l'image
        target_mask = np.ones_like(target_mask)

    with GradCAM(model=model, target_layers=[get_target_layer(model)]) as cam:
        heatmap = cam(input_tensor=x, targets=[SemanticSegmentationTarget(0, target_mask)])[0]

    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Les bandes noires ajoutées pour compléter l'image à 512x512 ne font pas partie de la photo :
    # on n'y affiche pas de chaleur (sinon artefacts de bord).
    padding = image_rgb.max(axis=2) < PADDING_LEVEL
    heatmap[padding] = 0
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    overlay = show_cam_on_image(image_rgb.astype(np.float32) / 255.0, heatmap, use_rgb=True)
    return overlay, heatmap.astype(np.float32)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python -m src.gradcam chemin/image.png")
        sys.exit(1)

    from src.predict import read_image, load_model

    image = read_image(sys.argv[1])
    model = load_model()
    print("Couche Grad-CAM :", get_target_layer(model).__class__.__name__)
    overlay, _ = gradcam_overlay(model, image)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(os.path.join("outputs", "gradcam.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Grad-CAM enregistré dans outputs/gradcam.png")
