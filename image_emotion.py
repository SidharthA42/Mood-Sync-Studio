import torch
import numpy as np
from PIL import Image
from utils import load_image_model, label_order_image
import cv2


# ── Emotion colour map (BGR for OpenCV drawing) ──────────────────────────────
_EMOTION_BGR = {
    "happy":    (129, 185, 16),   # green
    "sad":      (241, 102, 99),   # indigo
    "angry":    (68,  68,  239),  # red
    "fear":     (246, 92,  139),  # purple
    "disgust":  (22,  115, 249),  # orange
    "surprise": (8,   179, 234),  # yellow
    "neutral":  (139, 122, 107),  # slate
}


def detect_faces(img_np):
    """Detect faces in an RGB image and return boxes sorted by area."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(42, 42)
    )
    return sorted(
        [tuple(map(int, face)) for face in faces],
        key=lambda f: f[2] * f[3],
        reverse=True,
    )


def annotate_faces(image: Image.Image, label: str, confidence: float):
    """Draw face boxes and the predicted mood label on a copy of an uploaded image."""
    img_np = np.array(image.convert("RGB"))
    faces = detect_faces(img_np)
    annotated = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    color = (99, 102, 241)

    for index, (x, y, w, h) in enumerate(faces):
        line_color = color if index == 0 else (148, 163, 184)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), line_color, 2)
        text = f"{label.upper()} {confidence * 100:.0f}%" if index == 0 else "FACE"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(28, y - 12)
        cv2.rectangle(
            annotated,
            (x, label_y - th - 12),
            (x + tw + 18, label_y + 6),
            line_color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x + 9, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), faces


def predict_emotion_frame(frame_rgb: np.ndarray):
    """
    Run full ViT emotion inference on a single RGB frame (numpy array).
    Draws a face rectangle with the emotion label above it.

    Returns
    -------
    annotated_rgb : np.ndarray   — the frame with face box + label drawn on it
    emotion_label : str          — top predicted emotion
    confidence    : float        — confidence in [0, 1]
    probs_dict    : dict         — {emotion: probability} for all classes
    """
    processor, model = load_image_model()

    # ── Detect faces first ──────────────────────────────────────────────────
    faces = detect_faces(frame_rgb)

    # ── Run ViT on the whole frame (it handles its own crop internally) ──────
    pil_img = Image.fromarray(frame_rgb)
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = (
        torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    )
    pred_idx   = int(probs.argmax())
    emotion_label = model.config.id2label[pred_idx]
    confidence    = float(probs[pred_idx])
    probs_dict    = {
        model.config.id2label[i]: float(p) for i, p in enumerate(probs)
    }

    # ── Draw annotation ─────────────────────────────────────────────────────
    annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    color_bgr = _EMOTION_BGR.get(emotion_label.lower(), (212, 212, 45))

    if faces:
        for idx, (x, y, w, h) in enumerate(faces):
            box_color = color_bgr if idx == 0 else (148, 163, 184)
            # Rectangle around face
            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), box_color, 2)

            if idx == 0:
                # Label tag above rectangle
                tag_text = f"MOOD: {emotion_label.upper()}  {confidence * 100:.0f}%"
                (tw, th), baseline = cv2.getTextSize(
                    tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
                )
                tag_y = max(th + 18, y - 10)
                # filled badge background
                cv2.rectangle(
                    annotated_bgr,
                    (x, tag_y - th - 10),
                    (x + tw + 16, tag_y + 4),
                    box_color,
                    -1,
                )
                cv2.putText(
                    annotated_bgr,
                    tag_text,
                    (x + 8, tag_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    else:
        # No face — draw a status banner at the top
        banner = "No face detected — position your face in front of the camera"
        cv2.rectangle(annotated_bgr, (0, 0), (frame_rgb.shape[1], 46), (20, 20, 40), -1)
        cv2.putText(
            annotated_bgr,
            banner,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 200),
            1,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, emotion_label, confidence, probs_dict


def _detect_face_mask(img_np, target_size=224):
    """
    Returns a float mask (H x W) = 1 inside detected face bbox, 0 outside.
    Falls back to centre-weighted Gaussian if no face found.
    """
    faces = detect_faces(img_np)

    mask = np.zeros((target_size, target_size), dtype=np.float32)
    h_orig, w_orig = img_np.shape[:2]

    if len(faces) > 0:
        x, y, w, h = faces[0]
        sx = target_size / w_orig
        sy = target_size / h_orig
        x1 = max(0, int(x * sx))
        y1 = max(0, int(y * sy))
        x2 = min(target_size, int((x + w) * sx))
        y2 = min(target_size, int((y + h) * sy))
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(target_size, x2 + pad_x)
        y2 = min(target_size, y2 + pad_y)
        mask[y1:y2, x1:x2] = 1.0
        print(f"Face detected [{x1},{y1},{x2},{y2}]")
    else:
        print("No face detected – centre Gaussian fallback")
        cx, cy = target_size // 2, target_size // 2
        sigma = target_size * 0.25
        ys, xs = np.ogrid[:target_size, :target_size]
        mask = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))

    return mask


def predict_emotion(image: Image.Image, use_attn_heatmap=False):
    processor, model = load_image_model()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_idx = probs.argmax()
    emotion_label = model.config.id2label[pred_idx]
    confidence = float(probs[pred_idx])
    probs_dict = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}

    heatmap_img = None
    if use_attn_heatmap:
        try:
            with torch.no_grad():
                full_outputs = model(**inputs, output_attentions=True)

            # Attention rollout across all layers
            attentions = full_outputs.attentions
            rollout = torch.eye(attentions[0].shape[-1])
            for attn in attentions:
                avg_heads = attn.mean(dim=1).squeeze(0).cpu()
                aug = avg_heads + torch.eye(avg_heads.shape[0])
                aug = aug / aug.sum(dim=-1, keepdim=True)
                rollout = aug @ rollout

            cls_rollout = rollout[0, 1:].numpy()
            size = int(np.sqrt(len(cls_rollout)))
            cls_rollout = cls_rollout[:size * size]
            heatmap = cls_rollout.reshape(size, size)

            heatmap = cv2.resize(heatmap, (224, 224))

            # Full-image rollout: keep the original field of view and make low-attention
            # regions mostly transparent so the strongest contributing regions stand out.
            img_np = np.array(image.resize((224, 224), Image.Resampling.LANCZOS))

            # Normalise, smooth
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = np.clip((heatmap - 0.35) / 0.65, 0, 1)

            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            alpha = (0.18 + 0.52 * heatmap)[..., None]
            heatmap_img = ((1 - alpha) * img_np + alpha * heatmap_color).astype(np.uint8)
            print("Heatmap created (full-image rollout).")
        except Exception as e:
            print(f"Heatmap failed: {e}")

    return emotion_label, confidence, probs_dict, heatmap_img
