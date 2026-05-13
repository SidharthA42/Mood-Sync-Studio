import time
from collections import deque

import cv2
import numpy as np
from PIL import Image

try:
    import av
except Exception:
    av = None

try:
    from streamlit_webrtc import VideoProcessorBase
except Exception:
    VideoProcessorBase = object

from image_emotion import detect_faces, predict_emotion


EMOTION_COLORS = {
    "happy": (16, 185, 129),
    "sad": (99, 102, 241),
    "angry": (239, 68, 68),
    "fear": (139, 92, 246),
    "disgust": (249, 115, 22),
    "surprise": (234, 179, 8),
    "neutral": (107, 122, 153),
}


def _blend_predictions(history):
    totals = {}
    for probs in history:
        for label, value in probs.items():
            totals[label] = totals.get(label, 0.0) + float(value)
    if not totals:
        return "neutral", 0.0, {}

    count = len(history)
    averaged = {label: score / count for label, score in totals.items()}
    label = max(averaged, key=averaged.get)
    return label, averaged[label], averaged


class LiveEmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_inference = 0.0
        self.frame_interval = 0.45
        self.history = deque(maxlen=5)
        self.current_label = "Scanning"
        self.current_confidence = 0.0
        self.current_probs = {}
        self.face_count = 0
        self.latest_clean_rgb = None

    def recv(self, frame):
        if av is None:
            raise ImportError("The live webcam feature requires the `av` package. Run: pip install av")

        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.latest_clean_rgb = img_rgb.copy()
        faces = detect_faces(img_rgb)
        self.face_count = len(faces)

        now = time.time()
        if faces and now - self.last_inference >= self.frame_interval:
            x, y, w, h = faces[0]
            pad_x = int(w * 0.18)
            pad_y = int(h * 0.22)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_rgb.shape[1], x + w + pad_x)
            y2 = min(img_rgb.shape[0], y + h + pad_y)
            face_crop = Image.fromarray(img_rgb[y1:y2, x1:x2]).convert("RGB")

            try:
                label, confidence, probs, _ = predict_emotion(face_crop, use_attn_heatmap=False)
                self.history.append(probs)
                self.current_label, self.current_confidence, self.current_probs = _blend_predictions(self.history)
                if not self.current_label:
                    self.current_label = label
                    self.current_confidence = confidence
                    self.current_probs = probs
            except Exception:
                self.current_label = "Model warming"
                self.current_confidence = 0.0

            self.last_inference = now

        annotated = self._draw_overlay(img_bgr, faces)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _draw_overlay(self, img_bgr, faces):
        output = img_bgr.copy()
        if not faces:
            self._draw_status(output, "No face detected", (107, 122, 153))
            return output

        label = self.current_label or "Scanning"
        confidence = self.current_confidence
        color_rgb = EMOTION_COLORS.get(label.lower(), (99, 102, 241))
        color = (color_rgb[2], color_rgb[1], color_rgb[0])

        for index, (x, y, w, h) in enumerate(faces):
            line_color = color if index == 0 else (148, 163, 184)
            cv2.rectangle(output, (x, y), (x + w, y + h), line_color, 2)

            text = f"{label.upper()}  {confidence * 100:.0f}%" if index == 0 else "FACE"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            label_y = max(28, y - 12)
            cv2.rectangle(output, (x, label_y - th - 12), (x + tw + 18, label_y + 6), line_color, -1)
            cv2.putText(
                output,
                text,
                (x + 9, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        self._draw_status(output, f"Live mood: {label} | faces: {len(faces)}", color_rgb)
        return output

    def _draw_status(self, output, text, color_rgb):
        color = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(output, (18, 18), (18 + min(520, 24 + len(text) * 12), 56), (12, 18, 32), -1)
        cv2.rectangle(output, (18, 18), (18 + min(520, 24 + len(text) * 12), 56), color, 1)
        cv2.putText(output, text, (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 244, 255), 2, cv2.LINE_AA)


__all__ = ["LiveEmotionProcessor"]
