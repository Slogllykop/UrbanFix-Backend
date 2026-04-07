"""
YOLO ONNX detector – loads a YOLOv8-style ONNX model and runs inference.

Supports models exported from Ultralytics with a single detection head.
Output tensor shape expected: [1, 4 + num_classes, num_boxes].
"""

import numpy as np
import onnxruntime as ort
import cv2

import config


class YOLODetector:
    """Wraps an ONNX Runtime session for YOLOv8-style object detection."""

    def __init__(self, model_path: str):
        """
        Load the ONNX model and read its input metadata.

        Args:
            model_path: Filesystem path to the .onnx file.
        """
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name: str = input_meta.name
        # Expected shape: [batch, channels, height, width]
        self.img_height: int = input_meta.shape[2]
        self.img_width: int = input_meta.shape[3]

        self.conf_threshold: float = config.CONFIDENCE_THRESHOLD
        self.iou_threshold: float = config.IOU_THRESHOLD

    # ── Pre-processing ────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Resize, normalise, and reshape *image* for the model.

        Args:
            image: BGR uint8 image read by OpenCV.

        Returns:
            Float32 tensor of shape [1, 3, H, W] in range [0, 1].
        """
        img = cv2.resize(image, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)     # add batch dim
        return img

    # ── Post-processing ───────────────────────────────────────────────────

    def _postprocess(self, outputs: list[np.ndarray]) -> int:
        """
        Parse model output, apply confidence filter + NMS (if needed), return detection count.

        Args:
            outputs: Raw ONNX session outputs.

        Returns:
            Number of detections that survive thresholding.
        """
        output = outputs[0][0]

        # Check if the model already includes NMS -> shape (N, 6)
        # Format: [x1, y1, x2, y2, confidence, class_id]
        if output.ndim == 2 and output.shape[1] == 6:
            scores = output[:, 4]
            # Confidence filter
            mask = scores >= self.conf_threshold
            return int(np.sum(mask))

        # Standard raw YOLOv8 ONNX output shape without NMS: [4+C, N]
        output = output.T                 # [N, 4+C]

        boxes = output[:, :4]             # x_c, y_c, w, h
        scores = np.max(output[:, 4:], axis=1)

        # Confidence filter
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return 0

        # Convert centre-format to corner-format for NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        keep = self._nms(x1, y1, x2, y2, scores)
        return len(keep)

    def _nms(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """Greedy Non-Maximum Suppression."""
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            remaining = np.where(iou <= self.iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> int:
        """
        Run full detection pipeline on a single image.

        Args:
            image: BGR uint8 image (as returned by ``cv2.imdecode``).

        Returns:
            Number of detected objects.
        """
        tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        return self._postprocess(outputs)


# ── Module-level model cache ──────────────────────────────────────────────────
# Models are loaded once at import time and reused across requests.

_detectors: dict[str, YOLODetector] = {}


def get_detector(category: str) -> YOLODetector:
    """
    Return a cached YOLODetector for the given *category*.

    Args:
        category: ``"pothole"`` or ``"garbage"``.

    Returns:
        A ready-to-use YOLODetector instance.

    Raises:
        ValueError: If the category is unknown.
        FileNotFoundError: If the ONNX file is missing.
    """
    if category not in config.MODEL_PATHS:
        raise ValueError(f"Unknown category: {category!r}")

    if category not in _detectors:
        model_path = config.MODEL_PATHS[category]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _detectors[category] = YOLODetector(str(model_path))

    return _detectors[category]
