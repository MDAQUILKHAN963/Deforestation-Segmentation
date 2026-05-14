"""
metrics.py — Custom Keras metrics for binary segmentation evaluation.

Implements:
    - IoUScore: Intersection over Union (Jaccard Index)
    - FScore: F1 Score (Dice Coefficient)
    - PixelAccuracy: Overall pixel classification accuracy

All metrics are implemented as tf.keras.metrics.Metric subclasses
so they integrate seamlessly with model.compile() and model.fit().
"""

import tensorflow as tf


class IoUScore(tf.keras.metrics.Metric):
    """
    Intersection over Union (IoU / Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|
        = TP / (TP + FP + FN)

    The primary metric for segmentation quality.
    Target: >= 0.75 for this project.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6,
                 name: str = "iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config


class FScore(tf.keras.metrics.Metric):
    """
    F1 Score (Dice Coefficient) for binary segmentation.

    F1 = 2 * Precision * Recall / (Precision + Recall)
       = 2 * TP / (2 * TP + FP + FN)

    Equivalent to the Dice coefficient.
    Target: >= 0.85 for this project.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6,
                 name: str = "f_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1.0 - y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1.0 - y_pred_flat))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        return (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold, "smooth": self.smooth})
        return config


class PixelAccuracy(tf.keras.metrics.Metric):
    """
    Pixel Accuracy for binary segmentation.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Simple but can be misleading with class imbalance.
    Target: >= 88% for this project.
    """

    def __init__(self, threshold: float = 0.5,
                 name: str = "pixel_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        correct = tf.reduce_sum(tf.cast(tf.equal(y_true_flat, y_pred_flat), tf.float32))
        total = tf.cast(tf.size(y_true_flat), tf.float32)

        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return self.correct / (self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


# ─── Functional versions (for manual evaluation) ────────────────────────────────

def compute_iou(y_true: tf.Tensor, y_pred: tf.Tensor,
                threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute IoU as a single float value."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]) > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return float(iou.numpy())


def compute_f1(y_true: tf.Tensor, y_pred: tf.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute F1 score as a single float value."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]) > threshold, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))

    f1 = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    return float(f1.numpy())


def compute_pixel_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor,
                           threshold: float = 0.5) -> float:
    """Compute pixel accuracy as a single float value."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]) > threshold, tf.float32)

    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total = tf.cast(tf.size(y_true), tf.float32)

    return float((correct / total).numpy())
