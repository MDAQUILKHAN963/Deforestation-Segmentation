"""
losses.py — Custom loss functions for binary segmentation.

Implements:
    - DiceLoss: Region-based loss for class imbalance
    - BinaryFocalLoss: Pixel-level loss focusing on hard examples
    - CombinedLoss: Weighted sum of Dice + Focal
"""

import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    """
    Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice

    Handles class imbalance well by measuring overlap rather than
    per-pixel accuracy. Essential for satellite imagery where
    deforested regions may be a small fraction of the image.
    """

    def __init__(self, smooth: float = 1e-6, name: str = "dice_loss", **kwargs):
        """
        Args:
            smooth: Smoothing factor to prevent division by zero
                    and stabilize gradients when predictions are near-perfect.
        """
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Compute Dice Loss.

        Args:
            y_true: Ground truth masks, shape (B, H, W, 1), values in {0, 1}
            y_pred: Predicted masks, shape (B, H, W, 1), values in [0, 1] (after sigmoid)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten spatial dimensions per sample
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=-1)
        union = tf.reduce_sum(y_true_flat, axis=-1) + tf.reduce_sum(y_pred_flat, axis=-1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return tf.reduce_mean(1.0 - dice)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config


class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for pixel-level classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights well-classified pixels and focuses training on
    hard-to-segment boundaries and small deforestation patches.

    Args:
        alpha: Balancing factor for positive class (0-1). Higher values
               give more weight to the positive (deforested) class.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 name: str = "binary_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Compute Binary Focal Loss.

        Args:
            y_true: Ground truth masks, shape (B, H, W, 1)
            y_pred: Predicted probabilities, shape (B, H, W, 1)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute focal weights
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)

        # Binary cross-entropy
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # Focal loss
        focal_loss = focal_weight * bce

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


class CombinedLoss(tf.keras.losses.Loss):
    """
    Combined loss: DiceLoss + BinaryFocalLoss.

    The Dice loss handles region-level overlap while
    Focal loss handles pixel-level hard examples.
    Together they provide robust training signal for
    imbalanced satellite segmentation tasks.

    Args:
        dice_weight: Weight for Dice loss component (default 1.0)
        focal_weight: Weight for Focal loss component (default 1.0)
        dice_smooth: Smoothing factor for Dice loss
        focal_alpha: Alpha for Focal loss
        focal_gamma: Gamma for Focal loss
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        name: str = "combined_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def call(self, y_true, y_pred):
        """Compute weighted sum of Dice and Focal losses."""
        d_loss = self.dice_loss(y_true, y_pred)
        f_loss = self.focal_loss(y_true, y_pred)
        return self.dice_weight * d_loss + self.focal_weight * f_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
        })
        return config
