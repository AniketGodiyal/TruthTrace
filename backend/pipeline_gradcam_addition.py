"""
Robust Grad-CAM helper for VeriTrace/Xception.

Important:
- Never selects InputLayer as the Grad-CAM target.
- Prefers the last Conv2D/SeparableConv2D layer.
- Handles a nested Keras model if Xception is wrapped inside another model.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def _is_conv_layer(layer):
    name = layer.__class__.__name__.lower()
    return (
        "conv2d" in name
        or "separableconv" in name
        or name in {"conv1d", "conv3d"}
    )


def _find_direct_conv_layer(model):
    """
    Find the last real convolutional layer directly connected to model.inputs.
    InputLayer is deliberately excluded.
    """
    for layer in reversed(model.layers):
        if not _is_conv_layer(layer):
            continue

        try:
            if len(layer.output.shape) != 4:
                continue

            # Verify that this exact layer can be connected to the outer model.
            test_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[layer.output, model.output],
            )
            return test_model, layer
        except Exception:
            continue

    return None, None


def _find_nested_conv_model(model):
    """
    If the outer model contains Xception as a nested Keras Model,
    build Grad-CAM against the inner Xception model.
    """
    for outer in reversed(model.layers):
        if not isinstance(outer, tf.keras.Model):
            continue

        inner_model = outer

        # Prefer the last real convolution in the nested model.
        for layer in reversed(inner_model.layers):
            if not _is_conv_layer(layer):
                continue

            try:
                if len(layer.output.shape) != 4:
                    continue

                inner_grad_model = tf.keras.Model(
                    inputs=inner_model.inputs,
                    outputs=[layer.output, inner_model.output],
                )

                return inner_grad_model, layer, inner_model
            except Exception:
                continue

    return None, None, None


def _build_grad_model(model):
    direct, layer = _find_direct_conv_layer(model)
    if direct is not None:
        return direct, layer, model

    nested, layer, inner = _find_nested_conv_model(model)
    if nested is not None:
        return nested, layer, inner

    raise RuntimeError(
        "Could not find a connected Conv2D/SeparableConv2D layer "
        "for Grad-CAM."
    )


def _preprocess(face_crop_bgr):
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (299, 299), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(x, axis=0)


def make_gradcam(model, face_crop_bgr, output_path, alpha=0.45):
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        raise ValueError("Empty face crop supplied to Grad-CAM.")

    x = _preprocess(face_crop_bgr)

    grad_model, target_layer, prediction_model = _build_grad_model(model)

    # Keras sometimes expects the named input structure rather than a raw tensor.
    # Try the normal tensor first, then a single-item list.
    with tf.GradientTape() as tape:
        try:
            conv_outputs, predictions = grad_model(x, training=False)
        except (TypeError, ValueError):
            conv_outputs, predictions = grad_model([x], training=False)

        predictions = tf.convert_to_tensor(predictions)

        if predictions.shape.rank == 2:
            target = predictions[:, 0]
        else:
            target = tf.reshape(predictions, [-1])

    grads = tape.gradient(target, conv_outputs)

    if grads is None:
        # Retry with an explicit persistent tape. This also gives us a
        # deterministic error instead of silently returning no heatmap.
        with tf.GradientTape() as retry_tape:
            try:
                conv_outputs2, predictions2 = grad_model(x, training=False)
            except (TypeError, ValueError):
                conv_outputs2, predictions2 = grad_model([x], training=False)

            predictions2 = tf.convert_to_tensor(predictions2)
            target2 = (
                predictions2[:, 0]
                if predictions2.shape.rank == 2
                else tf.reshape(predictions2, [-1])
            )

        grads = retry_tape.gradient(target2, conv_outputs2)

        if grads is None:
            raise RuntimeError(
                f"Gradients are None for convolutional layer "
                f"'{target_layer.name}'. "
                f"Model output={predictions.numpy().tolist()}"
            )

        conv_outputs = conv_outputs2

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    # Global average pooling over spatial dimensions.
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = tf.reduce_sum(conv_outputs * weights, axis=-1)
    cam = tf.maximum(cam, 0)

    max_value = float(tf.reduce_max(cam).numpy())

    if max_value > 1e-12:
        heatmap = (cam / max_value).numpy().astype(np.float32)
    else:
        heatmap = np.zeros(
            (int(conv_outputs.shape[0]), int(conv_outputs.shape[1])),
            dtype=np.float32,
        )

    heatmap = cv2.resize(
        heatmap,
        (face_crop_bgr.shape[1], face_crop_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_u8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        face_crop_bgr,
        1.0 - alpha,
        colored,
        alpha,
        0,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Could not write Grad-CAM image: {output_path}")

    print(
        f"[INFO] Grad-CAM generated using convolutional layer "
        f"'{target_layer.name}': {output_path}"
    )

    return str(output_path)
