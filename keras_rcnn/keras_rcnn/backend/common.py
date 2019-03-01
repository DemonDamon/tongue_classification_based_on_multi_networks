import keras.backend

import keras_rcnn.backend


def anchor(base_size=16, ratios=None, scales=None):
    """
    Generates a regular grid of multi-aspect and multi-scale anchor boxes.
    """
    if ratios is None:
        ratios = keras.backend.cast([0.5, 1, 2], keras.backend.floatx())

    if scales is None:
        scales = keras.backend.cast([8, 16, 32], keras.backend.floatx())
    base_anchor = keras.backend.cast([1, 1, base_size, base_size], keras.backend.floatx()) - 1
    base_anchor = keras.backend.expand_dims(base_anchor, 0)

    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = _scale_enum(ratio_anchors, scales)

    return anchors


def bbox_transform(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = keras.backend.log(gt_widths / ex_widths)
    targets_dh = keras.backend.log(gt_heights / ex_heights)

    targets = keras.backend.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh))

    targets = keras.backend.transpose(targets)

    return keras.backend.cast(targets, 'float32')


def clip(boxes, shape):
    """
    Clips box coordinates to be within the width and height as defined in shape

    """
    indices = keras.backend.tile(keras.backend.arange(0, keras.backend.shape(boxes)[0]), [4])
    indices = keras.backend.reshape(indices, (-1, 1))
    indices = keras.backend.tile(indices, [1, keras.backend.shape(boxes)[1] // 4])
    indices = keras.backend.reshape(indices, (-1, 1))

    indices_coords = keras.backend.tile(keras.backend.arange(0, keras.backend.shape(boxes)[1], step = 4), [keras.backend.shape(boxes)[0]])
    indices_coords = keras.backend.concatenate([indices_coords, indices_coords + 1, indices_coords + 2, indices_coords + 3], 0)
    indices = keras.backend.concatenate([indices, keras.backend.expand_dims(indices_coords)], axis=1)

    updates = keras.backend.concatenate([keras.backend.maximum(keras.backend.minimum(boxes[:, 0::4], shape[1] - 1), 0),
                                     keras.backend.maximum(keras.backend.minimum(boxes[:, 1::4], shape[0] - 1), 0),
                                     keras.backend.maximum(keras.backend.minimum(boxes[:, 2::4], shape[1] - 1), 0),
                                     keras.backend.maximum(keras.backend.minimum(boxes[:, 3::4], shape[0] - 1), 0)], axis=0)
    updates = keras.backend.reshape(updates, (-1,))
    pred_boxes = keras_rcnn.backend.scatter_add_tensor(keras.backend.zeros_like(boxes), indices, updates)

    return pred_boxes


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    col1 = keras.backend.reshape(x_ctr - 0.5 * (ws - 1), (-1, 1))
    col2 = keras.backend.reshape(y_ctr - 0.5 * (hs - 1), (-1, 1))
    col3 = keras.backend.reshape(x_ctr + 0.5 * (ws - 1), (-1, 1))
    col4 = keras.backend.reshape(y_ctr + 0.5 * (hs - 1), (-1, 1))
    anchors = keras.backend.concatenate((col1, col2, col3, col4), axis=1)

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = keras.backend.round(keras.backend.sqrt(size_ratios))
    hs = keras.backend.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = keras.backend.expand_dims(w, 1) * scales
    hs = keras.backend.expand_dims(h, 1) * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[:, 2] - anchor[:, 0] + 1
    h = anchor[:, 3] - anchor[:, 1] + 1
    x_ctr = anchor[:, 0] + 0.5 * (w - 1)
    y_ctr = anchor[:, 1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def shift(shape, stride):
    """
    Produce shifted anchors based on shape of the map and stride size
    """
    shift_x = keras.backend.arange(0, shape[0]) * stride
    shift_y = keras.backend.arange(0, shape[1]) * stride

    shift_x, shift_y = keras_rcnn.backend.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = keras.backend.transpose(shifts)

    anchors = keras_rcnn.backend.anchor()

    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())

    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    iw = keras.backend.minimum(keras.backend.expand_dims(a[:, 2], 1), b[:, 2]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 0], 1), b[:, 0]) + 1
    ih = keras.backend.minimum(keras.backend.expand_dims(a[:, 3], 1), b[:, 3]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 1], 1), b[:, 1]) + 1

    iw = keras.backend.maximum(iw, 0)
    ih = keras.backend.maximum(ih, 0)

    ua = keras.backend.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1) + area - iw * ih

    ua = keras.backend.maximum(ua, 0.0001)

    intersection = iw * ih

    return intersection / ua


def smooth_l1_loss(y_true, y_pred):
    delta = 1

    x = keras.backend.abs(y_true - y_pred)

    p = x < delta
    q = 0.5 * x ** 2
    r = delta * (x - 0.5 * delta)

    x = keras.backend.switch(p, q, r)

    return keras.backend.sum(x)


def bbox_transform_inv(boxes, deltas):
    def shape_zero():
        x = keras.backend.int_shape(deltas)[-1]

        return keras.backend.zeros_like(x, dtype=keras.backend.floatx())

    def shape_non_zero():
        a = boxes[:, 2] - boxes[:, 0] + 1.0
        b = boxes[:, 3] - boxes[:, 1] + 1.0

        ctr_x = boxes[:, 0] + 0.5 * a
        ctr_y = boxes[:, 1] + 0.5 * b

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * a[:, keras_rcnn.backend.newaxis] + ctr_x[:, keras_rcnn.backend.newaxis]
        pred_ctr_y = dy * b[:, keras_rcnn.backend.newaxis] + ctr_y[:, keras_rcnn.backend.newaxis]

        pred_w = keras.backend.exp(dw) * a[:, keras_rcnn.backend.newaxis]
        pred_h = keras.backend.exp(dh) * b[:, keras_rcnn.backend.newaxis]


        indices = keras.backend.tile(keras.backend.arange(0, keras.backend.shape(deltas)[0]), [4])
        indices = keras.backend.reshape(indices, (-1, 1))
        indices = keras.backend.tile(indices, [1, keras.backend.shape(deltas)[-1] // 4])
        indices = keras.backend.reshape(indices, (-1, 1))
        indices_coords = keras.backend.tile(keras.backend.arange(0, keras.backend.shape(deltas)[1], step = 4), [keras.backend.shape(deltas)[0]])
        indices_coords = keras.backend.concatenate([indices_coords, indices_coords + 1, indices_coords + 2, indices_coords + 3], 0)
        indices = keras.backend.concatenate([indices, keras.backend.expand_dims(indices_coords)], axis=1)


        updates = keras.backend.concatenate([keras.backend.reshape(pred_ctr_x - 0.5 * pred_w, (-1,)),
                                             keras.backend.reshape(pred_ctr_y - 0.5 * pred_h, (-1,)),
                                             keras.backend.reshape(pred_ctr_x + 0.5 * pred_w, (-1,)),
                                             keras.backend.reshape(pred_ctr_y + 0.5 * pred_h, (-1,))], axis=0)
        pred_boxes = keras_rcnn.backend.scatter_add_tensor(keras.backend.zeros_like(deltas), indices, updates)
        return pred_boxes


    zero_boxes = keras.backend.equal(keras.backend.shape(deltas)[0], 0)

    pred_boxes = keras.backend.switch(zero_boxes, shape_zero, shape_non_zero)

    return pred_boxes


