import torch
import cv2
import numpy as np
import albumentations as A

def build_detection_transform(
    crop_size=512,
    use_hflip=False,
    use_rot=False,
    min_visibility=0.1
):
    transforms = [
        A.PadIfNeeded(
            min_height=crop_size,
            min_width=crop_size,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0
        ),
        A.RandomCrop(
            height=crop_size,
            width=crop_size,
            p=1.0
        )
    ]

    if use_hflip:
        transforms.append(A.HorizontalFlip(p=0.5))

    if use_rot:
        transforms.extend([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

    return A.Compose(
        transforms,
        additional_targets={'image_lq': 'image'},
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=min_visibility
        )
    )

def yolo_to_xyxy(label_raw, img_h, img_w):
    """
    Convert YOLO normalized xywh boxes to Pascal VOC xyxy boxes in pixels.

    Args:
        label_raw: np.ndarray [N, 5], each row is [cls, cx, cy, w, h].
    Returns:
        boxes: list of [x1, y1, x2, y2]
        class_labels: list of class ids
    """
    if label_raw is None or len(label_raw) == 0:
        return [], []

    label_raw = np.asarray(label_raw, dtype=np.float32).reshape(-1, 5)

    cls_ids = label_raw[:, 0].astype(np.int64).tolist()

    cx = label_raw[:, 1] * img_w
    cy = label_raw[:, 2] * img_h
    bw = label_raw[:, 3] * img_w
    bh = label_raw[:, 4] * img_h

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h - 1)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[keep]
    cls_ids = [cls_ids[i] for i in range(len(keep)) if keep[i]]

    return boxes.tolist(), cls_ids


def xyxy_to_yolo(boxes, class_labels, img_h, img_w, min_box_size=2):
    """
    Convert Pascal VOC xyxy boxes in pixels back to YOLO normalized xywh format.

    Args:
        boxes: list of [x1, y1, x2, y2]
        class_labels: list of class ids
    Returns:
        labels: np.ndarray [N, 5], each row is [cls, cx, cy, w, h].
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    class_labels = np.asarray(class_labels, dtype=np.float32).reshape(-1)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]

    keep = (bw >= min_box_size) & (bh >= min_box_size)
    if not np.any(keep):
        return np.zeros((0, 5), dtype=np.float32)

    boxes = boxes[keep]
    class_labels = class_labels[keep]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    labels = np.stack([class_labels, cx, cy, bw, bh], axis=1).astype(np.float32)
    labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)
    return labels


def sync_detection_aug_and_crop(
    img_gt,
    img_lq,
    label_raw,
    transform,
    min_box_size=2
):
    """
    Apply synchronized augmentation to GT, LQ, and bounding boxes.

    Workflow:
    - Convert YOLO xywh boxes to xyxy pixel boxes.
    - Apply the same pad/crop/flip/rotate transform to both images and boxes.
    - Convert transformed boxes back to YOLO xywh format.
    """

    # Convert grayscale images to HxWx1 before Albumentations.
    img_gt_hwc, gt_was_gray = _ensure_hwc_image(img_gt)
    img_lq_hwc, lq_was_gray = _ensure_hwc_image(img_lq)

    img_h, img_w = img_gt_hwc.shape[:2]

    bboxes, class_labels = yolo_to_xyxy(label_raw, img_h, img_w)

    transformed = transform(
        image=img_gt_hwc,
        image_lq=img_lq_hwc,
        bboxes=bboxes,
        class_labels=class_labels
    )

    img_gt_aug = transformed['image']
    img_lq_aug = transformed['image_lq']
    bboxes_aug = transformed['bboxes']
    class_labels_aug = transformed['class_labels']

    # Restore HxW if the original image was grayscale.
    img_gt_aug = _restore_gray_image(img_gt_aug, was_gray=gt_was_gray)
    img_lq_aug = _restore_gray_image(img_lq_aug, was_gray=lq_was_gray)

    new_h, new_w = img_gt_aug.shape[:2]
    label_aug = xyxy_to_yolo(
        bboxes_aug,
        class_labels_aug,
        new_h,
        new_w,
        min_box_size=min_box_size
    )

    return img_gt_aug, img_lq_aug, label_aug

def _to_uint8_rgb(img_tensor):
    """
    Convert an image tensor in [0, 1] to a writable uint8 RGB image.

    Args:
        img_tensor: [1,3,H,W] or [3,H,W]
    Returns:
        uint8 RGB image with shape [H,W,3]
    """
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.detach().cpu().float().clamp(0, 1)

        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]

        img = img_tensor.permute(1, 2, 0).contiguous().numpy()
    else:
        img = np.asarray(img_tensor, dtype=np.float32)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)

    # Ensure contiguous and writable memory for OpenCV drawing.
    img = np.ascontiguousarray(img).copy()
    return img


def _to_tensor(img_uint8):
    """
    Convert a uint8 RGB image [H,W,3] to a float tensor [1,3,H,W] in [0,1].
    """
    img_uint8 = np.ascontiguousarray(img_uint8)
    x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return x.unsqueeze(0)


def _class_name(cid, class_names=None):
    """Map a class id to a display name."""
    cid = int(cid)
    if class_names is None:
        return str(cid)
    if isinstance(class_names, dict):
        return class_names.get(cid, str(cid))
    if isinstance(class_names, (list, tuple)) and 0 <= cid < len(class_names):
        return class_names[cid]
    return str(cid)


def _draw_label(img, text, x1, y1, color, font_scale=0.5, thickness=1):
    """Draw a filled label box with text."""
    if text is None or len(text) == 0:
        return img

    img = np.ascontiguousarray(img)

    color = tuple(int(c) for c in color)

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    bx1 = max(0, int(x1))
    by1 = max(0, int(y1 - th - baseline - 4))
    bx2 = min(img.shape[1] - 1, bx1 + tw + 6)
    by2 = min(img.shape[0] - 1, by1 + th + baseline + 4)

    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(
        img,
        text,
        (bx1 + 3, by2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    return img


def draw_gt_boxes(image, cls, bboxes, class_names=None,
                  color=(0, 255, 0), thickness=2):
    """
    Draw ground-truth boxes on an image.

    Args:
        image : [1,3,H,W] or [3,H,W], value range [0,1]
        cls   : [N,1] or [N]
        bboxes: [N,4], YOLO normalized xywh
    """
    img = _to_uint8_rgb(image)
    img = np.ascontiguousarray(img).copy()

    H, W = img.shape[:2]
    color = tuple(int(c) for c in color)
    thickness = int(thickness)

    cls = torch.as_tensor(cls).view(-1).cpu().numpy() if cls is not None else np.zeros((0,))
    bboxes = torch.as_tensor(bboxes).reshape(-1, 4).cpu().numpy() if bboxes is not None else np.zeros((0, 4))

    font_scale = max(0.45, min(H, W) / 1024.0)

    for box, cid in zip(bboxes, cls):
        cx, cy, bw, bh = box.tolist()

        x1 = int(round((cx - bw / 2.0) * W))
        y1 = int(round((cy - bh / 2.0) * H))
        x2 = int(round((cx + bw / 2.0) * W))
        y2 = int(round((cy + bh / 2.0) * H))

        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        img = _draw_label(
            img,
            _class_name(cid, class_names),
            x1, y1,
            color,
            font_scale=font_scale,
            thickness=1
        )

    return _to_tensor(img)


def draw_pred_boxes(image, boxes, scores=None, classes=None, class_names=None,
                    score_thr=0.25, color=(255, 0, 0), thickness=2):
    """
    Draw predicted boxes on an image.

    Args:
        image : [1,3,H,W] or [3,H,W], value range [0,1]
        boxes : [N,4], xyxy in pixel coordinates
    """
    img = _to_uint8_rgb(image)
    img = np.ascontiguousarray(img).copy()

    H, W = img.shape[:2]
    color = tuple(int(c) for c in color)
    thickness = int(thickness)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1) if scores is not None else np.ones((len(boxes),), dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1) if classes is not None else np.zeros((len(boxes),), dtype=np.int64)

    font_scale = max(0.45, min(H, W) / 1024.0)

    for box, score, cid in zip(boxes, scores, classes):
        if float(score) < float(score_thr):
            continue

        x1, y1, x2, y2 = box.tolist()
        x1 = int(round(max(0, min(x1, W - 1))))
        y1 = int(round(max(0, min(y1, H - 1))))
        x2 = int(round(max(0, min(x2, W - 1))))
        y2 = int(round(max(0, min(y2, H - 1))))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        img = _draw_label(
            img,
            f"{_class_name(cid, class_names)} {float(score):.2f}",
            x1, y1,
            color,
            font_scale=font_scale,
            thickness=1
        )

    return _to_tensor(img)

def _ensure_2d_tensor(x, n_cols, dtype=torch.float32):
    """
    Convert the input to a 2D tensor.

    Supported cases:
    - None          -> [0, n_cols]
    - numpy array   -> tensor
    - list          -> tensor
    - 1D tensor     -> reshape to (-1, n_cols)
    - empty input   -> [0, n_cols]
    """
    if x is None:
        return torch.zeros((0, n_cols), dtype=dtype)

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)

    x = x.to(dtype=dtype)

    if x.numel() == 0:
        return torch.zeros((0, n_cols), dtype=dtype)

    if x.ndim == 1:
        if x.numel() % n_cols != 0:
            raise ValueError(f"Cannot reshape tensor of shape {tuple(x.shape)} to (-1, {n_cols})")
        x = x.view(-1, n_cols)

    return x

def make_yolo_targets_from_batch(batch):
    """
    Build YOLO loss targets directly from a dataloader batch.
    """
    return {
        'batch_idx': batch['batch_idx'],
        'cls': batch['cls'],
        'bboxes': batch['bboxes'],   # YOLO normalized xywh
    }


def select_targets_for_image(batch_idx, cls, bboxes, image_idx=0):
    """
    Select GT targets belonging to a specific image from a batched target set.
    Useful for visualization.
    """
    batch_idx = torch.as_tensor(batch_idx).view(-1)
    cls = torch.as_tensor(cls).view(-1, 1)
    bboxes = torch.as_tensor(bboxes).view(-1, 4)

    mask = (batch_idx == image_idx)

    cls_i = cls[mask]
    bboxes_i = bboxes[mask]
    return cls_i, bboxes_i

def _ensure_hwc_image(img):
    """
    Convert input image to HWC format.
    Grayscale HxW is converted to HxWx1.

    Returns:
        img_hwc, was_gray
    """
    if img is None:
        return None, False

    img = np.asarray(img)

    if img.ndim == 2:
        # HxW -> HxWx1
        img = img[..., None]
        was_gray = True
    elif img.ndim == 3 and img.shape[2] == 1:
        was_gray = True
    elif img.ndim == 3:
        was_gray = False
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.ascontiguousarray(img)
    return img, was_gray


def _restore_gray_image(img, was_gray=True):
    """
    Restore HxWx1 to HxW if the original image was grayscale.
    """
    if img is None:
        return None

    img = np.asarray(img)
    img = np.ascontiguousarray(img)

    if was_gray and img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    return np.ascontiguousarray(img)