"""
Marine Litter & Species Detection — Render.com deployment version
--------------------------------------------------------------------
Deploy this as a Render "Web Service". Free tier, no credit card,
no expiry. Spins down after ~15 min of no traffic and takes 30-60s
to wake back up on the next visit — a real but predictable trade-off
for permanent free hosting.

Repo layout expected:
  app.py                      (this file)
  requirements.txt
  best_v5_full_dataset.pt     (your trained weights, same folder)
"""

import os
import gradio as gr
from ultralytics import YOLO
import torch
from torchvision.ops import nms
import cv2

WEIGHTS_PATH = "best_v5_full_dataset.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

model = YOLO(WEIGHTS_PATH)
CLASS_NAMES = model.names

CLASS_CONF_THRESHOLDS = {
    "beverage_can": 0.35, "bottle_glass": 0.30, "brown_algae": 0.15, "fish": 0.30,
    "fishing_net": 0.15, "fishing_trap": 0.15, "marine_reef": 0.15,
    "metallic_bar_and_contrsuction_waste": 0.30, "other_debris": 0.35,
    "other_marine_species": 0.30, "other_marine_vegetation": 0.35,
    "plastic_and_other_bottles": 0.30, "plastic_debris": 0.35, "posidonia": 0.15,
    "rocky_ground": 0.15, "rope": 0.30, "sea_cucumber": 0.25, "sea_urchin": 0.20,
    "sponge": 0.30, "starfish": 0.20, "tire": 0.40, "water_column": 0.15, "wood": 0.30,
}

TILE_SIZE = 512
OVERLAP = 0.2
IOU_MERGE = 0.5


def tiled_predict(image_rgb, run_conf, iou, use_tta, class_ids):
    H, W = image_rgb.shape[:2]
    stride = int(TILE_SIZE * (1 - OVERLAP))
    all_boxes, all_scores, all_classes = [], [], []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x2, y2 = min(x + TILE_SIZE, W), min(y + TILE_SIZE, H)
            x1, y1 = max(0, x2 - TILE_SIZE), max(0, y2 - TILE_SIZE)
            tile = image_rgb[y1:y2, x1:x2]
            results = model.predict(source=tile, conf=run_conf, iou=iou,
                                     classes=class_ids, augment=use_tta, verbose=False)
            for box in results[0].boxes:
                bx = box.xyxy[0].tolist()
                all_boxes.append([bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1])
                all_scores.append(float(box.conf[0]))
                all_classes.append(int(box.cls[0]))

    if not all_boxes:
        return [], [], []

    boxes_t = torch.tensor(all_boxes)
    scores_t = torch.tensor(all_scores)
    classes_t = torch.tensor(all_classes)

    keep_indices = []
    for cls_id in classes_t.unique():
        idxs = torch.nonzero(classes_t == cls_id).squeeze(1)
        kept = nms(boxes_t[idxs], scores_t[idxs], IOU_MERGE)
        keep_indices.extend(idxs[kept].tolist())

    return boxes_t[keep_indices].tolist(), scores_t[keep_indices].tolist(), classes_t[keep_indices].tolist()


def draw_boxes(image_rgb, boxes, scores, classes):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img_bgr, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def detect(image, conf, iou, selected_classes, use_per_class_thresholds, use_tta, use_tiling):
    if image is None:
        return None, []

    class_ids = None
    if selected_classes:
        name_to_id = {v: k for k, v in CLASS_NAMES.items()}
        class_ids = [name_to_id[c] for c in selected_classes if c in name_to_id]

    run_conf = min(CLASS_CONF_THRESHOLDS.values()) if use_per_class_thresholds else conf

    if use_tiling:
        boxes, scores, classes = tiled_predict(image, run_conf, iou, use_tta, class_ids)
        rows = []
        keep_boxes, keep_scores, keep_classes = [], [], []
        for box, score, cls_id in zip(boxes, scores, classes):
            cls_name = CLASS_NAMES[cls_id]
            threshold = CLASS_CONF_THRESHOLDS.get(cls_name, 0.25) if use_per_class_thresholds else conf
            if score >= threshold:
                keep_boxes.append(box)
                keep_scores.append(score)
                keep_classes.append(cls_id)
                rows.append([cls_name, round(score, 3), [round(v, 1) for v in box]])
        annotated = draw_boxes(image, keep_boxes, keep_scores, keep_classes)
        return annotated, rows

    results = model.predict(
        source=image, conf=run_conf, iou=iou, classes=class_ids,
        augment=use_tta, verbose=False,
    )
    r = results[0]

    rows = []
    keep_indices = []
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id]
        box_conf = float(box.conf[0])
        threshold = CLASS_CONF_THRESHOLDS.get(cls_name, 0.25) if use_per_class_thresholds else conf
        if box_conf >= threshold:
            keep_indices.append(i)
            rows.append([cls_name, round(box_conf, 3), [round(x, 1) for x in box.xyxy[0].tolist()]])

    r_filtered = r[keep_indices] if keep_indices else r[[]]
    annotated = r_filtered.plot()[:, :, ::-1]

    return annotated, rows


with gr.Blocks(title="Marine Litter & Species Detector") as demo:
    gr.Markdown(
        "# 🌊 Marine Litter & Species Detection\n"
        "Upload an underwater image to detect litter and marine species. "
        "Adjust confidence/IoU thresholds and optionally filter to specific classes."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Input Image")
            conf_slider = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold (used when per-class thresholds are off)")
            iou_slider = gr.Slider(0.05, 0.95, value=DEFAULT_IOU, step=0.05, label="IoU threshold (NMS)")
            per_class_checkbox = gr.Checkbox(value=True, label="Use tuned per-class confidence thresholds")
            tta_checkbox = gr.Checkbox(value=True, label="Use test-time augmentation (TTA) — slower, usually more accurate")
            tiling_checkbox = gr.Checkbox(value=False, label="Use tiled inference (recommended for crowded/cluttered photos — slower)")
            class_filter = gr.Dropdown(
                choices=list(CLASS_NAMES.values()),
                multiselect=True,
                label="Filter to specific classes (optional, leave empty for all)",
            )
            run_btn = gr.Button("Detect", variant="primary")

        with gr.Column():
            image_output = gr.Image(type="numpy", label="Detections")
            table_output = gr.Dataframe(
                headers=["Class", "Confidence", "Box (x1,y1,x2,y2)"],
                label="Detected objects",
            )

    inputs = [image_input, conf_slider, iou_slider, class_filter, per_class_checkbox, tta_checkbox, tiling_checkbox]
    run_btn.click(fn=detect, inputs=inputs, outputs=[image_output, table_output])
    image_input.change(fn=detect, inputs=inputs, outputs=[image_output, table_output])

if __name__ == "__main__":
    # Render provides the PORT to bind to via an environment variable;
    # 0.0.0.0 makes the app reachable from outside the container.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
