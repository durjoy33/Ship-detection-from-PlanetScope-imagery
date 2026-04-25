# Ship-detection-from-PlanetScope-imagery
Satellite boat detection using fine-tuned YOLOv8m — mAP@50: 0.87 | 4-class | Google Colab pipeline
# 🚢 Satellite Boat Detection — YOLOv8m

Fine-tuned YOLOv8m model for detecting boats and ships in satellite imagery.
Trained on GeoTIFF tiles with YOLO-format annotations exported from a shapefile.

## Model Performance

| Metric | Score |
|---|---|
| mAP@50 | 0.8696 |
| mAP@50-95 | 0.5657 |
| Precision | 0.8793 |
| Recall | 0.8265 |

## Classes

| ID | Class | Description |
|---|---|---|
| 0 | `small_boat` | Small recreational or fishing vessels |
| 1 | `large_boat` | Large commercial vessels |
| 2 | `ship` | Ocean-going ships |
| 3 | `boat_cluster` | Dense groups of boats |

## Pipeline Overview
GeoTIFF tiles → Shapefile annotations → YOLO labels
→ Train/val split → YOLOv8m fine-tuning
→ Evaluation → GeoJSON export
## Notebook

`boat_detection_inference.ipynb` — full inference pipeline:
- Loads fine-tuned checkpoint from Google Drive
- Evaluates on validation set (mAP, per-class AP, confusion matrix)
- Visualises ground truth vs predictions side by side
- Runs inference on new images or tile folders
- Exports detections as GeoJSON (geo back-projected coordinates)
- Exports model to ONNX / TensorRT

## Quick Start

1. Open `boat_detection_inference.ipynb` in Google Colab
2. Set runtime to **GPU** (T4 or better)
3. Upload your checkpoint to Google Drive as `boat_best_v3_MMDD_HHMM.pt`
4. Edit the paths in **Cell 1 — Configuration**
5. Run all cells

## Requirements
ultralytics
rasterio
geopandas
shapely
torch
All installed automatically in Cell 0.

## Training Config (best run)

```python
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=256,
    batch=16,
    optimizer="AdamW",
    lr0=0.0003,
    freeze=3,
    mosaic=0.8,
    degrees=15.0,
    box=10.0,
    cls=0.3,
    patience=20,
)
```

## Output

- Annotated prediction images saved to `runs/detect/boat_inference/`
- GeoJSON file (`boat_detections.geojson`) openable in QGIS or geojson.io
- ONNX model for deployment

## Notes

- Tile size: 256×256 px
- Input format: GeoTIFF (`.tif`) with valid CRS for geo export
- Model: YOLOv8m fine-tuned from COCO pretrained weights
- Training hardware: Tesla T4 (Google Colab)
