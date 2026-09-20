import json
import uuid
from pathlib import Path

import boto3
import pandas as pd
import torch
import torch.nn as nn
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace
from PIL import Image, ImageStat
from torchvision import models, transforms

CLASSES = ["Positive", "Negative"]
DRIFT_SHARE_THRESHOLD = 0.5

with open("lambda_model/class_mapping.json") as f:
    CLASS_MAPPING = json.load(f)
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def compute_image_stats(image):
    grayscale = image.convert("L")
    r, g, b = image.split()
    return {
        "mean_brightness": ImageStat.Stat(grayscale).mean[0],
        "std_brightness": ImageStat.Stat(grayscale).stddev[0],
        "mean_r": ImageStat.Stat(r).mean[0],
        "mean_g": ImageStat.Stat(g).mean[0],
        "mean_b": ImageStat.Stat(b).mean[0],
    }


def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load("lambda_model/model_weights.pt", map_location="cpu"))
    model.eval()
    return model


model = load_model()


def get_reference_data(reference_dir="data/reference") -> pd.DataFrame:
    records = []

    for cls in CLASSES:
        class_folder = Path(reference_dir) / cls
        all_images = list(class_folder.glob("*.jpg"))

        assert len(all_images) > 0, f"No images in {class_folder}"

        for img_path in all_images:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            file_size = img_path.stat().st_size
            stats = compute_image_stats(image)

            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

            records.append(
                {
                    "request_id": str(uuid.uuid4()),
                    "image_width": w,
                    "image_height": h,
                    "file_size_bytes": file_size,
                    "mean_brightness": stats["mean_brightness"],
                    "std_brightness": stats["std_brightness"],
                    "mean_r": stats["mean_r"],
                    "mean_g": stats["mean_g"],
                    "mean_b": stats["mean_b"],
                    "prediction": IDX_TO_CLASS[pred_idx],
                    "confidence": round(confidence, 4),
                    "model_version": "1",
                }
            )

    return pd.DataFrame(records)


def get_current_data(bucket_name: str, prefix: str = "predictions/") -> pd.DataFrame:
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    records = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                file_content = response["Body"].read().decode("utf-8")
                records.append(json.loads(file_content))
            except Exception as e:
                print(f"Error while parsing file with key {key}: {e}")

    if not records:
        print(f"No records found in s3://{bucket_name}/{prefix}")
        return pd.DataFrame()

    return pd.DataFrame(records)


def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    numerical_columns = [
        "image_width",
        "image_height",
        "file_size_bytes",
        "mean_brightness",
        "std_brightness",
        "mean_r",
        "mean_g",
        "mean_b",
        "confidence",
    ]

    categorical_columns = ["prediction"]
    all_features = numerical_columns + categorical_columns

    ref_subset = reference_df[all_features].copy()
    cur_subset = current_df[all_features].copy()

    data_definition = DataDefinition(
        numerical_columns=numerical_columns, categorical_columns=categorical_columns
    )

    reference_dataset = Dataset.from_pandas(ref_subset, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(cur_subset, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    my_eval = report.run(current_dataset, reference_dataset)
    result = my_eval.dict()

    drift_metric = next(
        m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")
    )
    drift_share = drift_metric["value"]["share"]

    ws = Workspace.create("workspace")
    project = ws.search_project("crack-detection-monitoring")
    project = project[0] if project else ws.create_project("crack-detection-monitoring")
    ws.add_run(project.id, my_eval, include_data=False)

    return {
        "drift_share": drift_share,
        "drift_detected": drift_share >= DRIFT_SHARE_THRESHOLD,
    }


if __name__ == "__main__":
    BUCKET = "crack-detection-mlflow-s3-bucket"

    current_df = get_current_data(BUCKET)
    reference_df = get_reference_data()

    if current_df.empty:
        print("No production data.")
    else:
        result = check_drift(reference_df, current_df)
        print(f"Drift share: {result['drift_share']:.2%}")
        print(f"Drift detected: {result['drift_detected']}")
