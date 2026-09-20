import base64
import datetime
import io
import json
import os
import uuid
from pathlib import Path

import boto3
import torch
import torch.nn as nn
from PIL import Image, ImageStat
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent


def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(BASE_DIR / "model_weights.pt", map_location="cpu"))
    model.eval()

    return model


with open(BASE_DIR / "class_mapping.json") as f:
    CLASS_MAPPING = json.load(f)
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

model = load_model()

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


def handler(event, context):
    s3_bucket = os.environ.get("BUCKET_NAME", "crack-detection-mlflow-s3-bucket")
    s3_client = boto3.client("s3")

    body = json.loads(event["body"])
    image_bytes = base64.b64decode(body["image"])
    file_size = len(image_bytes)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    stats = compute_image_stats(image)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    request_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc)

    log_record = {
        "request_id": request_id,
        "timestamp": now.isoformat(),
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

    key = f"predictions/{now.date().isoformat()}/{request_id}.json"
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=json.dumps(log_record),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "prediction": IDX_TO_CLASS[pred_idx],
                "confidence": round(confidence, 4),
            }
        ),
    }
