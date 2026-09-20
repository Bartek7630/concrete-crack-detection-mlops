import base64
import io
import json

import boto3
from lambda_function import handler
from moto import mock_aws
from PIL import Image

BUCKET_NAME = "test-crack-detection-bucket"


def generate_base64_test_image() -> str:
    img = Image.new("RGB", (224, 224), color=(150, 150, 150))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@mock_aws
def test_lambda_handler_s3_integration(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("BUCKET_NAME", BUCKET_NAME)

    s3_client = boto3.client("s3", region_name="eu-central-1")
    s3_client.create_bucket(
        Bucket=BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
    )

    image_b64 = generate_base64_test_image()
    event = {
        "body": json.dumps({"image": image_b64}),
        "isBase64Encoded": False,
    }

    response = handler(event, context=None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert "prediction" in body
    assert "confidence" in body

    objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    assert "Contents" in objects
    assert len(objects["Contents"]) > 0

    log_key = objects["Contents"][0]["Key"]
    assert log_key.endswith(".json")

    s3_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=log_key)
    saved_data = json.loads(s3_obj["Body"].read().decode("utf-8"))

    assert "prediction" in saved_data
    assert "mean_brightness" in saved_data
