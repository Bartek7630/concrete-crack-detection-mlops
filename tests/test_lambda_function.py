from lambda_function import compute_image_stats
from PIL import Image


def test_compute_image_stats():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    stats = compute_image_stats(img)

    assert abs(stats["mean_brightness"] - 128.0) < 1.0
    assert abs(stats["std_brightness"] - 0.0) < 1.0
