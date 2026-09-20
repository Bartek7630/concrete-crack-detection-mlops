from prepare_data import flip_label, get_max_balanced_sample_size


def test_flip_label():
    assert flip_label(0) == 1
    assert flip_label(1) == 0


def test_get_max_balanced_sample_size(tmp_path):
    pos_dir = tmp_path / "Positive"
    neg_dir = tmp_path / "Negative"
    pos_dir.mkdir()
    neg_dir.mkdir()

    for i in range(7):
        (pos_dir / f"img_{i}.jpg").touch()

    for i in range(3):
        (neg_dir / f"img_{i}.jpg").touch()

    result = get_max_balanced_sample_size(str(tmp_path))

    assert result == 3
