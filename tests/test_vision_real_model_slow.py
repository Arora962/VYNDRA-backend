"""Optional slow integration test against the real YOLO weights and a local sample photo."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.slow
def test_real_vision_model_predicts_on_local_sample_photo():
    """Run actual best.pt inference when a local sample photo is available."""
    candidates = []
    own_root = ROOT / "data" / "test_images_own"
    if own_root.exists():
        candidates.extend(
            path for path in own_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            and not path.name.startswith("._")
        )
    uploads = ROOT / "uploads"
    if uploads.exists():
        candidates.extend(
            path for path in uploads.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            and not path.name.startswith("._")
        )
    if not candidates:
        pytest.skip("Add a real sample image under data/test_images_own/<approved class>/ to run this slow integration test.")
    from vyndra.vision import predict_image
    result = predict_image(sorted(candidates)[0], top_k=3)
    assert result["top_predictions"]
    assert len(result["top_predictions"]) <= 3
    assert all(0.0 <= item["confidence"] <= 1.0 for item in result["top_predictions"])
