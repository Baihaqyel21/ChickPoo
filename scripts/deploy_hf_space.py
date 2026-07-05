import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
SPACE_TEMPLATE_DIR = ROOT / "deployment" / "huggingface-space"
REQUIRED_MODELS = [
    "best_disease_classifier_model.keras",
    "best_object_validation_model.keras",
    "disease_classifier_metadata.json",
    "object_validation_metadata.json",
    "final_training_summary.json",
]


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def build_space_folder(target: Path) -> None:
    shutil.copy2(SPACE_TEMPLATE_DIR / "Dockerfile", target / "Dockerfile")
    shutil.copy2(SPACE_TEMPLATE_DIR / "README.md", target / "README.md")
    shutil.copy2(ROOT / "backend" / "requirements.txt", target / "requirements.txt")
    copy_tree(ROOT / "backend", target / "backend")

    models_target = target / "models"
    models_target.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_MODELS:
        source = ROOT / "models" / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing required model artifact: {source}")
        shutil.copy2(source, models_target / filename)

    (target / ".gitattributes").write_text("*.keras filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")


def main() -> None:
    token = os.environ["HF_TOKEN"]
    repo_id = os.environ.get("HF_SPACE_ID", "baihaqyel/chickpoo-api")
    cors_origins = os.environ.get("CORS_ORIGINS", "https://chickpoo.vercel.app")

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="chickpoo-hf-space-") as tmp_dir:
        staging_dir = Path(tmp_dir)
        build_space_folder(staging_dir)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(staging_dir),
            commit_message="Deploy ChickPoo API",
        )

    api.add_space_variable(repo_id=repo_id, key="CORS_ORIGINS", value=cors_origins)
    api.add_space_variable(repo_id=repo_id, key="TF_CPP_MIN_LOG_LEVEL", value="2")
    api.restart_space(repo_id=repo_id)
    print(f"Deployed Hugging Face Space: {repo_id}")


if __name__ == "__main__":
    main()
