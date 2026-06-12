from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_CONFIG = [
    {
        "slug": "healthy",
        "folder": "healthy",
        "display": "Healthy",
    },
    {
        "slug": "cocci",
        "folder": "cocci",
        "display": "Coccidiosis",
    },
    {
        "slug": "salmo",
        "folder": "salmo",
        "display": "Salmonella",
    },
    {
        "slug": "ncd",
        "folder": "ncd",
        "display": "Newcastle Disease",
    },
]

CLASS_NAMES = [item["display"] for item in CLASS_CONFIG]
SLUG_TO_ID = {item["slug"]: idx for idx, item in enumerate(CLASS_CONFIG)}
ID_TO_SLUG = {idx: item["slug"] for idx, item in enumerate(CLASS_CONFIG)}
SLUG_TO_DISPLAY = {item["slug"]: item["display"] for item in CLASS_CONFIG}
