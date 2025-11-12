from __future__ import annotations
import io, zipfile, requests
from pathlib import Path
import pandas as pd

MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def ensure_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    movies_fp, ratings_fp = data_dir / "movies.csv", data_dir / "ratings.csv"
    if not movies_fp.exists() or not ratings_fp.exists():
        print("Downloading MovieLens small dataset...")
        r = requests.get(MOVIELENS_SMALL_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for member in ("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv"):
                with zf.open(member) as src, open(data_dir / Path(member).name, "wb") as dst:
                    dst.write(src.read())
        print("Dataset downloaded.")
    movies, ratings = pd.read_csv(movies_fp), pd.read_csv(ratings_fp)
    return movies, ratings
