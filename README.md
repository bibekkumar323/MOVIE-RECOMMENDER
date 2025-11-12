# Movie Recommendation System (Content-Based)
A lightweight, API-free movie recommender using MovieLens ml-latest-small dataset.

## Quickstart
```
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
python src/main.py --download
python src/main.py --title "Toy Story (1995)" --topn 10
python src/main.py --keywords "space adventure future robots" --topn 10
```
