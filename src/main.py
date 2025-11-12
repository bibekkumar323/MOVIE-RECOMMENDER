from __future__ import annotations
import argparse
from pathlib import Path
from data_utils import ensure_data
from recommender import ContentRecommender

def build_model(data_dir: Path):
    movies, _ = ensure_data(data_dir)
    return ContentRecommender().fit(movies)

def main():
    parser = argparse.ArgumentParser(description="Content-based Movie Recommender (MovieLens)")
    parser.add_argument("--data", type=str, default="data", help="Data directory (default: ./data)")
    parser.add_argument("--download", action="store_true", help="Force dataset download")
    parser.add_argument("--title", type=str, help="Movie title to get recommendations")
    parser.add_argument("--keywords", type=str, help="Keywords to search for")
    parser.add_argument("--topn", type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()
    data_dir = Path(args.data)
    if args.download:
        ensure_data(data_dir)
    model = build_model(data_dir)
    if args.title:
        print(model.recommend_by_title(args.title, topn=args.topn))
    elif args.keywords:
        print(model.recommend_by_keywords(args.keywords, topn=args.topn))
    else:
        print("Use --title or --keywords.")

if __name__ == "__main__":
    main()
