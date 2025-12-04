from src.movie_recommender import MovieRecommender


def main():
    # Path to your dataset
    csv_path = "data/tmdb_5000_movies.csv"  

    recommender = MovieRecommender(csv_path)

    # Build the recommendation model
    recommender.fit()

    # Simple CLI loop
    while True:
        print("\nEnter a movie title to get recommendations (or 'q' to quit):")
        user_input = input("> ").strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye! 👋")
            break

        try:
            recs = recommender.recommend(user_input, top_n=10)
            print(f"\nTop recommendations similar to '{user_input}':\n")
            for idx, (title, score) in enumerate(recs, start=1):
                print(f"{idx}. {title}  (similarity: {score:.3f})")
        except ValueError as e:
            print(f"[ERROR] {e}")
            print("Try another exact movie title that exists in the dataset.")


if __name__ == "__main__":
    main()
