import os
from supabase import create_client


def collect_data(client):
    last_id = 0
    while True:
        response = (
            client.table("preferences")
            .select("*")
            .range(last_id, last_id + 1000)
            .execute()
        )
        if len(response.data) == 0:
            break
        last_id += len(response.data)
        yield from response.data


class EloRating:
    def __init__(self, initial_rating=400, client_weight_threshold=50):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.client_weight_threshold = client_weight_threshold
        self.client_counts = {}
        self.games_played = {}

    def process(self, row_iterator):
        for row in row_iterator:
            model_a = row["model_a"]
            model_b = row["model_b"]
            preference = int(row["preference"])
            client_id = row["client_ip"]

            if model_a not in self.ratings:
                self.ratings[model_a] = self.initial_rating
                self.games_played[model_a] = 0
            if model_b not in self.ratings:
                self.ratings[model_b] = self.initial_rating
                self.games_played[model_b] = 0

            if client_id not in self.client_counts:
                self.client_counts[client_id] = 0
            self.client_counts[client_id] += 1

            if (
                self.games_played[model_a] + self.games_played[model_b]
                >= self.client_weight_threshold
            ):
                weight = 1 / (1 + self.client_counts[client_id] - 1)
            else:
                weight = 1

            k_factor_a = self.calculate_k_factor(model_a)
            k_factor_b = self.calculate_k_factor(model_b)

            if preference == 0:
                self.update_ratings(
                    model_a, model_b, 1, 0, weight, k_factor_a, k_factor_b
                )
            elif preference == 1:
                self.update_ratings(
                    model_a, model_b, 0, 1, weight, k_factor_a, k_factor_b
                )
            else:
                self.update_ratings(
                    model_a, model_b, 0.5, 0.5, weight, k_factor_a, k_factor_b
                )

            self.games_played[model_a] += 1
            self.games_played[model_b] += 1

    def calculate_k_factor(self, model):
        Ne = self.calculate_Ne(model)
        games_played = self.games_played[model]
        denominator = Ne + games_played
        if denominator == 0:
            return 32
        return 800 / denominator

    def calculate_Ne(self, model):
        total_games = sum(self.games_played.values())
        model_games = self.games_played[model]
        return total_games - model_games

    def update_ratings(
        self, model_a, model_b, score_a, score_b, weight, k_factor_a, k_factor_b
    ):
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

        self.ratings[model_a] += weight * k_factor_a * (score_a - expected_a)
        self.ratings[model_b] += weight * k_factor_b * (score_b - expected_b)

    def get_ratings(self):
        return self.ratings


def main():
    supabase_client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    )

    elo_rating = EloRating()
    elo_rating.process(collect_data(supabase_client))
    ratings = elo_rating.get_ratings()

    for model_name, rating in ratings.items():
        supabase_client.table("ratings").insert(
            {
                "model_name": model_name,
                "rating": rating,
                "num_samples": elo_rating.games_played[model_name],
            }
        ).execute()


if __name__ == "__main__":
    main()
