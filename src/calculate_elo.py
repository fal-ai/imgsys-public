import os
import math
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.linear_model import LogisticRegression


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


# MLE computation code from Chatbot Arena LMSys paper
def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]
    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)
    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0
    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def main():
    supabase_client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY"),
    )
    raw_data = list(collect_data(supabase_client))
    df = pd.DataFrame(raw_data)
    df["winner"] = df["preference"].apply(
        lambda x: "model_a" if x == 0 else ("model_b" if x == 1 else "tie")
    )
    elo_scores = compute_mle_elo(df)
    for model, elo in elo_scores.items():
        supabase_client.table("ratings").insert(
            {
                "model_name": model,
                "rating": elo,
                "num_samples": int(df["model_a"].value_counts()[model]
                + df["model_b"].value_counts().get(model, 0)),
            }
        ).execute()

    print(elo_scores)


if __name__ == "__main__":
    main()
