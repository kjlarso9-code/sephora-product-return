"""
Simple Flask app for predicting Sephora product return risk.

- When the app starts, it trains a model on sephora_website_dataset.csv
  using a small set of features.
- The home page shows a form where a user can enter product info.
- The app returns the probability of high return risk.
"""

import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


CSV_PATH = "sephora_website_dataset.csv"

# -----------------------------
# Train model once at startup
# -----------------------------
def train_model():
    df = pd.read_csv(CSV_PATH)

    # Target engineering: high_return_risk from rating
    if "rating" not in df.columns:
        raise ValueError("Expected 'rating' column in dataset.")
    df = df.copy()
    df = df[~df["rating"].isna()]
    df = df[df["rating"] > 0]
    df["high_return_risk"] = (df["rating"] < 3.0).astype(int)

    # Use a small set of features that are easy for a user to input
    feature_cols = ["brand", "category", "price", "love", "number_of_reviews"]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in dataset.")

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df["high_return_risk"]

    num_cols = ["price", "love", "number_of_reviews"]
    cat_cols = ["brand", "category"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            ),
        ]
    )

    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    pipe.fit(X, y)
    return pipe


print("🔁 Training model at startup...")
model = train_model()
print("✅ Model trained and ready for predictions.")


# -----------------------------
# Flask app & HTML template
# -----------------------------
app = Flask(__name__)

FORM_HTML = """
<!doctype html>
<html>
  <head>
    <title>Sephora Return Risk Predictor</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; }
      label { display: block; margin-top: 10px; font-weight: bold; }
      input { width: 100%; padding: 6px; margin-top: 4px; }
      button { margin-top: 15px; padding: 8px 16px; }
      .result { margin-top: 20px; padding: 10px; border-radius: 4px; }
      .high { background-color: #ffe0e0; }
      .low  { background-color: #e0ffe5; }
    </style>
  </head>
  <body>
    <h1>Sephora Return Risk Predictor</h1>
    <p>Enter product details to estimate the probability that this product is <b>high return risk</b>.</p>

    <form method="post">
      <label>Brand</label>
      <input name="brand" required>

      <label>Category</label>
      <input name="category" placeholder="e.g., Fragrance, Skincare" required>

      <label>Price (USD)</label>
      <input name="price" type="number" step="0.01" min="0" required>

      <label>Number of Reviews</label>
      <input name="number_of_reviews" type="number" min="0" required>

      <label>"Loves" Count</label>
      <input name="love" type="number" min="0" required>

      <button type="submit">Predict Return Risk</button>
    </form>

    {% if prob is not none %}
      <div class="result {{ 'high' if label == 'High' else 'low' }}">
        <h2>Prediction: {{ label }} return risk</h2>
        <p>Estimated probability of high return risk: <b>{{ prob|round(3) }}</b></p>
      </div>
    {% endif %}
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prob = None
    label = None

    if request.method == "POST":
        try:
            brand = request.form.get("brand", "")
            category = request.form.get("category", "")
            price = float(request.form.get("price") or 0)
            num_reviews = float(request.form.get("number_of_reviews") or 0)
            love = float(request.form.get("love") or 0)

            row = pd.DataFrame(
                {
                    "brand": [brand],
                    "category": [category],
                    "price": [price],
                    "number_of_reviews": [num_reviews],
                    "love": [love],
                }
            )

            proba = float(model.predict_proba(row)[:, 1][0])
            prob = proba
            label = "High" if proba >= 0.5 else "Low"

        except Exception as e:
            prob = None
            label = f"Error: {e}"

    return render_template_string(FORM_HTML, prob=prob, label=label)


# For local debugging (not used on Render – Render will use gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
