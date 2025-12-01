"""
Sephora Return Risk Predictor – Improved UX Version

Changes vs previous version:
- Clearer explanation of what the prediction means.
- Concrete "next step" advice for high vs low return risk.
- Easier input: user can start typing a PRODUCT NAME and the app will
  look up that product in the dataset and automatically use its
  category, price, review count, and love count.

If the product name is not found, the app falls back to the manually
entered fields.
"""

import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

CSV_PATH = "sephora_website_dataset.csv"

# Globals that will be populated at startup
MODEL = None
PRODUCT_CATALOG = None
PRODUCT_NAME_COL = None
PRODUCT_NAME_SAMPLES = []  # for the autocomplete dropdown


def load_and_prepare_data():
    """Load dataset, create target, and build a clean catalog DataFrame."""
    df = pd.read_csv(CSV_PATH)

    # --- Target engineering ---
    if "rating" not in df.columns:
        raise ValueError("Expected 'rating' column in dataset.")

    df = df.copy()
    df = df[~df["rating"].isna()]
    df = df[df["rating"] > 0]
    df["high_return_risk"] = (df["rating"] < 3.0).astype(int)

    # We’ll try to find a reasonable product name column.
    # Many Sephora datasets use 'name' – adjust if yours is different.
    name_candidates = ["product_name", "name", "display_name", "product"]
    product_name_col = None
    for c in name_candidates:
        if c in df.columns:
            product_name_col = c
            break

    # This isn't fatal – the app still works without product-autofill.
    if product_name_col is None:
        print(
            "⚠️ Could not find a product name column. "
            "Autocomplete by product name will be disabled."
        )

    # We keep a "catalog" view for lookup by product name.
    # It must at least contain the columns we use for the model.
    required_cols = ["brand", "category", "price", "number_of_reviews", "love"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {required_cols}, but these are missing: {missing}"
        )

    # Drop rows that are missing any of the key feature columns
    df_catalog = df.dropna(subset=required_cols)

    return df_catalog, product_name_col


def train_model(df_catalog):
    """Train a simple model using a small set of easy-to-understand features."""
    feature_cols = ["brand", "category", "price", "love", "number_of_reviews"]

    X = df_catalog[feature_cols]
    y = df_catalog["high_return_risk"]

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

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


print("🔁 Loading data and training model at startup...")

PRODUCT_CATALOG, PRODUCT_NAME_COL = load_and_prepare_data()
MODEL = train_model(PRODUCT_CATALOG)

# Sample product names for the HTML datalist (autocomplete).
if PRODUCT_NAME_COL is not None:
    PRODUCT_NAME_SAMPLES = (
        PRODUCT_CATALOG[PRODUCT_NAME_COL]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .head(200)  # limit so page isn't huge
        .tolist()
    )
else:
    PRODUCT_NAME_SAMPLES = []

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
      body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; }
      label { display: block; margin-top: 10px; font-weight: bold; }
      input { width: 100%; padding: 6px; margin-top: 4px; box-sizing: border-box; }
      button { margin-top: 15px; padding: 8px 16px; }
      .result { margin-top: 24px; padding: 16px; border-radius: 6px; }
      .high { background-color: #ffe0e0; }
      .low  { background-color: #e0ffe5; }
      .hint { font-size: 0.9rem; color: #555; }
      .section-title { margin-top: 18px; font-weight: bold; }
      ul { margin-top: 6px; }
    </style>
  </head>
  <body>
    <h1>Sephora Return Risk Predictor</h1>
    <p>
      Enter a product name or basic details to estimate the probability that this product is
      <b>high return risk</b> based on historical ratings and reviews.
    </p>

    <form method="post">
      <label>Product name</label>
      <input list="product_names"
             name="product_name"
             placeholder="Start typing a product name..."
             value="{{ product_name or '' }}">
      {% if product_name_samples %}
        <datalist id="product_names">
          {% for n in product_name_samples %}
            <option value="{{ n }}">
          {% endfor %}
        </datalist>
      {% endif %}
      <p class="hint">
        Tip: start typing a known product name and pick it from the suggestions.
        The app will automatically use that product's category, price, reviews, and loves.
      </p>

      <div class="section-title">Or enter details manually</div>

      <label>Brand</label>
      <input name="brand" value="{{ brand or '' }}">

      <label>Category</label>
      <input name="category" placeholder="e.g., Fragrance, Skincare" value="{{ category or '' }}">

      <label>Price (USD)</label>
      <input name="price" type="number" step="0.01" min="0" value="{{ price or '' }}">

      <label>Number of Reviews</label>
      <input name="number_of_reviews" type="number" min="0" value="{{ number_of_reviews or '' }}">

      <label>"Loves" Count</label>
      <input name="love" type="number" min="0" value="{{ love or '' }}">

      <button type="submit">Predict Return Risk</button>
    </form>

    {% if prob is not none %}
      <div class="result {{ 'high' if label == 'High' else 'low' }}">
        <h2>Prediction: {{ label }} return risk</h2>
        <p>
          Estimated probability that this product is <b>high return risk</b>:
          <b>{{ prob|round(3) }}</b> ({{ (prob * 100)|round(1) }}%).
        </p>

        <p class="section-title">How to interpret this:</p>
        <p>{{ interpretation }}</p>

        <p class="section-title">Suggested next steps:</p>
        <ul>
          {% for step in next_steps %}
            <li>{{ step }}</li>
          {% endfor %}
        </ul>
      </div>
    {% endif %}

    {% if error %}
      <div class="result high">
        <h2>Something went wrong</h2>
        <p>{{ error }}</p>
      </div>
    {% endif %}
  </body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prob = None
    label = None
    interpretation = ""
    next_steps = []
    error = None

    # Values to keep in the form after submit
    product_name = ""
    brand = ""
    category = ""
    price = ""
    number_of_reviews = ""
    love = ""

    if request.method == "POST":
        try:
            # 1) Try to look up product by name in the catalog
            product_name = (request.form.get("product_name") or "").strip()

            row = None
            if product_name and PRODUCT_NAME_COL is not None:
                matches = PRODUCT_CATALOG[
                    PRODUCT_CATALOG[PRODUCT_NAME_COL]
                    .str.contains(product_name, case=False, na=False)
                ]
                if len(matches) > 0:
                    row = matches.iloc[0]
                    brand = row["brand"]
                    category = row["category"]
                    price = float(row["price"])
                    number_of_reviews = float(row["number_of_reviews"])
                    love = float(row["love"])

            # 2) If no match or name not provided, fall back to manual fields
            if row is None:
                brand = request.form.get("brand", "")
                category = request.form.get("category", "")
                price = float(request.form.get("price") or 0)
                number_of_reviews = float(request.form.get("number_of_reviews") or 0)
                love = float(request.form.get("love") or 0)

            # Build a single-row DataFrame for prediction
            input_df = pd.DataFrame(
                {
                    "brand": [brand],
                    "category": [category],
                    "price": [price],
                    "number_of_reviews": [number_of_reviews],
                    "love": [love],
                }
            )

            proba = float(MODEL.predict_proba(input_df)[:, 1][0])
            prob = proba
            label = "High" if proba >= 0.5 else "Low"

            # --- Human-friendly explanation & next steps ---
            if label == "High":
                interpretation = (
                    "This product is predicted to have a HIGH risk of being returned. "
                    "Based on similar products' ratings and reviews, customers often end up dissatisfied "
                    "and are more likely to send it back."
                )
                next_steps = [
                    "Review recent customer reviews to understand common complaints.",
                    "Check whether the product description, shade, size, or benefits might be misleading.",
                    "Consider improving photos, swatches, or usage instructions to better set expectations.",
                    "Monitor this product's return rate over time and consider adjusting inventory or promotion strategy.",
                ]
            else:
                interpretation = (
                    "This product is predicted to have a LOW risk of being returned. "
                    "Historically, similar products receive strong ratings and customers tend to keep them."
                )
                next_steps = [
                    "Feature this product confidently in marketing campaigns or recommendations.",
                    "Use it as a benchmark when comparing newer or higher-risk products.",
                    "Continue to monitor reviews, but no immediate action is needed.",
                ]

        except Exception as e:
            error = f"Error while making prediction: {e}"

    return render_template_string(
        FORM_HTML,
        prob=prob,
        label=label,
        interpretation=interpretation,
        next_steps=next_steps,
        error=error,
        product_name=product_name,
        brand=brand,
        category=category,
        price=price,
        number_of_reviews=number_of_reviews,
        love=love,
        product_name_samples=PRODUCT_NAME_SAMPLES,
    )


if __name__ == "__main__":
    # For local debugging; Render still uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
