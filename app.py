"""
Car Price Prediction - Multiple Linear Regression
Streamlit app with four sections: Overview, EDA, Model Information, Predict Price.

Run with:
    streamlit run Manthan_554_Lab1_Streamlit_App.py

The app expects the dataset (CarPrice_Assignment.csv) in the same folder.
If it is not found, the app asks you to upload it.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ----------------------------------------------------------------------------
# Settings taken from the notebook
# ----------------------------------------------------------------------------
CYLINDER_MAP = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}

CATEGORICAL = [
    "fueltype", "aspiration", "carbody", "drivewheel",
    "enginetype", "enginelocation", "doornumber", "fuelsystem",
]

# Features that showed a strong correlation with price (notebook, section 3)
NUMERICAL_FEATURES = [
    "wheelbase", "carlength", "carwidth", "curbweight",
    "cylindernumber", "enginesize", "boreratio", "horsepower",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

COLUMN_INFO = {
    "symboling": "Insurance risk rating (-3 safest to +3 riskiest)",
    "CarName": "Car company and model name",
    "fueltype": "Fuel type (gas / diesel)",
    "aspiration": "Engine aspiration (std / turbo)",
    "doornumber": "Number of doors (two / four)",
    "carbody": "Body style (sedan, hatchback, wagon, hardtop, convertible)",
    "drivewheel": "Drive type (fwd / rwd / 4wd)",
    "enginelocation": "Engine location (front / rear)",
    "wheelbase": "Distance between front and rear axles (inches)",
    "carlength": "Car length (inches)",
    "carwidth": "Car width (inches)",
    "carheight": "Car height (inches)",
    "curbweight": "Weight of the car without occupants (lbs)",
    "enginetype": "Engine type (ohc, ohcv, dohc, l, rotor ...)",
    "cylindernumber": "Number of cylinders",
    "enginesize": "Engine size (cubic inches)",
    "fuelsystem": "Fuel system (mpfi, 2bbl, idi ...)",
    "boreratio": "Bore ratio of the engine",
    "stroke": "Stroke of the engine",
    "compressionratio": "Compression ratio",
    "horsepower": "Engine power (hp)",
    "peakrpm": "Peak engine RPM",
    "citympg": "Mileage in the city (mpg)",
    "highwaympg": "Mileage on the highway (mpg)",
    "price": "Target: price of the car (USD)",
}


# ----------------------------------------------------------------------------
# Data + model helpers
# ----------------------------------------------------------------------------
def find_dataset():
    """Look for the CSV next to this script."""
    here = Path(__file__).resolve().parent
    for name in ["CarPrice_Assignment.csv", "CarPrice_Assignment (1).csv", "CarPrice_Assignment__1_.csv"]:
        if (here / name).exists():
            return here / name
    matches = sorted(here.glob("CarPrice*.csv"))
    return matches[0] if matches else None


@st.cache_data
def load_raw(source):
    return pd.read_csv(source)


def get_dataframe():
    path = find_dataset()
    if path is not None:
        return load_raw(path)
    st.warning("Dataset not found next to the app. Please upload CarPrice_Assignment.csv.")
    uploaded = st.file_uploader("Upload the car price CSV", type="csv")
    if uploaded is None:
        st.stop()
    return load_raw(uploaded)


def clean(df):
    """Drop the ID column and turn cylinder words into numbers."""
    data = df.drop(columns=["car_ID"]).copy()
    data["cylindernumber"] = data["cylindernumber"].str.lower().str.strip().map(CYLINDER_MAP)
    return data


@st.cache_resource
def train_model(raw_df):
    """Same pipeline as the notebook: encode -> split -> scale -> fit -> evaluate."""
    data = clean(raw_df)
    encoded = pd.get_dummies(data, columns=CATEGORICAL, drop_first=True, dtype=int)
    dummy_cols = [c for c in encoded.columns if c not in data.columns]

    X = encoded[NUMERICAL_FEATURES + dummy_cols]
    y = encoded["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    return {
        "model": model,
        "scaler": scaler,
        "columns": list(X.columns),
        "dummy_cols": dummy_cols,
        "y_test": y_test,
        "y_pred": y_pred,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_test, y_pred),
        },
        "coefficients": pd.DataFrame(
            {"Feature": X.columns, "Coefficient": model.coef_}
        ),
    }


def predict_price(bundle, numeric_values, category_values):
    """Build one row in the training column order, scale it, and predict."""
    row = pd.DataFrame(0, index=[0], columns=bundle["columns"], dtype=float)
    for name, value in numeric_values.items():
        row.loc[0, name] = value
    for column, value in category_values.items():
        dummy_name = f"{column}_{value}"
        if dummy_name in row.columns:  # the first category is the baseline (all zeros)
            row.loc[0, dummy_name] = 1
    row[NUMERICAL_FEATURES] = bundle["scaler"].transform(row[NUMERICAL_FEATURES])
    return float(bundle["model"].predict(row)[0])


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def money(x):
    return f"${x:,.0f}"


# ----------------------------------------------------------------------------
# Load everything once
# ----------------------------------------------------------------------------
raw_df = get_dataframe()
data = clean(raw_df)
bundle = train_model(raw_df)
metrics = bundle["metrics"]

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🚗 Car Price Prediction")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Information", "Predict Price"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Multiple Linear Regression  \n"
    f"{len(raw_df)} cars · {bundle['n_train']} train / {bundle['n_test']} test  \n"
    f"R² = {metrics['R2']:.4f}"
)

# ============================================================================
# 1. OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("🚗 Car Price Prediction")
    st.markdown(
        "Estimate the market price of a car from its physical and mechanical "
        "specifications using **Multiple Linear Regression**, trained on the "
        "Car Price Assignment dataset for the American automotive market."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cars in dataset", f"{len(raw_df):,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Average price", money(data["price"].mean()))
    c4.metric("Price range", f"{money(data['price'].min())} – {money(data['price'].max())}")

    st.subheader("Project workflow")
    steps = [
        ("1. Load & explore", "Check shape, data types, duplicates, missing values and summary statistics."),
        ("2. Preprocess", "Drop car_ID, convert cylinder words to numbers, one-hot encode 8 categorical columns."),
        ("3. EDA", "Study price distribution, scatter plots and the correlation matrix to pick strong features."),
        ("4. Build model", f"Select {len(NUMERICAL_FEATURES)} numerical + {len(bundle['dummy_cols'])} encoded features, "
                           f"split {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, scale, and fit LinearRegression."),
        ("5. Evaluate", "Measure MAE, MSE, RMSE and R² on the unseen test set; inspect residuals."),
        ("6. Predict", "Enter a car's specifications and get an estimated selling price."),
    ]
    cols = st.columns(3)
    for i, (title, text) in enumerate(steps):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(text)

    st.subheader("Model at a glance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² score", f"{metrics['R2']:.4f}", help="Share of price variance explained by the model on the test set")
    m2.metric("Mean absolute error", money(metrics["MAE"]))
    m3.metric("Root mean squared error", money(metrics["RMSE"]))

    st.subheader("What's in this app")
    st.markdown(
        "- **EDA** – data quality checks, price distribution, feature relationships and correlations\n"
        "- **Model Information** – pipeline details, evaluation metrics, coefficients and prediction diagnostics\n"
        "- **Predict Price** – enter a car's specifications and get a price estimate"
    )

    with st.expander("Column descriptions"):
        info = pd.DataFrame({"Column": list(COLUMN_INFO), "Description": list(COLUMN_INFO.values())})
        st.dataframe(info, hide_index=True, width="stretch")

# ============================================================================
# 2. EDA
# ============================================================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    tab_quality, tab_price, tab_rel, tab_cat = st.tabs(
        ["Data quality", "Price distribution", "Feature relationships", "Categorical features"]
    )

    # ---- Data quality ----
    with tab_quality:
        st.subheader("First records")
        st.dataframe(raw_df.head(), width="stretch")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows", raw_df.shape[0])
        q2.metric("Columns", raw_df.shape[1])
        q3.metric("Duplicate records", int(raw_df.duplicated().sum()))
        q4.metric("Missing values", int(raw_df.isna().sum().sum()))

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Data types")
            dtypes = pd.DataFrame({
                "Column": raw_df.columns,
                "Type": raw_df.dtypes.astype(str).values,
                "Non-null": raw_df.notna().sum().values,
            })
            st.dataframe(dtypes, hide_index=True, width="stretch", height=420)
        with right:
            st.subheader("Statistical summary")
            st.dataframe(raw_df.drop(columns=["car_ID"]).describe().T.round(2),
                         width="stretch", height=420)

    # ---- Price distribution ----
    with tab_price:
        st.subheader("Distribution of car prices")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Mean", money(data["price"].mean()))
        p2.metric("Median", money(data["price"].median()))
        p3.metric("Minimum", money(data["price"].min()))
        p4.metric("Maximum", money(data["price"].max()))

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            sns.boxplot(data=data, x="price", color="#7fb3d5", ax=ax)
            ax.set_title("Boxplot of Car Prices")
            ax.set_xlabel("Price ($)")
            show(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            sns.histplot(data["price"], bins=25, kde=True, color="#5499c7", ax=ax)
            ax.set_title("Histogram of Car Prices")
            ax.set_xlabel("Price ($)")
            show(fig)

        q1_, q3_ = data["price"].quantile([0.25, 0.75])
        upper = q3_ + 1.5 * (q3_ - q1_)
        n_out = int((data["price"] > upper).sum())
        st.info(
            f"Prices are right-skewed: the median ({money(data['price'].median())}) sits below the mean "
            f"({money(data['price'].mean())}). {n_out} cars lie above the boxplot's upper whisker "
            f"({money(upper)}); these luxury and high-performance cars pull the average up and are the "
            "hardest for a linear model to predict."
        )

    # ---- Feature relationships ----
    with tab_rel:
        st.subheader("Key features vs price")
        scatter_cols = st.columns(3)
        for col, feature in zip(scatter_cols, ["horsepower", "enginesize", "curbweight"]):
            with col:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(data[feature], data["price"], alpha=0.7, color="#2e86c1", edgecolor="white")
                slope, intercept = np.polyfit(data[feature], data["price"], 1)
                xs = np.linspace(data[feature].min(), data[feature].max(), 100)
                ax.plot(xs, slope * xs + intercept, color="firebrick", linestyle="--", linewidth=1.5)
                ax.set_title(f"{feature.title()} vs Price")
                ax.set_xlabel(feature.title())
                ax.set_ylabel("Price ($)")
                show(fig)

        # st.markdown("**Explore any numeric feature**")
        # numeric_cols = [c for c in data.select_dtypes(include=np.number).columns if c != "price"]
        # chosen = st.selectbox("Feature", numeric_cols, index=numeric_cols.index("wheelbase"))
        # fig, ax = plt.subplots(figsize=(8, 4))
        # sns.regplot(data=data, x=chosen, y="price", ax=ax,
        #             scatter_kws={"alpha": 0.6}, line_kws={"color": "firebrick"})
        # ax.set_title(f"{chosen} vs price (r = {data[chosen].corr(data['price']):.2f})")
        # show(fig)

        st.subheader("Correlation matrix")
        corr = data.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(13, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0,
                    annot_kws={"size": 8}, ax=ax)
        ax.set_title("Correlation Matrix of Numerical Features", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        show(fig)

        st.subheader("Correlation with price")
        price_corr = corr["price"].drop("price").sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#c0392b" if v < 0 else "#2e86c1" for v in price_corr]
        ax.barh(price_corr.index, price_corr.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Correlation with price")
        show(fig)

        strong = corr["price"].drop("price")
        strong_names = ", ".join(strong[strong >= 0.5].sort_values(ascending=False).index)
        weak_pos = ", ".join(strong[(strong > 0) & (strong < 0.5)].index)
        negative = ", ".join(strong[strong < 0].index)
        st.success(
            f"**Strong positive relationship (r ≥ 0.5):** {strong_names}. These are used as the numerical model features."
        )
        st.markdown(
            f"- **Weak positive (below 0.5):** {weak_pos}\n"
            f"- **Negative relationship:** {negative}"
        )

    # ---- Categorical features ----
    with tab_cat:
        st.subheader("Price by category")
        cat_choice = st.selectbox("Categorical feature", CATEGORICAL, index=CATEGORICAL.index("carbody"))
        order = data.groupby(cat_choice)["price"].median().sort_values(ascending=False).index

        cc1, cc2 = st.columns(2)
        with cc1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=data, x=cat_choice, y="price", order=order, hue=cat_choice, palette="Blues", legend=False, ax=ax)
            ax.set_title(f"Price by {cat_choice}")
            ax.set_ylabel("Price ($)")
            plt.xticks(rotation=30)
            show(fig)
        with cc2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=data, x=cat_choice, order=order, hue=cat_choice, palette="Blues", legend=False, ax=ax)
            ax.set_title(f"Number of cars by {cat_choice}")
            ax.set_ylabel("Count")
            plt.xticks(rotation=30)
            show(fig)

        summary = (
            data.groupby(cat_choice)["price"]
            .agg(Cars="count", Average="mean", Median="median", Min="min", Max="max")
            .round(0)
            .loc[order]
        )
        st.dataframe(summary, width="stretch")

# ============================================================================
# 3. MODEL INFORMATION
# ============================================================================
elif page == "Model Information":
    st.title("🧠 Model Information")

    st.subheader("Model summary")
    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown("**Algorithm:** Multiple Linear Regression (`sklearn.linear_model.LinearRegression`)")
            st.markdown("**Target:** `price`")
            st.markdown(
                f"**Features:** {len(bundle['columns'])} "
                f"({len(NUMERICAL_FEATURES)} numerical + {len(bundle['dummy_cols'])} one-hot encoded)"
            )
    with s2:
        with st.container(border=True):
            st.markdown(f"**Train / test split:** {int((1 - TEST_SIZE) * 100)}% / {int(TEST_SIZE * 100)}% "
                        f"({bundle['n_train']} / {bundle['n_test']} cars), `random_state={RANDOM_STATE}`")
            st.markdown("**Scaling:** `StandardScaler` on numerical features only (fit on training data)")
            st.markdown(f"**Intercept:** {money(bundle['model'].intercept_)}")

    with st.expander("Features used"):
        f1, f2 = st.columns(2)
        f1.markdown("**Numerical (scaled)**")
        f1.write(", ".join(NUMERICAL_FEATURES))
        f2.markdown("**One-hot encoded (first category of each column dropped)**")
        f2.write(", ".join(bundle["dummy_cols"]))

    # ---- Metrics ----
    st.subheader("Evaluation on the test set")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("R²", f"{metrics['R2']:.4f}")
    e2.metric("MAE", money(metrics["MAE"]))
    e3.metric("RMSE", money(metrics["RMSE"]))
    e4.metric("MSE", f"{metrics['MSE']:,.0f}")

    st.markdown(
        f"- **R² ({metrics['R2'] * 100:.2f}%)** – the model explains about {metrics['R2'] * 100:.0f}% of the "
        "variation in car prices, a solid linear baseline.\n"
        f"- **MAE ({money(metrics['MAE'])})** – on average, the estimate is off by this much from the actual price.\n"
        f"- **RMSE ({money(metrics['RMSE'])})** – higher than the MAE because squaring the errors penalises the "
        "few large misses, mainly on expensive cars.\n"
        "- **Scaling note:** standardising features does not change the predictions or metrics of a linear "
        "regression; it puts the coefficients on a common scale so they can be compared."
    )

    # ---- Coefficients ----
    st.subheader("Model coefficients")
    coef = bundle["coefficients"].copy()
    coef["Abs"] = coef["Coefficient"].abs()
    coef = coef.sort_values("Abs", ascending=False)

    top_n = st.slider("Number of features to show in the chart", 5, len(coef), len(coef))
    top = coef.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(top["Feature"], top["Coefficient"],
            color=["#c0392b" if v < 0 else "#2e86c1" for v in top["Coefficient"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (effect on price in $)")
    ax.set_title(f"Top {top_n} features by absolute coefficient")
    plt.tight_layout()
    show(fig)

    st.caption(
        "Numerical coefficients are per one standard deviation of the feature; "
        "categorical coefficients are the price difference against the dropped baseline category."
    )
    with st.expander("Full coefficient table"):
        st.dataframe(coef[["Feature", "Coefficient"]].round(2), hide_index=True, width="stretch")

    # ---- Diagnostics ----
    st.subheader("Prediction diagnostics")
    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]

    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=y_test, y=y_pred, color="royalblue", edgecolor="w", s=70, ax=ax,
                        label="Actual vs Predicted")
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color="firebrick", linestyle="--", linewidth=2,
                label="Perfect prediction (y = x)")
        ax.set_xlabel("Actual price ($)")
        ax.set_ylabel("Predicted price ($)")
        ax.set_title(f"Actual vs Predicted (R² = {metrics['R2']:.4f})")
        ax.legend(loc="upper left")
        show(fig)
    with d2:
        residual = y_test - y_pred
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=y_pred, y=residual, color="royalblue", edgecolor="w", s=70, ax=ax,
                        label="Residuals")
        ax.axhline(0, color="firebrick", linestyle="--", linewidth=2, label="Zero residual")
        ax.set_xlabel("Predicted price ($)")
        ax.set_ylabel("Residual = actual − predicted ($)")
        ax.set_title("Residual plot")
        ax.legend(loc="upper left")
        show(fig)

    with st.expander("Test-set predictions"):
        table = pd.DataFrame({
            "Actual price": y_test.values,
            "Predicted price": y_pred,
        })
        table["Error"] = table["Actual price"] - table["Predicted price"]
        st.dataframe(table.round(0), hide_index=True, width="stretch")

# ============================================================================
# 4. PREDICT PRICE
# ============================================================================
else:
    st.title("💰 Predict Car Price")
    st.write("Enter the specifications below. Sliders are limited to the range seen in the training data.")

    def slider(label, column, step, integer=False):
        low, high = data[column].min(), data[column].max()
        default = data[column].median()
        if integer:
            return st.slider(label, int(low), int(high), int(default), step=step)
        return st.slider(label, float(low), float(high), float(round(default, 2)), step=step)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Size & weight")
            wheelbase = slider("Wheelbase (in)", "wheelbase", 0.1)
            carlength = slider("Car length (in)", "carlength", 0.1)
            carwidth = slider("Car width (in)", "carwidth", 0.1)
            curbweight = slider("Curb weight (lbs)", "curbweight", 10, integer=True)

        with col2:
            st.markdown("##### Engine & performance")
            enginesize = slider("Engine size (cu in)", "enginesize", 1, integer=True)
            horsepower = slider("Horsepower", "horsepower", 1, integer=True)
            boreratio = slider("Bore ratio", "boreratio", 0.01)
            cyl_options = sorted(
                data["cylindernumber"].dropna().unique().astype(int).tolist()
            )
            cylindernumber = st.selectbox(
                "Number of cylinders", cyl_options, index=cyl_options.index(4)
            )

        with col3:
            st.markdown("##### Configuration")
            choices = {}
            for column, label in [
                ("carbody", "Body style"),
                ("drivewheel", "Drive wheel"),
                ("fueltype", "Fuel type"),
                ("aspiration", "Aspiration"),
                ("doornumber", "Doors"),
                ("enginelocation", "Engine location"),
                ("enginetype", "Engine type"),
                ("fuelsystem", "Fuel system"),
            ]:
                options = sorted(raw_df[column].unique())
                default = raw_df[column].mode()[0]
                choices[column] = st.selectbox(label, options, index=options.index(default))

        submitted = st.form_submit_button("Predict price", type="primary", width="stretch")

    if submitted:
        numeric_values = {
            "wheelbase": wheelbase,
            "carlength": carlength,
            "carwidth": carwidth,
            "curbweight": curbweight,
            "cylindernumber": cylindernumber,
            "enginesize": enginesize,
            "boreratio": boreratio,
            "horsepower": horsepower,
        }
        price = predict_price(bundle, numeric_values, choices)

        st.markdown("---")
        if price <= 0:
            st.error(
                "The model returned a non-positive price for this combination. Linear regression can "
                "extrapolate badly for unusual specification mixes, so treat this as unreliable."
            )
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Estimated price", money(price))
            r2.metric("Likely range (± MAE)", f"{money(max(price - metrics['MAE'], 0))} – {money(price + metrics['MAE'])}")
            percentile = (data["price"] < price).mean() * 100
            r3.metric("Compared with the dataset", f"Higher than {percentile:.0f}% of cars",
                      delta=f"{money(price - data['price'].mean())} vs average", delta_color="off")

            fig, ax = plt.subplots(figsize=(9, 3.2))
            sns.histplot(data["price"], bins=25, color="#aed6f1", ax=ax)
            ax.axvline(price, color="firebrick", linewidth=2.5, label=f"Your car: {money(price)}")
            ax.axvline(data["price"].mean(), color="black", linestyle="--", linewidth=1.2,
                       label=f"Dataset average: {money(data['price'].mean())}")
            ax.set_xlabel("Price ($)")
            ax.set_ylabel("Number of cars")
            ax.legend()
            show(fig)

            if choices["enginelocation"] == "rear":
                st.warning(
                    "Only 3 cars in the training data have a rear engine, so estimates for rear-engine "
                    "cars are much less reliable."
                )
            st.caption(
                f"The range is the estimate ± the model's mean absolute error on the test set "
                f"({money(metrics['MAE'])}); it is a rough guide, not a confidence interval."
            )
    else:
        st.info("Set the specifications and click **Predict price**.")
