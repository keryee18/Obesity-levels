"""Streamlit dashboard for obesity-level classification.

Run locally with: streamlit run app.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Obesity Levels Predictor", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
TARGET = "NObeyesdad"
RANDOM_STATE = 42

# Shared blue-tone palette used across every chart in the dashboard.
MACARON_COLORS = [
    "#E27D8C",  # dusty rose
    "#E8A87C",  # terracotta peach
    "#D4B85A",  # mustard yellow
    "#7FB685",  # sage mint
    "#5B8DB8",  # dusty blue
    "#9C7FB8",  # muted lavender
    "#C97B8C",  # mauve
]
# Two-tone blue gradient for continuous ("Count"-style) scales, e.g. the confusion matrix.
MACARON_GRADIENT = ["#E3F2FD", "#0B3D91"]


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the bundled data, or a user-uploaded CSV with the same schema."""
    if uploaded_bytes is not None:
        from io import BytesIO

        data = pd.read_csv(BytesIO(uploaded_bytes))
    else:
        data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


@st.cache_resource(show_spinner="Training and evaluating the four models…")
def train_models(data: pd.DataFrame):
    features = data.drop(columns=TARGET)
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    fitted, summaries = {}, []
    for name, classifier in models.items():
        pipeline = Pipeline(
            [("preprocess", make_preprocessor(features)), ("model", classifier)]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted[name] = pipeline
        summaries.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Weighted F1": f1_score(y_test, predictions, average="weighted"),
                "Predictions": predictions,
            }
        )
    return fitted, pd.DataFrame(summaries), y_test, x_test


def filtered_data(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Data filters")
    selected_classes = st.sidebar.multiselect(
        "Obesity level", sorted(data[TARGET].unique()), default=sorted(data[TARGET].unique())
    )
    selected_genders = st.sidebar.multiselect(
        "Gender", sorted(data["Gender"].unique()), default=sorted(data["Gender"].unique())
    )
    age_min, age_max = float(data["Age"].min()), float(data["Age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max), 0.5)
    return data[
        data[TARGET].isin(selected_classes)
        & data["Gender"].isin(selected_genders)
        & data["Age"].between(*age_range)
    ].copy()


def section_overview(data: pd.DataFrame, subset: pd.DataFrame) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Filtered records", f"{len(subset):,}")
    b.metric("Total records", f"{len(data):,}")
    c.metric("Input features", data.shape[1] - 1)
    d.metric("Obesity classes", data[TARGET].nunique())
    st.caption("Filters in the sidebar apply to this dashboard's data and chart tabs.")
    st.subheader("Filtered dataset")
    st.dataframe(subset, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        subset.to_csv(index=False).encode("utf-8"),
        "filtered_obesity_data.csv",
        "text/csv",
    )


def section_charts(subset: pd.DataFrame) -> None:
    if subset.empty:
        st.warning("No records match the selected filters.")
        return

    counts = subset[TARGET].value_counts()

    bar_col, pie_col = st.columns(2)

    with bar_col:
        st.subheader("Obesity-level distribution")
        counts_df = counts.rename_axis("Obesity level").reset_index(name="Count")
        chart = alt.Chart(counts_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Obesity level:N", sort="-y", title=None),
            y=alt.Y("Count:Q", title="People"),
            color=alt.Color(
                "Obesity level:N",
                legend=None,
                scale=alt.Scale(range=MACARON_COLORS),
            ),
            tooltip=["Obesity level", "Count"],
        ).properties(height=340)
        st.altair_chart(chart, width="stretch")

    with pie_col:
        st.subheader("Distribution of Obesity Levels (Pie Chart)")
        counts_df = counts.rename_axis("Obesity level").reset_index(name="Count")
        counts_df["Percent"] = counts_df["Count"] / counts_df["Count"].sum()
        counts_df = counts_df.sort_values("Count", ascending=False).reset_index(drop=True)
        pie_base = alt.Chart(counts_df).encode(
            theta=alt.Theta("Count:Q", stack=True, sort=None),
            order=alt.Order("Count:Q", sort="descending"),
        )
        pie_chart = pie_base.mark_arc(stroke="white", strokeWidth=1).encode(
            color=alt.Color(
                "Obesity level:N",
                sort=counts_df["Obesity level"].tolist(),
                legend=alt.Legend(title="Obesity level", orient="right"),
                scale=alt.Scale(range=MACARON_COLORS),
            ),
            tooltip=["Obesity level", "Count", alt.Tooltip("Percent:Q", format=".1%")],
        ).properties(height=340)
        pie_labels = pie_base.mark_text(radius=115, size=11, color="white", fontWeight="bold").encode(
            text=alt.Text("Percent:Q", format=".1%"),
        )
        st.altair_chart(pie_chart + pie_labels, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Numeric variable distribution")
        numeric_choices = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        variable = st.selectbox("Choose a variable", numeric_choices)
        histogram = alt.Chart(subset).mark_bar(opacity=0.85).encode(
            x=alt.X(f"{variable}:Q", bin=alt.Bin(maxbins=24)),
            y=alt.Y("count():Q", title="People"),
            color=alt.Color(
                f"{TARGET}:N",
                title="Obesity level",
                scale=alt.Scale(range=MACARON_COLORS),
            ),
            tooltip=[alt.Tooltip("count():Q", title="People")],
        ).properties(height=300)
        st.altair_chart(histogram, width="stretch")

    with right:
        st.subheader("Alcohol consumption levels by gender")
        calc_order = [level for level in ["no", "Sometimes", "Frequently", "Always"] if level in subset["CALC"].unique()]
        calc_counts = subset.groupby(["Gender", "CALC"]).size().reset_index(name="Count")
        alcohol_chart = alt.Chart(calc_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Gender:N", title="Gender"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "CALC:N",
                title="CALC",
                sort=calc_order,
                scale=alt.Scale(domain=calc_order, range=MACARON_COLORS[: len(calc_order)]),
            ),
            xOffset=alt.XOffset("CALC:N", sort=calc_order),
            tooltip=["Gender", "CALC", "Count"],
        ).properties(height=300)
        st.altair_chart(alcohol_chart, width="stretch")

    st.subheader("Lifestyle factor by obesity level")
    lifestyle = st.selectbox(
        "Choose a lifestyle factor", ["CAEC", "CALC", "FAVC", "SMOKE", "SCC", "MTRANS", "family_history_with_overweight"]
    )
    grouped = subset.groupby([lifestyle, TARGET]).size().reset_index(name="Count")
    lifestyle_chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X(f"{lifestyle}:N", title=lifestyle),
        y=alt.Y("Count:Q", stack="normalize", title="Proportion"),
        color=alt.Color(
            f"{TARGET}:N",
            title="Obesity level",
            scale=alt.Scale(range=MACARON_COLORS),
        ),
        tooltip=[lifestyle, TARGET, "Count"],
    ).properties(height=340)
    st.altair_chart(lifestyle_chart, width="stretch")


def section_models(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    models, results, y_test, _ = train_models(data)
    ranking = results[["Model", "Accuracy", "Weighted F1"]].sort_values("Weighted F1", ascending=False)
    st.subheader("Model comparison")
    st.caption("All models use the same stratified 80/20 split (random state 42). Categorical inputs are one-hot encoded; numeric inputs are standardized.")
    st.dataframe(
        ranking.style.format({"Accuracy": "{:.2%}", "Weighted F1": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )
    performance = ranking.melt("Model", var_name="Metric", value_name="Score")
    chart = alt.Chart(performance).mark_bar().encode(
        x=alt.X("Model:N", sort="-y"),
        y=alt.Y("Score:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Metric:N", scale=alt.Scale(range=MACARON_COLORS)),
        xOffset="Metric:N",
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2%")],
    ).properties(height=340)
    st.altair_chart(chart, width="stretch")

    chosen = st.selectbox("Inspect model", ranking["Model"].tolist())
    predicted = results.loc[results["Model"] == chosen, "Predictions"].iloc[0]
    report = pd.DataFrame(classification_report(y_test, predicted, output_dict=True)).T
    st.subheader(f"{chosen}: class-level metrics")
    st.dataframe(report[["precision", "recall", "f1-score", "support"]].style.format({"precision": "{:.2%}", "recall": "{:.2%}", "f1-score": "{:.2%}", "support": "{:.0f}"}), width="stretch")
    matrix = confusion_matrix(y_test, predicted, labels=sorted(data[TARGET].unique()))
    labels = sorted(data[TARGET].unique())
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("Actual").reset_index().melt("Actual", var_name="Predicted", value_name="Count")
    heatmap = alt.Chart(matrix_df).mark_rect().encode(
        x=alt.X("Predicted:N", sort=labels), y=alt.Y("Actual:N", sort=labels),
        color=alt.Color(
            "Count:Q",
            scale=alt.Scale(range=MACARON_GRADIENT),
        ),
        tooltip=["Actual", "Predicted", "Count"]
    ).properties(height=360, title="Confusion matrix")
    st.altair_chart(heatmap, width="stretch")
    return models, results


def section_prediction(data: pd.DataFrame) -> None:
    models, results, _, _ = train_models(data)
    best_model = results.sort_values("Weighted F1", ascending=False).iloc[0]["Model"]
    st.subheader("Predict an obesity level")
    selected_model = st.selectbox("Prediction model", list(models), index=list(models).index(best_model))
    st.caption(f"Recommended by weighted F1 on the hold-out test set: {best_model}.")
    features = data.drop(columns=TARGET)
    values = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for i, column in enumerate(features.columns):
            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(features[column]):
                    minimum, maximum = float(features[column].min()), float(features[column].max())
                    default = float(features[column].median())
                    step = 0.01 if column in {"Height", "Weight"} else 0.1
                    values[column] = st.number_input(column, min_value=minimum, max_value=maximum, value=default, step=step)
                else:
                    options = sorted(features[column].astype(str).unique())
                    values[column] = st.selectbox(column, options, index=options.index(str(features[column].mode().iloc[0])))
        submitted = st.form_submit_button("Predict obesity level", type="primary")
    if submitted:
        person = pd.DataFrame([values])
        model = models[selected_model]
        prediction = model.predict(person)[0]
        probabilities = model.predict_proba(person)[0]
        probability_table = pd.DataFrame({"Obesity level": model.classes_, "Probability": probabilities}).sort_values("Probability", ascending=False)
        st.success(f"Predicted obesity level: **{prediction}**")
        st.altair_chart(
            alt.Chart(probability_table).mark_bar().encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Obesity level:N", sort="-x"),
                color=alt.Color(
                    "Obesity level:N",
                    legend=None,
                    scale=alt.Scale(range=MACARON_COLORS),
                ),
                tooltip=["Obesity level", alt.Tooltip("Probability:Q", format=".2%")],
            ).properties(height=260, title="Prediction probabilities"),
            width="stretch",
        )


def main() -> None:
    st.title("Obesity Levels Prediction Dashboard")
    st.write("Explore the eating-habit and physical-condition data, compare classifiers, and make an individual prediction.")
    uploaded = st.sidebar.file_uploader("Use a different obesity CSV (optional)", type="csv")
    try:
        data = load_data(uploaded.getvalue() if uploaded else None)
    except Exception as error:
        st.error(f"Could not load the CSV: {error}")
        st.stop()
    if TARGET not in data.columns:
        st.error(f"The CSV must include a '{TARGET}' target column.")
        st.stop()
    subset = filtered_data(data)
    overview, charts, comparison, prediction = st.tabs(["Data explorer", "Charts", "Model comparison", "Make a prediction"])
    with overview:
        section_overview(data, subset)
    with charts:
        section_charts(subset)
    with comparison:
        section_models(data)
    with prediction:
        section_prediction(data)
    st.caption("Educational dashboard — predictions are estimates from the supplied dataset, not medical advice.")


if __name__ == "__main__":
    main()
