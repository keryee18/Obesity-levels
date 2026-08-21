from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Obesity Levels Predictor", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "ObesityDataSet_Cleaned_Outliers_Capped.csv"
RAW_DATA_PATH = Path(__file__).parent / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
TUNING_RESULTS_PATH = Path(__file__).parent / "data" / "best_hyperparameters.csv"
TARGET = "NObeyesdad"
MODEL_EXCLUDED_COLUMNS = ["BMI"]
RANDOM_STATE = 42

# Explicitly defining custom orders for Gender and Yes/No variables
GENDER_ORDER = ["Male", "Female"]
YES_NO_ORDER = ["yes", "no"]

DEFAULT_PARAMS = {
    "rf_n_estimators": 250,
    "rf_max_depth": 16,
    "rf_min_samples_split": 2,
    "rf_min_samples_leaf": 1,
    "rf_max_features": "sqrt",
    "rf_max_leaf_nodes": None,
    "dt_max_depth": 12,
    "dt_min_samples_split": 2,
    "dt_min_samples_leaf": 1,
    "dt_criterion": "entropy",
    "dt_max_leaf_nodes": 100,
    "lr_c": 100.0,
    "lr_max_iter": 1000,
    "lr_solver": "lbfgs",
    "lr_class_weight": None,
    "lr_tol": 0.0001,
    "knn_neighbors": 3,
    "knn_weights": "uniform",
    "knn_p": 1,
    "knn_leaf_size": 20,
    "smote_enabled": True,
    "smote_k_neighbors": 3,
    "smote_k_neighbors_by_model": {
        "Random Forest": 5,
        "Decision Tree": 3,
        "Logistic Regression": 3,
        "K-Nearest Neighbors": 5,
    },
}

MACARON_COLORS = [
    "#E27D8C",  # dusty rose
    "#E8A87C",  # terracotta peach
    "#D4B85A",  # mustard yellow
    "#7FB685",  # sage mint
    "#5B8DB8",  # dusty blue
    "#9C7FB8",  # muted lavender
    "#C97B8C",  # mauve
]
MACARON_GRADIENT = ["#E3F2FD", "#0B3D91"]
GENDER_PALETTE = ["#5B8DB8", "#D4B85A"]
YES_NO_PALETTE = ["#9C7FB8", "#E27D8C"]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    data = pd.read_csv(RAW_DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


@st.cache_data(show_spinner=False)
def load_tuning_results() -> pd.DataFrame:
    return pd.read_csv(TUNING_RESULTS_PATH)


@st.cache_data(show_spinner=False)
def load_tuned_params() -> dict:
    """Load the cross-validation-selected parameters, with safe defaults as fallback."""
    params = DEFAULT_PARAMS.copy()
    params["smote_k_neighbors_by_model"] = DEFAULT_PARAMS[
        "smote_k_neighbors_by_model"
    ].copy()

    if not TUNING_RESULTS_PATH.exists():
        return params

    results = load_tuning_results().set_index("Model")

    def value(model: str, column: str, fallback):
        selected = results.at[model, column]
        return fallback if pd.isna(selected) else selected

    def integer(model: str, column: str, fallback):
        return int(value(model, column, fallback))

    def optional_integer(model: str, column: str, fallback):
        selected = value(model, column, fallback)
        return None if pd.isna(selected) else int(selected)

    params.update(
        {
            "rf_n_estimators": integer("Random Forest", "n_estimators", params["rf_n_estimators"]),
            "rf_max_depth": optional_integer("Random Forest", "max_depth", params["rf_max_depth"]),
            "rf_min_samples_split": integer("Random Forest", "min_samples_split", params["rf_min_samples_split"]),
            "rf_min_samples_leaf": integer("Random Forest", "min_samples_leaf", params["rf_min_samples_leaf"]),
            "rf_max_features": value("Random Forest", "max_features", params["rf_max_features"]),
            "rf_max_leaf_nodes": optional_integer("Random Forest", "max_leaf_nodes", params["rf_max_leaf_nodes"]),
            "dt_max_depth": optional_integer("Decision Tree", "max_depth", params["dt_max_depth"]),
            "dt_min_samples_split": integer("Decision Tree", "min_samples_split", params["dt_min_samples_split"]),
            "dt_min_samples_leaf": integer("Decision Tree", "min_samples_leaf", params["dt_min_samples_leaf"]),
            "dt_criterion": value("Decision Tree", "criterion", params["dt_criterion"]),
            "dt_max_leaf_nodes": optional_integer("Decision Tree", "max_leaf_nodes", params["dt_max_leaf_nodes"]),
            "lr_c": float(value("Logistic Regression", "C", params["lr_c"])),
            "lr_max_iter": integer("Logistic Regression", "max_iter", params["lr_max_iter"]),
            "lr_solver": value("Logistic Regression", "solver", params["lr_solver"]),
            "lr_class_weight": value("Logistic Regression", "class_weight", params["lr_class_weight"]),
            "lr_tol": float(value("Logistic Regression", "tol", params["lr_tol"])),
            "knn_neighbors": integer("K-Nearest Neighbors", "n_neighbors", params["knn_neighbors"]),
            "knn_weights": value("K-Nearest Neighbors", "weights", params["knn_weights"]),
            "knn_p": integer("K-Nearest Neighbors", "p", params["knn_p"]),
            "knn_leaf_size": integer("K-Nearest Neighbors", "leaf_size", params["knn_leaf_size"]),
        }
    )
    params["smote_k_neighbors_by_model"].update(
        {
            model: integer(model, "k_neighbors", params["smote_k_neighbors_by_model"][model])
            for model in params["smote_k_neighbors_by_model"]
        }
    )
    return params

# Exclude Target (NObeyesdad) & MODEL_EXECUTED_COLUMNS (BMI) from model transformation
def get_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return input features used for model training and prediction."""
    return data.drop(columns=[TARGET, *MODEL_EXCLUDED_COLUMNS], errors="ignore")


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    # Binary categories are represented as 0/1 for training.
    binary_categories = {
        "Gender": ["Female", "Male"],
        "family_history_with_overweight": ["no", "yes"],
        "FAVC": ["no", "yes"],
        "SMOKE": ["no", "yes"],
        "SCC": ["no", "yes"],
    }
    binary_features = [
        column for column in binary_categories if column in features.columns
    ]

    # Multi-class lifestyle categories are one-hot encoded.
    one_hot_features = [
        column for column in ["CAEC", "CALC", "MTRANS"] if column in features.columns
    ]

    numeric = [
        column
        for column in features.select_dtypes(include=np.number).columns
        if column not in binary_features and column not in one_hot_features
    ]

    return ColumnTransformer(
        [
            # Standardize numeric features so they are on a comparable scale.
            ("numeric", StandardScaler(), numeric),
            (
                "binary",
                OrdinalEncoder(
                    categories=[binary_categories[column] for column in binary_features],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                binary_features,
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot_features),
        ]
    )


@st.cache_data(show_spinner=False)
def make_transformed_explorer_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return the feature table after the same transformations used for training."""
    features = get_model_features(data)
    preprocessor = make_preprocessor(features)
    transformed_features = preprocessor.fit_transform(features)
    if hasattr(transformed_features, "toarray"):
        transformed_features = transformed_features.toarray()

    transformed_data = pd.DataFrame(
        transformed_features,
        columns=preprocessor.get_feature_names_out(),
        index=data.index,
    )
    target_one_hot = pd.get_dummies(data[TARGET], prefix=TARGET, dtype=int)
    transformed_data = pd.concat([transformed_data, target_one_hot], axis=1)
    return transformed_data


def sidebar_tuning(params: dict) -> tuple[str, dict]:
    st.sidebar.header("Chart Data Source")

    source_option = st.sidebar.radio(
        "Display distributions based on:",
        options=[
            "Ground Truth (Actual Labels)",
            "Predicted: Random Forest",
            "Predicted: Decision Tree",
            "Predicted: Logistic Regression",
            "Predicted: K-Nearest Neighbors",
        ],
        index=0,
    )

    return source_option, params



@st.cache_data(show_spinner="Training and evaluating models with updated hyperparameters…")
def train_models(data: pd.DataFrame, params: dict):
    features = get_model_features(data)
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            max_features=params["rf_max_features"],
            max_leaf_nodes=params["rf_max_leaf_nodes"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=params["dt_max_depth"],
            min_samples_split=params["dt_min_samples_split"],
            min_samples_leaf=params["dt_min_samples_leaf"],
            criterion=params["dt_criterion"],
            max_leaf_nodes=params["dt_max_leaf_nodes"],
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            C=params["lr_c"],
            max_iter=params["lr_max_iter"],
            solver=params["lr_solver"],
            class_weight=params["lr_class_weight"],
            tol=params["lr_tol"],
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=params["knn_neighbors"],
            weights=params["knn_weights"],
            p=params["knn_p"],
            leaf_size=params["knn_leaf_size"],
        ),
    }
    fitted, summaries = {}, []
    for name, classifier in models.items():
        steps = [("preprocess", make_preprocessor(features))]
        if params["smote_enabled"]:
            # SMOTE is fitted only on x_train/y_train inside this pipeline.
            steps.append(
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        k_neighbors=params.get("smote_k_neighbors_by_model", {}).get(
                            name, params["smote_k_neighbors"]
                        ),
                    ),
                )
            )
        steps.append(("model", classifier))
        pipeline = ImbPipeline(steps)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted[name] = pipeline
        summaries.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, average="macro"),
                "Recall": recall_score(y_test, predictions, average="macro"),
                "F1-score": f1_score(y_test, predictions, average="macro"),
                "Predictions": predictions,
            }
        )
    return fitted, pd.DataFrame(summaries), y_test, x_test


def render_data_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Data filters")
    selected_classes = st.sidebar.multiselect(
        "Obesity level", sorted(data[TARGET].unique()), default=sorted(data[TARGET].unique())
    )
    selected_genders = st.sidebar.multiselect(
        "Gender", sorted(data["Gender"].unique()), default=sorted(data["Gender"].unique())
    )
    age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max), 1)
    
    return data[
        data[TARGET].isin(selected_classes)
        & data["Gender"].isin(selected_genders)
        & data["Age"].between(*age_range)
    ].copy()


def section_overview(data: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    target_one_hot_columns = [
        column for column in data.columns if column.startswith(f"{TARGET}_")
    ]
    obesity_class_count = (
        data[TARGET].nunique() if TARGET in data.columns else len(target_one_hot_columns)
    )
    target_display = TARGET if TARGET in data.columns else f"{TARGET} (one-hot)"
    target_column_count = 1 if TARGET in data.columns else len(target_one_hot_columns)
    a, b, c, d = st.columns(4)
    a.metric("Total records", f"{len(data):,}")
    b.metric("Input features", data.shape[1] - target_column_count)
    c.metric("Obesity classes", obesity_class_count)
    d.metric("Target variable", target_display)
    st.subheader("Dataset Preview")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button(
        "Download dataset as CSV",
        data.to_csv(index=False).encode("utf-8"),
        "obesity_data.csv",
        "text/csv",
        key=f"download_{title}",
    )


def section_charts(subset: pd.DataFrame, models: dict, source_option: str) -> None:
    if subset.empty:
        st.warning("No records match the selected filters.")
        return

    chart_subset = subset.copy()
    target_col = TARGET

    if source_option != "Ground Truth (Actual Labels)":
        chosen_model_name = source_option.replace("Predicted: ", "")
        model_pipeline = models[chosen_model_name]
        chart_subset["Predicted_Target"] = model_pipeline.predict(
            get_model_features(chart_subset)
        )
        target_col = "Predicted_Target"

    counts = chart_subset[target_col].value_counts()

    overview_tab, demographics_tab, habits_tab, body_tab = st.tabs(
        ["Overview", "Demographics & Mobility", "Diet & Daily Habits", "Body & Physical Activity"]
    )

    overview_tab.subheader("Obesity-level distribution")
    bar_col, pie_col = overview_tab.columns(2)

    with bar_col:
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
                legend=alt.Legend(
                    title="Obesity level", 
                    orient="right", 
                    symbolType="square"
                ),
                scale=alt.Scale(range=MACARON_COLORS),
            ),
            tooltip=["Obesity level", "Count", alt.Tooltip("Percent:Q", format=".1%")],
        ).properties(height=340)
        
        pie_labels = pie_base.mark_text(radius=115, size=11, color="white", fontWeight="bold").encode(
            text=alt.Text("Percent:Q", format=".1%"),
        )
        
        final_pie = (pie_chart + pie_labels).configure_legend(
            symbolSize=100,
            symbolStrokeWidth=0
        )
        
        st.altair_chart(final_pie, width="stretch")

    left = overview_tab.container()
    right = habits_tab.container()
    with left:
        st.subheader("Numeric variable distribution")
        numeric_choices = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        variable = st.selectbox("Choose a variable", numeric_choices)
        histogram = alt.Chart(chart_subset).mark_bar(opacity=0.85).encode(
            x=alt.X(f"{variable}:Q", bin=alt.Bin(maxbins=24)),
            y=alt.Y("count():Q", title="People"),
            color=alt.Color(
                f"{target_col}:N",
                title="Obesity level",
                scale=alt.Scale(range=MACARON_COLORS),
            ),
            tooltip=[alt.Tooltip("count():Q", title="People")],
        ).properties(height=300)
        st.altair_chart(histogram, width="stretch")

    with right:
        st.subheader("Alcohol consumption levels by gender")
        calc_order = [level for level in ["no", "Sometimes", "Frequently", "Always"] if level in chart_subset["CALC"].unique()]
        calc_counts = chart_subset.groupby(["Gender", "CALC"]).size().reset_index(name="Count")
        alcohol_chart = alt.Chart(calc_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Gender:N", title="Gender", sort=GENDER_ORDER),
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

    overview_tab.subheader("Lifestyle factor by obesity level")
    lifestyle = overview_tab.selectbox(
        "Choose a lifestyle factor", ["CAEC", "CALC", "FAVC", "SMOKE", "SCC", "MTRANS", "family_history_with_overweight"]
    )
    grouped = chart_subset.groupby([lifestyle, target_col]).size().reset_index(name="Count")
    lifestyle_chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X(f"{lifestyle}:N", title=lifestyle),
        y=alt.Y("Count:Q", stack="normalize", title="Proportion"),
        color=alt.Color(
            f"{target_col}:N",
            title="Obesity level",
            scale=alt.Scale(range=MACARON_COLORS),
        ),
        tooltip=[lifestyle, target_col, "Count"],
    ).properties(height=340)
    overview_tab.altair_chart(lifestyle_chart, width="stretch")

    transport_col, scatter_col = demographics_tab.columns(2)
    with transport_col:
        st.subheader("Transportation methods by gender")
        transport_counts = chart_subset.groupby(["MTRANS", "Gender"]).size().reset_index(name="Count")
        transport_chart = alt.Chart(transport_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("MTRANS:N", title="Transportation mode (MTRANS)", sort="-y"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "Gender:N",
                title="Gender",
                sort=GENDER_ORDER,
                scale=alt.Scale(domain=GENDER_ORDER, range=GENDER_PALETTE),
            ),
            xOffset=alt.XOffset("Gender:N", sort=GENDER_ORDER),
            tooltip=["MTRANS", "Gender", "Count"],
        ).properties(height=520)
        st.altair_chart(transport_chart, width="stretch")

    with scatter_col:
        st.subheader("Age vs. weight by gender")
        scatter_chart = alt.Chart(chart_subset).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("Age:Q", title="Age"),
            y=alt.Y("Weight:Q", title="Weight (kilogram)"),
            color=alt.Color(
                "Gender:N",
                title="Gender",
                sort=GENDER_ORDER,
                scale=alt.Scale(domain=GENDER_ORDER, range=GENDER_PALETTE),
                legend=alt.Legend(symbolType="square"),
            ),
            tooltip=["Gender", "Age", "Weight", target_col],
        ).properties(height=520).interactive()
        st.altair_chart(scatter_chart, width="stretch")

    gender_col, facet_col = demographics_tab.columns(2)
    with gender_col:
        st.subheader("Obesity levels by gender")
        obesity_gender_counts = chart_subset.groupby([target_col, "Gender"]).size().reset_index(name="Count")
        obesity_gender_chart = alt.Chart(obesity_gender_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(f"{target_col}:N", title="Obesity level", sort="-y"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "Gender:N",
                title="Gender",
                sort=GENDER_ORDER,
                scale=alt.Scale(domain=GENDER_ORDER, range=GENDER_PALETTE),
            ),
            xOffset=alt.XOffset("Gender:N", sort=GENDER_ORDER),
            tooltip=[target_col, "Gender", "Count"],
        ).properties(height=520)
        st.altair_chart(obesity_gender_chart, width="stretch")

    with facet_col:
        st.subheader("Age distribution by gender and smoking status")
        facet_chart = alt.Chart(chart_subset).mark_bar(opacity=0.85).encode(
            x=alt.X("Age:Q", bin=alt.Bin(maxbins=15), title="Age"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("Gender:N", legend=None, sort=GENDER_ORDER, scale=alt.Scale(domain=GENDER_ORDER, range=GENDER_PALETTE)),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        ).properties(width=180, height=220).facet(
            row=alt.Row("Gender:N", title=None, sort=GENDER_ORDER),
            column=alt.Column("SMOKE:N", title="Smokes", sort=YES_NO_ORDER),
        )
        st.altair_chart(facet_chart)

    caec_col, calc_col = habits_tab.columns(2)
    with caec_col:
        st.subheader("Eating between meals (CAEC) by gender")
        caec_counts = chart_subset.groupby(["CAEC", "Gender"]).size().reset_index(name="Count")
        caec_chart = alt.Chart(caec_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("CAEC:N", title="Do you eat any food between meals", sort="-y"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "Gender:N",
                title="Gender",
                sort=GENDER_ORDER,
                scale=alt.Scale(domain=GENDER_ORDER, range=GENDER_PALETTE),
            ),
            xOffset=alt.XOffset("Gender:N", sort=GENDER_ORDER),
            tooltip=["CAEC", "Gender", "Count"],
        ).properties(height=420)
        st.altair_chart(caec_chart, width="stretch")

    with calc_col:
        st.subheader("Alcohol consumption (CALC) by family history")
        calc_family_counts = chart_subset.groupby(["CALC", "family_history_with_overweight"]).size().reset_index(name="Count")
        calc_family_chart = alt.Chart(calc_family_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("CALC:N", title="How often do you drink alcohol", sort="-y"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "family_history_with_overweight:N",
                title="Family history with overweight",
                sort=YES_NO_ORDER,
                scale=alt.Scale(domain=YES_NO_ORDER, range=YES_NO_PALETTE),
            ),
            xOffset=alt.XOffset("family_history_with_overweight:N", sort=YES_NO_ORDER),
            tooltip=["CALC", "family_history_with_overweight", "Count"],
        ).properties(height=420)
        st.altair_chart(calc_family_chart, width="stretch")

    body_tab.subheader("Physical activity levels across weight categories")
    weight_order = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]
    weight_order = [level for level in weight_order if level in chart_subset[target_col].unique()]
    activity_means = chart_subset.groupby(target_col)["FAF"].mean().reindex(weight_order).reset_index(name="Average FAF")
    activity_chart = alt.Chart(activity_means).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X(f"{target_col}:N", title=None, sort=weight_order),
        y=alt.Y("Average FAF:Q", title="Average physical activity score (FAF)"),
        color=alt.Color(
            f"{target_col}:N",
            legend=None,
            sort=weight_order,
            scale=alt.Scale(domain=weight_order, range=MACARON_COLORS[: len(weight_order)]),
        ),
        tooltip=[target_col, alt.Tooltip("Average FAF:Q", format=".2f")],
    ).properties(height=380)
    body_tab.altair_chart(activity_chart, width="stretch")

    bmi_col = body_tab.container()
    favc_col = habits_tab.container()
    with bmi_col:
        st.subheader("BMI distribution across weight categories by gender")
        bmi_chart = alt.Chart(chart_subset).mark_boxplot(size=28, outliers=True).encode(
            x=alt.X(f"{target_col}:N", title="Weight category", sort=weight_order),
            y=alt.Y("BMI:Q", title="Body Mass Index (BMI)"),
            color=alt.Color(
                "Gender:N",
                title="Gender",
                sort=GENDER_ORDER,
                scale=alt.Scale(domain=GENDER_ORDER, range=GENDER_PALETTE),
                legend=alt.Legend(symbolType="square", symbolSize=30),
            ),
            xOffset=alt.XOffset("Gender:N", sort=GENDER_ORDER),
            tooltip=["Gender", target_col, alt.Tooltip("BMI:Q", format=".1f")],
        ).properties(height=420)
        st.altair_chart(bmi_chart, width="stretch")

    with favc_col:
        st.subheader("High caloric food habit by weight category")
        favc_means = (
            chart_subset.groupby([target_col, "FAVC"])["FAF"]
            .mean()
            .reindex(weight_order, level=0)
            .reset_index(name="Average FAF")
        )
        favc_chart = alt.Chart(favc_means).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X(f"{target_col}:N", title="Weight category", sort=weight_order),
            y=alt.Y("Average FAF:Q", title="Average physical activity score (FAF)"),
            color=alt.Color(
                "FAVC:N",
                title="High calorie intake (FAVC)",
                sort=YES_NO_ORDER,
                scale=alt.Scale(domain=YES_NO_ORDER, range=YES_NO_PALETTE),
            ),
            xOffset=alt.XOffset("FAVC:N", sort=YES_NO_ORDER),
            tooltip=[target_col, "FAVC", alt.Tooltip("Average FAF:Q", format=".2f")],
        ).properties(height=420)
        st.altair_chart(favc_chart, width="stretch")

    # Hydration (CH2O) and Meal Frequency (NCP) by Obesity Category
    habits_tab.subheader("Hydration (CH2O) and Meal Frequency (NCP) by Obesity Category")

    ch2o_ncp_df = chart_subset.copy()
    ch2o_ncp_df["NCP_Group"] = ch2o_ncp_df["NCP"].round().astype(int).astype(str) + " Meals/Day"

    grouped_df = (
        ch2o_ncp_df.groupby(["NCP_Group", target_col])["CH2O"]
        .mean()
        .reset_index(name="Average Water Intake (CH2O)")
    )

    ncp_order = [f"{i} Meals/Day" for i in range(1, 5)]
    ncp_order = [lbl for lbl in ncp_order if lbl in grouped_df["NCP_Group"].unique()]

    hydration_chart = (
        alt.Chart(grouped_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("NCP_Group:N", title="Daily Meals (NCP)", sort=ncp_order, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("Average Water Intake (CH2O):Q", title="Average Water Intake (CH2O)"),
            color=alt.Color(
                f"{target_col}:N",
                title="Obesity Level",
                scale=alt.Scale(range=MACARON_COLORS),
            ),
            xOffset=alt.XOffset(f"{target_col}:N"),
            tooltip=[
                "NCP_Group",
                target_col,
                alt.Tooltip("Average Water Intake (CH2O):Q", format=".2f"),
            ],
        )
        .properties(height=450)
    )

    habits_tab.altair_chart(hydration_chart, width="stretch")


def section_models(data: pd.DataFrame, params: dict) -> tuple[dict, pd.DataFrame]:
    models, results, y_test, _ = train_models(data, params)

    ranking = results[
        ["Model", "Accuracy", "F1-score", "Precision", "Recall"]
    ].sort_values("F1-score", ascending=False)

    st.subheader("Model comparison")
    st.caption(
        "All models use validation-selected hyperparameters and the same stratified "
        "80/20 split (random state 42). BMI is excluded from training; SMOTE is "
        "applied to training data only."
    )
    if TUNING_RESULTS_PATH.exists():
        st.subheader("Exhaustive hyperparameter tuning results")
        st.caption("Best macro-F1 result from every tested parameter combination (3-fold CV).")
        st.dataframe(
            load_tuning_results()
            .sort_values("Best CV macro F1", ascending=False)
            .style.format({"Best CV macro F1": "{:.2%}"}),
            width="stretch",
            hide_index=True,
        )
    st.subheader("Hold-out test set model performance")
    st.caption("Metrics from the same stratified 20% test set, ranked by macro F1-score.")
    st.dataframe(
        ranking.style.format(
            {
                "Accuracy": "{:.2%}",
                "F1-score": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    performance = ranking.melt("Model", var_name="Metric", value_name="Score")
    chart = (
        alt.Chart(performance)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort="-y"),
            y=alt.Y(
                "Score:Q",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("Metric:N", scale=alt.Scale(range=MACARON_COLORS)),
            xOffset="Metric:N",
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2%")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")

    chosen = st.selectbox("Inspect model", ranking["Model"].tolist())
    predicted = results.loc[results["Model"] == chosen, "Predictions"].iloc[0]

    report = pd.DataFrame(
        classification_report(y_test, predicted, output_dict=True)
    ).T

    class_report = report.drop(
        index=["accuracy", "macro avg", "weighted avg"]
    ).copy()
    class_report["Accuracy"] = accuracy_score(y_test, predicted)

    class_report = class_report.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-score",
        }
    )

    class_report = class_report[
        ["Accuracy", "F1-score", "Precision", "Recall"]
    ]

    st.subheader(f"{chosen}: class-level metrics")
    st.dataframe(
        class_report.style.format(
            {
                "Accuracy": "{:.2%}",
                "F1-score": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
            }
        ),
        width="stretch",
    )

    matrix = confusion_matrix(
        y_test, predicted, labels=sorted(data[TARGET].unique())
    )
    labels = sorted(data[TARGET].unique())
    matrix_df = (
        pd.DataFrame(matrix, index=labels, columns=labels)
        .rename_axis("Actual")
        .reset_index()
        .melt("Actual", var_name="Predicted", value_name="Count")
    )
    color_scale = alt.Scale(range=MACARON_GRADIENT)
    heatmap_cells = (
        alt.Chart(matrix_df)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", sort=labels, title="Predicted Label"),
            y=alt.Y("Actual:N", sort=labels, title="True Label"),
            color=alt.Color("Count:Q", scale=color_scale, title="Count"),
            tooltip=["Actual", "Predicted", "Count"],
        )
    )
    heatmap_values = (
        alt.Chart(matrix_df)
        .mark_text(fontSize=14)
        .encode(
            x=alt.X("Predicted:N", sort=labels),
            y=alt.Y("Actual:N", sort=labels),
            text=alt.Text("Count:Q", format="d"),
            color=alt.condition(
                alt.datum.Count > matrix.max() * 0.5,
                alt.value("white"),
                alt.value("#1f1f1f"),
            ),
        )
    )
    heatmap = (heatmap_cells + heatmap_values).properties(
        height=500, title=f"Confusion Matrix - {chosen}"
    )
    st.altair_chart(heatmap, width="stretch")

    return models, results


def section_prediction(data: pd.DataFrame, params: dict) -> None:
    models, results, _, _ = train_models(data, params)
    
    best_model = results.sort_values("F1-score", ascending=False).iloc[0]["Model"]
    
    st.subheader("Predict an obesity level")
    selected_model = st.selectbox("Prediction model", list(models), index=list(models).index(best_model))
    st.caption(f"Recommended by F1-score on the hold-out test set: {best_model}.")
    
    features = get_model_features(data)
    values = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for i, column in enumerate(features.columns):
            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(features[column]):
                    minimum, maximum = float(features[column].min()), float(features[column].max())
                    default = float(features[column].median())
                    
                    if column == "Age":
                        values[column] = st.number_input(
                            column,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default),
                            step=1,
                            format="%d"
                        )
                    else:
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
    st.title("Obesity Levels Prediction")
    try:
        data = load_data()
        raw_data = load_raw_data()
    except Exception as error:
        st.error(f"Could not load the CSV: {error}")
        st.stop()
    if TARGET not in data.columns:
        st.error(f"The CSV must include a '{TARGET}' target column.")
        st.stop()
    tuned_params = load_tuned_params()

    # Apply global theme overrides and transform st.segmented_control into underlined tabs
    st.markdown(
        """
        <style>
        /* --- Streamlit Global Theme Variables Overrides --- */
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stSidebar"],
        .stApp {
            --primary-color: #000000 !important;
            --text-color: #000000 !important;
        }

        /* Hide st.segmented_control border and pill backgrounds to match native tab style */
        [data-testid="stSegmentedControl"] {
            border: none !important;
            border-bottom: 2px solid #E0E0E0 !important;
            border-radius: 0px !important;
            gap: 24px !important;
            background: transparent !important;
            padding: 0px !important;
            margin-bottom: 1rem !important;
        }

        /* Style individual control items as tab text */
        [data-testid="stSegmentedControl"] button {
            border: none !important;
            background: transparent !important;
            border-radius: 0px !important;
            color: #666666 !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            padding: 8px 0px !important;
            box-shadow: none !important;
            border-bottom: 2px solid transparent !important;
            margin-bottom: -2px !important;
        }

        /* Selected Tab Underline Accent */
        [data-testid="stSegmentedControl"] button[aria-selected="true"] {
            color: #000000 !important;
            font-weight: 800 !important;
            border-bottom: 3px solid #000000 !important;
            background: transparent !important;
        }

        /* Hover behavior for unselected tabs */
        [data-testid="stSegmentedControl"] button:hover {
            color: #000000 !important;
        }

        /* Slider Track, Thumbs & Inputs */
        [data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"],
        [data-testid="stSlider"] div[role="slider"] ~ div,
        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background-color: #000000 !important;
            background: #000000 !important;
        }

        [data-testid="stSlider"] [role="slider"],
        [data-testid="stSlider"] div[role="slider"] {
            background-color: #000000 !important;
            border-color: #000000 !important;
            box-shadow: none !important;
        }

        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"],
        [data-testid="stSlider"] [data-testid="stThumbValue"],
        [data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p,
        [data-testid="stSlider"] span {
            color: #000000 !important;
        }

        /* Primary Buttons */
        button[kind="primary"] {
            background-color: #000000 !important;
            border-color: #000000 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render native segmented control styled like underlined tabs
    selected_tab = st.segmented_control(
        "Navigation",
        options=["Data explorer", "Charts", "Model comparison", "Make a prediction"],
        default="Data explorer",
        label_visibility="collapsed",
    )

    # Only charts need interactive sidebar controls.
    if selected_tab != "Charts":
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Conditionally execute page blocks
    if selected_tab == "Data explorer":
        raw_tab, prepared_tab, transformed_tab = st.tabs(
            ["Raw dataset", "Cleaned dataset", "Training transformation"]
        )
        with raw_tab:
            section_overview(raw_data, "Raw dataset")
        with prepared_tab:
            section_overview(
                data,
                "Cleaned dataset: duplicates removed, outliers capped, BMI added",
            )
        with transformed_tab:
            st.caption(
                "BMI is excluded from training. Binary fields are 0/1; CAEC, CALC, "
                "MTRANS, and NObeyesdad are one-hot encoded."
            )
            section_overview(
                make_transformed_explorer_data(data),
                "Dataset after training transformation",
            )

    elif selected_tab == "Charts":
        source_option, params = sidebar_tuning(tuned_params)
        subset = render_data_filters(data)
        models, _, _, _ = train_models(data, params)
        section_charts(subset, models, source_option)

    elif selected_tab == "Model comparison":
        section_models(data, tuned_params)

    elif selected_tab == "Make a prediction":
        section_prediction(data, tuned_params)


if __name__ == "__main__":
    main()
