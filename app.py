
# ============================================================
# UK ROAD ACCIDENT SEVERITY PREDICTION DASHBOARD
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UK Road Accident ML Dashboard",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# 2. ROADINSIGHTS LOGO AND DASHBOARD HEADER
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "roadinsights_logo.png"

# Reduce unnecessary spacing at top of dashboard
st.markdown("""
<style>
.block-container {
    padding-top: 0.75rem;
    padding-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Compact horizontal layout
logo_col, title_col = st.columns(
    [1, 4],
    vertical_alignment="center"
)

with logo_col:
    if os.path.isfile(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            st.image(f.read(), width=200)
    else:
        st.error("RoadInsights logo not found")
        st.caption(LOGO_PATH)

with title_col:
    st.title("UK Road Accident Severity Prediction Dashboard")

    st.markdown(
        "**RoadInsights | Data Driven Safer Roads**"
    )

    st.caption(
        "Interactive dashboard using the "
        "B5 Weighted Gradient-Boosted Trees model "
        "to predict Low and High road accident severity."
    )

st.divider()

# ============================================================
# 3. LOAD DASHBOARD DATA
# ============================================================

DATA_PATH = BASE_DIR / "interactive_dashboard_data.parquet"


@st.cache_data
def load_data():

    df = pd.read_parquet(DATA_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    return df


df = load_data()


# ============================================================
# 4. SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")


# ------------------------------------------------------------
# Year
# ------------------------------------------------------------

years = sorted(
    df["Year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

selected_years = st.sidebar.multiselect(
    "Year",
    options=years,
    default=years
)


# ------------------------------------------------------------
# Road Type
# ------------------------------------------------------------

road_types = sorted(
    df["Road_Type_Grouped"]
    .dropna()
    .unique()
    .tolist()
)

selected_roads = st.sidebar.multiselect(
    "Road Type",
    options=road_types,
    default=road_types
)


# ------------------------------------------------------------
# Speed Category
# ------------------------------------------------------------

speed_categories = sorted(
    df["Speed_Category"]
    .dropna()
    .unique()
    .tolist()
)

selected_speeds = st.sidebar.multiselect(
    "Speed Category",
    options=speed_categories,
    default=speed_categories
)


# ------------------------------------------------------------
# Area Type
# ------------------------------------------------------------

area_types = sorted(
    df["Area_Type"]
    .dropna()
    .unique()
    .tolist()
)

selected_areas = st.sidebar.multiselect(
    "Area Type",
    options=area_types,
    default=area_types
)


# ------------------------------------------------------------
# Actual Severity
# ------------------------------------------------------------

severity_classes = sorted(
    df["Actual_Severity"]
    .dropna()
    .unique()
    .tolist()
)

selected_severity = st.sidebar.multiselect(
    "Actual Severity",
    options=severity_classes,
    default=severity_classes
)


# ============================================================
# 5. APPLY FILTERS
# ============================================================

filtered = df[
    df["Year"].isin(selected_years)
& df["Road_Type_Grouped"].isin(selected_roads)
& df["Speed_Category"].isin(selected_speeds)
& df["Area_Type"].isin(selected_areas)
& df["Actual_Severity"].isin(selected_severity)
].copy()


if filtered.empty:

    st.warning(
        "No collisions match the selected filters."
    )

    st.stop()


# ============================================================
# 6. CALCULATE PERFORMANCE METRICS
# ============================================================

total = len(filtered)

actual_high = int(
    (filtered["Actual_Class"] == 1).sum()
)

predicted_high = int(
    (filtered["prediction"] == 1).sum()
)


tp = int(
    (
        (filtered["Actual_Class"] == 1)
& (filtered["prediction"] == 1)
    ).sum()
)

tn = int(
    (
        (filtered["Actual_Class"] == 0)
& (filtered["prediction"] == 0)
    ).sum()
)

fp = int(
    (
        (filtered["Actual_Class"] == 0)
& (filtered["prediction"] == 1)
    ).sum()
)

fn = int(
    (
        (filtered["Actual_Class"] == 1)
& (filtered["prediction"] == 0)
    ).sum()
)


accuracy = (
    (tp + tn) / total
    if total > 0
    else np.nan
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else np.nan
)


# ============================================================
# 7. KPI CARDS
# ============================================================

st.subheader("Model Performance Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Collisions",
    f"{total:,}"
)

col2.metric(
    "Actual High Severity",
    f"{actual_high:,}"
)

col3.metric(
    "Predicted High Severity",
    f"{predicted_high:,}"
)


col4, col5, col6 = st.columns(3)

col4.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col5.metric(
    "High-Severity Recall",
    f"{recall:.2%}"
)

col6.metric(
    "High-Severity Precision",
    f"{precision:.2%}"
)


st.divider()


# ============================================================
# 8. DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Model Performance",
        "Collision Analysis",
        "Prediction Data"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        "Actual vs Predicted Severity"
    )

    severity_summary = pd.DataFrame(
        {
            "Actual": [
                total - actual_high,
                actual_high
            ],
            "Predicted": [
                total - predicted_high,
                predicted_high
            ]
        },
        index=[
            "Low Severity",
            "High Severity"
        ]
    )

    st.bar_chart(
        severity_summary
    )


    st.subheader(
        "Prediction Summary"
    )

    summary_table = pd.DataFrame(
        {
            "Measure": [
                "True Negatives",
                "False Positives",
                "False Negatives",
                "True Positives"
            ],
            "Count": [
                tn,
                fp,
                fn,
                tp
            ]
        }
    )

    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================

with tab2:

    st.subheader(
        "Confusion Matrix"
    )

    confusion = np.array(
        [
            [tn, fp],
            [fn, tp]
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    image = ax.imshow(
        confusion
    )

    labels = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                (
                    f"{labels[i][j]}\n"
                    f"{confusion[i, j]:,}"
                ),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold"
            )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Predicted Low",
            "Predicted High"
        ]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        [
            "Actual Low",
            "Actual High"
        ]
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "B5 Weighted GBT Confusion Matrix"
    )

    plt.colorbar(
        image,
        ax=ax,
        label="Number of Collisions"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    st.subheader(
        "High-Severity Prediction Probability"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        filtered[
            "High_Severity_Probability"
        ].dropna(),
        bins=30
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=2,
        label="Classification Threshold = 0.50"
    )

    ax.set_xlabel(
        "Predicted Probability of High Severity"
    )

    ax.set_ylabel(
        "Number of Collisions"
    )

    ax.set_title(
        "Distribution of High-Severity Prediction Probability"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# TAB 3 — COLLISION ANALYSIS
# ============================================================

with tab3:

    col_left, col_right = st.columns(2)


    # --------------------------------------------------------
    # Road type
    # --------------------------------------------------------

    with col_left:

        st.subheader(
            "Collisions by Road Type"
        )

        road_summary = (
            filtered[
                "Road_Type_Grouped"
            ]
            .value_counts()
            .rename_axis(
                "Road Type"
            )
            .reset_index(
                name="Collisions"
            )
        )

        st.bar_chart(
            road_summary,
            x="Road Type",
            y="Collisions"
        )


    # --------------------------------------------------------
    # Speed category
    # --------------------------------------------------------

    with col_right:

        st.subheader(
            "Collisions by Speed Category"
        )

        speed_summary = (
            filtered[
                "Speed_Category"
            ]
            .value_counts()
            .rename_axis(
                "Speed Category"
            )
            .reset_index(
                name="Collisions"
            )
        )

        st.bar_chart(
            speed_summary,
            x="Speed Category",
            y="Collisions"
        )


    # --------------------------------------------------------
    # Area type
    # --------------------------------------------------------

    st.subheader(
        "Collisions by Area Type"
    )

    area_summary = (
        filtered[
            "Area_Type"
        ]
        .value_counts()
        .rename_axis(
            "Area Type"
        )
        .reset_index(
            name="Collisions"
        )
    )

    st.bar_chart(
        area_summary,
        x="Area Type",
        y="Collisions"
    )


    # --------------------------------------------------------
    # Day of week
    # --------------------------------------------------------

    if "Day_Name" in filtered.columns:

        st.subheader(
            "Collisions by Day of Week"
        )

        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day_summary = (
            filtered[
                "Day_Name"
            ]
            .value_counts()
            .reindex(
                day_order,
                fill_value=0
            )
        )

        st.bar_chart(
            day_summary
        )


# ============================================================
# TAB 4 — PREDICTION DATA
# ============================================================

with tab4:

    st.subheader(
        "Filtered Prediction Records"
    )

    display_columns = [
        "Date",
        "Actual_Severity",
        "Predicted_Severity",
        "High_Severity_Probability",
        "Prediction_Result",
        "Road_Type_Grouped",
        "Speed_Category",
        "Area_Type",
        "Day_Name"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # CSV download
    # --------------------------------------------------------

    csv = filtered[
        display_columns
    ].to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="Download Filtered Results",
        data=csv,
        file_name=(
            "filtered_accident_predictions.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UK Road Accident Machine Learning Analysis | "
    "B5 Weighted Gradient-Boosted Trees | "
    "Final Test / Streaming Prediction Dataset"
)
