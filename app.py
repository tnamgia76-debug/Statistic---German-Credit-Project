import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

model = joblib.load(
    ARTIFACT_DIR / "final_model.joblib"
)

metadata = joblib.load(
    ARTIFACT_DIR / "model_metadata.joblib"
)


FINAL_FEATURES = metadata["features"]
FINAL_THRESHOLD = metadata["threshold"]

NUMERIC_FEATURES = metadata[
    "numeric_features"
]

CATEGORICAL_FEATURES = metadata[
    "categorical_features"
]

CATEGORY_VALUES = metadata[
    "category_values"
]

NUMERIC_DEFAULTS = metadata[
    "numeric_defaults"
]


# =========================================================
# HEADER
# =========================================================

st.title("🏦 Credit Risk Assessment")

st.caption(
    "German Credit Risk Classification — Academic Demonstration"
)

st.info(
    "This application is an academic risk-screening demo. "
    "It is not intended to make real-world lending decisions."
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "👤 Single Applicant",
    "📁 Batch Prediction",
    "ℹ️ Model Information"
])


# =========================================================
# TAB 1 — SINGLE APPLICANT
# =========================================================

with tab1:

    st.subheader(
        "Applicant Information"
    )

    st.write(
        "Enter the applicant's information below "
        "and run the trained credit-risk model."
    )

    applicant_data = {}

    with st.form("credit_form"):

        col1, col2 = st.columns(2)

        for index, feature in enumerate(
            FINAL_FEATURES
        ):

            current_column = (
                col1
                if index % 2 == 0
                else col2
            )

            with current_column:

                # Numeric feature
                if feature in NUMERIC_FEATURES:

                    applicant_data[feature] = (
                        st.number_input(
                            label=feature.replace(
                                "_", " "
                            ).title(),

                            value=float(
                                NUMERIC_DEFAULTS[
                                    feature
                                ]
                            )
                        )
                    )

                # Categorical feature
                else:

                    applicant_data[feature] = (
                        st.selectbox(
                            label=feature.replace(
                                "_", " "
                            ).title(),

                            options=CATEGORY_VALUES[
                                feature
                            ]
                        )
                    )

        submitted = st.form_submit_button(
            "Assess Credit Risk",
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        applicant_df = pd.DataFrame(
            [applicant_data]
        )

        applicant_df = applicant_df[
            FINAL_FEATURES
        ]


        probability_bad = (
            model.predict_proba(
                applicant_df
            )[0, 1]
        )


        prediction_bad = int(
            probability_bad
            >= FINAL_THRESHOLD
        )


        st.divider()

        st.subheader(
            "Risk Assessment Result"
        )


        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )


        with result_col1:

            if prediction_bad == 1:

                st.error(
                    "BAD CREDIT RISK"
                )

            else:

                st.success(
                    "GOOD CREDIT RISK"
                )


        with result_col2:

            st.metric(
                "Probability of Bad",
                f"{probability_bad:.2%}"
            )


        with result_col3:

            st.metric(
                "Decision Threshold",
                f"{FINAL_THRESHOLD:.2%}"
            )


        st.progress(
            float(probability_bad)
        )


        if prediction_bad == 1:

            st.warning(
                "The predicted probability of Bad "
                "is greater than or equal to the "
                "selected decision threshold."
            )

        else:

            st.success(
                "The predicted probability of Bad "
                "is below the selected decision "
                "threshold."
            )


# =========================================================
# TAB 2 — BATCH PREDICTION
# =========================================================

with tab2:

    st.subheader(
        "Batch Credit Risk Prediction"
    )

    st.write(
        "Upload a CSV file containing applicant "
        "information to score multiple applications."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        batch_data = pd.read_csv(
            uploaded_file
        )


        st.write(
            "Uploaded data preview"
        )

        st.dataframe(
            batch_data.head()
        )


        missing_features = [
            feature
            for feature in FINAL_FEATURES
            if feature not in batch_data.columns
        ]


        if missing_features:

            st.error(
                "Missing required features: "
                + ", ".join(
                    missing_features
                )
            )

        else:

            X_batch = batch_data[
                FINAL_FEATURES
            ].copy()


            batch_probability = (
                model.predict_proba(
                    X_batch
                )[:, 1]
            )


            batch_prediction = (
                batch_probability
                >= FINAL_THRESHOLD
            ).astype(int)


            results = batch_data.copy()

            results[
                "probability_bad"
            ] = batch_probability

            results[
                "predicted_bad"
            ] = batch_prediction

            results[
                "credit_risk"
            ] = results[
                "predicted_bad"
            ].map({
                0: "Good",
                1: "Bad"
            })


            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                results
            )


            bad_count = int(
                batch_prediction.sum()
            )

            good_count = (
                len(batch_prediction)
                - bad_count
            )


            col1, col2, col3 = (
                st.columns(3)
            )


            col1.metric(
                "Applications",
                len(results)
            )

            col2.metric(
                "Predicted Good",
                good_count
            )

            col3.metric(
                "Predicted Bad",
                bad_count
            )


            csv_output = (
                results
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )


            st.download_button(
                label="Download Predictions",
                data=csv_output,
                file_name=(
                    "credit_risk_predictions.csv"
                ),
                mime="text/csv"
            )


# =========================================================
# TAB 3 — MODEL INFORMATION
# =========================================================

with tab3:

    st.subheader(
        "Model Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Model",
            metadata["model_name"]
        )

        st.metric(
            "Selected Features",
            len(FINAL_FEATURES)
        )


    with col2:

        st.metric(
            "Decision Threshold",
            f"{FINAL_THRESHOLD:.2%}"
        )

        st.metric(
            "Business Cost",
            "5 × FN + FP"
        )


    st.write(
        "**Model parameters**"
    )

    st.json(
        metadata["params"]
    )


    st.write(
        "**Selected features**"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": FINAL_FEATURES
        }),
        hide_index=True
    )


    st.divider()

    st.caption(
        "The model was developed for an academic "
        "German Credit classification project. "
        "The prediction should be interpreted as "
        "a model-based risk screening result rather "
        "than an automated lending decision."
    )