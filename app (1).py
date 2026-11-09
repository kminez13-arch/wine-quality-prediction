import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Required input columns
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Prediction System")
st.write(
    "Upload a wine dataset to predict whether each wine is "
    "**Good Quality** or **Needs Improvement**."
)

# Dataset information
st.header("1. Dataset Information")

try:
    dataset = pd.read_csv("wine_quality.csv")
    st.write("Training dataset overview:")
    st.write(f"Number of rows: {dataset.shape[0]}")
    st.write(f"Number of columns: {dataset.shape[1]}")
    st.dataframe(dataset.head())
except FileNotFoundError:
    st.info("The original dataset file is not available in this app folder.")

# Upload file
st.header("2. Upload a CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file containing wine measurements",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(input_df.head())

        # Check required columns
        missing_columns = [
            column for column in FEATURE_COLUMNS
            if column not in input_df.columns
        ]

        if missing_columns:
            st.error(
                "Your CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )
        else:
            # Select only model input features.
            # The original quality column is not used for prediction.
            prediction_data = input_df[FEATURE_COLUMNS].copy()

            # Convert feature values to numeric
            for column in FEATURE_COLUMNS:
                prediction_data[column] = pd.to_numeric(
                    prediction_data[column], errors="coerce"
                )

            if prediction_data.isnull().all(axis=1).any():
                st.warning(
                    "Some rows contain no usable measurement values. "
                    "Check your uploaded CSV."
                )

            model = load_model()

            # Make predictions
            predictions = model.predict(prediction_data)

            # Add readable prediction labels
            results_df = input_df.copy()
            results_df["Predicted Quality"] = [
                "Good Quality" if prediction == 1
                else "Needs Improvement"
                for prediction in predictions
            ]

            st.header("3. Prediction Results")
            st.dataframe(results_df)

            # Prediction summary
            good_count = sum(predictions == 1)
            improvement_count = sum(predictions == 0)

            col1, col2 = st.columns(2)
            col1.metric("Good Quality", int(good_count))
            col2.metric("Needs Improvement", int(improvement_count))

            # Download results
            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name="wine_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"An error occurred while processing the file: {error}")

st.caption(
    "Note: This system provides preliminary predictions based on "
    "the trained model and the supplied wine measurements."
)
