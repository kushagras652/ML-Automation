import streamlit as st
import pandas as pd
import tempfile
from graph import build_graph

st.set_page_config(
    page_title="Agentic AutoML",
    layout="wide"
)

st.title("🧠 Agentic AutoML & EDA System")
st.caption("Upload a CSV → Get EDA, ML models, and AI insights")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    if st.button("🚀 Run Analysis"):
        with st.spinner("Running agentic pipeline..."):
            app = build_graph()

            initial_state = {
                "csv_path": csv_path
            }

            final_state = app.invoke(initial_state)

        # --- Results ---
        st.success("Analysis completed!")

        # Dataset Info
        st.subheader("📊 Dataset Summary")
        st.write("Shape:", final_state["eda_summary"]["shape"])
        st.write("Target:", final_state["target"])
        st.write("Task Type:", final_state["task_type"])

        # Missing Values
        st.subheader("❗ Missing Values")
        st.json(final_state["eda_summary"]["missing_values"])

        # Class Balance (if classification)
        if final_state["task_type"] == "classification":
            st.subheader("⚖️ Class Balance")
            st.bar_chart(final_state["eda_summary"]["class_balance"])

        # Model Leaderboard
        st.subheader("🏆 Model Performance")
        st.json(final_state["model_results"])

        st.markdown(f"### 🥇 Best Model: `{final_state['best_model']}`")

        # Final Insights
        st.subheader("🧠 AI-Generated Insights")
        st.write(final_state["final_insights"])
