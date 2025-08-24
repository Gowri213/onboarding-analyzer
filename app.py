import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.cluster import KMeans

# 🔑 Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.title("Onboarding Analyzer Agent")
st.write("""
This intelligent agent connects with your product analytics CSV, identifies friction points in onboarding journeys,
clusters drop-off reasons, and recommends UX fixes weekly.
""")

# 📂 CSV upload
option = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])

df = None
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your onboarding CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_onboarding.csv")
    st.success("Loaded sample onboarding data ✅")

# -----------------------------
# Dynamic Column Mapping (for any dataset)
# -----------------------------
if df is not None:
    st.subheader("⚙️ Map Your Columns")
    student_col = st.selectbox("Select Student ID Column", df.columns)
    stage_col = st.selectbox("Select Stage Column", df.columns)
    status_col = st.selectbox("Select Status/Drop-off Column", df.columns)
    reason_col = st.selectbox("Select Reason Column (optional)", ["None"] + list(df.columns))
    date_col = st.selectbox("Select Date Column (optional, for trends)", ["None"] + list(df.columns))

    # -----------------------------
    # Sanity Checks
    # -----------------------------
    missing_cols = []
    for col in [student_col, stage_col]:
        if df[col].isnull().all():
            missing_cols.append(col)
    if missing_cols:
        st.error(f"The following required column(s) are empty: {', '.join(missing_cols)}")
        st.stop()

    if reason_col != "None" and df[reason_col].isnull().all():
        st.warning("The selected Reason column is empty. Reason-based clustering will be skipped.")

    if date_col != "None" and df[date_col].isnull().all():
        st.warning("The selected Date column is empty. Trend chart will be skipped.")

    # -----------------------------
    # KPI Cards for instant metrics
    # -----------------------------
    total_students = len(df)
    top_stage = df[stage_col].value_counts().idxmax()
    top_stage_count = df[stage_col].value_counts().max()
    top_stage_percent = round(top_stage_count / total_students * 100, 2)

    if reason_col != "None":
        top_reason = df[reason_col].value_counts().idxmax()
        top_reason_count = df[reason_col].value_counts().max()
    else:
        top_reason = "N/A"
        top_reason_count = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Overall Drop-off %", f"{top_stage_percent}%")
    col3.markdown(
        f"<div style='background-color:#F0F2F6;padding:10px;border-radius:5px;'><strong>Top Stage Drop-off:</strong><br>{top_stage} ({top_stage_count})</div>",
        unsafe_allow_html=True
    )
    col4.markdown(
        f"<div style='background-color:#F0F2F6;padding:10px;border-radius:5px;'><strong>Top Reason:</strong><br>{top_reason} ({top_reason_count})</div>",
        unsafe_allow_html=True
    )

    # -----------------------------
    # Stage-wise Drop-off
    # -----------------------------
    st.write("### Stage-wise Drop-Off Count")
    stage_counts = df[stage_col].value_counts()
    stage_percent = (stage_counts / total_students * 100).round(2)

    fig, ax = plt.subplots()
    stage_counts.plot(kind='bar', ax=ax, color='skyblue')
    for i, val in enumerate(stage_counts):
        ax.text(i, val + 0.5, f"{stage_percent[i]}%", ha='center')
    ax.set_ylabel("Number of Students")
    ax.set_xlabel("Stages")
    ax.set_title("Stage-wise Drop-Off")
    st.pyplot(fig)

    max_stage = stage_percent.idxmax()
    st.write(f"⚠️ Stage with highest drop-off: **{max_stage} ({stage_percent[max_stage]}%)**")

    # -----------------------------
    # Drop-off Clustering
    # -----------------------------
    st.write("### Drop-Off Clustering")
    df_cluster = df.copy()
    df_cluster['Stage_num'] = df_cluster[stage_col].astype('category').cat.codes
    k = min(3, len(df_cluster[stage_col].unique()))
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[['Stage_num']])
    st.write(df_cluster[[stage_col, 'Cluster']].head(10))

    fig2, ax2 = plt.subplots()
    for cluster in sorted(df_cluster['Cluster'].unique()):
        cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
        ax2.scatter(cluster_data.index, cluster_data['Stage_num'], label=f'Cluster {cluster}')
    ax2.set_xlabel("Student Index")
    ax2.set_ylabel("Stage (numeric)")
    ax2.set_title("Drop-off Clustering")
    ax2.legend()
    st.pyplot(fig2)

    # -----------------------------
    # Drop-off Reason Clustering
    # -----------------------------
    if reason_col != "None":
        st.write("### Drop-off Reason Clustering")
        reason_counts = df[reason_col].value_counts()
        st.write(reason_counts.head(10))
    else:
        st.info("No 'Reason' column selected. Add it for reason-based clustering.")

    # -----------------------------
    # Trend Over Time
    # -----------------------------
    if date_col != "None":
        st.write("### Drop-off Trend Over Time")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        trend = df.groupby([date_col, stage_col]).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(10,5))
        trend.plot(ax=ax3)
        ax3.set_ylabel("Number of Students")
        ax3.set_xlabel("Date")
        ax3.set_title("Drop-off Trend Over Time")
        st.pyplot(fig3)
    else:
        st.info("No 'Date' column selected. Trend chart requires a Date column.")

    # -----------------------------
    # Weekly AI Recommendations
    # -----------------------------
    st.write("### Weekly AI UX Recommendations & Friction Points Summary")
    if st.button("Get Weekly AI Suggestions"):
        prompt = f"""
        You are an AI product analytics agent. Analyze this onboarding dataset sample:
        {df.head(10).to_string(index=False)}

        Provide:
        1. Top 3 friction points causing drop-offs.
        2. Cluster drop-off reasons and patterns.
        3. Weekly actionable UX recommendations to reduce drop-offs.
        4. Stage-specific fixes to improve onboarding experience.
        Format the output clearly for easy reading.
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            st.write(response.text)

            # 🔹 Highlight top friction points with badges
            st.write("#### Top Friction Points & UX Fixes (Summary)")
            lines = response.text.split('\n')
            summary = []
            count = 0
            for line in lines:
                if any(keyword in line.lower() for keyword in ['friction', 'drop-off', 'ux', 'recommend']):
                    summary.append(line.strip())
                    count += 1
                if count >= 6:
                    break
            for item in summary:
                st.markdown(f"<span style='background-color:#FFD700;color:black;padding:4px;border-radius:4px'>{item}</span>", unsafe_allow_html=True)

            # 🔹 Download summary as CSV
            summary_df = pd.DataFrame({'Summary': summary})
            st.download_button("Download AI Summary CSV", summary_df.to_csv(index=False), file_name="ai_summary.csv")

        except Exception as e:
            st.error(f"Error: {e}")
