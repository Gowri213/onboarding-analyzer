import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.cluster import KMeans

# 🔑 Configure Gemini API
genai.configure(api_key="AIzaSyBBl4xVJ-DJfq_IPJhcJnLFA9ypDk6nRjA")

st.title("Onboarding Analyzer Agent")
st.write("""
This intelligent agent connects with your product analytics CSV, identifies friction points in onboarding journeys,
clusters drop-off reasons, and recommends UX fixes weekly.
""")

# 📂 CSV upload
uploaded_file = st.file_uploader("Upload Product Analytics CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data (first 10 rows)")
    st.write(df.head(10))

    # ✅ KPI Cards for instant metrics
    if 'Stage' in df.columns:
        total_students = len(df)
        top_stage = df['Stage'].value_counts().idxmax()
        top_stage_count = df['Stage'].value_counts().max()
        top_stage_percent = round(top_stage_count / total_students * 100, 2)

        if 'Reason' in df.columns:
            top_reason = df['Reason'].value_counts().idxmax()
            top_reason_count = df['Reason'].value_counts().max()
        else:
            top_reason = "N/A"
            top_reason_count = 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", total_students)
        col2.metric("Overall Drop-off %", f"{top_stage_percent}%")

        # ✅ Full Top Stage Drop-off displayed with wrapping
        col3.markdown(
            f"""
            <div style="background-color:#F0F2F6;padding:10px;border-radius:5px;">
                <strong>Top Stage Drop-off:</strong><br>{top_stage} ({top_stage_count})
            </div>
            """,
            unsafe_allow_html=True
        )

        # ✅ Full Top Reason displayed with wrapping
        col4.markdown(
            f"""
            <div style="background-color:#F0F2F6;padding:10px;border-radius:5px;">
                <strong>Top Reason:</strong><br>{top_reason} ({top_reason_count})
            </div>
            """,
            unsafe_allow_html=True
        )

    # 1️⃣ Stage-wise drop-off
    st.write("### Stage-wise Drop-Off Count")
    stage_counts = df['Stage'].value_counts()
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

    # 2️⃣ Drop-off Clustering
    st.write("### Drop-Off Clustering")
    df_cluster = df.copy()
    df_cluster['Stage_num'] = df_cluster['Stage'].astype('category').cat.codes

    k = min(3, len(df_cluster['Stage'].unique()))
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[['Stage_num']])
    st.write(df_cluster[['Stage', 'Cluster']].head(10))

    fig2, ax2 = plt.subplots()
    for cluster in sorted(df_cluster['Cluster'].unique()):
        cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
        ax2.scatter(cluster_data.index, cluster_data['Stage_num'], label=f'Cluster {cluster}')
    ax2.set_xlabel("Student Index")
    ax2.set_ylabel("Stage (numeric)")
    ax2.set_title("Drop-off Clustering")
    ax2.legend()
    st.pyplot(fig2)

    # 3️⃣ Drop-off reason clustering
    if 'Reason' in df.columns:
        st.write("### Drop-off Reason Clustering")
        reason_counts = df['Reason'].value_counts()
        st.write(reason_counts.head(10))
    else:
        st.info("No 'Reason' column found. Add it for reason-based clustering.")

    # 4️⃣ Trend over time chart
    if 'Date' in df.columns:
        st.write("### Drop-off Trend Over Time")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        trend = df.groupby(['Date', 'Stage']).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(10,5))
        trend.plot(ax=ax3)
        ax3.set_ylabel("Number of Students")
        ax3.set_xlabel("Date")
        ax3.set_title("Drop-off Trend Over Time")
        st.pyplot(fig3)
    else:
        st.info("No 'Date' column found. Trend chart requires 'Date'.")

    # 5️⃣ Weekly Gemini AI recommendations + summary box
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