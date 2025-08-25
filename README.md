Onboarding Drop-off Analyzer 🚀

An AI-powered Streamlit app that helps identify where and why users drop off during an onboarding process.
Useful for universities, training platforms, SaaS products, and businesses that want to improve user engagement.

🔹 Features

📂 Upload any CSV dataset with onboarding/staged process data

👀 Preview the uploaded dataset (first 10 rows)

📊 Overall drop-off percentage calculation

🏷️ Stage-wise drop-off analysis (which stage loses the most users)

💡 AI-powered clustering of drop-off reasons (e.g., lack of guidance, unclear instructions)

📉 Drop-off funnel visualization

🤖 AI Suggestions for reducing drop-offs using Gemini


🔹 How It Works

1. Upload a CSV file with onboarding data.

Recommended columns:

User ID / Student ID

Stage / Step Name

Status / Drop-off Reason


Example:

user_id	stage	status	reason

101	Registration	Dropped Off	Lack of interest
102	Orientation	Completed	—
103	Stage 1	Dropped Off	Unclear instructions




2. The analyzer automatically:

Detects drop-off percentage

Identifies stage with highest drop-off

Clusters common reasons

Provides AI recommendations




🔹 Live Demo

Try the app here: 👉 

🔹 Repository

GitHub Repo: 👉 

🔹 Setup (for developers)

If you want to run it locally:

# Clone the repo
git clone <your-repo-url>
cd onboarding-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🔹 Notes

This tool works best with datasets that represent onboarding or multi-stage processes.

If you upload a dataset without stages or drop-off information, results may not be meaningful.

Designed for educational and research purposes (college hackathons, demos, and prototype building).

