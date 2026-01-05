import streamlit as st
import json
import requests
import os

from generate import GenerateEmail
generator = GenerateEmail(model=os.getenv("DEPLOYMENT_NAME"))

# helper funciton
# load emails
def load_emails_from_jsonl(path):
    emails = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                emails.append(json.loads(line))
    return emails

from datetime import datetime

def log_email_edit(
    operation,
    original_email,
    edited_email,
    reviews,
    log_path="logs/email_edits.jsonl",
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "operation": operation,
        "original_email": original_email,
        "edited_email": edited_email,
        "reviews": reviews,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



# --- CONFIG ---
st.set_page_config(page_title="AI Email Editor", page_icon="📧", layout="wide")
DATA_DIR = "datasets"
datasets = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]

# --- UI HEADER ---
st.title("📧 AI Email Editing Tool")
st.write("Select an email record by ID and use AI to refine it.")

# --- CHOOSE DATASET ---
st.sidebar.markdown("### 📁 Select Dataset")
dataset = st.sidebar.selectbox(
    "📂 Select Dataset",
    options=["lengthen.jsonl", "shorten.jsonl", "tone.jsonl"],
)

dataset_path = os.path.join(DATA_DIR, dataset)
emails = load_emails_from_jsonl(dataset_path)

if not emails:
    st.warning("No emails found in your JSONL file.")
    st.stop()

# --- ID NAVIGATION BAR ---
email_ids = [email["id"] for email in emails]
selected_id = st.sidebar.selectbox("📂 Select Email ID", options=email_ids)

# Find the selected email
selected_email = next((email for email in emails if email["id"] == selected_id), None)
if not selected_email:
    st.error(f"No email found with ID {selected_id}.")
    st.stop()

# --- DISPLAY SELECTED EMAIL ---
st.markdown(f"### ✉️ Email ID: `{selected_id}`")
st.markdown(f"**From:** {selected_email.get('sender', '(unknown)')}")
st.markdown(f"**Subject:** {selected_email.get('subject', '(no subject)')}")

email_text = st.text_area(
    "Email Content",
    value=selected_email.get("content", ""),
    height=250,
    key=f"email_text_{selected_id}",
)

# --- NEW BUTTON ---
st.markdown("### temp_action_button")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("elaborate"):
        result = generator.generate(
            "elaborate",
            selected_text=email_text,
        )

        analysis_faith = generator.generate(
            "faithfulness_judge",
            selected_text=email_text,
            model_response=result,
        )
        analysis_complete = generator.generate(
            "completeness_check",
            selected_text=email_text,
            model_response=result,
        )

        log_email_edit(
            operation="elaborate",
            original_email=selected_email,
            edited_email=result,
            reviews={
                "faithfulness": analysis_faith,
                "completeness": analysis_complete,
            },
        )

        st.session_state.result = result
        st.session_state.analysis_faith = analysis_faith
        st.session_state.analysis_complete = analysis_complete



with col2:
    if st.button("shorten"):
        result = generator.generate(
            "shorten",
            selected_text=email_text,
        )
        analysis_faith = generator.generate(
            "faithfulness_judge",
            selected_text=email_text,
            model_response=result,
        )
        analysis_complete = generator.generate(
            "completeness_check",
            selected_text=email_text,
            model_response=result,
        )

        log_email_edit(
            operation="shorten",
            original_email=selected_email,
            edited_email=result,
            reviews={
                "faithfulness": analysis_faith,
                "completeness": analysis_complete,
            },
        )

        st.session_state.result = result
        st.session_state.analysis_faith = analysis_faith
        st.session_state.analysis_complete = analysis_complete

with col3:
    if st.button("tone"):
        result = generator.generate(
            "tone",
            selected_text=email_text,
        )
        analysis_faith = generator.generate(
            "faithfulness_judge",
            selected_text=email_text,
            model_response=result,
        )
        analysis_complete = generator.generate(
            "completeness_check",
            selected_text=email_text,
            model_response=result,
        )

        log_email_edit(
            operation="tone",
            original_email=selected_email,
            edited_email=result,
            reviews={
                "faithfulness": analysis_faith,
                "completeness": analysis_complete,
            },
        )

        st.session_state.result = result
        st.session_state.analysis_faith = analysis_faith
        st.session_state.analysis_complete = analysis_complete
    
if "result" in locals():
    st.text_area("Result", result, height=300)
    st.text_area("Faithfulness Analysis", analysis_faith, height=200)
    st.text_area("Completeness Analysis", analysis_complete, height=200)

