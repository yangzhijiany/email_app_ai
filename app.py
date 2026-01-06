import streamlit as st
import json
import requests
import os
import pandas as pd
import re

from generate import GenerateEmail

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


def log_batch_result(
    email_id,
    operation,
    model,
    dataset_path,
    original_email,
    edited_email,
    faithfulness_review,
    completeness_review,
    log_path="logs/batch_results.jsonl",
):
    """Log batch processing results"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    record = {
        "email_id": email_id,
        "operation": operation,
        "model": model,
        "dataset_path": dataset_path,
        "original_email": original_email,
        "edited_email": edited_email,
        "faithfulness_review": faithfulness_review,
        "completeness_review": completeness_review,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def search_batch_result(email_id, operation=None, log_path="logs/batch_results.jsonl"):
    """Search batch processing results"""
    if not os.path.exists(log_path):
        return None
    
    results = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if record.get("email_id") == email_id:
                        if operation is None or record.get("operation") == operation:
                            results.append(record)
                except json.JSONDecodeError:
                    continue
    
    return results if results else None


def parse_review_rating(review_str):
    """Parse review JSON string to extract rating"""
    try:
        review_dict = json.loads(review_str)
        return review_dict.get("rating")
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, try to find rating number
        rating_match = re.search(r'"rating":\s*(\d+)', review_str)
        return int(rating_match.group(1)) if rating_match else None


def load_batch_results(log_path="logs/batch_results.jsonl"):
    """Load all batch processing results"""
    if not os.path.exists(log_path):
        return []
    
    results = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    # Parse ratings
                    faithfulness_rating = parse_review_rating(record.get("faithfulness_review", ""))
                    completeness_rating = parse_review_rating(record.get("completeness_review", ""))
                    
                    record["faithfulness_rating"] = faithfulness_rating
                    record["completeness_rating"] = completeness_rating
                    results.append(record)
                except json.JSONDecodeError:
                    continue
    
    return results


def calculate_average_scores(results, model=None, operation=None):
    """Calculate average scores for filtered results"""
    filtered = results
    
    if model and model != "All":
        filtered = [r for r in filtered if r.get("model") == model]
    
    if operation and operation != "All":
        filtered = [r for r in filtered if r.get("operation") == operation]
    
    if not filtered:
        return None
    
    faithfulness_ratings = [r["faithfulness_rating"] for r in filtered if r.get("faithfulness_rating") is not None]
    completeness_ratings = [r["completeness_rating"] for r in filtered if r.get("completeness_rating") is not None]
    
    avg_faithfulness = sum(faithfulness_ratings) / len(faithfulness_ratings) if faithfulness_ratings else None
    avg_completeness = sum(completeness_ratings) / len(completeness_ratings) if completeness_ratings else None
    
    return {
        "avg_faithfulness": avg_faithfulness,
        "avg_completeness": avg_completeness,
        "count": len(filtered)
    }



# --- CONFIG ---
st.set_page_config(page_title="AI Email Editor", page_icon="📧", layout="wide")
DATA_DIR = "datasets"
NEW_DATA_DIR = "new_datasets"

# Load datasets from both directories
datasets = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]
new_datasets = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith(".jsonl")] if os.path.exists(NEW_DATA_DIR) else []

# Create a mapping of dataset names to their directories
dataset_paths = {}
for dataset_file in datasets:
    dataset_paths[dataset_file] = DATA_DIR
for dataset_file in new_datasets:
    dataset_paths[dataset_file] = NEW_DATA_DIR

# Combine all dataset options
all_datasets = sorted(datasets + new_datasets)

# --- UI HEADER ---
st.title("📧 AI Email Editing Tool")
st.write("Select an email record by ID and use AI to refine it.")

st.sidebar.markdown("### Select Model")
model_options = ["gpt-4.1", "gpt-4o-mini"]

if "selected_model" not in st.session_state:
    default_model = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
    if default_model not in model_options:
        default_model = model_options[0]
    st.session_state.selected_model = default_model

current_index = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    options=model_options,
    index=current_index,
)

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.generator = None

if "generator" not in st.session_state or st.session_state.generator is None:
    st.session_state.generator = GenerateEmail(model=st.session_state.selected_model)

generator = st.session_state.generator

# --- CHOOSE DATASET ---
st.sidebar.markdown("### 📁 Select Dataset")
dataset = st.sidebar.selectbox(
    "📂 Select Dataset",
    options=all_datasets if all_datasets else ["lengthen.jsonl", "shorten.jsonl", "tone.jsonl"],
)

# Get the correct directory for the selected dataset
dataset_dir = dataset_paths.get(dataset, DATA_DIR)
dataset_path = os.path.join(dataset_dir, dataset)
emails = load_emails_from_jsonl(dataset_path)

if not emails:
    st.warning("No emails found in your JSONL file.")
    st.stop()

# --- BATCH PROCESSING ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Batch Processing")
batch_operation = st.sidebar.selectbox(
    "Select Operation",
    options=["elaborate", "shorten", "tone"],
    key="batch_operation"
)

if st.sidebar.button("Process All Emails"):
    st.session_state.batch_processing = True
    st.session_state.batch_progress = 0
    st.session_state.batch_total = len(emails)
    st.session_state.batch_completed = False
    st.session_state.batch_processed = 0
    st.session_state.batch_errors = 0
    st.rerun()

# Batch processing logic
if st.session_state.get("batch_processing", False) and not st.session_state.get("batch_completed", False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_container = st.container()
    
    processed = 0
    errors = 0
    error_messages = []
    
    for idx, email in enumerate(emails):
        email_id = email.get("id", f"unknown_{idx}")
        email_content = email.get("content", "")
        
        status_text.text(f"Processing: Email ID {email_id} ({idx+1}/{len(emails)})")
        
        try:
            # Generate edited email
            edited_email = generator.generate(
                batch_operation,
                selected_text=email_content,
            )
            
            # Get reviews
            faithfulness_review = generator.generate(
                "faithfulness_judge",
                selected_text=email_content,
                model_response=edited_email,
            )
            
            completeness_review = generator.generate(
                "completeness_check",
                selected_text=email_content,
                model_response=edited_email,
            )
            
            # Log result
            log_batch_result(
                email_id=email_id,
                operation=batch_operation,
                model=st.session_state.selected_model,
                dataset_path=dataset_path,
                original_email=email,
                edited_email=edited_email,
                faithfulness_review=faithfulness_review,
                completeness_review=completeness_review,
            )
            
            processed += 1
        except Exception as e:
            errors += 1
            error_msg = f"Error processing Email ID {email_id}: {str(e)}"
            error_messages.append(error_msg)
        
        # Update progress
        progress = (idx + 1) / len(emails)
        progress_bar.progress(progress)
    
    # Display error messages
    if error_messages:
        with error_container:
            for msg in error_messages:
                st.error(msg)
    
    # All processing completed
    status_text.text(f"Completed! Success: {processed}, Failed: {errors}")
    st.session_state.batch_completed = True
    st.session_state.batch_processing = False
    st.session_state.batch_processed = processed
    st.session_state.batch_errors = errors
    st.success(f"Batch processing completed! Successfully processed {processed}, failed {errors}")

# If batch processing is completed, show notification
if st.session_state.get("batch_completed", False):
    st.sidebar.markdown("---")
    st.sidebar.success("Batch processing completed!")

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
st.markdown("### Actions")
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
    
# Display results
if "result" in st.session_state:
    st.text_area("Result", st.session_state.result, height=300)
    st.text_area("Faithfulness Analysis", st.session_state.analysis_faith, height=200)
    st.text_area("Completeness Analysis", st.session_state.analysis_complete, height=200)

# --- METRICS VISUALIZATION ---
st.markdown("---")
st.markdown("### 📊 Metrics Visualization")

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    viz_model = st.selectbox(
        "Select Model",
        options=["All", "gpt-4.1", "gpt-4o-mini"],
        key="viz_model"
    )
with viz_col2:
    viz_operation = st.selectbox(
        "Select Operation",
        options=["All", "elaborate", "shorten", "tone"],
        key="viz_operation"
    )

if st.button("📈 Show Metrics", key="show_metrics_button"):
    batch_results = load_batch_results()
    
    if not batch_results:
        st.warning("No batch processing results found. Please run batch processing first.")
    else:
        if viz_model == "All":
            # Show side-by-side charts for both models
            model_col1, model_col2 = st.columns(2)
            
            with model_col1:
                st.markdown("#### gpt-4.1")
                scores_41 = calculate_average_scores(batch_results, model="gpt-4.1", operation=viz_operation)
                
                if scores_41:
                    chart_data = pd.DataFrame({
                        "Metric": ["Faithfulness", "Completeness"],
                        "Average Score": [scores_41["avg_faithfulness"], scores_41["avg_completeness"]]
                    })
                    st.bar_chart(chart_data.set_index("Metric"))
                    st.caption(f"Based on {scores_41['count']} records")
                    st.metric("Avg Faithfulness", f"{scores_41['avg_faithfulness']:.2f}")
                    st.metric("Avg Completeness", f"{scores_41['avg_completeness']:.2f}")
                else:
                    st.info("No data available for gpt-4.1 with the selected filters.")
            
            with model_col2:
                st.markdown("#### gpt-4o-mini")
                scores_mini = calculate_average_scores(batch_results, model="gpt-4o-mini", operation=viz_operation)
                
                if scores_mini:
                    chart_data = pd.DataFrame({
                        "Metric": ["Faithfulness", "Completeness"],
                        "Average Score": [scores_mini["avg_faithfulness"], scores_mini["avg_completeness"]]
                    })
                    st.bar_chart(chart_data.set_index("Metric"))
                    st.caption(f"Based on {scores_mini['count']} records")
                    st.metric("Avg Faithfulness", f"{scores_mini['avg_faithfulness']:.2f}")
                    st.metric("Avg Completeness", f"{scores_mini['avg_completeness']:.2f}")
                else:
                    st.info("No data available for gpt-4o-mini with the selected filters.")
        else:
            # Show single chart for selected model
            scores = calculate_average_scores(batch_results, model=viz_model, operation=viz_operation)
            
            if scores:
                chart_data = pd.DataFrame({
                    "Metric": ["Faithfulness", "Completeness"],
                    "Average Score": [scores["avg_faithfulness"], scores["avg_completeness"]]
                })
                st.bar_chart(chart_data.set_index("Metric"))
                st.caption(f"Based on {scores['count']} records")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Avg Faithfulness", f"{scores['avg_faithfulness']:.2f}")
                with metric_col2:
                    st.metric("Avg Completeness", f"{scores['avg_completeness']:.2f}")
            else:
                st.info(f"No data available for {viz_model} with the selected filters.")

# --- SEARCH BATCH RESULTS ---
st.markdown("---")
st.markdown("### 🔍 Search Batch Results")

search_col1, search_col2 = st.columns([2, 1])
with search_col1:
    search_email_id = st.text_input("Enter Email ID", key="search_email_id", placeholder="e.g.: 1")
with search_col2:
    search_operation = st.selectbox(
        "Operation Type (Optional)",
        options=[None, "elaborate", "shorten", "tone"],
        format_func=lambda x: "All" if x is None else x,
        key="search_operation"
    )

if st.button("🔍 Search", key="search_button"):
    if search_email_id:
        try:
            email_id = int(search_email_id) if search_email_id.isdigit() else search_email_id
            results = search_batch_result(email_id, search_operation)
            
            if results:
                st.session_state.search_results = results
                st.success(f"Found {len(results)} result(s)")
            else:
                st.warning(f"No results found for Email ID {email_id}")
                st.session_state.search_results = None
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.session_state.search_results = None
    else:
        st.warning("Please enter Email ID")

# Initialize search results
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# Display search results
if st.session_state.get("search_results"):
    st.markdown("---")
    st.markdown("### 📊 Search Results")
    
    for idx, result in enumerate(st.session_state.search_results):
        with st.expander(f"Email ID: {result['email_id']} - {result['operation']} ({result.get('model', 'N/A')})"):
            st.markdown(f"**Operation Type:** {result['operation']}")
            st.markdown(f"**Model:** {result.get('model', 'N/A')}")
            st.markdown(f"**Processing Time:** {result.get('timestamp', 'N/A')}")
            
            st.markdown("#### 📥 Input (Original Email)")
            original_email = result.get('original_email', {})
            if isinstance(original_email, dict):
                st.text_area(
                    "Original Email Content",
                    value=original_email.get('content', ''),
                    height=150,
                    key=f"input_{idx}",
                    disabled=True
                )
            else:
                st.text_area(
                    "Original Email Content",
                    value=str(original_email),
                    height=150,
                    key=f"input_{idx}",
                    disabled=True
                )
            
            st.markdown("#### 📤 Output (Processed Email)")
            st.text_area(
                "Processed Email",
                value=result.get('edited_email', ''),
                height=200,
                key=f"output_{idx}",
                disabled=True
            )
            
            st.markdown("#### 📋 Review Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Faithfulness Review**")
                st.text_area(
                    "Faithfulness",
                    value=result.get('faithfulness_review', ''),
                    height=150,
                    key=f"faith_{idx}",
                    disabled=True
                )
            with col2:
                st.markdown("**Completeness Review**")
                st.text_area(
                    "Completeness",
                    value=result.get('completeness_review', ''),
                    height=150,
                    key=f"complete_{idx}",
                    disabled=True
                )

