"""
Refactored utility functions for the Vibe Test Uploader application.
This code integrates and refines functionalities based on vibe_test_upload_query_phase4_v2.py.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- Embedded Prompt Library -------------------------
prompt_library = [
  {
    "prompt":"Show me the top 5 sellers with the highest internal risk rating and most escalations.",
    "intent_description":"Ranking with CASE + Sorting",
    "expected_code":"df.sort_values(by=['internal_risk_rating', 'escalation_count'], ascending=[False, False]).head(5)",
    "expected_output":"Top 5 sellers sorted by risk rating and escalation count",
    "output_type":"Value",
    "category":"Sort / Ranking",
    "python_code":"df.sort_values(by=['internal_risk_rating', 'escalation_count'], ascending=[False, False]).head(5)",
    "complexity_level":"Nested"
  },
  {
    "prompt":"Calculate the month-over-month percent change in ratings.",
    "intent_description":"Percent Change",
    "expected_code":"df = df.sort_values(by='date'); df['pct_change'] = df['rating'].pct_change()",
    "expected_output":"DataFrame with pct_change column",
    "output_type":"DataFrame",
    "category":"Datetime Logic",
    "python_code":"df = df.sort_values(by='date'); df['pct_change'] = df['rating'].pct_change()",
    "complexity_level":"Intermediate"
  }
  # Add more diverse examples from your actual library if available
]

# ------------------------- Helper Functions for Data Processing -------------------------

def _log_processing_message(log_expander, message, level):
    """Helper to log messages to a Streamlit expander."""
    if level == "success":
        log_expander.success(message)
    elif level == "error":
        log_expander.error(message)
    elif level == "warning":
        log_expander.warning(message.replace("&nbsp;&nbsp;&nbsp;&nbsp;", " " * 4)) # For indentation
    else:
        log_expander.info(message)

def _try_convert_to_numeric(df_series, col_name, log_messages):
    """Attempts to convert a Pandas Series to numeric type."""
    original_dtype_str = str(df_series.dtype)
    original_col_data_for_revert = None
    if df_series.dtype == "object":
        original_col_data_for_revert = df_series.copy()

    col_str_series = df_series.astype(str)
    cleaned_for_num_test = col_str_series.str.replace(r"[$,]", "", regex=True)
    cleaned_for_num_test = cleaned_for_num_test.str.replace("%", "", regex=True).str.strip()
    cleaned_for_num_test_no_empty = cleaned_for_num_test.replace("", np.nan)

    # Use original series for mask to correctly identify originally non-null values
    source_for_mask = original_col_data_for_revert if original_col_data_for_revert is not None else df_series
    original_non_null_mask = source_for_mask.notna() & (source_for_mask.astype(str).str.strip() != "")

    if original_non_null_mask.sum() > 0: # Only proceed if there are non-null values to test
        test_convert_series = pd.to_numeric(cleaned_for_num_test_no_empty[original_non_null_mask], errors="coerce")
        num_successfully_coerced = test_convert_series.notna().sum()
        num_original_non_null_to_test = original_non_null_mask.sum()

        if num_original_non_null_to_test > 0 and (num_successfully_coerced / num_original_non_null_to_test) >= 0.80:
            try:
                had_percentage_sign_mask = col_str_series.str.contains("%", na=False)
                numeric_col = pd.to_numeric(cleaned_for_num_test_no_empty, errors="coerce") # Convert the whole series
                
                # Normalize percentage values
                condition_for_division = had_percentage_sign_mask & numeric_col.notna()
                numeric_col.loc[condition_for_division] = numeric_col.loc[condition_for_division] / 100.0
                
                # Check how many originally non-null values became NaN after conversion
                newly_coerced_to_nan_mask = original_non_null_mask & numeric_col.isna()
                coerced_count = newly_coerced_to_nan_mask.sum()
                
                log_msg_main = f"✅ Column **'{col_name}'**: Converted to numeric (was {original_dtype_str})."
                if condition_for_division.any(): 
                    log_msg_main += " Percentage values normalized."
                log_messages.append(("success", log_msg_main))
                if coerced_count > 0: 
                    log_messages.append(("warning", f"&nbsp;&nbsp;&nbsp;&nbsp;⚠️ In '{col_name}', {coerced_count} original non-empty value(s) became missing (NaN) after numeric conversion."))
                return numeric_col, True
            except Exception as e_conv:
                log_messages.append(("error", f"❌ Column **'{col_name}'**: Numeric conversion failed during final attempt: {e_conv}. Reverted."))
                if original_col_data_for_revert is not None:
                    return original_col_data_for_revert, False
                return df_series, False
    return df_series, False

def _try_convert_to_datetime(df_series, col_name, log_messages):
    """Attempts to convert a Pandas Series to datetime type."""
    original_dtype_str = str(df_series.dtype)
    original_for_date_conv = df_series.copy() # Keep original for potential revert

    # Convert to string for pattern matching, handle potential mixed types gracefully
    try:
        col_str_series_for_date = df_series.astype(str)
    except Exception: # If astype(str) fails (highly unlikely for a series), skip
        return df_series, False

    is_potential_date_col_by_name = any(k in col_name.lower() for k in ["date", "dt", "timestamp", "period"])
    
    # Regex for common date patterns
    # Covers YYYY-MM-DD, DD-MM-YYYY, MM/DD/YYYY etc. with various separators
    date_pattern_regex = r"(\b\d{4}[-/.\s]\d{1,2}[-/.\s]\d{1,2}\b|\b\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}\b)"
    date_pattern_matches = col_str_series_for_date.str.match(date_pattern_regex, na=False).sum()
    
    num_non_null_in_series_for_date = col_str_series_for_date.replace("",np.nan).dropna().shape[0]
    has_sufficient_date_like_patterns = False
    if num_non_null_in_series_for_date > 0: 
        has_sufficient_date_like_patterns = (date_pattern_matches / num_non_null_in_series_for_date) >= 0.50

    if is_potential_date_col_by_name or has_sufficient_date_like_patterns:
        try:
            # Test conversion on non-null values first to gauge success rate
            non_null_original_for_date_test = df_series.dropna()
            if not non_null_original_for_date_test.empty:
                converted_dates_test = pd.to_datetime(non_null_original_for_date_test, errors="coerce")
                datetime_conversion_success_rate = converted_dates_test.notna().mean()
                
                if datetime_conversion_success_rate >= 0.80: # Threshold for attempting full conversion
                    converted_series = pd.to_datetime(df_series, errors="coerce")
                    num_null_after_conversion = converted_series.isnull().sum()
                    num_null_before_conversion = df_series.isnull().sum()

                    log_messages.append(("success", f"✅ Column **'{col_name}'**: Converted to datetime (was {original_dtype_str})."))
                    if (num_null_after_conversion - num_null_before_conversion) > 0:
                         newly_nulled_count = num_null_after_conversion - num_null_before_conversion
                         log_messages.append(("warning", f"&nbsp;&nbsp;&nbsp;&nbsp;⚠️ In '{col_name}', {newly_nulled_count} additional value(s) became missing (NaT) after datetime conversion."))
                    return converted_series, True
        except Exception as e_date: # Catch any error during the datetime conversion process
            log_messages.append(("error", f"❌ Column **'{col_name}'**: Date conversion attempt failed: {e_date}. Reverted."))
            return original_for_date_conv, False # Return original series and failure status
    return df_series, False


@st.cache_data(ttl=3600, show_spinner=False)
def process_dataframe(df_to_process, cleaning_enabled=True):
    """Process and clean dataframe based purely on content analysis."""
    if df_to_process is None:
        return None
        
    processed_df = df_to_process.copy()

    # Initial minimal cleaning (moved from main app load)
    processed_df.columns = [str(col).strip() for col in processed_df.columns]
    for col in processed_df.select_dtypes(include=['object', 'string']).columns:
        try:
            if processed_df[col].apply(type).eq(str).mean() > 0.5 or processed_df[col].dtype == 'string':
                processed_df[col] = processed_df[col].str.strip()
            else: # Handle mixed types or non-string objects by converting to string first
                processed_df[col] = processed_df[col].astype(str).str.strip().replace(['nan', 'NaN', 'None', '<NA>'], np.nan, regex=False)
        except Exception: # Silently pass if stripping fails on a column
            pass # log_messages.append(("warning", f"Initial strip attempt failed on '{col}': {e_strip}"))

    if not cleaning_enabled:
        st.info("Auto data type cleaning (numeric/datetime conversion) is disabled. Minimal string stripping applied.")
        return processed_df

    log_expander = st.expander("Data Type Cleaning & Processing Log", expanded=False)
    log_messages_list = [] 

    for col in processed_df.columns:
        current_col_data = processed_df[col]
        # Only attempt conversion if the column is of object type
        if current_col_data.dtype == "object":
            # Attempt numeric conversion first
            converted_numeric_series, success_numeric = _try_convert_to_numeric(current_col_data, col, log_messages_list)
            if success_numeric:
                processed_df[col] = converted_numeric_series
                continue # Move to next column if numeric conversion was successful
            
            # If numeric conversion failed or wasn't applicable, attempt datetime conversion
            converted_datetime_series, success_datetime = _try_convert_to_datetime(current_col_data, col, log_messages_list)
            if success_datetime:
                processed_df[col] = converted_datetime_series
    
    if log_messages_list:
        for level, msg in log_messages_list:
            _log_processing_message(log_expander, msg, level)
    else:
        _log_processing_message(log_expander, "No specific numeric/datetime type conversions applied (or column types were already appropriate).", "info")
    return processed_df

# ------------------------- Data Quality Assessment Functions -------------------------

def _check_missing_values(df):
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        missing_counts = df.isna().sum()
        missing_cols_details = []
        for col, count in missing_counts[missing_counts > 0].items():
            missing_cols_details.append(f"{col} ({count} missing, {count/len(df):.1%})")
        return f"  - ⚠️ Missing values: {missing_total} total detected. Details: {'; '.join(missing_cols_details)}"
    return None

def _check_numeric_type_hints(df_column, column_name):
    if df_column.dtype == "object":
        col_str_series = df_column.astype(str)
        cleaned_for_test = col_str_series.str.replace(r"[$,%]", "", regex=True).str.strip().replace("",np.nan)
        original_non_null_mask = df_column.notna() & (df_column.astype(str).str.strip() != "")
        if original_non_null_mask.sum() > 0:
            test_convert_series = pd.to_numeric(cleaned_for_test[original_non_null_mask], errors="coerce")
            # Check if a high percentage of *original non-null values* could be converted
            if (test_convert_series.notna().sum() / original_non_null_mask.sum()) >= 0.80:
                return f"  - ℹ️ Hint: Column '{column_name}' (text) appears mostly numeric. Auto-cleaning would attempt conversion if enabled."
    return None

def _check_duplicate_values(df_column, column_name):
    """Checks for duplicates, more relevant for potential ID columns."""
    col_dropna = df_column.dropna()
    if not col_dropna.empty:
        # Heuristic: high uniqueness might indicate an ID or key column
        # Consider it for duplicate check if more than 50% unique or many unique values
        is_potentially_id_like = (col_dropna.nunique() / len(col_dropna) > 0.7 and len(col_dropna) > 10) or col_dropna.nunique() > 1000
        
        if is_potentially_id_like and col_dropna.duplicated().any():
            duplicate_count = col_dropna.duplicated().sum()
            return f"  - ⚠️ Potential ID duplicates: {duplicate_count} duplicate value(s) found in column '{column_name}' (which has high uniqueness)."
    return None


def _check_date_type_hints(df_column, column_name):
    if df_column.dtype == "object":
        date_keywords_heuristic = ["date", "dt", "timestamp", "period", "time", "year", "month", "day"]
        is_potential_date_col_by_name = any(keyword in column_name.lower() for keyword in date_keywords_heuristic)
        
        col_str_series = df_column.astype(str)
        # Regex for common date patterns
        date_pattern_regex = r"(\b\d{4}[-/.\s]\d{1,2}[-/.\s]\d{1,2}\b|\b\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}\b)"
        date_pattern_matches = col_str_series.str.match(date_pattern_regex, na=False).sum()
        
        num_non_null_in_series = col_str_series.replace("",np.nan).dropna().shape[0]
        has_sufficient_date_like_patterns = False
        if num_non_null_in_series > 0:
            has_sufficient_date_like_patterns = (date_pattern_matches / num_non_null_in_series) >= 0.50
            
        if is_potential_date_col_by_name or has_sufficient_date_like_patterns:
            return f"  - ℹ️ Hint: Column '{column_name}' (text) has date-like patterns or name. Auto-cleaning would attempt conversion if enabled."
    return None

@st.cache_data(ttl=3600)
def get_data_quality_issues(df, cleaning_enabled_for_context):
    """Identify data quality issues/hints in the raw uploaded file."""
    issues = []
    if df is None or df.empty:
        return ["No data loaded to assess quality."]

    issues.append("**Data Quality Hints (based on Raw Uploaded Data):**")
    
    missing_issue = _check_missing_values(df)
    if missing_issue: issues.append(missing_issue)

    for col_name in df.columns: # Iterate through all columns for these checks
        # Numeric hint only if cleaning is disabled (otherwise it would have been attempted)
        if not cleaning_enabled_for_context:
            num_hint = _check_numeric_type_hints(df[col_name], col_name)
            if num_hint: issues.append(num_hint)
        
        # Date hint only if cleaning is disabled
        if not cleaning_enabled_for_context:
            date_hint = _check_date_type_hints(df[col_name], col_name)
            if date_hint: issues.append(date_hint)

        # Duplicate hints are general
        dup_hint = _check_duplicate_values(df[col_name], col_name)
        if dup_hint: issues.append(dup_hint)
        

    if len(issues) == 1: # Only header was added
        issues.append("  - ✅ No major quality hints found via basic checks, or auto-cleaning is enabled (which would address some hints).")
    return issues

# ------------------------- Prompt Example Formatting -------------------------
def format_prompt_examples(library, num_examples=8):
    """Selects and formats diverse examples from the library for the prompt."""
    if not library: return "# Prompt library not loaded or empty.\n"
    examples_str = ""
    selected_examples = []
    
    if not library: return "# Prompt library is empty.\n" # Should not happen with embedded

    # Try to get diverse examples by complexity
    complexities = sorted(list(set(ex.get("complexity_level", "Unknown") for ex in library)))
    
    # Try to get at least one example from each complexity level if possible
    for comp_level in complexities:
        comp_examples = [ex for ex in library if ex.get("complexity_level") == comp_level]
        if comp_examples:
            selected_examples.append(random.choice(comp_examples))
    
    # Fill remaining slots randomly from unique examples
    remaining_needed = num_examples - len(selected_examples)
    if remaining_needed > 0:
        # Create a pool of examples not yet selected
        remaining_pool = [ex for ex in library if ex not in selected_examples]
        if remaining_pool: # Check if remaining_pool is not empty
            fill_count = min(remaining_needed, len(remaining_pool))
            selected_examples.extend(random.sample(remaining_pool, fill_count))
            
    if not selected_examples: return "# No examples selected from library.\n"

    # Shuffle the final selection for variety in presentation
    random.shuffle(selected_examples)

    examples_str += "# --- START: Analytical Pattern Examples (Selected from Library) ---\n"
    examples_str += "# Learn from these patterns. Adapt the LOGIC using the ACTUAL columns/types from the schema above.\n\n"
    for i, ex in enumerate(selected_examples[:num_examples]): # Ensure we don't exceed num_examples
        examples_str += f"{i+1}. # Category: {ex.get('category', 'N/A')}, Complexity: {ex.get('complexity_level', 'N/A')}\n"
        examples_str += f"   # User Hint Example: '{ex.get('prompt', 'N/A')}'\n"
        examples_str += f"   # Pandas Logic Example (ADAPT to current schema!): \n   #   {ex.get('python_code', '# N/A')}\n"
        examples_str += f"   # Description: {ex.get('intent_description', 'N/A')}\n\n"
    examples_str += "# Remember: Adapt the *logic* from these patterns using the *actual columns/types* from the current schema.\n"
    examples_str += "# --- END: Analytical Pattern Examples ---\n\n"
    return examples_str

# ------------------------- File Loading -------------------------
def load_dataframe_from_uploaded_file(uploaded_file_object):
    """Loads a DataFrame from an uploaded file object (CSV or Excel)."""
    if uploaded_file_object is None:
        return None
        
    filename = uploaded_file_object.name
    file_bytes = uploaded_file_object.getvalue()
    df_loaded = None
    try:
        if filename.lower().endswith(".csv"):
            df_loaded = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        elif filename.lower().endswith(('.xls', '.xlsx')):
            # Try with default engine, then openpyxl if it fails (for broader compatibility)
            try:
                df_loaded = pd.read_excel(io.BytesIO(file_bytes))
            except Exception: # pylint: disable=broad-except
                df_loaded = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        else:
            st.error(f"Unsupported file type: {filename}. Please upload CSV or Excel.")
            return None
    except Exception as e: # pylint: disable=broad-except
        st.error(f"Error loading file '{filename}': {e}")
        return None
    return df_loaded

# ------------------------- Session State Management -------------------------
def initialize_session_state():
    """Initializes required session state variables if they don't exist."""
    defaults = {
        "df_processed": None,
        "df_original": None,
        "query_log": [],
        "user_query_input": "", # Changed from generated_question for clarity
        "client": None, # To store OpenAI client if initialized in main
        "last_uploaded_filename": None,
        # Feature toggles - sidebar will manage their state but good to have defaults
        "debug_mode": False,
        "enable_df_pagination": True,
        "enable_result_pagination": True,
        "enable_multilingual": False,
        "enable_graph_suggestions": True,
        "enable_data_cleaning_toggle": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ------------------------- Sidebar UI Rendering -------------------------
def render_sidebar(prompt_lib): # prompt_lib is passed but not directly used here anymore if embedded
    """Renders all sidebar elements and returns their current state/values."""
    st.sidebar.title("App Controls & Info") 

    if st.sidebar.button("🔁 Reset Session State", help="Clears uploaded data and query history."):
        # Preserve client if it's managed outside and passed in, or re-init in main
        # For this refactoring, client is initialized in main_app.
        # Keys to preserve across reset:
        preserve_keys = ['client'] # Add any other keys that should absolutely persist
        
        current_keys = list(st.session_state.keys())
        for key in current_keys:
            if key not in preserve_keys:
                del st.session_state[key]
        initialize_session_state() # Re-initialize to defaults
        st.experimental_rerun()

    # Use st.session_state for feature toggles to persist their values
    st.session_state.debug_mode = st.sidebar.checkbox(
        "🔍 Debug Mode", value=st.session_state.get("debug_mode", False)
    )
    
    with st.sidebar.expander("⚙️ Feature Toggles", expanded=True):
        st.session_state.enable_df_pagination = st.checkbox(
            "🔢 Data Preview Pagination", value=st.session_state.get("enable_df_pagination", True)
        )
        st.session_state.enable_result_pagination = st.checkbox(
            "🔢 Query Result Pagination", value=st.session_state.get("enable_result_pagination", True)
        )
        st.session_state.enable_multilingual = st.checkbox(
            "🌐 Multilingual Prompting", value=st.session_state.get("enable_multilingual", False), help="Experimental: LLM attempts to interpret non-English queries."
        )
        st.session_state.enable_graph_suggestions = st.checkbox(
            "📈 Auto Graph Suggestions", value=st.session_state.get("enable_graph_suggestions", True), help="Suggests graph types for query results."
        )
        st.session_state.enable_data_cleaning_toggle = st.checkbox(
            "🧹 Auto Data Type Cleaning", value=st.session_state.get("enable_data_cleaning_toggle", True), help="Content-driven numeric/datetime conversion."
        )

    # Prompt Library Load Status
    if not prompt_library: # Check the embedded one
        st.sidebar.warning("Embedded prompt library is empty or failed to load. LLM examples will be limited.")
    else:
        st.sidebar.success(f"Using {len(prompt_library)} embedded examples for LLM.")

    # Export Query Log Button
    if st.sidebar.button("📥 Export Query Log"):
        if 'query_log' in st.session_state and st.session_state.query_log:
            try:
                query_log_df = pd.DataFrame(st.session_state.query_log)
                csv_data_log = query_log_df.to_csv(index=False).encode('utf-8')
                # Use a unique key for the download button if it's re-rendered
                st.sidebar.download_button(
                    label="Download Query Log as CSV", 
                    data=csv_data_log, 
                    file_name=f"query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                    mime="text/csv", 
                    key="export_query_log_button" 
                )
            except Exception as e_log: 
                st.sidebar.error(f"Error exporting log: {e_log}")
        else: 
            st.sidebar.info("No queries have been logged yet.")
            
    # Return a dictionary of the current feature toggle states
    return {
        "debug_mode": st.session_state.debug_mode,
        "enable_df_pagination": st.session_state.enable_df_pagination,
        "enable_result_pagination": st.session_state.enable_result_pagination,
        "enable_multilingual": st.session_state.enable_multilingual,
        "enable_graph_suggestions": st.session_state.enable_graph_suggestions,
        "enable_data_cleaning_toggle": st.session_state.enable_data_cleaning_toggle
    }

# ------------------------- Data Schema and Exploration Utilities -------------------------

def get_schema_summary_df(df):
    """Generates a DataFrame summarizing the schema of the input DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame() # Return empty DataFrame if input is None or empty
        
    schema_info_list = []
    for col_name_schema in df.columns:
        col_data = df[col_name_schema]
        non_null_count = col_data.notnull().sum()
        total_count = len(col_data)
        try:
            unique_count = col_data.nunique()
        except Exception: # pylint: disable=broad-except
            unique_count = "Error" # Handle cases like unhashable types
        
        # Get sample values, ensuring they are stringified for display
        sample_vals_schema = []
        try:
            sample_vals_schema = [str(v) for v in col_data.dropna().unique()[:3]]
        except Exception: # pylint: disable=broad-except
            sample_vals_schema = ["Error getting samples"]

        sample_vals_str_schema = ', '.join(sample_vals_schema) if sample_vals_schema else "N/A (All Nulls or Empty)"
        
        schema_info_list.append({
            "Column Name": col_name_schema,
            "Data Type": str(col_data.dtype),
            "Non-Null Count": f"{non_null_count}/{total_count} ({non_null_count / total_count:.1%})" if total_count > 0 else "0/0",
            "Unique Values": unique_count,
            "Sample Values": sample_vals_str_schema
        })
    return pd.DataFrame(schema_info_list).reset_index(drop=True)


def generate_column_distribution_plots(st_object, df, numeric_cols_to_plot, categorical_cols_to_plot):
    """Generates and displays distribution plots for selected columns using st_object for Streamlit elements."""
    if df is None or df.empty:
        st_object.info("No data available for plotting distributions.")
        return

    if numeric_cols_to_plot:
        st_object.markdown("#### 🔢 Numeric Column Distributions")
        num_cols_per_row_viz = min(3, len(numeric_cols_to_plot)) # Adjust for better layout
        for i in range(0, len(numeric_cols_to_plot), num_cols_per_row_viz):
            cols_viz = st_object.columns(num_cols_per_row_viz)
            for j, col_name_viz_num in enumerate(numeric_cols_to_plot[i:i+num_cols_per_row_viz]):
                with cols_viz[j % num_cols_per_row_viz]:
                    st_object.markdown(f"**{col_name_viz_num}**")
                    fig_num, ax_num = plt.subplots(figsize=(4, 3)) # Slightly larger
                    try:
                        sns.histplot(df[col_name_viz_num].dropna(), kde=True, ax=ax_num, bins=20)
                        ax_num.tick_params(axis='x', labelsize=8, rotation=30)
                        ax_num.tick_params(axis='y', labelsize=8)
                        ax_num.set_xlabel('')
                        ax_num.set_ylabel('')
                        plt.tight_layout()
                        st_object.pyplot(fig_num)
                    except Exception as e_p_num: # pylint: disable=broad-except
                        st_object.warning(f"Plot failed for '{col_name_viz_num}': {e_p_num}")
                    finally:
                        plt.close(fig_num)
    
    if categorical_cols_to_plot:
        st_object.markdown("#### 🅰️ Categorical Column Distributions (Top 10)")
        cat_cols_per_row_viz = min(3, len(categorical_cols_to_plot)) # Adjust for better layout
        for i in range(0, len(categorical_cols_to_plot), cat_cols_per_row_viz):
            cols_viz_cat = st_object.columns(cat_cols_per_row_viz)
            for j, col_name_viz_cat in enumerate(categorical_cols_to_plot[i:i+cat_cols_per_row_viz]):
                with cols_viz_cat[j % cat_cols_per_row_viz]:
                    st_object.markdown(f"**{col_name_viz_cat}**")
                    fig_cat, ax_cat = plt.subplots(figsize=(4.5, 3.5)) # Slightly larger for labels
                    try:
                        # Ensure column is treated as string for value_counts, especially if mixed types
                        top_vals = df[col_name_viz_cat].astype(str).value_counts().nlargest(10)
                        if not top_vals.empty:
                            sns.barplot(x=top_vals.index, y=top_vals.values, ax=ax_cat, palette="viridis")
                            ax_cat.tick_params(axis='x', rotation=75, labelsize=8, ha='right')
                            ax_cat.tick_params(axis='y', labelsize=8)
                            ax_cat.set_xlabel('')
                            ax_cat.set_ylabel('')
                            plt.tight_layout()
                            st_object.pyplot(fig_cat)
                        else:
                            st_object.caption("No values to plot.")
                    except Exception as e_p_cat: # pylint: disable=broad-except
                        st_object.warning(f"Plot failed for '{col_name_viz_cat}': {e_p_cat}")
                    finally:
                        plt.close(fig_cat)

# ------------------------- Graphing Utilities for Query Results -------------------------
def get_suggested_chart_types(df_result):
    """Suggests chart types based on the result DataFrame structure."""
    if df_result is None or df_result.empty:
        return ["Table View"]

    graph_types_sugg = ["Table View"]
    num_rows_res = df_result.shape[0]
    num_cols_res = df_result.shape[1]

    if num_cols_res == 0: # Should not happen with valid results
        return ["Table View"]

    col1_name_res = df_result.columns[0]
    col1_dtype_res = pd.api.types.infer_dtype(df_result[col1_name_res], skipna=True)

    if num_cols_res >= 2:
        col2_name_res = df_result.columns[1]
        col2_dtype_res = pd.api.types.infer_dtype(df_result[col2_name_res], skipna=True)

        if num_rows_res > 0 and num_rows_res <= 50: # Limit suggestions for very large result sets for performance
            if col2_dtype_res in ['integer', 'floating', 'decimal']: # Y-axis is numeric
                if col1_dtype_res in ['string', 'categorical', 'boolean'] and df_result[col1_name_res].nunique() <= 30:
                    graph_types_sugg.extend(["Bar Chart", "Horizontal Bar Chart"])
                if col1_dtype_res in ['datetime', 'datetime64', 'date', 'period'] or \
                   (col1_dtype_res in ['integer', 'floating'] and df_result[col1_name_res].nunique() > 5): # X-axis is date/time or continuous numeric
                    graph_types_sugg.extend(["Line Chart"])
                if col1_dtype_res in ['integer', 'floating', 'decimal']: # Both X and Y numeric
                    graph_types_sugg.extend(["Scatter Plot"])
        
        # For multi-series, typically index is categorical/datetime, other columns are numeric
        numeric_cols_count_res = sum(1 for i in range(1, min(6, num_cols_res)) if pd.api.types.infer_dtype(df_result.iloc[:, i], skipna=True) in ['integer', 'floating', 'decimal'])
        if numeric_cols_count_res >= 1 and col1_dtype_res in ['string', 'categorical', 'datetime', 'datetime64', 'date', 'period']: # Index/first col is suitable for grouping
             if num_cols_res > 2 : # Needs at least one categorical and two numeric, or one index + multiple numeric
                graph_types_sugg.extend(["Multi-Series Line Chart", "Stacked Bar Chart", "Grouped Bar Chart"])


    elif num_cols_res == 1: # Single column result
        if pd.api.types.is_numeric_dtype(df_result.iloc[:,0]):
            graph_types_sugg.extend(["Histogram", "Line Chart (Index vs Value)"])
        elif df_result.iloc[:,0].nunique() <=30 : # if categorical with few unique values
            graph_types_sugg.extend(["Bar Chart (Counts)"])


    return sorted(list(set(graph_types_sugg)))


def plot_suggested_chart(st_object, selected_chart, result_df):
    """Plots the selected chart type for the result_df using st_object."""
    if selected_chart == "Table View" or result_df is None or result_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_successful = False
    try:
        df_to_plot = result_df.copy()
        num_cols = df_to_plot.shape[1]
        col1_name = df_to_plot.columns[0]
        
        # Ensure first column (often x-axis) is string for bar charts if it's not distinctly numeric/datetime
        if selected_chart in ["Bar Chart", "Horizontal Bar Chart"] and num_cols >=1:
             if not pd.api.types.is_datetime64_any_dtype(df_to_plot[col1_name]) and \
                df_to_plot[col1_name].nunique() <=30 : # and not pd.api.types.is_numeric_dtype(df_to_plot[col1_name])
                 df_to_plot[col1_name] = df_to_plot[col1_name].astype(str)


        if selected_chart == "Bar Chart":
            if num_cols >= 2: sns.barplot(x=col1_name, y=df_to_plot.columns[1], data=df_to_plot, ax=ax, palette="mako"); ax.tick_params(axis='x', rotation=45, ha='right')
            elif num_cols == 1: counts = df_to_plot[col1_name].value_counts().nlargest(30); sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="mako"); ax.tick_params(axis='x', rotation=45, ha='right'); ax.set_ylabel("Count")
            else: st_object.info("Bar chart needs at least 1 column (for counts) or 2 columns (for x, y)."); plt.close(fig); return
        elif selected_chart == "Horizontal Bar Chart":
            if num_cols >= 2: sns.barplot(x=df_to_plot.columns[1], y=col1_name, data=df_to_plot, ax=ax, orient='h', palette="mako")
            elif num_cols == 1: counts = df_to_plot[col1_name].value_counts().nlargest(30); sns.barplot(x=counts.values, y=counts.index, ax=ax, orient='h', palette="mako"); ax.set_xlabel("Count")
            else: st_object.info("Horizontal Bar chart needs at least 1 column (for counts) or 2 columns (for x, y)."); plt.close(fig); return
        elif selected_chart == "Line Chart":
            if num_cols >= 2:
                if pd.api.types.is_datetime64_any_dtype(df_to_plot[col1_name]) or pd.api.types.is_numeric_dtype(df_to_plot[col1_name]):
                    df_to_plot = df_to_plot.sort_values(by=col1_name) # Sort by X for proper line plot
                ax.plot(df_to_plot[col1_name], df_to_plot[df_to_plot.columns[1]], marker='o')
                ax.tick_params(axis='x', rotation=45, ha='right')
            else: st_object.info("Line chart typically needs at least 2 columns (X and Y)."); plt.close(fig); return
        elif selected_chart == "Scatter Plot":
            if num_cols >= 2: ax.scatter(df_to_plot[col1_name], df_to_plot[df_to_plot.columns[1]]); ax.tick_params(axis='x', rotation=45, ha='right')
            else: st_object.info("Scatter plot needs at least 2 columns (X and Y)."); plt.close(fig); return
        elif selected_chart == "Histogram" and num_cols == 1:
            sns.histplot(df_to_plot.iloc[:,0].dropna(), kde=True, ax=ax, bins=20)
        elif selected_chart == "Line Chart (Index vs Value)" and num_cols == 1:
            ax.plot(df_to_plot.index, df_to_plot.iloc[:,0], marker='o')
        elif selected_chart in ["Multi-Series Line Chart", "Stacked Bar Chart", "Grouped Bar Chart"] and num_cols > 1:
            df_multi_plot = df_to_plot.set_index(col1_name)
            numeric_cols_for_plot = [col for col in df_multi_plot.columns if pd.api.types.is_numeric_dtype(df_multi_plot[col])]
            if not numeric_cols_for_plot: st_object.warning("No suitable numeric columns found for multi-series chart after setting index."); plt.close(fig); return
            
            if len(numeric_cols_for_plot) > 7: # Limit number of series plotted for clarity
                numeric_cols_for_plot = numeric_cols_for_plot[:7]
                st_object.caption(f"Plotting first {len(numeric_cols_for_plot)} numeric series.")

            if selected_chart == "Multi-Series Line Chart": df_multi_plot[numeric_cols_for_plot].plot(ax=ax, marker='o')
            elif selected_chart == "Stacked Bar Chart": df_multi_plot[numeric_cols_for_plot].plot(kind='bar', stacked=True, ax=ax)
            elif selected_chart == "Grouped Bar Chart": df_multi_plot[numeric_cols_for_plot].plot(kind='bar', stacked=False, ax=ax)
            ax.legend(title="Series", bbox_to_anchor=(1.05, 1), loc='upper left'); ax.tick_params(axis='x', rotation=45, ha='right')
        else:
            st_object.info(f"Chart type '{selected_chart}' is not applicable or not fully configured for this data structure.")
            plt.close(fig)
            return

        ax.set_title(f"{selected_chart} of Query Result", fontsize=12)
        # Smartly set x and y labels
        if num_cols >= 1 : ax.set_xlabel(col1_name, fontsize=10)
        if num_cols >=2 and selected_chart not in ["Multi-Series Line Chart", "Stacked Bar Chart", "Grouped Bar Chart"]:
            ax.set_ylabel(df_to_plot.columns[1], fontsize=10)
        elif num_cols == 1 and selected_chart != "Histogram":
             ax.set_ylabel(col1_name, fontsize=10) # For Line Chart (Index vs Value)
        elif selected_chart == "Histogram":
            ax.set_ylabel("Frequency", fontsize=10)


        plt.tight_layout(rect=[0, 0, 0.85, 1] if "Multi-Series" in selected_chart or "Grouped Bar" in selected_chart or "Stacked Bar" in selected_chart else [0,0,1,1]) # Adjust layout for legend
        st_object.pyplot(fig)
        plot_successful = True

    except Exception as e_chart: # pylint: disable=broad-except
        st_object.error(f"Error creating chart '{selected_chart}': {e_chart}")
    finally:
        if fig: # Ensure fig is defined
            plt.close(fig)
    return plot_successful