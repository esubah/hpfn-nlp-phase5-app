#--- START OF FILE vibe_main_phase4_v1.py (Corrected) ---

"""
Main Streamlit application file for the Vibe Test Uploader.
This script imports and uses refactored functions from refactored_vibe_phase4.py.
It aims to implement the full functionality of vibe_test_upload_query_phase4_v2.py.
"""
import streamlit as st
import pandas as pd
import numpy as np # For exec context
import traceback
from openai import OpenAI # RateLimitError, APIConnectionError, OpenAIError imported for direct use if needed
import io
from datetime import datetime
import re # For code cleaning
# No os or tempfile directly used here unless specific need arises

# Import refactored utility functions
import refactored_vibe_phase4 as rvu

# ------------------------- Application Setup -------------------------
st.set_page_config(page_title="HPFN Data Query Tool (V1)", layout="wide") # Changed title

# --- Initialize Session State (must be one of the first Streamlit commands) ---
rvu.initialize_session_state() # Ensures all necessary keys are present

# --- OpenAI Client (Initialize in the main app, as it uses secrets) ---
if "client" not in st.session_state or st.session_state.client is None: # Ensure client is initialized if not already
    try:
        # This line will try to get the key from st.secrets
        api_key_to_use = st.secrets["OPENAI_API_KEY"] # This will raise KeyError if not found
        st.session_state.client = OpenAI(api_key=api_key_to_use)
        st.sidebar.success("OpenAI Client Initialized via st.secrets!")
    except KeyError: 
        st.sidebar.warning("OPENAI_API_KEY NOT FOUND in st.secrets. Attempting initialization without explicit key (may use ENV VAR).")
        try:
            st.session_state.client = OpenAI() # Initialize without explicit key
            if hasattr(st.session_state.client, 'api_key') and st.session_state.client.api_key:
                 st.sidebar.success("OpenAI Client Initialized (likely via ENV VAR, as not found in st.secrets).")
            else:
                 # This path might be hard to reach if OpenAI() without key just fails earlier
                 st.error("OpenAI client initialization failed: No API key found in st.secrets or discoverable from ENV by OpenAI lib.")
                 st.stop()
        except Exception as e_env_init: 
            st.error(f"OpenAI client initialization failed after st.secrets miss: {e_env_init}")
            st.stop()
    except Exception as e_client_init: 
        st.error(f"General OpenAI client initialization failed: {e_client_init}. Please ensure OPENAI_API_KEY is set correctly.")
        st.stop()
client = st.session_state.client


# --- Render Sidebar and Get Feature Toggles ---
# The prompt_library is defined in rvu, so we can access it directly if needed,
# but render_sidebar handles its status display.
sidebar_config = rvu.render_sidebar(rvu.prompt_library) 
# Update session state with returned toggle values (render_sidebar already updates session_state)
# This is more for clarity if you want to use these variables directly in main_app logic
debug_mode = st.session_state.debug_mode
enable_df_pagination = st.session_state.enable_df_pagination
enable_result_pagination = st.session_state.enable_result_pagination
enable_multilingual = st.session_state.enable_multilingual
enable_graph_suggestions = st.session_state.enable_graph_suggestions
enable_data_cleaning_toggle = st.session_state.enable_data_cleaning_toggle

# ------------------------- Main Application UI -------------------------
# Title was already set by st.set_page_config, but if you want a main area title:
st.title("📊 HPFN Data Query Tool (V1)") # This can be redundant if page_title in set_page_config is similar


# --- File Upload Section ---
uploaded_file = st.file_uploader(
    "Upload a structured data file (CSV or Excel only)", 
    type=["csv", "xlsx", "xls"],
    key="file_uploader_key" # Add a key for stability
)

# Check if a new file is uploaded or if it's the same as before
new_file_uploaded_this_run = False
if uploaded_file is not None:
    if "last_uploaded_filename" not in st.session_state or \
       st.session_state.last_uploaded_filename != uploaded_file.name:
        new_file_uploaded_this_run = True
        st.session_state.last_uploaded_filename = uploaded_file.name
        # Clear previous data if a new file is truly uploaded
        st.session_state.df_original = None
        st.session_state.df_processed = None
        st.session_state.query_log = [] # Reset query log for new file
else: # No file currently in uploader widget
    if "last_uploaded_filename" in st.session_state and st.session_state.last_uploaded_filename is not None:
        # This means a file was there, but now it's removed from the uploader
        # To signify this, we can clear the last_uploaded_filename so next upload is "new"
        # Or let user continue with session data if df_original exists.
        # For now, just pass. If user re-uploads same name, it won't be "new_file_uploaded_this_run"
        pass


# Process file if df_original is not set (new upload) or if a new file is detected
if uploaded_file and (st.session_state.get("df_original") is None or new_file_uploaded_this_run):
    st.info(f"Loading and processing `{uploaded_file.name}`...")
    df_loaded = rvu.load_dataframe_from_uploaded_file(uploaded_file)

    if df_loaded is not None:
        st.session_state.df_original = df_loaded.copy()
        # Processing (including minimal cleaning) is now inside rvu.process_dataframe
        st.session_state.df_processed = rvu.process_dataframe(
            st.session_state.df_original.copy(), # Pass a copy to process_dataframe
            cleaning_enabled=enable_data_cleaning_toggle
        )
        st.success(f"Successfully loaded and initially processed `{uploaded_file.name}` ({st.session_state.df_original.shape[0]} rows, {st.session_state.df_original.shape[1]} columns).")
    else:
        # Error is already handled by load_dataframe_from_uploaded_file
        st.session_state.df_original = None # Ensure it's cleared on load failure
        st.session_state.df_processed = None


# --- Data Exploration UI (if data is loaded) ---
if st.session_state.get("df_original") is not None: # Use .get for safety
    st.markdown("---")
    st.write("#### 🔬 Data Exploration")

    # Display Data Quality Hints for the original data
    with st.expander("💡 Raw Data Quality Hints", expanded=False):
        quality_issues = rvu.get_data_quality_issues(st.session_state.df_original, enable_data_cleaning_toggle)
        if quality_issues and len(quality_issues) > 1:
            for issue in quality_issues[1:]: # Skip the header
                if "⚠️" in issue: st.warning(issue)
                elif "ℹ️" in issue: st.info(issue)
                elif "✅" in issue: st.success(issue)
                else: st.markdown(issue)
        else:
            st.info("No specific quality hints found, or quality checks passed.")
        st.caption("Hints based on raw data. 'Processed Data' reflects cleaning attempts if enabled.")

    # Data View Selection (Raw vs Processed)
    view_options = ["Processed Data (Used for Queries)", "Raw Uploaded Data"]
    
    default_view_index = 0
    if st.session_state.get("df_processed") is None :
        default_view_index = 1 # Default to Raw if processed is None

    selected_view_option = st.radio(
        "Select data view for exploration:",
        view_options,
        index=default_view_index,
        horizontal=True,
        key="data_view_selector"
    )

    df_to_explore = None
    if selected_view_option == "Processed Data (Used for Queries)":
        df_to_explore = st.session_state.get("df_processed")
        if df_to_explore is None:
            st.warning("Processed data is not available. This might happen if the initial processing failed or the file was just removed. Showing Raw Data instead if available.")
            df_to_explore = st.session_state.get("df_original") # Fallback
            if df_to_explore is not None: selected_view_option = "Raw Uploaded Data" # Update displayed option
    else: # Raw Uploaded Data
        df_to_explore = st.session_state.get("df_original")

    if df_to_explore is not None and not df_to_explore.empty:
        st.write(f"Displaying: **{selected_view_option}** (Shape: {df_to_explore.shape[0]} rows, {df_to_explore.shape[1]} columns)")
        
        # Data Preview
        st.markdown("##### Data Preview")
        if enable_df_pagination and df_to_explore.shape[0] > 10:
            top_n_preview = st.slider("Show top N rows for preview:", 5, min(100, df_to_explore.shape[0]), 10, key="top_n_preview_slider_main")
            st.dataframe(df_to_explore.head(top_n_preview), use_container_width=True)
            with st.expander("Show bottom 5 rows of preview"):
                st.dataframe(df_to_explore.tail(5), use_container_width=True)
        else:
            st.dataframe(df_to_explore.head(min(20, df_to_explore.shape[0])), use_container_width=True) # Show a bit more if not paginated

        # Schema Summary
        st.markdown("##### Data Schema Summary")
        schema_df = rvu.get_schema_summary_df(df_to_explore)
        if not schema_df.empty:
            st.dataframe(schema_df, use_container_width=True)
        else:
            st.info("Could not generate schema summary for the selected view.")

        # Column Distributions
        st.markdown("##### Column Distributions")
        numeric_cols_dist = df_to_explore.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_dist = df_to_explore.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist() # Include boolean
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            cols_to_plot_num = st.multiselect(
                "Select numeric columns for histograms:", numeric_cols_dist, 
                default=numeric_cols_dist[:min(3, len(numeric_cols_dist))], 
                key="dist_num_select_main"
            )
        with col_sel2:
            cols_to_plot_cat = st.multiselect(
                "Select categorical/boolean columns for bar plots:", categorical_cols_dist, 
                default=categorical_cols_dist[:min(3, len(categorical_cols_dist))], 
                key="dist_cat_select_main"
            )
        
        rvu.generate_column_distribution_plots(st, df_to_explore, cols_to_plot_num, cols_to_plot_cat)

        # Descriptive Statistics
        st.markdown("##### Descriptive Statistics (Numeric Columns Only)")
        if numeric_cols_dist:
            try:
                st.dataframe(df_to_explore[numeric_cols_dist].describe().T, use_container_width=True)
            except Exception as e_desc: # pylint: disable=broad-except
                st.warning(f"Could not generate numeric descriptive statistics: {e_desc}")
        else:
            st.info("No numeric columns in this data view to describe.")

    elif df_to_explore is None and st.session_state.get("df_original") is not None : # Original was loaded but selected view (processed) is None
        st.warning("Processed data is not available (e.g. processing failed). Please check cleaning logs if auto-cleaning was enabled.")
    elif uploaded_file and df_to_explore is None : # File was uploaded but df_to_explore ended up None (e.g. load failed)
        st.warning("Data could not be loaded or processed from the uploaded file.")
    # else: # df_original is None (no file uploaded yet) or df_to_explore is empty
    #    if st.session_state.get("df_original") is not None and df_to_explore is not None and df_to_explore.empty:
    #        st.warning(f"The '{selected_view_option}' is empty. This might indicate an issue with the uploaded file or the cleaning process.")


# --- NLP Query Interface ---
query_df_for_nlp = st.session_state.get('df_processed')

if query_df_for_nlp is not None and not query_df_for_nlp.empty:
    st.markdown("---")
    st.subheader("🧠 Ask a question about your data")
    query_data_source_msg = "Processed Data (after type cleaning if enabled)" if enable_data_cleaning_toggle else "Minimally Cleaned Data"
    st.info(f"Queries will run on: **{query_data_source_msg}**.")

    with st.expander("📝 Example Queries (Adapt to your data!)", expanded=False):
        st.markdown("""
        - "What is the total sum of `[SalesColumn]`?"
        - "Show the average `[ProfitColumn]` for each `[CategoryColumn]`."
        - "Group by `[RegionColumn]` and then `[ProductTypeColumn]` and count records."
        - "List the top 10 `[ProductColumn]` by total `[RevenueColumn]` in descending order."
        - "Filter for `[StatusColumn]` equal to 'Shipped' and `[OrderDateColumn]` in 2023."
        - "Summarize `[NumericColumn]`." (Shows descriptive stats)
        - "How many unique `[CustomerIDColumn]` are there?"
        """)
        st.caption("Replace `[ColumnName]` with actual column names from your data. Be specific!")

    user_question = st.text_input(
        "Type your question here:", 
        value=st.session_state.get("user_query_input", ""), 
        key="user_query_main_input",
        placeholder="e.g., 'What is the average sales per region?'"
    )
    st.session_state.user_query_input = user_question # Persist input across reruns

    if st.button("🚀 Get Answer", key="get_answer_button") and user_question:
        with st.spinner("Thinking and crunching numbers..."):
            try:
                # --- Prepare Prompt Context ---
                column_info_list = []
                for col_name in query_df_for_nlp.columns: # Renamed col to col_name for clarity
                    col_data = query_df_for_nlp[col_name] # Get Series data
                    dtype_str = str(col_data.dtype)
                    if pd.api.types.is_datetime64_any_dtype(col_data):
                        dtype_str = "datetime"
                    elif pd.api.types.is_numeric_dtype(col_data):
                        if pd.api.types.is_integer_dtype(col_data):
                            dtype_str = "integer"
                        else:
                            dtype_str = "float"
                    elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data): # Added object
                         dtype_str = "string/categorical" # Consolidate object here

                    column_info_list.append(f"`{col_name}` (type: {dtype_str})")
                column_info = ', '.join(column_info_list)

                sample_rows = query_df_for_nlp.head(min(3, len(query_df_for_nlp)))
                sample_data_str = "Sample Data (first few rows):\n" + sample_rows.to_string(index=False)
                
                language_instruction = "User query is in English."
                if enable_multilingual: 
                    language_instruction = "User query might be multilingual. Prioritize interpreting the analytical intent."

                formatted_examples = rvu.format_prompt_examples(rvu.prompt_library, num_examples=6)

                prompt = (
                    f"You are a Python data analysis expert using the pandas library. Your task is to generate Python code to answer a user's question about a dataframe named `df`.\n"
                    f"{language_instruction}\n\n"
                    f"The dataframe `df` has the following schema (column names and their inferred types):\n{column_info}\n\n"
                    f"{sample_data_str}\n\n"
                    f"{formatted_examples}" 
                    f"Now, based on the user's question below, generate **only** the executable Python code using pandas. Assign the final result (DataFrame, Series, or single value) to a variable named `result`.\n"
                    f"User Question: \"{user_question}\"\n\n"
                    f"**Important Instructions for Code Generation:**\n"
                    f"1.  **DataFrame Name:** Use `df` to refer to the dataframe.\n"
                    f"2.  **Column Access:** Use the exact column names provided in the schema (e.g., `df['ActualColumnName']`). Column names are case-sensitive.\n"
                    f"3.  **Type Handling:** If the query implies operations on specific data types (e.g., date calculations, numeric aggregations), ensure the code respects these types. The schema provides inferred types. If a user query suggests a type mismatch (e.g. summing a text column), the code should ideally fail or handle it gracefully if an obvious conversion is implied (e.g., text '123' to numeric 123 for sum). Prefer direct operations if type fits. Explicit conversions like `pd.to_datetime(df['col'], errors='coerce')` or `pd.to_numeric(df['col'], errors='coerce')` can be used if strongly implied and safe.\n"
                    f"4.  **Missing Data:** Be mindful of missing data (NaN/NaT). Use methods like `.dropna()`, `.fillna()`, or checks like `.isnull()`/`.notnull()` appropriately if the query context suggests sensitivity to missing values (e.g., before aggregations if NaNs should be excluded, or when filtering).\n"
                    f"5.  **Ambiguity & Clarification:**\n"
                    f"    - If the user's query is highly ambiguous or lacks critical information (e.g., 'show data' without specifying columns or conditions), and you cannot make a reasonable inference, set `result` to a string asking for clarification: `result = \"Clarification needed: Your query is too ambiguous. Please specify columns, conditions, or the type of summary you need.\"`\n"
                    f"    - **'Summarize' Keyword Special Handling:**\n"
                    f"        - If query is 'summarize `[COLUMN_NAME]`': If `[COLUMN_NAME]` is numeric, use `df['[COLUMN_NAME]'].describe()`. If categorical/object, use `df['[COLUMN_NAME]'].value_counts().reset_index().rename(columns={{'index':'Value', '[COLUMN_NAME]':'Count'}})`. If datetime, provide min, max, and count.\n"
                    f"        - If query is 'summarize a_few_words_column_name': Infer the column name from the schema. If no match, ask for clarification.\n"
                    f"        - If query is 'summarize' (general, no column): Use `df.describe(include='all').T`.\n"
                    f"        - If query uses 'summarize' **AND** an explicit aggregation like 'total', 'average' for a specific column (e.g., 'summarize total sales'), this is ambiguous. Set `result` to this exact string, replacing `[COLUMN_NAME]` and `[AGG_TERM]`: `result = \"Ambiguous request: Term 'summarize' combined with aggregation '[AGG_TERM]' for column '[COLUMN_NAME]' is unclear. Do you want 'Calculate [AGG_TERM] [COLUMN_NAME]' or 'Show descriptive statistics for [COLUMN_NAME]'? Please rephrase.\"`\n"
                    f"6.  **Aggregations:** For explicit aggregations (sum, mean, median, count, min, max, nunique), generate direct pandas code (e.g., `df['Sales'].sum()`, `df.groupby('Category')['Profit'].mean()`).\n"
                    f"7.  **Grouping:** If grouping is requested:\n"
                    f"    - If specific aggregations are mentioned (e.g., 'total sales by region'), perform that: `df.groupby('Region')['Sales'].sum().reset_index()`.\n"
                    f"    - If no specific aggregation is mentioned for numeric columns (e.g., 'group by Region'), apply `.describe()` to all other available numeric columns in each group: `df.groupby('Region')[['NumericCol1', 'NumericCol2']].describe()`.\n"
                    f"    - If grouping by a column and then requesting counts of another, use `.size()` or `.count()`: `df.groupby('Category')['Product'].count().reset_index(name='ProductCount')`.\n"
                    f"8.  **Output:** Provide **only the Python code block**. No explanations, comments, or ```python ... ``` markers.\n"
                    f"9.  **Result Variable:** The final output of the code (DataFrame, Series, or scalar) MUST be assigned to a variable named `result`.\n"
                    f"Ensure the generated code is a single block of executable Python."
                )

                generated_code = ""
                api_error = None
                try:
                    if not client: 
                        st.error("OpenAI client not available.")
                        st.stop()
                    
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo-0125",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    if not completion or not completion.choices:
                        api_error = "No response or choices from OpenAI."
                    else:
                        generated_code = completion.choices[0].message.content.strip()
                        generated_code = re.sub(r"^```python\n?", "", generated_code, flags=re.MULTILINE)
                        generated_code = re.sub(r"\n?```$", "", generated_code, flags=re.MULTILINE)
                        generated_code = generated_code.strip()
                
                except Exception as e_api: 
                    api_error = f"OpenAI API Error: {e_api}"
                
                if api_error:
                    st.error(api_error)
                    st.session_state.query_log.append({"timestamp": datetime.now().isoformat(), "question": user_question, "code": "API Error", "success": False, "error": api_error, "result_preview": None})
                    st.stop()

                if not generated_code:
                    st.warning("LLM did not return any code. Please try rephrasing your question.")
                    st.session_state.query_log.append({"timestamp": datetime.now().isoformat(), "question": user_question, "code": "No code generated", "success": False, "error": "No code returned by LLM", "result_preview": None})
                    st.stop()

                st.markdown("##### Generated Code:")
                st.code(generated_code, language="python")

                exec_namespace = {'df': query_df_for_nlp.copy(), 'pd': pd, 'np': np} 
                query_result = None
                exec_error_msg = None
                try:
                    exec(generated_code, exec_namespace) 
                    query_result = exec_namespace.get('result', None)
                except Exception as e_exec: 
                    exec_error_msg = str(e_exec)
                    if debug_mode:
                        st.error(f"Code Execution Trace:\n{traceback.format_exc()}")
                    else:
                        st.error(f"Code execution failed: {exec_error_msg}")
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "question": user_question,
                    "code": generated_code,
                    "success": exec_error_msg is None,
                    "error": exec_error_msg,
                    "result_preview": None 
                }

                if exec_error_msg: 
                    pass 
                elif query_result is None:
                    st.warning("Query executed, but the `result` variable was not found or was None.")
                    log_entry["error"] = "Result variable not found or None"
                    log_entry["success"] = False 
                else: 
                    st.success("✅ Query Executed Successfully!")
                    
                    if isinstance(query_result, str) and ("Clarification needed:" in query_result or "Ambiguous request:" in query_result):
                        st.info(f"🤔 LLM Asks: {query_result}")
                        log_entry["result_preview"] = query_result
                        log_entry["error"] = "Clarification requested by LLM" 
                    else:
                        result_df_display = None
                        if isinstance(query_result, pd.DataFrame):
                            result_df_display = query_result.copy()
                        elif isinstance(query_result, pd.Series):
                            result_df_display = query_result.reset_index()
                            if len(result_df_display.columns) == 2 and ('index' in str(result_df_display.columns[0]).lower() or result_df_display.columns[0] == 0):
                                col0_name = query_result.index.name if query_result.index.name else 'Index'
                                col1_name = query_result.name if query_result.name else 'Value'
                                result_df_display.columns = [col0_name, col1_name]
                        elif isinstance(query_result, (int, float, str, bool, np.generic)): 
                            result_df_display = pd.DataFrame({'Result': [query_result]})
                        else: 
                            st.warning(f"Result type ({type(query_result)}) may not be fully displayable. Attempting to show as string.")
                            result_df_display = pd.DataFrame({'Result': [str(query_result)]})

                        if result_df_display is not None and not result_df_display.empty:
                            st.markdown("##### Query Result:")
                            if enable_result_pagination and result_df_display.shape[0] > 20:
                                st.write(f"Showing top 20 of {result_df_display.shape[0]} rows.")
                                st.dataframe(result_df_display.head(20), use_container_width=True)
                                with st.expander("Show all rows of result"):
                                    st.dataframe(result_df_display, use_container_width=True)
                            else:
                                st.dataframe(result_df_display.head(min(100, len(result_df_display))), use_container_width=True)
                            
                            log_entry["result_preview"] = result_df_display.head(5).to_string()

                            if enable_graph_suggestions:
                                st.markdown("---")
                                st.markdown("##### 📊 Auto Graph Recommendation")
                                suggested_types = rvu.get_suggested_chart_types(result_df_display)
                                if len(suggested_types) > 1: 
                                    selected_chart_type = st.selectbox(
                                        "Choose a chart type to visualize the result:",
                                        suggested_types,
                                        index=0, 
                                        key="result_chart_selector"
                                    )
                                    if selected_chart_type != "Table View":
                                        rvu.plot_suggested_chart(st, selected_chart_type, result_df_display)
                                else:
                                    st.info("No specific graph types strongly suggested for this result structure beyond the table view.")

                            st.markdown("---")
                            st.markdown("##### Export Result")
                            export_col1, export_col2 = st.columns(2)
                            with export_col1:
                                try:
                                    csv_export_data = result_df_display.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        "⬇️ Download as CSV", csv_export_data, 
                                        f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                                        "text/csv", key="csv_download_result_button"
                                    )
                                except Exception as e_csv_export: 
                                    st.error(f"CSV Export Error: {e_csv_export}")
                            with export_col2:
                                excel_buffer = io.BytesIO()
                                try:
                                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                        result_df_display.to_excel(writer, sheet_name='QueryResult', index=False)
                                    excel_buffer.seek(0)
                                    st.download_button(
                                        "⬇️ Download as Excel", excel_buffer,
                                        f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="excel_download_result_button"
                                    )
                                except ImportError:
                                    st.warning("Excel export requires 'xlsxwriter'. Install with: pip install xlsxwriter")
                                except Exception as e_excel_export: 
                                    st.error(f"Excel Export Error: {e_excel_export}")
                        
                        elif result_df_display is not None and result_df_display.empty:
                            st.info("Query executed successfully, but the result is an empty table.")
                            log_entry["result_preview"] = "Empty DataFrame"
                        else: 
                            st.warning("Query result could not be displayed in a standard table format.")
                            log_entry["result_preview"] = "Non-standard display"
                
                st.session_state.query_log.append(log_entry)

            except Exception as e_outer_query: 
                st.error(f"An unexpected error occurred during query processing: {e_outer_query}")
                if debug_mode: st.error(f"Outer Query Trace:\n{traceback.format_exc()}")
                st.session_state.query_log.append({
                    "timestamp": datetime.now().isoformat(), "question": user_question, 
                    "code": generated_code if 'generated_code' in locals() else "Error before code generation",
                    "success": False, "error": str(e_outer_query), "result_preview": None
                })

elif st.session_state.get('df_original') is None and not uploaded_file: 
    st.info("👋 Welcome! Please upload a CSV or Excel file to begin your analysis.")
elif query_df_for_nlp is not None and query_df_for_nlp.empty:
    st.warning("The data available for querying is empty. This could be due to an empty uploaded file or issues during the data cleaning process. Please check the uploaded file or data cleaning logs.")
elif query_df_for_nlp is None and st.session_state.get('df_original') is not None:
     st.warning("Processed data is not available for querying. This might indicate a failure during the data processing step. Check logs if available.")


# --- Display Query Log (if any and in debug mode) ---
if debug_mode and "query_log" in st.session_state and st.session_state.query_log:
    with st.expander("📜 Query Log (Debug View - Last 10)", expanded=False):
        reversed_log = st.session_state.query_log[::-1]
        st.json(reversed_log[:10]) 

st.markdown("---")
st.caption("End of application.")