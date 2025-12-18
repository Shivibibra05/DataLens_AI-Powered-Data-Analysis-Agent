import streamlit as st
import pandas as pd
import duckdb
import tempfile
import csv
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import google.generativeai as genai
import openpyxl 


def apply_custom_theme():
    
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Remove white header background */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a4d6d 0%, #0a2540 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1a1f2e;
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    /* Text */
    p, label, .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #263244 !important;
        color: #ffffff !important;
        border: 1px solid #3a4a5e !important;
        border-radius: 5px;
        caret-color: #ffffff !important;
    }
    
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #8b95a5 !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border: 1px solid #00d4ff !important;
        box-shadow: 0 0 0 1px #00d4ff !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00a8cc 0%, #0086a8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #00c4e6 0%, #00a8cc 100%) !important;
        box-shadow: 0 4px 12px rgba(0, 168, 204, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Download button specific styling */
    .stDownloadButton button {
        background: linear-gradient(90deg, #16a34a 0%, #15803d 100%) !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%) !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #263244 !important;
        border: 2px dashed #3a4a5e !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Data tables - using Streamlit's height parameter instead of CSS */
    [data-testid="stDataFrame"] {
        background-color: #2d3748 !important;
        border-radius: 8px !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e2936 !important;
        border: 1px solid #3a4a5e !important;
        border-radius: 5px !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background-color: #263244 !important;
        border-left: 4px solid #00d4ff !important;
        color: #e0e0e0 !important;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #1a3d2e !important;
        color: #4ade80 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }
    
    /* Divider */
    hr {
        border-color: #3a4a5e !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #263244 !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------
# Utility: File Preprocessing
# --------------------------------
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        elif file.name.endswith('.txt'):
            df = pd.read_csv(file, delimiter='\t', na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("❌ Unsupported file format. Please upload a CSV, Excel, JSON, or TXT file.")
            return None, None, None

        
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        st.success(f"✅ Loaded {file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        return temp_path, df.columns.tolist(), df

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
        return None, None, None

# Utility: Extract SQL and Explanation
# --------------------------------
def extract_sql_and_answer(response_text):
    sql_match = re.search(r"```sql(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    sql_query = sql_match.group(1).strip() if sql_match else None
    explanation = re.sub(r"```sql(.*?)```", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
    return sql_query, explanation

# --------------------------------
# Streamlit Setup
# --------------------------------
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide", initial_sidebar_state="expanded")

# Apply custom theme
apply_custom_theme()

st.title("🤖 Data Lens")
st.markdown("### AI-powered Data Analysis ")
st.markdown("*Upload your dataset and ask  questions*")

# --- MODIFICATION: API Key Handling ---
with st.sidebar:
    st.header("Quick Guide")
    st.info("""
    **Steps to get started:**
    
    1️⃣ Upload your data file
    
    2️⃣ Ask questions about your data
    
    **Example queries:**
    - "Show top 10 sales by region"
    - "Calculate average revenue"
    - "Find customers with orders > 1000"
    """)
    
   
# 2. ADD check for Streamlit Secrets
def run_app():
    # --------------------------------
    # File Upload Section
    # --------------------------------
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx", "json", "txt"], help="Supported formats: CSV, Excel, JSON, TXT")

    if uploaded_file is not None:
        temp_path, columns, df = preprocess_and_save(uploaded_file)

        if temp_path:
            st.markdown("---")
            st.subheader(" Data Preview")
            
            
            html_table = df.head(10).to_html(index=False, escape=False, classes='styled-table')
            styled_table = f"""
            <div style="background-color: #2d3748; padding: 20px; border-radius: 8px; overflow-x: auto;">
                <style>
                    .styled-table {{ width: 100%; border-collapse: collapse; color: #e0e0e0; table-layout: auto; }}
                    .styled-table thead tr {{ background-color: #1e293b; }}
                    .styled-table th {{ background-color: #1e293b; color: #00d4ff; padding: 12px 15px; text-align: left; font-weight: 600; white-space: nowrap; }}
                    .styled-table td {{ background-color: #2d3748; padding: 10px 15px; border-bottom: 1px solid #3a4a5e; text-align: left; }}
                    .styled-table tbody tr:hover {{ background-color: #374151; }}
                    .styled-table tbody tr:hover td {{ background-color: #374151; }}
                </style>
                {html_table}
            </div>
            """
            st.markdown(styled_table, unsafe_allow_html=True)

            # Create DuckDB Table
            conn = duckdb.connect(database=':memory:')
            conn.execute(f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{temp_path}')")

            # --------------------------------
            # Query Section
            # --------------------------------
            st.markdown("---")
            st.subheader("💬 Ask a Question About Your Data")
            user_query = st.text_area(
                "Enter your query:",
                placeholder="Example: 'Show average sales by region' or 'Top 5 customers by revenue'",
                height=100
            )

            if st.button(" Analyze Data"):
                if not user_query.strip():
                    st.warning("⚠️ Please enter a query.")
                else:
                    with st.spinner("🔍 Analyzing your data..."):
                        prompt = f"""
                        You are a skilled data analyst.
                        Convert this natural language question into a valid SQL query for a table named 'uploaded_data'
                        with columns: {list(df.columns)}.
                        Format the output exactly as:

                        ```sql
                        <SQL query here>
                        ```
                        <Brief explanation of what this query does>

                        Question: {user_query}
                        """

                        try:
                            # 3. CONFIGURE Gemini with the secret key
                            
                            model = genai.GenerativeModel("models/gemini-2.5-flash")
                            response = model.generate_content(prompt)
                            content = response.text

                            
                            sql_query, explanation = extract_sql_and_answer(content)

                            st.markdown("---")
                            st.markdown("### AI Response")

                            if sql_query:
                                st.markdown("**Generated SQL Query:**")
                                st.code(sql_query, language="sql")

                            if explanation:
                                st.markdown(f"**Explanation:** {explanation}")

                            if sql_query:
                                try:
                                    result = conn.execute(sql_query).fetchdf()

                                    if not result.empty:
                                        st.markdown("---")
                                        st.markdown("###  Query Results")
                                        total_rows = len(result)
                                        st.info(f" Found **{total_rows}** records")
                                        
                                        display_limit = 50
                                        if total_rows > display_limit:
                                            st.warning(f"⚠️ Showing first {display_limit} rows out of {total_rows}. Download full results if needed.")
                                            display_df = result.head(display_limit)
                                        else:
                                            display_df = result
                                        
                                        html_result = display_df.to_html(index=False, escape=False, classes='styled-table')
                                        styled_result = f"""
                                        <div style="background-color: #2d3748; padding: 20px; border-radius: 8px; overflow-x: auto;">
                                            <style>
                                                .styled-table {{ width: 100%; border-collapse: collapse; color: #e0e0e0; table-layout: auto; }}
                                                .styled-table thead tr {{ background-color: #1e293b; }}
                                                .styled-table th {{ background-color: #1e293b; color: #00d4ff; padding: 12px 15px; text-align: left; font-weight: 600; white-space: nowrap; }}
                                                .styled-table td {{ background-color: #2d3748; padding: 10px 15px; border-bottom: 1px solid #3a4a5e; text-align: left; }}
                                                .styled-table tbody tr:hover {{ background-color: #374151; }}
                                                .styled-table tbody tr:hover td {{ background-color: #374151; }}
                                            </style>
                                            {html_result}
                                        </div>
                                        """
                                        st.markdown(styled_result, unsafe_allow_html=True)
                                        
                                        if total_rows > display_limit:
                                            csv = result.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Full Results (CSV)",
                                                data=csv,
                                                file_name="query_results.csv",
                                                mime="text/csv"
                                            )

                                        # --- VISUALIZATIONS (with updated text colors) ---
                                        st.markdown("---")
                                        st.markdown("###  Visualizations")
                                        num_cols = result.select_dtypes(include=['number']).columns
                                        cat_cols = result.select_dtypes(exclude=['number']).columns

                                        if len(result) > 100:
                                            st.info(" **Tip:** For better visualizations, try aggregate queries.")
                                        
                                        if len(num_cols) >= 2 and len(num_cols) <= 10 and len(result) <= 100:
                                            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#263244')
                                            ax.set_facecolor('#263244')
                                            ax.tick_params(axis='x', colors='#e0e0e0')
                                            ax.tick_params(axis='y', colors='#e0e0e0')
                                            heatmap = sns.heatmap(result[num_cols].corr(), annot=True, cmap="Blues", ax=ax, cbar_kws={'label': 'Correlation'}, fmt='.2f')
                                            cbar = heatmap.collections[0].colorbar
                                            cbar.ax.yaxis.label.set_color('#e0e0e0')
                                            cbar.ax.tick_params(colors='#e0e0e0')
                                            plt.title("Correlation Heatmap", color='#00d4ff', pad=20, weight='bold')
                                            st.pyplot(fig)

                                        elif len(cat_cols) >= 1 and len(num_cols) >= 1:
                                            x_col, y_col = cat_cols[0], num_cols[0]
                                            if len(result) > 50:
                                                plot_data = result.groupby(x_col)[y_col].sum().reset_index().head(20)
                                            else:
                                                plot_data = result.head(20)
                                            
                                            fig = px.bar(plot_data, x=x_col, y=y_col, title=f"{y_col} by {x_col} (Top 20)", color=x_col, template="plotly_dark")
                                            fig.update_layout(paper_bgcolor='#263244', plot_bgcolor='#263244', showlegend=False, title_font_color='#00d4ff',
                                                              xaxis=dict(title_font_color='#e0e0e0', tickfont_color='#e0e0e0'),
                                                              yaxis=dict(title_font_color='#e0e0e0', tickfont_color='#e0e0e0'))
                                            st.plotly_chart(fig, use_container_width=True)

                                            unique_cats = plot_data[x_col].nunique()
                                            if unique_cats <= 10:
                                                fig_pie = px.pie(plot_data, names=x_col, values=y_col, title=f"{y_col} Distribution by {x_col}", template="plotly_dark")
                                                fig_pie.update_layout(paper_bgcolor='#263244', title_font_color='#00d4ff', legend_font_color='#e0e0e0')
                                                fig_pie.update_traces(textfont_color='#ffffff', textposition='auto')
                                                st.plotly_chart(fig_pie, use_container_width=True)

                                        elif len(num_cols) == 1:
                                            fig = px.histogram(result.head(1000), x=num_cols[0], nbins=20, title=f"Distribution of {num_cols[0]}", template="plotly_dark")
                                            fig.update_layout(paper_bgcolor='#263244', plot_bgcolor='#263244', title_font_color='#00d4ff',
                                                              xaxis=dict(title_font_color='#e0e0e0', tickfont_color='#e0e0e0'),
                                                              yaxis=dict(title_font_color='#e0e0e0', tickfont_color='#e0e0e0'))
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info("ℹ️ No suitable numeric columns for visualization.")

                                    else:
                                        st.warning("⚠️ No results found for this query.")

                                except Exception as sql_err:
                                    st.error(f"⚠️ SQL Execution Error: {sql_err}")

                            else:
                                st.warning("⚠️ No SQL query detected in Gemini response.")

                        except Exception as genai_err:
                            st.error(f"⚠️ Gemini API Error: {genai_err}")

# --- Main execution ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("🛑 Gemini API Key not found.")
    st.info("Please add your Gemini API Key to Streamlit Secrets to run this app.")
    st.code("GEMINI_API_KEY = 'your_api_key_here'", language="toml")
    st.stop()
else:
    # Configure the API
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        run_app()
    except Exception as e:
        st.error(f"⚠️ Error configuring Gemini API: {e}")
        st.stop()


