#**DataLens: AI-Powered Data Analysis**

DataLens is an intelligent data analytics application that allows users to upload datasets, ask questions in plain English, and instantly receive analytical results, SQL queries, and visualizations. The system uses Google Gemini’s Large Language Model to interpret natural language and generate accurate SQL queries, making data analysis accessible to both technical and non-technical users.

**Features**

1. Natural Language Querying
2. Users can ask questions such as “Top 10 customers by revenue” or “Average sales by region” without needing SQL knowledge.
3. Automatic SQL Generation
4. Queries are converted into SQL statements using Gemini LLM, ensuring interpretability and transparency in analysis.
5. Multi-format Data Upload
6. Supports CSV, Excel, JSON, and TXT files.
7. In-Memory Query Execution
8. Uses DuckDB for fast, efficient SQL execution on local datasets.

**Data Visualization**

1. Automatically generates charts such as bar charts, histograms, heatmaps, and pie charts using Plotly and Seaborn.
2. Interactive Web Interface
3. Built entirely with Streamlit for a clean, real-time analytical workflow.

**Tech Stack**

Python 3.10+
Streamlit (Frontend UI)
Gemini 2.5 Flash (LLM for SQL generation)
DuckDB (In-memory database engine)
Pandas (Data handling)
Plotly & Seaborn (Visualization)
