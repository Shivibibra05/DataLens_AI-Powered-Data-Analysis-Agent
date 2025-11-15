DataLens: AI-Powered Data Analysis

DataLens is an intelligent data analytics application that allows users to upload datasets, ask questions in plain English, and instantly receive analytical results, SQL queries, and visualizations. The system uses Google Gemini’s Large Language Model to interpret natural language and generate accurate SQL queries, making data analysis accessible to both technical and non-technical users.

Features
Natural Language Querying

Users can ask questions such as “Top 10 customers by revenue” or “Average sales by region” without needing SQL knowledge.

Automatic SQL Generation

Queries are converted into SQL statements using Gemini LLM, ensuring interpretability and transparency in analysis.

Multi-format Data Upload

Supports CSV, Excel, JSON, and TXT files.

In-Memory Query Execution

Uses DuckDB for fast, efficient SQL execution on local datasets.

Data Visualization

Automatically generates charts such as bar charts, histograms, heatmaps, and pie charts using Plotly and Seaborn.

Interactive Web Interface

Built entirely with Streamlit for a clean, real-time analytical workflow.

Tech Stack

Python 3.10+

Streamlit (Frontend UI)

Gemini 2.5 Flash (LLM for SQL generation)

DuckDB (In-memory database engine)

Pandas (Data handling)

Plotly & Seaborn (Visualization)
