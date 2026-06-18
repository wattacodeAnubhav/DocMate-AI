import sqlite3
import pandas as pd
import re
import os

DB_PATH = "./data/docmate.db"

def load_csv_to_sqlite(csv_path: str, raw_filename: str):
    """Loads a CSV into SQLite, automatically cleaning column and table names."""
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Clean the table name (e.g., "Financial Data Q3.csv" -> "financial_data_q3")
    table_name = re.sub(r'[^a-zA-Z0-9_]', '', raw_filename.replace('.csv', '').replace(' ', '_').lower())
    
    # 2. Read the CSV and clean the column headers for SQL safety
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    
    # 3. Push to SQLite
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return table_name

def get_db_schema() -> str:
    """Extracts the live schema (tables and columns) to feed to the LLM."""
    if not os.path.exists(DB_PATH):
        return "No database found."
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        return "No tables available."
        
    schema = ""
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        schema += f"Table: {table_name} ({', '.join(columns)})\n"
        
    conn.close()
    return schema

def execute_sql(query: str):
    """Executes the AI-generated SQL and returns a Pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        result_df = pd.read_sql(query, conn)
        conn.close()
        return result_df, None
    except Exception as e:
        conn.close()
        return None, str(e)