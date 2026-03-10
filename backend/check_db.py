import sqlite3
import pandas as pd
import glob
import os

dbs = glob.glob(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend\database\*.db')
if not dbs:
    print("No databases found in backend/database/")
    # Fallback to backend root
    dbs = glob.glob(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend\*.db')

if dbs:
    print(f"Using DB: {dbs[0]}")
    conn = sqlite3.connect(dbs[0])
    try:
        df = pd.read_sql('SELECT id, depression_risk FROM survey_responses ORDER BY timestamp DESC LIMIT 5', conn)
        for _, row in df.iterrows():
            print(f"ID: {row['id']}")
            risk = row['depression_risk']
            # Only print first 2 lines of prediction
            if risk:
                print('\n'.join(risk.split('\n')[:2]))
            else:
                print("None")
            print("-" * 40)
    except Exception as e:
        print("Query failed:", e)
else:
    print("No DB found")
