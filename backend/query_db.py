import sqlite3

conn = sqlite3.connect(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend\backend\database\predictions_v2.db')
cursor = conn.cursor()
cursor.execute("SELECT id, depression_risk FROM survey_responses ORDER BY timestamp DESC LIMIT 5")
rows = cursor.fetchall()
with open('db_dump.txt', 'w', encoding='utf-8') as f:
    for row in rows:
        f.write(f"ID: {row[0]}\n")
        f.write(f"RISK: {row[1]}\n")
        f.write("-" * 50 + "\n")
