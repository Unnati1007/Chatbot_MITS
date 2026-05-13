import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "feedback.db")

def force_migrate():
    print(f"Checking database at: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print("Error: Database file not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('ALTER TABLE feedback ADD COLUMN answer_id INTEGER')
        print("Success: Added answer_id column.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("Info: Column already exists.")
        else:
            print(f"Error: {e}")
            
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    force_migrate()
