import sqlite3
import os
from datetime import datetime

# Build absolute path to feedback.db in the backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'feedback.db')

def init_db():
    """Initialize the database and ensure the schema is up to date"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create feedback table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            bot_response TEXT,
            rating INTEGER,
            remark TEXT,
            timestamp TEXT,
            answer_id INTEGER
        )
    ''')
    
    # Force Migration: Check if answer_id column exists, if not add it
    cursor.execute("PRAGMA table_info(feedback)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'answer_id' not in columns:
        print("🔧 Migrating database: Adding answer_id column...")
        cursor.execute("ALTER TABLE feedback ADD COLUMN answer_id INTEGER")
    
    conn.commit()
    conn.close()

def save_feedback(query, response, rating, answer_id=None, remark=""):
    """Save user feedback to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Standardize timestamp to ISO format
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO feedback (answer_id, user_query, bot_response, rating, remark, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (answer_id, query, response, rating, remark, timestamp))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error saving feedback: {e}")
        return False

def get_all_feedback():
    """Fetch all feedback records from the database using explicit columns"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, user_query, bot_response, rating, remark, timestamp, answer_id 
            FROM feedback 
            ORDER BY timestamp DESC
        ''')
        rows = cursor.fetchall()
        
        feedback_list = []
        for row in rows:
            feedback_list.append({
                "id": row[0],
                "query": row[1],
                "response": row[2],
                "rating": row[3],
                "remark": row[4],
                "timestamp": row[5],
                "answer_id": row[6]
            })
            
        conn.close()
        return feedback_list
    except Exception as e:
        print(f"❌ Error fetching feedback: {e}")
        return []

def get_rl_stats():
    """Get aggregated RL stats using explicit column logic"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Aggregated stats for RL boosting
        cursor.execute('''
            SELECT answer_id, AVG(rating), COUNT(*) 
            FROM feedback 
            WHERE answer_id IS NOT NULL
            GROUP BY answer_id
        ''')
        rows = cursor.fetchall()
        
        stats = []
        for row in rows:
            avg_rating = round(row[1], 2)
            impact = 0
            if avg_rating >= 4.0: 
                impact = 0.1
                status = "Rewarded (Boosted)"
            elif avg_rating < 3.0: 
                impact = -0.1
                status = "Penalized"
            else:
                impact = 0
                status = "Normal"
            
            stats.append({
                "answer_id": row[0],
                "avg_rating": avg_rating,
                "total_feedback": row[2],
                "rl_impact": impact,
                "status": status
            })
            
        conn.close()
        return stats
    except Exception as e:
        print(f"❌ Error fetching RL stats: {e}")
        return []

# Initialize on import
init_db()
