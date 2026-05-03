import sqlite3
import os

DB_NAME = "chats.db"
VECTOR_STORE_DIR = "vector_stores"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    # Ensure the vector_stores directory exists
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table for chat session metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        chat_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        video_url TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Table for individual chat messages
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES chat_sessions (chat_id)
    );
    """)
    
    conn.commit()
    conn.close()

def save_chat_session(chat_id, title, video_url):
    """Saves a new chat session to the database."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO chat_sessions (chat_id, title, video_url) VALUES (?, ?, ?)",
            (chat_id, title, video_url)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def save_message(chat_id, role, content):
    """Saves a chat message to the database."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def load_chat_sessions():
    """Loads all chat sessions from the database, most recent first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, title, video_url FROM chat_sessions ORDER BY created_at DESC")
    sessions = cursor.fetchall()
    conn.close()
    
    # Format as the app expects: {chat_id: {title: ..., video_url: ...}}
    chat_sessions_dict = {}
    for session in sessions:
        chat_sessions_dict[session['chat_id']] = {
            "title": session['title'],
            "video_url": session['video_url']
            # Messages will be loaded on-demand when a chat is opened
        }
    return chat_sessions_dict

def load_messages(chat_id):
    """Loads all messages for a specific chat_id from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_messages WHERE chat_id = ? ORDER BY created_at ASC",
        (chat_id,)
    )
    messages = cursor.fetchall()
    conn.close()
    
    # Format as list of dictionaries
    return [{"role": msg['role'], "content": msg['content']} for msg in messages]