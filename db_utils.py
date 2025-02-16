import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    document_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            conn.commit()
    
    def add_document(self, filename: str, path: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (filename, path) VALUES (?, ?)",
                (filename, path)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_documents(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
            return cursor.fetchall()
    
    def add_chat_message(self, role: str, content: str, document_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (role, content, document_id) VALUES (?, ?, ?)",
                (role, content, document_id)
            )
            conn.commit()
    
    def get_chat_history(self, document_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM chat_history WHERE document_id = ? ORDER BY timestamp",
                (document_id,)
            )
            return [{
                "role": role,
                "content": content
            } for role, content in cursor.fetchall()]
    
    def clear_chat_history(self, document_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE document_id = ?", (document_id,))
            conn.commit()
    
    def clear_all_documents(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete all chat history first due to foreign key constraint
            cursor.execute("DELETE FROM chat_history")
            # Then delete all documents
            cursor.execute("DELETE FROM documents")
            conn.commit()