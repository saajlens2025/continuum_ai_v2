import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

class MemoryEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.db_path = "memory.db"
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_deleted BOOLEAN DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # CSV uploads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS csv_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                row_count INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Training data table (soft deleted items)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_table TEXT,
                original_id INTEGER,
                data_snapshot TEXT,
                deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str, user_id: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)",
            (session_id, user_id)
        )
        conn.commit()
        conn.close()
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return sessions
    
    def store_message(self, session_id: str, user_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO messages (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (session_id, user_id, role, content)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )
        conn.commit()
        conn.close()
        
        # Also store in vector store for semantic search
        self.vector_store.add_memory(
            session_id=session_id,
            text=content,
            metadata={"role": role, "user_id": user_id}
        )
    
    def process_csv_upload(self, session_id: str, user_id: str, df: pd.DataFrame, filename: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Record upload metadata
        cursor.execute(
            "INSERT INTO csv_uploads (session_id, user_id, filename, row_count) VALUES (?, ?, ?, ?)",
            (session_id, user_id, filename, len(df))
        )
        upload_id = cursor.lastrowid
        
        # Process each row
        rows_processed = 0
        for idx, row in df.iterrows():
            row_text = f"CSV {filename} row {idx}: {row.to_dict()}"
            
            # Store in vector store for semantic search
            self.vector_store.add_memory(
                session_id=session_id,
                text=row_text,
                metadata={
                    "type": "csv_row",
                    "upload_id": upload_id,
                    "row_index": idx,
                    "filename": filename
                }
            )
            rows_processed += 1
        
        conn.commit()
        conn.close()
        
        return rows_processed
    
    def retrieve_context(self, session_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        context = {
            "recent_messages": [],
            "relevant_csv": [],
            "semantic_matches": []
        }
        
        # Get recent messages from SQLite
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? AND is_deleted = 0 ORDER BY timestamp DESC LIMIT 10",
            (session_id,)
        )
        recent = [dict(row) for row in cursor.fetchall()]
        context["recent_messages"] = list(reversed(recent))
        conn.close()
        
        # Get semantic matches from vector store
        semantic_results = self.vector_store.search(query, session_id, limit=limit)
        context["semantic_matches"] = semantic_results
        
        # Extract CSV rows from semantic matches
        for match in semantic_results:
            if match.get("metadata", {}).get("type") == "csv_row":
                context["relevant_csv"].append(match)
        
        return context
    
    def get_session_memory(self, session_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? AND is_deleted = 0 ORDER BY timestamp",
            (session_id,)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return messages
    
    def soft_delete_message(self, session_id: str, message_id: int):
        conn = sqlite3.connect(self.db_path)
        
        # Get message data
        cursor = conn.execute(
            "SELECT * FROM messages WHERE id = ? AND session_id = ?",
            (message_id, session_id)
        )
        message = cursor.fetchone()
        
        if message:
            # Move to training data
            expires_at = datetime.now() + timedelta(days=30)
            conn.execute(
                "INSERT INTO training_data (original_table, original_id, data_snapshot, expires_at) VALUES (?, ?, ?, ?)",
                ("messages", message_id, json.dumps(dict(message)), expires_at)
            )
            
            # Soft delete from messages
            conn.execute(
                "UPDATE messages SET is_deleted = 1 WHERE id = ?",
                (message_id,)
            )
            conn.commit()
        
        conn.close()
    
    def get_training_data(self, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM training_data WHERE expires_at > CURRENT_TIMESTAMP LIMIT ?",
            (limit,)
        )
        data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return data
    
    def get_system_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE is_deleted = 0")
        messages = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM csv_uploads")
        uploads = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_data")
        training = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_sessions": sessions,
            "total_messages": messages,
            "total_csv_uploads": uploads,
            "training_data_count": training
        }