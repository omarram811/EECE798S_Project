"""
Migration script to add the 'announcement' column to the agents table.
Run this once after updating the code to add the announcement feature.

Usage:
    python migrate_add_announcement.py
"""

import sqlite3
import os

DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Nothing to migrate.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if the column already exists
    cursor.execute("PRAGMA table_info(agents)")
    columns = [row[1] for row in cursor.fetchall()]

    if "announcement" in columns:
        print("Column 'announcement' already exists. No migration needed.")
        conn.close()
        return

    # Add the announcement column
    print("Adding 'announcement' column to agents table...")
    cursor.execute("ALTER TABLE agents ADD COLUMN announcement TEXT")
    conn.commit()
    print("Migration completed successfully!")

    conn.close()


if __name__ == "__main__":
    migrate()
