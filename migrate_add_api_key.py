"""
Migration script to add api_key column to Agent table.
Run this once to update your existing database.
"""
from app.db import engine
from sqlalchemy import text

def migrate():
    with engine.connect() as conn:
        # Check if column exists
        result = conn.execute(text("PRAGMA table_info(agents)"))
        columns = [row[1] for row in result]
        
        if 'api_key' not in columns:
            print("Adding api_key column to agents table...")
            conn.execute(text("ALTER TABLE agents ADD COLUMN api_key VARCHAR"))
            conn.commit()
            print("✓ Migration completed successfully!")
        else:
            print("✓ api_key column already exists, no migration needed.")

if __name__ == "__main__":
    migrate()
