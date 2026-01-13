"""
Migration Script: Add Summary TTL Fields
========================================

WEEK 1: Add summary_created_at and summary_expires_at to existing conversations

Run once after deploying Week 1 changes.

Usage:
    python scripts/migrate_summary_ttl.py
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.memory.mongo_store import db_manager
from config import settings


async def migrate_summary_ttl():
    """Add summary_expires_at to all existing summaries"""
    
    print("🔄 Starting summary TTL migration...")
    
    # Connect to MongoDB
    await db_manager.connect(
        settings.mongodb_uri,
        settings.mongodb_database
    )
    
    collection = db_manager.db.conversations
    
    # Find all documents with summary but no summary_expires_at
    query = {
        "summary": {"$exists": True, "$ne": None, "$ne": ""},
        "summary_expires_at": {"$exists": False}
    }
    
    count = await collection.count_documents(query)
    print(f"📊 Found {count} conversations with summaries to migrate")
    
    if count == 0:
        print("✅ No migration needed - all summaries have TTL fields")
        return
    
    # Update with 30-day TTL from created_at (or updated_at if created_at missing)
    result = await collection.update_many(
        query,
        [{
            "$set": {
                "summary_created_at": {
                    "$ifNull": ["$created_at", "$updated_at"]
                },
                "summary_expires_at": {
                    "$add": [
                        {"$ifNull": ["$created_at", "$updated_at"]},
                        30 * 24 * 60 * 60 * 1000  # 30 days in milliseconds
                    ]
                }
            }
        }]
    )
    
    print(f"✅ Migrated {result.modified_count} documents")
    print(f"   - Added summary_created_at")
    print(f"   - Added summary_expires_at (30 days from creation)")
    
    # Verify migration
    remaining = await collection.count_documents(query)
    if remaining > 0:
        print(f"⚠️  Warning: {remaining} documents still need migration")
    else:
        print("🎉 Migration complete!")
    
    await db_manager.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(migrate_summary_ttl())
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
