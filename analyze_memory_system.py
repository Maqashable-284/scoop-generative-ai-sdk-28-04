#!/usr/bin/env python3
"""Analyze recent MongoDB activity and user profiles"""
import pymongo
import json
from datetime import datetime, timedelta, timezone

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://scoop_admin:W6AuJLLnYrPnq.3@scoop.xbbeory.mongodb.net/?appName=Scoop")
db = client.scoop_db

print("=" * 100)
print("🔍 MEMORY SYSTEM ANALYSIS - Last 5 Minutes")
print("=" * 100)

# Get timestamp for 5 minutes ago
five_min_ago = datetime.now(timezone.utc) - timedelta(minutes=5)

print(f"\n⏰ Current Time: {datetime.now(timezone.utc).isoformat()}")
print(f"⏰ Looking for activity after: {five_min_ago.isoformat()}")

# 1. Check recent users
print("\n" + "=" * 100)
print("👥 RECENT USERS (created/updated in last 5 min)")
print("=" * 100)

recent_users = list(db.users.find({
    "$or": [
        {"created_at": {"$gte": five_min_ago}},
        {"updated_at": {"$gte": five_min_ago}}
    ]
}).sort("updated_at", -1).limit(5))

if not recent_users:
    print("❌ No users created/updated in last 5 minutes")
else:
    for user in recent_users:
        profile = user.get("profile", {})
        print(f"\n📋 User: {user.get('user_id')}")
        print(f"   Created: {user.get('created_at')}")
        print(f"   Updated: {user.get('updated_at', 'N/A')}")
        print(f"   Profile: {json.dumps(profile, ensure_ascii=False)}")
        print(f"   Stats: {user.get('stats', {})}")

# 2. Check recent chat sessions
print("\n" + "=" * 100)
print("💬 RECENT CHAT SESSIONS (last 5 min)")
print("=" * 100)

recent_sessions = list(db.conversations.find({
    "updated_at": {"$gte": five_min_ago}
}).sort("updated_at", -1).limit(10))

if not recent_sessions:
    print("❌ No chat sessions in last 5 minutes")
else:
    for session in recent_sessions:
        print(f"\n📝 Session: {session.get('session_id')}")
        print(f"   User: {session.get('user_id')}")
        print(f"   Title: {session.get('title', 'N/A')}")
        print(f"   Messages: {len(session.get('messages', []))}")
        print(f"   Updated: {session.get('updated_at')}")
        
        # Show last 2 messages
        messages = session.get('messages', [])
        if messages:
            print(f"   Last message preview:")
            last_msg = messages[-1]
            content = last_msg.get('content', '')[:80]
            print(f"     - {last_msg.get('role')}: {content}...")

# 3. Check ALL users (to see what profiles exist)
print("\n" + "=" * 100)
print("📊 ALL USER PROFILES IN DATABASE")
print("=" * 100)

all_users = list(db.users.find({}, {"user_id": 1, "profile": 1, "stats": 1}).limit(10))

if not all_users:
    print("❌ No users in database at all!")
else:
    print(f"Total users found: {len(all_users)}\n")
    for user in all_users:
        profile = user.get("profile", {})
        print(f"👤 {user.get('user_id')}: {profile}")

# 4. Analysis
print("\n" + "=" * 100)
print("🧠 MEMORY SYSTEM ARCHITECTURE")
print("=" * 100)

print("""
Memory Storage Layers:
┌─────────────────────────────────────────────────────────────────┐
│ 1. EPHEMERAL (Chat History - in Gemini context)                │
│    - Lives in active chat session                               │
│    - Includes: Full conversation, user statements               │
│    - Gemini "remembers" from this during chat                   │
│    - Lost when: Session ends, cache expires, user deletes data  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CONTEXT CACHE (Gemini Cached Content)                       │
│    - System prompt + catalog cached for 60 minutes              │
│    - Saves 85% token costs                                      │
│    - Does NOT include user profile or chat history              │
│    - Just: System instructions + product catalog                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. PERSISTENT (MongoDB)                                         │
│    - user.profile: name, allergies, goals, fitness_level        │
│    - conversations: Full chat history                           │
│    - Retrieved via: get_user_profile() function call            │
│    - Only these 4 fields saved to profile!                      │
└─────────────────────────────────────────────────────────────────┘

⚠️  CRITICAL LIMITATION:
────────────────────────────────────────────────────────────────
update_user_profile() ONLY saves:
  ✅ name
  ✅ allergies  
  ✅ goals
  ✅ fitness_level

NOT saved to MongoDB (only in ephemeral chat history):
  ❌ age (40 წელი)
  ❌ occupation (ბანკში მუშაობს)
  ❌ workout_frequency (კვირაში 5-ჯერ)
  ❌ experience_years (3 წელია)

This is why the second chat couldn't recall job/workout details!
""")

client.close()
