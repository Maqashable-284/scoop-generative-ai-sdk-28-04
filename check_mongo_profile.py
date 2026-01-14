#!/usr/bin/env python3
"""Check what's stored in MongoDB for user Dato"""
import pymongo
import json
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://scoop_admin:W6AuJLLnYrPnq.3@scoop.xbbeory.mongodb.net/?appName=Scoop")
db = client.scoop_db

print("=" * 80)
print("MongoDB User Profile Analysis - დათო")
print("=" * 80)

# Find user with name დათო
user = db.users.find_one({"profile.name": "დათო"})

if not user:
    print("❌ User 'დათო' not found in MongoDB!")
else:
    print(f"\n✅ Found user: {user.get('user_id')}\n")
    
    print("📋 Full MongoDB Document:")
    print(json.dumps({
        "user_id": user.get("user_id"),
        "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
        "profile": user.get("profile", {}),
        "stats": user.get("stats", {}),
    }, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)
    print("🔍 Profile Fields Available:")
    print("=" * 80)
    
    profile = user.get("profile", {})
    for key, value in profile.items():
        print(f"  ✅ {key:20s} = {value}")
    
    print("\n" + "=" * 80)
    print("❌ Missing Fields (NOT in MongoDB):")
    print("=" * 80)
    missing = ["age", "occupation", "workout_frequency", "experience_years", "fitness_experience"]
    for field in missing:
        if field not in profile:
            print(f"  ❌ {field}")
    
    print("\n" + "=" * 80)
    print("💡 Conclusion:")
    print("=" * 80)
    print("""
The update_user_profile() function only supports 4 fields:
  1. name
  2. allergies
  3. goals
  4. fitness_level
  
Additional information like age, occupation, workout_frequency
are NOT being saved to MongoDB!

Gemini mentions these details in responses but they are stored
ONLY in the ephemeral chat history (context cache), not in
the persistent MongoDB profile.
""")

client.close()
