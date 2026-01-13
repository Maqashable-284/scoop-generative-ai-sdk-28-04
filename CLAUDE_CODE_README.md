# 🚀 For Claude Code: SDK Migration Task

**Status**: Week 1 ✅ | Week 2-4 ⏸️ Needs Your Help!

---

## 📋 Quick Start

### What's Done:
✅ **Week 1**: Summary injection fix + 30-day TTL (tested and working!)

### What You Need to Do:
🔨 **Week 2-4**: Migrate `google.generativeai` → `google.genai`

---

## 📖 Full Instructions

**READ THIS FIRST**: [Claude Code Handoff Document](file:///Users/maqashable/.gemini/antigravity/brain/721a3a61-b954-40c9-a09d-d75ba0a2c37c/claude_code_handoff.md)

Contains:
- All blockers and questions
- Step-by-step migration guide
- Code examples
- Testing requirements
- Week 1 features to preserve

---

## 🎯 Your Mission

1. Research new `google.genai` SDK API
2. Migrate 3 files: `requirements.txt`, `main.py`, `mongo_store.py`
3. Preserve Week 1 summary injection fix
4. Test everything
5. (Optional) Implement context caching (Week 4)

---

## ⚡ Test Locally

```bash
cd /Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026

# Install dependencies (after you update requirements.txt)
pip install -r requirements.txt

# Run server
python3 main.py

# Test
curl http://localhost:8080/health
```

---

## 📚 Key Resources

- **Repository**: https://github.com/Maqashable-284/scoop-genai-project-2026
- **Branch**: `claude/review-memory-system-THN9J`
- **New SDK Docs**: https://ai.google.dev/gemini-api/docs/sdks?lang=python

---

## ❓ Critical Questions to Answer

1. How to create chat model in new SDK?
2. How to start chat with history?
3. How to send messages (async)?
4. How to use function calling?
5. How to implement context caching?

**All details in the handoff doc!**

---

**Good luck! 🚀**
