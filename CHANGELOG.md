# Changelog - Memory Optimization 2026

All notable changes to the memory system will be documented here.

---

## [Week 1] - 2026-01-13

### Fixed

#### 🐛 Summary Injection Bug
- **Issue**: Summary was logged but never passed to Gemini model
- **Location**: `main.py:292-294`
- **Fix**: Prepend summary as context message in history
- **Impact**: AI can now actually recall summarized conversations

**Before**:
```python
if summary and not history:
    logger.info(f"Injecting summary...")  # Does nothing!
```

**After**:
```python
if summary:
    summary_message = {
        "role": "user",
        "parts": [{"text": f"[წინა საუბრის კონტექსტი: {summary}]"}]
    }
    gemini_history = [summary_message] + gemini_history
```

### Changed

#### 📦 Summary Retention Extended
- **Schema Update**: Added `summary_expires_at` field
- **TTL Change**: 7 days → 30 days for summaries
- **Rationale**: Summaries are cheap (~500 tokens) vs raw history (~20k tokens)
- **Files Modified**:
  - `app/memory/mongo_store.py` (schema, indexes)
  - `scripts/migrate_summary_ttl.py` (new migration script)

### Added

#### 📄 New Files
- `CHANGELOG.md` - This file
- `scripts/migrate_summary_ttl.py` - Database migration for TTL update
- `docs/WEEK_1_VERIFICATION.md` - Testing procedures

---

## [Unreleased]

### Week 2 - SDK Migration (Planned)
- Replace `google.generativeai` with `google.genai`
- Update all imports and model initialization
- Comprehensive regression testing

### Week 3 - LLM Summarization (Planned)
- New `ConversationSummarizer` class
- Replace keyword extraction with semantic understanding
- A/B testing vs old approach

### Week 4 - Context Caching (Planned)
- Implement Gemini context caching
- Cache catalog + system prompt (60k tokens)
- Target: 75% token cost reduction

---

## Migration Notes

### Week 1 → Production
- Run `python scripts/migrate_summary_ttl.py` before deployment
- No breaking changes, backward compatible
- Monitor MongoDB TTL index creation

---

**Format**: Based on [Keep a Changelog](https://keepachangelog.com/)
