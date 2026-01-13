# Memory System Review - Claude Code

**Reviewer:** Claude Code (Opus 4.5)
**Date:** 2026-01-13
**Branch:** `claude/review-memory-system-THN9J`

---

## Executive Summary

The current memory system demonstrates solid engineering fundamentals with a well-structured MongoDB persistence layer, proper session management via `ContextVar`, and reasonable token management strategies. However, the implementation has **three critical gaps**: (1) the summarization logic in `_prune_history()` is naive keyword extraction rather than LLM-powered summarization, (2) there's no semantic retrieval capability for relevant context selection, and (3) the 7-day TTL policy may conflict with a summarization strategy that requires longer retention of compressed context. The architecture is production-ready for the current scale (~$15/mo ops) but will need these enhancements before token costs become a significant concern at scale.

---

## Current Architecture Assessment

### Strengths

| Component | Implementation | Assessment |
|-----------|----------------|------------|
| **MongoDB Schema** | Separate collections (conversations, users) | Excellent design for independent scaling |
| **Session Isolation** | `ContextVar` for user_id | Correctly prevents user ID hallucination bug |
| **Token Estimation** | `len(text) // 4` heuristic | Reasonable approximation |
| **History Format** | Native BSON with `proto_to_native()` conversion | Properly handles Gemini protobuf types |
| **TTL Management** | 7-day auto-expiration via MongoDB index | Works but may conflict with summarization |

### Current Limitations

```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT FLOW (mongo_store.py:489-519)                       │
│                                                              │
│ if messages > 100 OR tokens > 50000:                        │
│     old_messages = history[:-50]     ← Simple slice         │
│     summary = keyword_extract(old)   ← NOT LLM summary!     │
│     keep = history[-50:]                                     │
└─────────────────────────────────────────────────────────────┘
```

The `_generate_simple_summary()` method (lines 521-543) only does **keyword extraction** for topics like "პროტეინი", "კრეატინი", "ალერგია" - this is placeholder code, not actual summarization.

---

## Priority Assessment

### Recommended Priority Order

| Priority | Recommendation | Current Doc Priority | My Assessment | Rationale |
|----------|----------------|---------------------|---------------|-----------|
| **P0** | SDK Migration (`google.generativeai` → `google.genai`) | Not mentioned | **AGREE as P0** | Deprecated SDK is a blocker for new features |
| **P1** | LLM-Powered Summarization | P0 | **Downgrade to P1** | Depends on stable SDK |
| **P2** | Smart Context Window (semantic retrieval) | P0 | **Downgrade to P2** | Nice-to-have, complex implementation |
| **P3** | User Preference Learning | P2 | **AGREE** | Advanced feature, lower ROI |

### Rationale for Changes

1. **SDK Migration First**: The code uses `google.generativeai` (deprecated). Before adding complex features, migrate to `google.genai` to avoid technical debt.

2. **Summarization Before Semantic Search**: Conversation summarization provides ~80% of the token savings with ~20% of the implementation complexity. Smart context window requires embeddings infrastructure.

3. **Preference Learning is Premature**: Until you have more users and conversation data, pattern detection will be unreliable.

---

## Technical Concerns

### 🚨 Critical

**1. Deprecated SDK Usage**
```python
# main.py:59 - DEPRECATED
import google.generativeai as genai

# Should migrate to:
from google import genai
```

**Impact**: New features may not be available. Breaking changes possible.
**Effort**: 2-3 days for full migration + testing

**2. TTL Conflict with Summarization**
```python
# mongo_store.py:140
expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=7))
```

If you summarize conversations, the **summary** should persist longer than 7 days, but the **raw history** should expire. Current schema doesn't support this.

**Fix**: Add separate `summary_expires_at` field or store summaries in a separate collection.

**3. No Error Handling for Summarization API Calls**
```python
# mongo_store.py:521-543 - Current "summary" is just keyword extraction
def _generate_simple_summary(self, messages):
    # This doesn't call any LLM API
    topics = []
    for msg in messages:
        if "პროტეინ" in text.lower():
            topics.append("პროტეინი")
    # ...
```

When you implement real LLM summarization, you need:
- Timeout handling
- Retry logic
- Fallback to keyword extraction if API fails
- Token budget for summarization calls

### ⚠️ Important

**4. No Semantic Search Infrastructure**
Current search is exact match only:
```python
# user_tools.py:308-315
mongo_query = {
    "$or": [
        {"name": {"$regex": safe_term, "$options": "i"}},
        # ... regex matching only
    ]
}
```

For smart context window, you need:
- Embedding model (Gemini embedding-001 or ada-002)
- MongoDB Atlas Vector Search OR separate vector store
- Embedding generation on message save

**5. Multi-Language Summarization Challenge**
System prompt and user messages are Georgian, but your keyword extraction looks for Georgian keywords. LLM summarization should:
- Summarize in the same language as the conversation
- Handle code-switching (Georgian + English product names)

**6. Context Injection Gap**
```python
# main.py:292-295
if summary and not history:
    # Add summary as context
    logger.info(f"Injecting summary for {user_id}: {summary[:100]}...")
```

This only logs! It doesn't actually inject the summary into the conversation. You need to prepend the summary as a system message or user context message.

### 💡 Nice to Have

**7. Message Deduplication**
No check for duplicate messages if user rapidly clicks send. Consider idempotency keys.

**8. Conversation Title Generation**
Currently extracts first 30 chars:
```python
# mongo_store.py:576
title = text[:30] + "..." if len(text) > 30 else text
```

Could use LLM to generate better titles for sidebar display.

---

## Cost-Benefit Analysis

### Token Cost Reality Check

**Current State (from config.py):**
- Max history messages: 100
- Max history tokens: 50,000
- Catalog context: ~60k tokens (315 products)

**Estimated Current Costs:**
```
Per conversation:
- System prompt: ~1,200 tokens
- Catalog context: ~60,000 tokens
- Average history: ~20,000 tokens
- User message: ~50 tokens
- Total input: ~81,250 tokens

With Gemini 2.5 Flash pricing ($0.075/1M input):
- Per message: ~$0.006
- 100 users × 20 messages/day = $12/day = ~$360/month
```

**With Summarization (estimated):**
```
Per conversation (after pruning):
- System prompt: ~1,200 tokens
- Catalog context: ~60,000 tokens (can also compress!)
- Summary: ~500 tokens
- Recent history (10 msgs): ~4,000 tokens
- Total input: ~65,700 tokens

Savings: ~19% reduction in context tokens
Monthly savings at scale: ~$70/month
```

**Reality Check:** The 90% token savings claim is **optimistic**. The catalog context (60k tokens) dominates the token budget. True savings require:
1. Catalog compression (only include relevant products)
2. History summarization (implemented)
3. Selective context loading (semantic search)

**Combined realistic estimate: 30-50% savings, not 90%**

### Hidden Costs

| Feature | Hidden Cost | Estimate |
|---------|-------------|----------|
| Summarization API calls | Extra Gemini calls for summarization | +$0.001/summary |
| Embedding generation | Gemini embedding-001 calls | +$0.00002/message |
| MongoDB Atlas Vector Search | Requires M10+ cluster | +$60/month minimum |

---

## Implementation Roadmap

### Phase 0: Foundation (Week 1) - CRITICAL

```
[ ] 1. SDK Migration Spike
    - Create branch: feature/sdk-migration
    - Replace: google.generativeai → google.genai
    - Update: GenerativeModel → Client pattern
    - Test: All existing functionality
    - Effort: 2-3 days

[ ] 2. Fix Summary Injection Bug
    - File: main.py:292-295
    - Actually prepend summary to conversation
    - Test: Verify summary appears in chat context
    - Effort: 2 hours

[ ] 3. Add Summary Persistence Schema
    - Add: summary_expires_at field (30 days)
    - Add: summary_token_count field
    - Keep raw history TTL at 7 days
    - Effort: 4 hours
```

### Phase 1: Conversation Summarization (Week 2)

```
[ ] 4. Implement LLM Summarization
    - Create: app/memory/summarizer.py
    - Method: async def summarize_conversation(messages: List[Dict]) -> str
    - Prompt: Georgian-aware summarization
    - Fallback: Keyword extraction on API failure
    - Effort: 1 day

[ ] 5. Add Summarization Config
    - SUMMARIZE_THRESHOLD_MESSAGES: int = 50
    - SUMMARIZE_THRESHOLD_TOKENS: int = 30000
    - SUMMARY_MAX_TOKENS: int = 500
    - Effort: 2 hours

[ ] 6. Integrate with save_history()
    - Call summarize_conversation() when thresholds exceeded
    - Store summary in MongoDB
    - Preserve recent N messages
    - Effort: 4 hours

[ ] 7. Add Summarization Tests
    - Test: Georgian text summarization quality
    - Test: API failure fallback
    - Test: Token counting accuracy
    - Effort: 4 hours
```

### Phase 2: Smart Context Window (Weeks 3-4)

```
[ ] 8. Evaluate Vector Search Options
    - Option A: MongoDB Atlas Vector Search ($60+/mo)
    - Option B: Pinecone free tier (limited)
    - Option C: In-memory FAISS (no persistence)
    - Decision: Based on budget and scale needs
    - Effort: 1 day research

[ ] 9. Add Message Embeddings
    - Generate embeddings on message save
    - Store in chosen vector store
    - Batch embed for existing messages
    - Effort: 2 days

[ ] 10. Implement Semantic Retrieval
    - Create: app/memory/retriever.py
    - Method: get_relevant_context(query, top_k=5)
    - Combine: Recent + Relevant messages
    - Effort: 1 day

[ ] 11. Update Context Building
    - Modify: get_or_create_session()
    - Add: Relevant context injection
    - Test: Quality of retrieved context
    - Effort: 1 day
```

### Phase 3: Optimization (Week 5+)

```
[ ] 12. Catalog Compression
    - Only inject relevant product categories
    - Based on detected user intent
    - Potential savings: 40-50k tokens
    - Effort: 2 days

[ ] 13. User Preference Learning (Optional)
    - Track: Product views, purchases, questions
    - Detect: Patterns in preferences
    - Store: In user profile
    - Effort: 3-5 days
```

---

## Alternative Approaches

### Instead of MongoDB Atlas Vector Search, consider:

**Option A: Gemini Context Caching (Recommended)**
```python
# Available in google.genai SDK
cache = client.caches.create(
    model="gemini-2.0-flash",
    contents=[catalog_context, user_profile],
    ttl="1h"
)
```
- **Pro**: Native Gemini feature, no extra infrastructure
- **Pro**: 75% token cost reduction for cached content
- **Con**: 1-hour TTL minimum, 32k token minimum

**Option B: Hybrid Keyword + Embedding**
```python
# Quick keyword filter first, then semantic ranking
candidates = keyword_search(query, limit=20)  # Fast MongoDB
ranked = semantic_rank(query, candidates)     # Small embedding model
return ranked[:5]
```
- **Pro**: No vector store needed
- **Pro**: Works with current MongoDB
- **Con**: Slightly worse relevance than pure semantic

### Instead of LLM Summarization, consider:

**Option: Sliding Window + Key Facts Extraction**
```python
# Extract structured facts, not prose summary
facts = {
    "user_name": "გიორგი",
    "allergies": ["lactose"],
    "discussed_products": ["Gold Standard Whey", "Creatine Monohydrate"],
    "stated_goals": ["muscle_gain"],
    "budget_mentioned": "150 ლარი"
}
```
- **Pro**: More reliable than LLM prose
- **Pro**: Cheaper (rule-based extraction)
- **Con**: Loses conversational nuance

---

## Testing Strategy

### Unit Tests Required

```python
# tests/test_summarization.py
def test_summarize_georgian_conversation():
    """Summary should be in Georgian"""

def test_summarize_handles_api_failure():
    """Should fall back to keyword extraction"""

def test_summarize_respects_token_limit():
    """Summary should not exceed SUMMARY_MAX_TOKENS"""
```

### Integration Tests Required

```python
# tests/test_memory_integration.py
def test_full_conversation_lifecycle():
    """Create session → Send messages → Hit threshold → Summarize → Load → Verify summary injected"""

def test_session_recovery_with_summary():
    """User returns after pruning, should see summary context"""
```

### Manual Testing Checklist

- [ ] Georgian text summarization produces readable Georgian output
- [ ] Summary injection visible in Gemini responses ("როგორც წინა საუბარში...")
- [ ] No token count regression after summarization
- [ ] Allergies/preferences preserved through summarization

---

## Questions for Product/Business

Before implementing, clarify:

1. **Budget for vector search?** MongoDB Atlas Vector Search requires M10+ cluster (~$60/mo minimum). Is this acceptable?

2. **Summary quality requirements?** Should summaries be human-readable or just machine context?

3. **Retention policy?** Current 7-day TTL for all data. Should summaries persist longer for returning users?

4. **Privacy review needed?** Storing AI-generated summaries of user conversations may have GDPR implications. Legal review recommended.

5. **Multi-session UX?** Current sidebar shows sessions. Should summaries be visible to users?

---

## Final Recommendations

### Immediate Actions (This Week)

1. **Fix the summary injection bug** (main.py:292-295) - 2 hours
2. **Start SDK migration research** - 4 hours
3. **Add summary persistence schema** - 4 hours

### Short-term (Next 2 Weeks)

1. **Complete SDK migration** - blocks all other work
2. **Implement basic LLM summarization** - biggest token savings

### Defer Until Scale Requires

1. **Vector search infrastructure** - wait until >1000 daily users
2. **User preference learning** - needs more data first

---

## Appendix: Code Snippets for Quick Fixes

### Fix 1: Summary Injection (main.py)

```python
# Replace lines 292-295 with:
if summary and not history:
    # Inject summary as first message context
    summary_context = {
        "role": "user",
        "parts": [{"text": f"[წინა საუბრის რეზიუმე: {summary}]"}]
    }
    gemini_history = [summary_context]
    logger.info(f"Injected summary for {user_id}")
```

### Fix 2: Summary Schema (mongo_store.py)

```python
# Add to ConversationDocument dataclass:
summary_token_count: int = 0
summary_expires_at: datetime = field(
    default_factory=lambda: datetime.utcnow() + timedelta(days=30)
)
```

### Fix 3: Basic LLM Summarization

```python
# app/memory/summarizer.py
async def summarize_conversation(
    messages: List[Dict],
    model: genai.GenerativeModel,
    max_tokens: int = 500
) -> str:
    """Generate Georgian summary of conversation"""

    # Extract text only
    text_content = "\n".join(
        part.get("text", "")
        for msg in messages
        for part in msg.get("parts", [])
        if "text" in part
    )

    prompt = f"""შეაჯამე ეს საუბარი 2-3 წინადადებით ქართულად.
გაითვალისწინე: მომხმარებლის სახელი, ალერგიები, მიზნები, განხილული პროდუქტები.

საუბარი:
{text_content[:10000]}

რეზიუმე:"""

    try:
        response = await model.generate_content_async(prompt)
        return response.text[:max_tokens * 4]  # Rough char limit
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # Fallback to keyword extraction
        return _generate_simple_summary(messages)
```

---

**Review Complete.** Ready for implementation once priorities are confirmed.
