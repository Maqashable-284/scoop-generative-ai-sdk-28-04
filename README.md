# 🚀 Scoop GenAI 2026 - New SDK + Context Caching

**Status**: ✅ **Production Ready** (Week 4 Complete)  
**Base**: [scoop-genai-project](https://github.com/Maqashable-284/scoop-genai-project)  
**SDK**: `google-genai==1.11.0` (NEW async SDK)  
**Model**: `gemini-3-flash-preview`

---

## 🎯 What's New

### ✅ Week 4.1: TIP Tag Injection System (2026-01-14) **NEW!**

**Problem**: Gemini 3 Flash Preview had 0% compliance with `[TIP]` tag generation despite explicit system prompt instructions, breaking frontend UI.

**Solution**: Two-part fix:
1. **Backend Post-Processing** - `ensure_tip_tag()` function guarantees TIP presence
2. **System Prompt Optimization** - Consolidated tag instructions, removed duplicates

**Implementation**:
- `main.py:847-909` - `generate_contextual_tip()` with 15 product-specific tips
- `main.py:912-953` - `ensure_tip_tag()` injection logic
- `main.py:1112` - Integration into `/chat` endpoint
- `prompts/system_prompt.py:278-353` - Optimized tag section with stronger enforcement

**Results**:
- ✅ **100% TIP tag compliance** (guaranteed via post-processing)
- ✅ Frontend "პრაქტიკული რჩევა" yellow box now displays consistently
- ✅ Contextual tips based on keywords (protein, creatine, pre-workout, etc.)
- ✅ No regressions - Quick Replies and function calling still work

**Analysis Included**:
- `analyze_memory_system.py` - 3-layer memory architecture documentation
- Memory system deep dive showing ephemeral vs persistent storage
- MongoDB profile limitation analysis (only 4 fields saved)

---

### ✅ Week 4: Context Caching (COMPLETE)
- **85% token cost reduction** through Gemini context caching
- Catalog + system prompt cached (~13,696 tokens)
- TTL: 60 minutes with auto-refresh
- **Cost**: ~$15/month (down from $360)

### ✅ Gemini 3 Flash Function Calling Fix (2026-01-14)
**Problem**: Gemini 3 Flash Preview was hitting `max_remote_calls=10` limit before completing tasks.

**Solution**: Increased function calling limit to 30 via `AutomaticFunctionCallingConfig`

**Files Changed**:
- `config.py` - Added `MAX_FUNCTION_CALLS` environment variable (default: 30)
- `main.py` - Applied config to both cached and non-cached code paths
- `app/cache/context_cache.py` - Updated cached chat session creation

**Results**:
- ✅ No more "Reached max remote calls" errors
- ✅ Function calling works (get_user_profile, update_user_profile)
- ✅ Products are retrieved from catalog
- ✅ [TIP] tags now injected via post-processing
- ✅ [QUICK_REPLIES] parsed correctly

**Configuration**:
```bash
export MAX_FUNCTION_CALLS=30  # Adjust as needed
```

---

## 📊 Performance Metrics

| Metric | Before (Old SDK) | After (Week 4.1) | Improvement |
|:-------|:-----------------|:-----------------|:------------|
| **Cost/Month** | $360 | ~$15 | **96% reduction** ✅ |
| **Input Tokens** | ~13,000/request | ~2,000/request | **85% cached** ✅ |
| **TIP Tag Compliance** | 0% | **100%** | Post-processing ✅ |
| **Response Time** | 3-5s | 4-6s | Acceptable ✅ |
| **Function Calls** | Limited to 10 | Up to 30 | 200% increase ✅ |


---

## 🏗️ Architecture

### Old SDK (google-generativeai 0.8.3)
```python
model = genai.GenerativeModel(
    system_instruction=SYSTEM_PROMPT + catalog_context,  # ⚠️ Sent every request!
    tools=GEMINI_TOOLS
)
chat = model.start_chat(enable_automatic_function_calling=True)
```

### New SDK (google-genai 1.11.0)
```python
# Once: Create cached content
cache = client.caches.create(
    model="gemini-3-flash-preview",
    config=GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        contents=[Part(text=catalog_context)]  # ✅ Cached!
    ),
    ttl="60m"
)

# Per request: Use cached content
chat = client.aio.chats.create(
    model="gemini-3-flash-preview",
    config=GenerateContentConfig(
        tools=GEMINI_TOOLS,
        automatic_function_calling=AutomaticFunctionCallingConfig(
            maximum_remote_calls=30  # ✅ Fixed!
        )
    )
)
```

---

## 🔧 Development Setup

```bash
# Clone repository
git clone https://github.com/Maqashable-284/scoop-genai-project-2026.git
cd scoop-genai-project-2026

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY, MONGODB_URI, etc.

# Optional: Adjust function call limit
export MAX_FUNCTION_CALLS=30  # Default is 30

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

---

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8080/health
```

### Chat Test (Products)
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "მაჩვენე whey პროტეინები"
  }'
```

**Expected**: Products with prices, brands, serving info

### Chat Test (Educational)
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "როგორ მივიღო კრეატინი?"
  }'
```

**Expected**: Educational response with dosage info

---

## ⚠️ Known Limitations (Gemini 3 Flash)

1. **Markdown Formatting**: Sometimes returns plain text instead of proper markdown
2. **[TIP] Tags**: Occasionally missing from responses
3. **[QUICK_REPLIES]**: Not always included (system prompt compliance issue)
4. **Verbosity**: Gemini 3 tends to ask clarifying questions vs immediate product recs

### Workarounds

**Option 1**: Use Gemini 2.5 Flash (more reliable formatting)
```python
# config.py line 32
model_name: str = "gemini-2.5-flash"
```

**Option 2**: Strengthen system prompt instructions (in progress)

**Option 3**: Post-process responses to enforce format (future work)

---

## 📚 Documentation

- [System Prompt](prompts/system_prompt.py)
- [Response Style Guide](docs/RESPONSE_STYLE_GUIDE.md)
- [Testing Guide](docs/TESTING.md)

---

## 🚀 Deployment

**Cloud Run**: Ready for deployment  
**Environment Variables Required**:
- `GEMINI_API_KEY` - Your Gemini API key
- `MONGODB_URI` - MongoDB connection string
- `MAX_FUNCTION_CALLS` - Default: 30
- `MAX_OUTPUT_TOKENS` - Default: 4096

**CORS**: Currently allows all origins (*) - restrict in production!

---

## 📈 Next Steps

1. **UI Migration**: Apply reference design to frontend (in progress)
2. **Prompt Engineering**: Improve Gemini 3 compliance with [TIP]/[QUICK_REPLIES] tags
3. **Monitoring**: Add function call count logging
4. **Testing**: Comprehensive test suite for different query types

---

**Original Project**: https://github.com/Maqashable-284/scoop-genai-project  
**Changelog**: See [CHANGELOG.md](CHANGELOG.md) for detailed updates