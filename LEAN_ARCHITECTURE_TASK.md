# 🏗️ LEAN ARCHITECTURE REFACTORING TASK

## 📋 ASSIGNMENT FOR CLAUDE CODE

**Date:** 2026-01-14
**Priority:** HIGH
**Estimated Time:** 1 hour
**Type:** Architecture Optimization

---

## 🎯 OBJECTIVE

Transform the current "Heavy" architecture (65k tokens cached) into a "Lean" architecture (~2k tokens cached) that **forces** Gemini to call `search_products()` for ALL product-related queries.

**Problem Being Solved:**
- Currently, Gemini has full product catalog (~60k tokens) in context cache
- When asked about products, Gemini often writes from cached memory instead of calling `search_products()`
- This results in plain text instead of formatted ProductCards
- Frontend cannot render products without proper markdown format
- We're paying for 60k tokens cache that creates redundancy with MongoDB

**Goal:**
- Reduce cache from ~65k to ~2k tokens
- Force ALL product queries through `search_products()` function
- Guarantee ProductCards render on frontend
- Reduce costs by ~97%

---

## 📁 PROJECT STRUCTURE

```
/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/
├── main.py                           # FastAPI app, session management
├── config.py                         # Settings, imports SYSTEM_PROMPT
├── prompts/
│   └── system_prompt.py              # SYSTEM_PROMPT string (~5k tokens)
├── app/
│   ├── catalog/
│   │   └── loader.py                 # CatalogLoader - MAIN TARGET
│   ├── cache/
│   │   └── context_cache.py          # ContextCacheManager
│   ├── tools/
│   │   └── user_tools.py             # search_products() function
│   └── memory/
│       └── mongo_store.py            # MongoDB operations
└── requirements.txt
```

---

## 🔍 STEP 1: ANALYZE THESE FILES FIRST (DO NOT MODIFY YET)

### 1.1 Read and understand these files:

| File | Purpose | What to Look For |
|------|---------|------------------|
| `app/catalog/loader.py` | Formats catalog for cache | `format_catalog_context()` method |
| `prompts/system_prompt.py` | System instructions | Lines 65-101 (product recommendations) |
| `main.py` | Startup logic | How catalog is loaded and cached (lines 540-634) |
| `app/cache/context_cache.py` | Cache management | `create_cache()` method |
| `app/tools/user_tools.py` | Product search | `search_products()` function signature |

### 1.2 Current Data Flow (Understand This):

```
STARTUP:
1. CatalogLoader.load_products() → MongoDB → 315 products
2. CatalogLoader.format_catalog_context() → ~60k tokens of product text
3. ContextCacheManager.create_cache(system_prompt + catalog_context)
4. Result: ~65k tokens cached

CHAT REQUEST:
1. User asks: "რა პროტეინები გვაქვს?"
2. Gemini has full catalog in context
3. Option A: Gemini writes from cache (BAD - plain text)
4. Option B: Gemini calls search_products() (GOOD - formatted)
```

---

## 🔧 STEP 2: MODIFICATIONS REQUIRED

### 2.1 FILE: `app/catalog/loader.py`

**Current** (`format_catalog_context` method, ~lines 180-228):
```python
def format_catalog_context(self, products: List[Dict[str, Any]]) -> str:
    # Formats FULL product details for each of 315 products
    # Results in ~60,000 tokens!
```

**New** (ADD a new method `format_catalog_summary`):
```python
def format_catalog_summary(self, products: List[Dict[str, Any]]) -> str:
    """
    Generate MINIMAL catalog summary for Gemini context.
    
    Lean Architecture: Only provide metadata, NOT full product data.
    Forces Gemini to call search_products() for actual product info.
    
    Target: ~500-1000 tokens (vs 60k currently)
    """
    if not products:
        return "პროდუქტების კატალოგი ცარიელია."
    
    # Collect unique values
    categories = set()
    brands = set()
    prices = []
    
    for p in products:
        if cat := p.get("category"):
            categories.add(cat)
        if brand := p.get("brand"):
            brands.add(brand)
        if price := p.get("price"):
            prices.append(price)
    
    # Category translations
    category_names = {
        "protein": "პროტეინი",
        "creatine": "კრეატინი", 
        "bcaa": "BCAA/ამინომჟავები",
        "pre_workout": "პრე-ვორკაუთი",
        "vitamin": "ვიტამინები",
        "gainer": "გეინერი",
        "fat_burner": "ცხიმის მწვავი",
    }
    
    cat_list = [f"- {category_names.get(c, c)}" for c in sorted(categories)]
    brand_list = sorted(brands)[:10]  # Top 10 brands only
    
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    
    summary = f"""# Scoop.ge კატალოგის მიმოხილვა

## კატეგორიები ({len(categories)}):
{chr(10).join(cat_list)}

## ბრენდები (ტოპ 10):
{', '.join(brand_list)}

## ფასების დიაპაზონი:
{min_price:.0f}₾ - {max_price:.0f}₾

## სტატისტიკა:
- სულ პროდუქტი: {len(products)}
- მარაგში: {sum(1 for p in products if p.get('in_stock', False))}

---

⚠️ **CRITICAL INSTRUCTION:**
ზემოთ მოცემული მხოლოდ კატალოგის ᲛᲘᲛᲝᲮᲘᲚᲕᲐᲐ, არა სრული პროდუქტების ინფორმაცია!

პროდუქტის რეკომენდაციისთვის, ფასებისთვის, ან დეტალებისთვის **ᲐᲣᲪᲘᲚᲔᲑᲚᲐᲓ** გამოიძახე `search_products()` ფუნქცია!

არასოდეს არ დაწერო პროდუქტის სახელი, ფასი, ან buylink `search_products()` გამოძახების გარეშე!
"""
    return summary
```

**Also modify `get_catalog_context` method** (~line 234):
```python
async def get_catalog_context(self, force_refresh: bool = False, lean: bool = True) -> str:
    """
    Get catalog context with caching
    
    Args:
        force_refresh: Force reload from MongoDB
        lean: If True, return minimal summary (default). If False, return full catalog.
    """
    # ... existing cache check logic ...
    
    products = await self.load_products()
    
    # NEW: Use lean summary by default
    if lean:
        context = self.format_catalog_summary(products)
    else:
        context = self.format_catalog_context(products)
    
    # ... rest of caching logic ...
```

---

### 2.2 FILE: `prompts/system_prompt.py`

**Location:** Lines 65-101

**Replace the product recommendation section with:**

```python
## 🎯 SALES-FIRST MANDATORY RULE

**⚠️ CRITICAL:** შენ არ გაქვს პროდუქტების დეტალები კონტექსტში!
მხოლოდ კატეგორიები და ბრენდები იცი.

### ⚡ ᲐᲣᲪᲘᲚᲔᲑᲚᲐᲓ გამოიძახე `search_products()` როცა მომხმარებელი:

**პროდუქტზე კითხულობს:**
- "რომელი X ჯობია?" → `search_products("X")`
- "რა X მირჩევ?" → `search_products("X")`
- "რა პროდუქტები გვაქვს?" → `search_products("")`

**ფასზე კითხულობს:**
- "რამდენი ღირს X?" → `search_products("X")`
- "100₾-მდე ვარიანტები" → `search_products("protein", max_price=100)`

**ბრენდზე კითხულობს:**
- "Optimum Nutrition" → `search_products("Optimum Nutrition")`
- "რა ბრენდები გვაქვს?" → `search_products("")`

**სარგებელზე კითხულობს:**
- "რა სარგებელი აქვს X-ს?" → ᲯᲔᲠ `search_products("X")`, ᲛᲔᲠᲔ ახსნა!

### 🚫 ᲐᲙᲠᲫᲐᲚᲣᲚᲘᲐ:

- ❌ პროდუქტის სახელის დაწერა `search_products()` გარეშე
- ❌ ფასის დასახელება ფუნქციის გამოძახების გარეშე  
- ❌ buylink-ის გენერაცია ფუნქციის გამოძახების გარეშე
- ❌ პროდუქტების ჩამოთვლა მეხსიერებიდან

### ✅ ᲡᲬᲝᲠᲘ WORKFLOW:

1. მომხმარებელი კითხულობს პროდუქტზე
2. **ᲞᲘᲠᲕᲔᲚᲘ:** `search_products(query)` გამოძახება
3. **ᲛᲔᲝᲠᲔ:** შედეგების დაფორმატება frontend-ისთვის
4. **ᲛᲔᲡᲐᲛᲔ:** საგანმანათლებლო კონტექსტის დამატება

### 💡 რატომ?

- შენ არ გაქვს ფასები / სტოკი კონტექსტში (მხოლოდ კატეგორიები)
- `search_products()` გიბრუნებს LIVE მონაცემებს MongoDB-დან
- Frontend მოელის სპეციფიკურ markdown ფორმატს ProductCards-ისთვის
- ფუნქციის გარეშე პროდუქტები plain text-ად გამოჩნდება
```

---

### 2.3 FILE: `main.py`

**Location:** Startup lifespan function (~lines 560-570)

**Current:**
```python
catalog_context = await catalog_loader.get_catalog_context()
logger.info(f"Loaded catalog: ~{len(catalog_context)//4} tokens")
```

**Change to:**
```python
# Lean Architecture: Use minimal catalog summary
catalog_context = await catalog_loader.get_catalog_context(lean=True)
logger.info(f"Loaded lean catalog summary: ~{len(catalog_context)//4} tokens")
```

---

## ✅ STEP 3: VERIFICATION

After making changes, verify:

### 3.1 Restart Backend:
```bash
# Kill existing server
lsof -ti:8080 | xargs kill -9

# Start fresh
cd /Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080
```

### 3.2 Check Startup Logs:
```
Expected:
- "Loaded lean catalog summary: ~500 tokens" (NOT ~11000!)
- "Context cache created successfully (~2000 tokens cached)" (NOT ~13000!)
```

### 3.3 Test Query:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"რა პროტეინები გვაქვს?","user_id":"lean_test"}'
```

**Expected in logs:**
```
🔍 Calling search_products with: {"query": "protein"}
📦 Extracted 3 products from search_products calls
✅ Products already in correct markdown format (or "injected")
```

**Expected in response:**
- Markdown with `**Product Name**`, `*Brand*`, `**Price ₾**`
- NOT plain text paragraphs

### 3.4 Frontend Test:
1. Open http://localhost:3000
2. Ask: "რა პროტეინები გვაქვს?"
3. Verify: Horizontal ProductCards render (not plain text)

---

## 📊 SUCCESS METRICS

| Metric | Before (Heavy) | After (Lean) | Target |
|--------|---------------|--------------|--------|
| Cache Size | ~65,000 tokens | ~2,000 tokens | ✅ <3,000 |
| Cache Cost | $0.001/hour | $0.00003/hour | ✅ <$0.0001 |
| search_products calls | Sometimes | Always | ✅ 100% |
| ProductCards render | Random | Guaranteed | ✅ 100% |

---

## ⚠️ IMPORTANT NOTES

1. **DO NOT remove** `format_catalog_context()` method - keep it for potential future use
2. **DO NOT modify** `search_products()` function in user_tools.py
3. **DO NOT change** MongoDB schema or queries
4. **PRESERVE** all existing functionality for [TIP] and [QUICK_REPLIES] tags
5. **TEST** after each file modification before moving to next

---

## 🔗 RELATED FILES (Read-Only Reference)

These files should NOT be modified but may be useful for context:
- `app/tools/user_tools.py` - Understand search_products() return format
- `app/cache/context_cache.py` - Understand how cache is created
- `config.py` - Settings and environment variables

---

## 📝 SUMMARY OF CHANGES

| File | Action | Lines |
|------|--------|-------|
| `app/catalog/loader.py` | ADD `format_catalog_summary()` method | New method |
| `app/catalog/loader.py` | MODIFY `get_catalog_context()` | Add `lean` param |
| `prompts/system_prompt.py` | REPLACE lines 65-101 | Product section |
| `main.py` | MODIFY startup | ~line 564 |
| `config.py` | MODIFY setting | `enable_context_caching = False` |

---

## 🔒 STEP 4: DISABLE CACHING (Keep Code for Future)

### Why Disable?
- Lean Architecture = ~5,500 tokens total
- Google Caching API MINIMUM = 32,768 tokens
- **Caching won't work** with less than 32k tokens
- Keep code for future when catalog grows to 1000+ products

### 4.1 FILE: `config.py`

**Find the setting** (search for `enable_context_caching`):
```python
# CURRENT:
enable_context_caching: bool = True

# CHANGE TO:
enable_context_caching: bool = False
```

### 4.2 Verify in `main.py` (lines ~585-618)

The existing code already handles disabled caching:
```python
# Week 4: Initialize context caching for 85% token savings
if settings.enable_context_caching:  # ← This will be False now
    logger.info("🚀 Week 4: Initializing context caching...")
    # ... caching logic ...
else:
    logger.info("Context caching disabled via settings")  # ← This will run
    context_cache_manager = None
```

**No code change needed in main.py for caching** - just the config flag!

### 4.3 Expected Startup Logs (After Changes):

```
INFO - Starting Scoop GenAI server...
INFO - Connected to MongoDB
INFO - Loaded lean catalog summary: ~500 tokens    # ← NEW (was ~11000)
INFO - Context caching disabled via settings       # ← NEW
INFO - Application startup complete
INFO - Uvicorn running on http://0.0.0.0:8080
```

**NOT expected anymore:**
```
❌ INFO - Context cache created successfully (~13798 tokens cached)
❌ INFO - Started context cache refresh task
```

---

## 📊 FINAL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEAN ARCHITECTURE (Final)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PER REQUEST:                                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  System Prompt (~5k tokens)                               │  │
│   │  + Catalog Summary (~500 tokens)                          │  │
│   │  + User message + History                                 │  │
│   │  ────────────────────────────────                        │  │
│   │  Total: ~6k tokens input per request                      │  │
│   │  Cost: ~$0.00045 per request                              │  │
│   │  Daily (1000 req): ~$0.45                                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   Gemini MUST call search_products()                            │
│   (no product details in context!)                              │
│                              │                                   │
│                              ▼                                   │
│   MongoDB → LIVE Products → Formatted Markdown                  │
│                              │                                   │
│                              ▼                                   │
│   Frontend: ProductCards ALWAYS render ✅                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔮 FUTURE: When to Re-enable Caching

Re-enable caching when:
- Catalog grows to 500+ products (~80k tokens)
- System prompt grows significantly
- Total context > 32,768 tokens

To re-enable:
1. Set `enable_context_caching = True` in config.py
2. Use `get_catalog_context(lean=False)` for full catalog
3. Restart backend

**The caching code is preserved in:**
- `app/cache/context_cache.py` - ContextCacheManager
- `app/catalog/loader.py` - Full catalog formatting
- `main.py` - Caching initialization logic

---

**🚀 BEGIN IMPLEMENTATION - ANALYZE FIRST, THEN MODIFY!**
