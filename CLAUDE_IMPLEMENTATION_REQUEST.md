# 🔧 Implementation Request: TIP Tag Injection System

**Date:** 2026-01-14  
**Priority:** HIGH  
**Estimated Time:** 30-45 minutes  
**Risk Level:** LOW

---

## 📊 Executive Summary

**Problem:** Gemini 3 Flash Preview does NOT reliably generate `[TIP]...[/TIP]` tags despite explicit system prompt instructions.

**Root Cause:** Model instruction-following limitation (documented Gemini 3 Flash Preview issue)

**Solution:** Two-part fix:
1. ✅ **Backend Post-Processing** (CRITICAL) - Inject [TIP] tags when missing
2. ✅ **System Prompt Optimization** (ENHANCEMENT) - Consolidate and strengthen instructions

**Impact:** Guaranteed 95%+ TIP compliance + improved model behavior

---

## 🔍 Critical Analysis Findings

### ✅ What ALREADY Works (DO NOT TOUCH!)

#### 1. Quick Replies System
**Status:** ✅ **WORKING PERFECTLY**

**Evidence:**
- Screenshot shows 4 Quick Reply buttons rendering correctly
- Backend `parse_quick_replies()` function (main.py:753-809) works correctly
- Frontend reads from `data.quick_replies` JSON field (Chat.tsx:297)

**How it works:**
```python
# Backend parses [QUICK_REPLIES] tags from text
pattern = r'\[QUICK_REPLIES\](.*?)\[/QUICK_REPLIES\]'
quick_replies = [{"title": line, "payload": line} for line in ...]
# Returns JSON array to frontend
```

**⚠️ CRITICAL:** DO NOT modify `parse_quick_replies()` - it's working!

---

#### 2. Function Calling
**Status:** ✅ Working

- `search_products()` called correctly
- `get_user_profile()` / `update_user_profile()` work
- `automatic_function_calling` config with 30 max calls works

---

#### 3. Context Caching
**Status:** ✅ Working

- Saves 96% costs ($360 → $15/mo)
- System prompt properly cached
- TTL: 60 minutes with auto-refresh

---

### ❌ What NEEDS Fixing

#### TIP Tags - Missing from Responses
**Status:** ❌ **0% COMPLIANCE**

**Problem:**
- Gemini doesn't generate `[TIP]...[/TIP]` tags in response text
- Frontend expects tags (parseProducts.ts:107 parses them client-side)
- Without tags, yellow "პრაქტიკული რჩევა" box doesn't appear

**Evidence from Analysis:**

**System Prompt (lines 103-113 AND 252-273):**
```markdown
### 3️⃣ მესამე: დაამატე [TIP] section
**ყველა პასუხის ბოლოს ᲐᲣᲪᲘᲚᲔᲑᲚᲐᲓ:**
[TIP]
პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში.
[/TIP]
```

**Frontend Parsing (parseProducts.ts:106-112):**
```typescript
const tipPattern = /\[TIP\]([\s\S]*?)\[\/TIP\]/;
const tipMatch = markdown.match(tipPattern);
if (tipMatch) {
    tip = tipMatch[1].trim();  // Frontend expects this!
    markdown = markdown.replace(tipPattern, '').trim();
}
```

**Issue Causes (from Analysis):**
1. **Long prompt (382 lines)** - Gemini may "forget" instructions
2. **Duplicate TIP instructions** - Appears at lines 103-113 AND 252-273 (confusing)
3. **No enforcement** - Model can ignore without consequence
4. **Model limitation** - Known Gemini 3 Flash Preview issue

---

## 🎯 Implementation Plan

### Solution 1: Backend TIP Injection (REQUIRED)

**Files to Modify:** `main.py`

**Effort:** Low | **Risk:** Low | **Expected Impact:** 95%+ compliance

---

#### Step 1.1: Add `generate_contextual_tip()` Function

**Location:** Add after line 825 (after `clean_leaked_function_calls()` function)

**Code:**

```python
def generate_contextual_tip(text: str) -> str:
    """
    Generate contextual tip based on response content.
    
    Args:
        text: The response text to analyze
        
    Returns:
        Contextual tip string (1-2 sentences)
    """
    text_lower = text.lower()
    
    # Contextual tips mapped to keywords
    contextual_tips = {
        # Protein-related
        'პროტეინ': 'პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.',
        'whey': 'whey პროტეინი საუკეთესოდ აღიწოვს ვარჯიშის შემდეგ.',
        'isolate': 'isolate უფრო სწრაფად აღიწოვს და შეიცავს ნაკლებ ლაქტოზას.',
        
        # Creatine-related
        'კრეატინ': 'კრეატინი ყოველდღიურად მიიღეთ 3-5 გრამი, ვარჯიშის დღეებშიც და დასვენების დღეებშიც.',
        'creatine': 'კრეატინის loading ფაზა არ არის სავალდებულო, შეგიძლიათ დაიწყოთ 3-5g/დღე.',
        
        # Pre-workout
        'პრე-ვორკ': 'პრე-ვორკაუთი ვარჯიშამდე 20-30 წუთით ადრე მიიღეთ.',
        'pre-work': 'თავიდან აარიდეთ პრე-ვორკაუთი საღამოს, რათა ძილი არ დაირღვეს.',
        
        # BCAA
        'bcaa': 'BCAA ეფექტურია ცარიელ კუჭზე ვარჯიშის დროს.',
        'ამინომჟავ': 'ამინომჟავები საუკეთესოდ მუშაობს ვარჯიშის დროს და შემდეგ.',
        
        # Gainer
        'გეინერ': 'გეინერი მიიღეთ ვარჯიშის შემდეგ და საჭიროების მიხედვით კვებებს შორის.',
        'gainer': 'გეინერი 2-3 დოზად დაყავით დღეში კუჭის დისკომფორტის თავიდან ასაცილებლად.',
        
        # Vitamins
        'ვიტამინ': 'ვიტამინები უმჯობესია საკვებთან ერთად მიიღოთ შეწოვის გასაუმჯობესებლად.',
        'vitamin': 'მულტივიტამინები დილით საკვებთან ერთად მიიღეთ.',
        
        # Fat burners
        'fat burn': 'fat burner-ების ეფექტურობისთვის აუცილებელია კალორიული დეფიციტი.',
        'წონის კლება': 'წონის კლებისთვის მთავარია კალორიული დეფიციტი - დანამატები დამხმარე საშუალებაა.',
        
        # General weight
        'წონა': 'წონის ცვლილებისთვის მთავარია კალორიების ბალანსი - დანამატები დამხმარე საშუალებაა.',
        'მასა': 'კუნთოვანი მასის მოსაპოვებლად საჭიროა კალორიული სუფიციტი და საკმარისი პროტეინი.',
        
        # Hydration
        'წყალი': 'დღეში მინიმუმ 2-3 ლიტრი წყალი მიიღეთ, განსაკუთრებით კრეატინის მიღებისას.',
    }
    
    # Find matching tip
    for keyword, tip in contextual_tips.items():
        if keyword in text_lower:
            logger.info(f"💡 Generated contextual tip for keyword: {keyword}")
            return tip
    
    # Default fallback tip
    logger.info("💡 Using default generic tip")
    return 'რეკომენდაციებთან დაკავშირებით კითხვების შემთხვევაში მოგვწერეთ support@scoop.ge'
```

---

#### Step 1.2: Add `ensure_tip_tag()` Function

**Location:** Add right after `generate_contextual_tip()`

**Code:**

```python
def ensure_tip_tag(response_text: str) -> str:
    """
    Ensure response has [TIP] tag. If missing, inject contextual tip.
    
    This is a safety net for Gemini 3 Flash Preview which doesn't reliably
    generate [TIP] tags despite explicit system prompt instructions.
    
    Args:
        response_text: The model's response text
        
    Returns:
        Response text with guaranteed [TIP] tag
    """
    # Check if TIP tag already exists
    if '[TIP]' in response_text and '[/TIP]' in response_text:
        logger.info("✅ [TIP] tag already present in response")
        return response_text
    
    logger.warning("⚠️ [TIP] tag missing from Gemini response - injecting")
    
    # Generate contextual tip based on response content
    tip = generate_contextual_tip(response_text)
    
    # Determine injection point
    # CRITICAL: Inject BEFORE [QUICK_REPLIES] if it exists
    if '[QUICK_REPLIES]' in response_text:
        # Split at QUICK_REPLIES and insert TIP before it
        parts = response_text.split('[QUICK_REPLIES]', 1)
        injected = f"{parts[0].rstrip()}\n\n[TIP]\n{tip}\n[/TIP]\n\n[QUICK_REPLIES]{parts[1]}"
        logger.info(f"💉 Injected TIP before [QUICK_REPLIES]: {tip[:50]}...")
    else:
        # Append TIP at the very end
        injected = f"{response_text.rstrip()}\n\n[TIP]\n{tip}\n[/TIP]"
        logger.info(f"💉 Appended TIP at end: {tip[:50]}...")
    
    return injected
```

---

#### Step 1.3: Integrate into `/chat` Endpoint

**Location:** Find the `/chat` endpoint (around lines 940-1050)

**What to find:**
Look for where `response_text_geo` is extracted from the API response.

**Current code (approximately):**
```python
# Extract response text
response_text_geo = data.get('response_text_geo') or data.get('response') or data.get('text') or ''
```

**Modified code:**
```python
# Extract response text
response_text_geo = data.get('response_text_geo') or data.get('response') or data.get('text') or ''

# CRITICAL FIX: Ensure [TIP] tag is present (inject if missing)
# Gemini 3 Flash Preview doesn't reliably generate [TIP] tags
response_text_geo = ensure_tip_tag(response_text_geo)
```

**⚠️ IMPORTANT:** Add this BEFORE calling `parse_quick_replies()` so the full text (with TIP) gets parsed.

---

### Solution 2: System Prompt Optimization (ENHANCEMENT)

**Files to Modify:** `prompts/system_prompt.py`

**Effort:** Low | **Risk:** Low | **Expected Impact:** 20-30% model improvement

---

#### Step 2.1: Remove Duplicate TIP Instructions

**Action:** DELETE lines 103-113

**Reason:** TIP instructions appear twice (103-113 AND 252-273) which confuses the model.

**Lines to remove:**
```markdown
### 3️⃣ მესამე: დაამატე [TIP] section

**ყველა პასუხის ბოლოს ᲐᲣᲪᲘᲚᲔᲑᲚᲐᲓ:**
```
[TIP]
პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.
[/TIP]
```

**TIP-ის გარეშე პასუხი არასრულია!**
```

---

#### Step 2.2: Consolidate Tag Instructions at END

**Action:** REPLACE lines 252-345 with the optimized version below

**Reason:** 
- Instructions at END of prompt are more likely to be followed (recency bias)
- Consolidates scattered instructions into one clear block
- Adds stronger enforcement language

**New consolidated section:**

```markdown
---

## 🚨 MANDATORY OUTPUT FORMAT - YOU MUST OBEY!

**CRITICAL:** ყოველი პასუხი ᲐᲣᲪᲘᲚᲔᲑᲚᲐᲓ უნდა დასრულდეს ამ ორი სექციით ზუსტად ამ თანმიმდევრობით:

### 1️⃣ პრაქტიკული რჩევა [TIP]

**ფორმატი (200% დაიცავი):**

```
[TIP]
მოკლე, პრაქტიკული რჩევა 1-2 წინადადებით.
[/TIP]
```

**კონკრეტული მაგალითები:**

პროტეინის შესახებ პასუხში:
```
[TIP]
პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.
[/TIP]
```

კრეატინის შესახებ პასუხში:
```
[TIP]
კრეატინი ყოველდღიურად მიიღეთ 3-5 გრამი, ვარჯიშის დღეებშიც და დასვენების დღეებშიც.
[/TIP]
```

პრე-ვორკაუთის შესახებ პასუხში:
```
[TIP]
პრე-ვორკაუთი ვარჯიშამდე 20-30 წუთით ადრე მიიღეთ.
[/TIP]
```

**⛔ [TIP] tag-ის გარეშე პასუხი ᲐᲠᲐᲡᲠᲣᲚᲘᲐ და ᲣᲐᲠᲧᲝᲤᲘᲚᲘ იქნება!**

---

### 2️⃣ შემდეგი ნაბიჯები [QUICK_REPLIES]

**ფორმატი (ზუსტად 4 ოფცია - არანაკლები, არამეტი):**

```
[QUICK_REPLIES]
ოფცია 1 - პროდუქტი/გაყიდვა
ოფცია 2 - პროდუქტი/გაყიდვა
ოფცია 3 - განათლება/ცოდნა
ოფცია 4 - განათლება/ცოდნა
[/QUICK_REPLIES]
```

**სავალდებულო სტრატეგია 2+2:**
- **პირველი 2** = გაყიდვაზე ორიენტირებული (მაგ: "ამ პროდუქტის შეძენა", "100₾-მდე ვარიანტები")
- **მეორე 2** = საგანმანათლებლო (მაგ: "როგორ მივიღო?", "რა დოზა მჭირდება?")

**მაგალითი #1 - პროტეინის რეკომენდაცია:**
```
[QUICK_REPLIES]
ამ პროტეინის შეძენა
100₾-მდე ალტერნატივები
როგორ მივიღო პროტეინი?
whey vs isolate განსხვავება
[/QUICK_REPLIES]
```

**მაგალითი #2 - კრეატინის რეკომენდაცია:**
```
[QUICK_REPLIES]
ამ კრეატინის შეძენა
პრე-ვორკაუთიც მჭირდება?
როგორ მივიღო კრეატინი?
loading ფაზა საჭიროა?
[/QUICK_REPLIES]
```

**⛔ [QUICK_REPLIES] tag-ის გარეშე პასუხი ᲐᲠᲐᲡᲠᲣᲚᲘᲐ და ᲣᲐᲠᲧᲝᲤᲘᲚᲘ იქნება!**

---

## ⚠️ რა მოხდება Tags-ების გარეშე

თუ არ დააგენერირებ [TIP] და [QUICK_REPLIES] tags-ებს:

❌ **Frontend UI incomplete:**
- პრაქტიკული რჩევის ყვითელი box არ გამოჩნდება
- Follow-up action ღილაკები არ გამოჩნდება
- მომხმარებელი მიიღებს არასრულ UX-ს

❌ **Backend post-processing:**
- სისტემა იძულებულია დაამატოს generic TIP (არასასურველია!)
- ნაკლებად კონტექსტური რჩევა

❌ **Quality degradation:**
- მომხმარებლის engagement დაბალი
- Conversion rate დაბალი (არ ხედავს "ყიდვის" ღილაკებს)

**🎯 გადაწყვეტა:** ᲧᲝᲕᲔᲚᲗᲕᲘᲡ დაამთავრე პასუხი ორივე tag-ით. EXCEPTIONS არ არსებობს!
```

---

## 🧪 Testing Plan

### Test 1: Verify TIP Injection Works

**Purpose:** Confirm that `ensure_tip_tag()` successfully injects TIP tags

**Command:**
```bash
curl -s -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "test_tip_inject", "message": "მაჩვენე whey პროტეინები"}' | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
text = data.get('response_text_geo', '')
has_tip = '[TIP]' in text and '[/TIP]' in text

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('Test 1: TIP Tag Injection')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('Status:', '✅ PASS' if has_tip else '❌ FAIL')

if has_tip:
    tip_start = text.index('[TIP]') + 5
    tip_end = text.index('[/TIP]')
    tip_content = text[tip_start:tip_end].strip()
    print(f'TIP Content: {tip_content}')
else:
    print('ERROR: [TIP] tag not found in response!')
    print('Response preview:', text[:200], '...')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
"
```

**Expected Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test 1: TIP Tag Injection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ PASS
TIP Content: პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Test 2: Verify Quick Replies Still Work

**Purpose:** Ensure Quick Replies functionality wasn't broken

**Command:**
```bash
curl -s -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "test_qr_stable", "message": "რომელი პროტეინი ჯობია?"}' | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
qr = data.get('quick_replies', [])

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('Test 2: Quick Replies Stability')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print(f'Status:', '✅ PASS' if len(qr) >= 2 else '❌ FAIL')
print(f'Count: {len(qr)} replies found')
for i, r in enumerate(qr[:4], 1):
    print(f'  {i}. {r.get(\"title\", \"N/A\")}')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
"
```

**Expected Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test 2: Quick Replies Stability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ PASS
Count: 4 replies found
  1. ამ პროტეინის შეძენა
  2. 100₾-მდე ალტერნატივები
  3. როგორ მივიღო პროტეინი?
  4. whey vs isolate განსხვავება
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Test 3: Multiple Query Types

**Purpose:** Test TIP injection across different content types

**Command:**
```bash
queries=(
  "მინდა კუნთის მასის მომატება"
  "როგორ მივიღო კრეატინი?"
  "რა პროტეინი მირჩიე?"
  "100 ლარამდე ვარიანტები"
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Multi-Query TIP Detection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for q in "${queries[@]}"; do
  response=$(curl -s -X POST http://localhost:8080/chat \
    -H 'Content-Type: application/json' \
    -d "{\"user_id\": \"test_multi_$(date +%s)\", \"message\": \"$q\"}")
  
  has_tip=$(echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('✅' if '[TIP]' in data.get('response_text_geo', '') else '❌')
" 2>/dev/null)
  
  printf "%-40s %s\n" "$q" "$has_tip"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

**Expected Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test 3: Multi-Query TIP Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
მინდა კუნთის მასის მომატება            ✅
როგორ მივიღო კრეატინი?                ✅
რა პროტეინი მირჩიე?                    ✅
100 ლარამდე ვარიანტები                  ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Test 4: Check Injection Logs

**Purpose:** Monitor that injection is happening and being logged

**Command:**
```bash
# Start monitoring logs (in separate terminal)
# Adjust path to your actual log file or use docker logs
tail -f /path/to/backend.log | grep -E "(TIP tag|💡|💉|✅|⚠️)"

# Or if using stdout:
# Monitor the backend process output
```

**Expected Log Output:**
```
2026-01-14 16:45:23 - INFO - ⚠️ [TIP] tag missing from Gemini response - injecting
2026-01-14 16:45:23 - INFO - 💡 Generated contextual tip for keyword: პროტეინ
2026-01-14 16:45:23 - INFO - 💉 Injected TIP before [QUICK_REPLIES]: პროტეინი მიიღეთ ვარჯიშის შემდეგ...
```

---

### Test 5: Regression - Function Calling

**Purpose:** Ensure function calling still works

**Command:**
```bash
# Monitor logs for function calls
tail -f /path/to/backend.log | grep -E "(search_products|get_user_profile|Function call)"
```

**Expected:** See function call logs when queries require product search

---

### Test 6: Frontend Visual Test

**Steps:**
1. Open browser at `http://localhost:3000` (or your frontend URL)
2. Send message: "მაჩვენე whey პროტეინები"
3. Visual inspection:
   - ✅ Yellow "პრაქტიკული რჩევა" box appears
   - ✅ Contains relevant tip about protein
   - ✅ 4 Quick Reply buttons appear below
   - ✅ Product cards render correctly

---

## ✅ Success Criteria

Implementation is complete when ALL of these are true:

### Code Changes:
- [  ] `generate_contextual_tip()` function added to main.py
- [  ] `ensure_tip_tag()` function added to main.py
- [  ] `/chat` endpoint calls `ensure_tip_tag()` before returning
- [  ] Duplicate TIP instructions removed from system_prompt.py (lines 103-113)
- [  ] Tag instructions consolidated at end of system_prompt.py (lines 252-345)

### Test Results:
- [  ] Test 1 passes: TIP tag present in response
- [  ] Test 2 passes: Quick Replies still return 4 options
- [  ] Test 3 passes: All 4 query types have TIP tags
- [  ] Test 4 passes: Logs show injection messages
- [  ] Test 5 passes: Function calling still works
- [  ] Test 6 passes: Frontend renders TIP box correctly

### No Regressions:
- [  ] `parse_quick_replies()` unchanged and working
- [  ] Context caching still active (check `/health` endpoint)
- [  ] No errors in backend logs
- [  ] Response times acceptable (<5s)

---

## 📋 Implementation Checklist

**Before starting:**
- [  ] Read this entire document
- [  ] Understand what works (Quick Replies) vs what needs fixing (TIP tags)
- [  ] Backup current `main.py` and `system_prompt.py`

**During implementation:**
- [  ] Add `generate_contextual_tip()` function
- [  ] Add `ensure_tip_tag()` function
- [  ] Integrate `ensure_tip_tag()` into `/chat` endpoint
- [  ] Remove duplicate TIP instructions (lines 103-113)
- [  ] Replace lines 252-345 with consolidated version
- [  ] Add logging statements for debugging

**After implementation:**
- [  ] Restart backend server
- [  ] Run Test 1 (TIP injection)
- [  ] Run Test 2 (Quick Replies stability)
- [  ] Run Test 3 (Multiple queries)
- [  ] Check logs (Test 4)
- [  ] Verify function calling (Test 5)
- [  ] Visual check frontend (Test 6)

**Reporting back:**
- [  ] List all modified files with line numbers
- [  ] Share test outputs (all 6 tests)
- [  ] Share relevant code snippets
- [  ] Report any warnings/errors encountered

---

## ⚠️ Critical Reminders

### DO NOT:
- ❌ Touch `parse_quick_replies()` function - it's working!
- ❌ Modify function calling config (`automatic_function_calling`)
- ❌ Change context caching setup
- ❌ Remove or modify frontend TIP parsing (parseProducts.ts)
- ❌ Change Quick Replies JSON structure

### DO:
- ✅ Add comprehensive logging (`logger.info()`, `logger.warning()`)
- ✅ Use contextual tips from `generate_contextual_tip()`
- ✅ Test with multiple query types
- ✅ Check that TIP appears BEFORE [QUICK_REPLIES] in text
- ✅ Verify frontend renders correctly after changes

---

## 📤 How to Report Back

After completing implementation, provide:

### 1. Summary of Changes
```markdown
## Modified Files:

### main.py
- Line XXX: Added generate_contextual_tip() function
- Line YYY: Added ensure_tip_tag() function
- Line ZZZ: Integrated ensure_tip_tag() into /chat endpoint

### prompts/system_prompt.py
- Lines 103-113: Removed (duplicate TIP instructions)
- Lines 252-345: Replaced with consolidated tag section
```

### 2. Test Results
```markdown
## Test Results:

✅ Test 1: PASS - TIP tag present
✅ Test 2: PASS - 4 Quick Replies found
✅ Test 3: PASS - 4/4 queries have TIP
✅ Test 4: PASS - Injection logs visible
✅ Test 5: PASS - Function calls working
✅ Test 6: PASS - Frontend displays correctly
```

### 3. Code Snippets

Share:
- Final `ensure_tip_tag()` function
- Where it's integrated in `/chat` endpoint
- New system prompt tag section (lines 252-345)

### 4. Any Issues Encountered

Report:
- Errors or warnings
- Unexpected behavior
- Questions or clarifications needed

---

**Priority:** HIGH  
**Deadline:** ASAP  
**Estimated Time:** 30-45 minutes  
**Point of Contact:** Return to Gemini for code review after implementation

Good luck! 🚀

Hi Claude Code! Based on my analysis, please implement **Solution 1 + Solution 2** to fix the missing `[TIP]` tags issue.

---

## 📋 Context Summary

**Problem:** Gemini 3 Flash Preview doesn't consistently generate `[TIP]...[/TIP]` tags despite system prompt instructions.

**Root Cause:** Model instruction-following limitation (known Gemini 3 issue)

**Solution:** Two-part fix:
1. **Backend post-processing** - Inject missing TIP tags (guaranteed compliance)
2. **System prompt optimization** - Consolidate and strengthen tag instructions (improve model behavior)

---

## 🎯 Implementation Tasks

### Task 1: Backend Post-Processing (Priority: HIGH)

**File:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/main.py`

#### Step 1.1: Add `generate_contextual_tip()` function

Add this function after line 825 (after `clean_leaked_function_calls`):

```python
def generate_contextual_tip(text: str) -> str:
    """
    Generate contextual tip based on response content.
    Returns appropriate tip for the topic.
    """
    text_lower = text.lower()
    
    # Product-specific tips
    contextual_tips = {
        'პროტეინ': 'პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.',
        'whey': 'whey პროტეინი საუკეთესოდ აღიწოვს ვარჯიშის შემდეგ.',
        'კრეატინ': 'კრეატინი ყოველდღიურად მიიღეთ 3-5 გრამი, ვარჯიშის დღეებშიც და დასვენების დღეებშიც.',
        'creatine': 'კრეატინის loading ფაზა არ არის სავალდებულო, შეგიძლიათ დაიწყოთ 3-5g/დღე.',
        'პრე-ვორკ': 'პრე-ვორკაუთი ვარჯიშამდე 20-30 წუთით ადრე მიიღეთ.',
        'pre-work': 'თავიდან აარიდეთ პრე-ვორკაუთი საღამოს, რათა ძილი არ დაირღვეს.',
        'bcaa': 'BCAA ეფექტურია ცარიელ კუჭზე ვარჯიშის დროს.',
        'გეინერ': 'გეინერი მიიღეთ ვარჯიშის შემდეგ და საჭიროების მიხედვით კვებებს შორის.',
        'gainer': 'გეინერი 2-3 დოზად დაყავით დღეში კუჭის დისკომფორტის თავიდან ასაცილებლად.',
        'ვიტამინ': 'ვიტამინები უმჯობესია საკვებთან ერთად მიიღოთ შეწოვის გასაუმჯობესებლად.',
        'fat burn': 'fat burner-ების ეფექტურობისთვის აუცილებელია კალორიული დეფიციტი.',
        'წონა': 'წონის ცვლილებისთვის მთავარია კალორიების ბალანსი - დანამატები დამხმარე საშუალებაა.',
    }
    
    # Find matching tip
    for keyword, tip in contextual_tips.items():
        if keyword in text_lower:
            return tip
    
    # Default generic tip
    return 'რეკომენდაციებთან დაკავშირებით კითხვების შემთხვევაში მოგვწერეთ support@scoop.ge'
```

#### Step 1.2: Add `ensure_tip_tag()` function

Add this function right after `generate_contextual_tip()`:

```python
def ensure_tip_tag(response_text: str) -> str:
    """
    Ensure response has [TIP] tag. If missing, inject contextual tip.
    
    Args:
        response_text: The model's response text
        
    Returns:
        Response text with guaranteed [TIP] tag
    """
    # Check if TIP tag already exists
    if '[TIP]' in response_text and '[/TIP]' in response_text:
        logger.info("✅ TIP tag already present in response")
        return response_text
    
    # Generate contextual tip based on response content
    tip = generate_contextual_tip(response_text)
    
    # Inject TIP tag at the end (before QUICK_REPLIES if exists)
    if '[QUICK_REPLIES]' in response_text:
        # Insert TIP before QUICK_REPLIES
        parts = response_text.split('[QUICK_REPLIES]')
        injected = f"{parts[0].strip()}\n\n[TIP]\n{tip}\n[/TIP]\n\n[QUICK_REPLIES]{parts[1]}"
    else:
        # Append TIP at the end
        injected = f"{response_text.strip()}\n\n[TIP]\n{tip}\n[/TIP]"
    
    logger.info(f"⚠️ TIP tag was missing - injected contextual tip: {tip[:50]}...")
    return injected
```

#### Step 1.3: Integrate into `/chat` endpoint

Find the `/chat` endpoint (around line 940-1050) and locate where `response_text_geo` is set.

**Before:**
```python
response_text_geo = data.response_text_geo || data.response || data.text || ''
```

**After:**
```python
response_text_geo = data.response_text_geo or data.response or data.text or ''

# Ensure TIP tag is present (inject if missing)
response_text_geo = ensure_tip_tag(response_text_geo)
```

---

### Task 2: System Prompt Optimization (Priority: MEDIUM)

**File:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/prompts/system_prompt.py`

#### Step 2.1: Remove duplicate TIP instructions

**Remove lines 103-113** (first TIP instruction block - keep only the second one)

#### Step 2.2: Consolidate tag instructions at END of prompt

**Replace lines 252-345** (entire TIP + Quick Replies section) with this optimized version:

```python
---

## 🚨 MANDATORY OUTPUT FORMAT - NEVER SKIP THIS!

**CRITICAL REQUIREMENT:** ყოველი პასუხი ᲐᲣᲪᲘᲚᲔᲑᲚᲐᲓ უნდა დასრულდეს ამ ორი სექციით ზუსტად ამ თანმიმდევრობით:

### 1. პრაქტიკული რჩევა [TIP]

**ფორმატი (200% დაიცავი):**

```
[TIP]
მოკლე, პრაქტიკული რჩევა 1-2 წინადადებით.
[/TIP]
```

**მაგალითები:**
- პროტეინის შესახებ: "პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის."
- კრეატინის შესახებ: "კრეატინი ყოველდღიურად მიიღეთ 3-5 გრამი, ვარჯიშის დღეებშიც და დასვენების დღეებშიც."
- პრე-ვორკაუთის შესახებ: "პრე-ვორკაუთი ვარჯიშამდე 20-30 წუთით ადრე მიიღეთ."

**⛔ [TIP] tag-ის გარეშე პასუხი ᲐᲠᲐᲡᲠᲣᲚᲘᲐ და ᲣᲐᲠᲧᲝᲤᲘᲚᲘᲐ!**

---

### 2. Quick Replies [QUICK_REPLIES]

**ფორმატი (ზუსტად 4 ოფცია):**

```
[QUICK_REPLIES]
ოფცია 1 - პროდუქტი/გაყიდვა
ოფცია 2 - პროდუქტი/გაყიდვა
ოფცია 3 - განათლება/ინფო
ოფცია 4 - განათლება/ინფო
[/QUICK_REPLIES]
```

**სტრატეგია 2+2:**
- პირველი 2 = გაყიდვაზე ორიენტირებული ("ამ პროდუქტის შეძენა", "შევადარო ბრენდებს", "100₾-მდე ვარიანტები")
- მეორე 2 = საგანმანათლებლო ("როგორ მივიღო?", "რა დოზა მჭირდება?", "whey vs isolate განსხვავება")

**მაგალითები:**

პროტეინის რეკომენდაციის შემდეგ:
```
[QUICK_REPLIES]
ამ პროტეინის შეძენა
100₾-მდე ალტერნატივები
როგორ მივიღო პროტეინი?
whey vs isolate განსხვავება
[/QUICK_REPLIES]
```

კრეატინის რეკომენდაციის შემდეგ:
```
[QUICK_REPLIES]
ამ კრეატინის შეძენა
პრე-ვორკაუთიც მჭირდება?
როგორ მივიღო კრეატინი?
loading ფაზა საჭიროა?
[/QUICK_REPLIES]
```

**⛔ [QUICK_REPLIES] tag-ის გარეშე პასუხი ᲐᲠᲐᲡᲠᲣᲚᲘᲐ და ᲣᲐᲠᲧᲝᲤᲘᲚᲘᲐ!**

---

## ⚠️ შედეგები Tag-ების გარეშე

თუ პასუხს არ აქვს [TIP] და [QUICK_REPLIES] tags:
- ❌ Frontend UI ვერ გამოაჩენს პრაქტიკული რჩევის სექციას
- ❌ მომხმარებელი ვერ დაინახავს follow-up ღილაკებს
- ❌ UX დამწყებული ჩაითვლება
- ❌ სისტემა დაამატებს generic tags-ებს (არასასურველია!)

**გახსოვდეს:** Tags არის ᲐᲣᲪᲘᲚᲔᲑᲔᲚᲘ, არა ოფციონალური!
```

---

### Task 3: Context Cache Refresh

**File:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/app/cache/context_cache.py`

The system prompt changes require cache refresh. You don't need to modify code - just note that:

1. When backend restarts, cache will auto-refresh within 60min TTL
2. Or manually delete cache using `/admin/cache/refresh` endpoint (if exists)
3. Cache metrics will log the refresh

---

## 🧪 Testing Instructions

After implementation, run these tests:

### Test 1: Verify TIP Injection Works

```bash
# Test against localhost
curl -s -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "test_tip_inject", "message": "მაჩვენე whey პროტეინები"}' | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
text = data.get('response_text_geo', '')
has_tip = '[TIP]' in text and '[/TIP]' in text
print('✅ TIP tag present' if has_tip else '❌ TIP tag MISSING')
if has_tip:
    tip_start = text.index('[TIP]') + 5
    tip_end = text.index('[/TIP]')
    tip_content = text[tip_start:tip_end].strip()
    print(f'TIP content: {tip_content}')
"
```

### Test 2: Verify Quick Replies Still Work

```bash
curl -s -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "test_qr", "message": "რომელი პროტეინი ჯობია?"}' | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
qr = data.get('quick_replies', [])
print(f'Quick Replies: {len(qr)} found')
for i, r in enumerate(qr[:4], 1):
    print(f'  {i}. {r.get(\"title\", \"N/A\")}')"
```

### Test 3: Check Logs for TIP Injection

```bash
# Monitor backend logs for TIP injection messages
tail -f /path/to/backend.log | grep -E "(TIP tag already present|TIP tag was missing)"
```

### Expected Output:

```
✅ TIP tag present
TIP content: პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.

Quick Replies: 4 found
  1. ამ პროტეინის შეძენა
  2. 100₾-მდე ალტერნატივები
  3. როგორ მივიღო პროტეინი?
  4. whey vs isolate განსხვავება
```

---

## ⚠️ Important Notes

### DO NOT:
- ❌ Remove or disable `parse_quick_replies()` function - it's working correctly!
- ❌ Change function calling configuration - `automatic_function_calling` is fine
- ❌ Modify context caching TTL or structure
- ❌ Touch `parseProducts.ts` in frontend - TIP parsing works correctly

### DO:
- ✅ Add logging for TIP injection (helps debugging)
- ✅ Use `logger.info()` when injecting tips
- ✅ Keep injected tips contextual (use `generate_contextual_tip()`)
- ✅ Test with multiple query types (product search, educational, general)

---

## 📊 Success Criteria

Implementation is complete when:

1. ✅ `ensure_tip_tag()` function added to main.py
2. ✅ `generate_contextual_tip()` function added to main.py
3. ✅ `/chat` endpoint calls `ensure_tip_tag()` before returning response
4. ✅ System prompt optimized (duplicate removed, consolidated at end)
5. ✅ Test 1 passes: TIP tag present in response
6. ✅ Test 2 passes: Quick Replies still work (4 options)
7. ✅ Logs show "TIP tag was missing - injected" when needed
8. ✅ No regressions (function calling, product search, context caching still work)

---

## 🎯 How to Report Back

After implementation, please provide:

1. **Changes Made:**
   - List of modified files
   - Line numbers where changes were made
   - Brief description of each change

2. **Test Results:**
   - Output of Test 1 (TIP tag check)
   - Output of Test 2 (Quick Replies check)
   - Any warnings/errors from logs

3. **Code Snippets:**
   - Show the final `ensure_tip_tag()` function
   - Show where it's integrated in `/chat` endpoint
   - Show the new system prompt section

---

**Priority:** HIGH  
**Estimated Time:** 30-45 minutes  
**Risk Level:** LOW (no breaking changes, backward compatible)

Good luck! 🚀
