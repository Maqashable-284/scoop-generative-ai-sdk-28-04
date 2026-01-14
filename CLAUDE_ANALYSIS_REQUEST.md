# 🔴 URGENT: Gemini Tag Compliance Analysis Request

Hi Claude! I need your help analyzing a critical issue with our Scoop AI backend.

## 🎯 Problem Statement

Our Gemini 3 Flash Preview model has **mixed compliance** with system prompt formatting:
- ❌ `[TIP]` tags - missing from response text (but frontend parses them client-side)
- ✅ `quick_replies` - **WORKS** via backend JSON field (not text tags!)

## 🔍 CRITICAL DISCOVERY: Frontend Architecture

**Important:** The frontend has TWO different mechanisms:

### 1. Quick Replies (JSON-based) ✅ WORKING!
**Frontend Code:** `Chat.tsx` line 297
```typescript
const backendQuickReplies: QuickReply[] = data.quick\_replies || [];
```

**How it works:**
- Backend sends `quick_replies` as a JSON array in API response
- Frontend reads from `data.quick_replies` directly
- No text parsing needed!

**Evidence:** Screenshot shows Quick Replies buttons working correctly  
**Conclusion:** `parse_quick_replies()` function in backend IS working!

---

### 2. TIP Tags (Text-based parsing) ❓ UNCLEAR STATUS
**Frontend Code:** `parseProducts.ts` line 107-112
```typescript
const tipPattern = /\[TIP\]([\s\S]*?)\[\/TIP\]/;
const tipMatch = markdown.match(tipPattern);
if (tipMatch) {
    tip = tipMatch[1].trim();
    markdown = markdown.replace(tipPattern, '').trim();
}
```

**How it works:**
- Frontend expects `[TIP]...[/TIP]` tags IN the markdown text
- Client-side regex extracts the tip
- Displays in yellow "პრაქტიკული რჩევა" box

**Unknown:** Are TIP tags present in response text but not visible in test output?

## 📊 Test Evidence

**IMPORTANT:** My initial test was against `localhost:8080` which showed:
- API returned: `"quick_replies": []` (empty array)
- Response text had no `[TIP]...[/TIP]` tags visible

**HOWEVER:** Screenshots from production frontend show Quick Replies **ARE working!**

**This means:**
1. ✅ The `parse_quick_replies()` backend function (main.py line 753-807) IS working correctly
2. ✅ Backend IS returning `quick_replies` JSON array to frontend
3. ❓ **Unknown:** Is my localhost backend old/different from production?
4. ❓ **Unknown:** Are `[TIP]` tags present in `response_text_geo` field?

**Need to investigate:**
- Is my local backend up-to-date with production code?
- Are `[TIP]` tags in the response text but just not being found by my test grep?

Full test results: `/Users/maqashable/.gemini/antigravity/brain/419a75f7-a647-4661-83cd-5c54938de69d/prompt_compliance_analysis.md`

## 🔍 What I Need From You

**Phase 1: ANALYSIS ONLY (don't modify any code yet!)**

**NEW PRIMARY OBJECTIVE:** Understand why my test showed empty `quick_replies` but production has them working.

**SECONDARY OBJECTIVE:** Verify if `[TIP]` tags are present in response text.

### Step 1: Review System Prompt
**File:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/prompts/system_prompt.py`

Check:
- Lines 103-113: Are `[TIP]` tag instructions clear enough?
- Lines 277-345: Are `[QUICK_REPLIES]` instructions clear enough?
- Is the prompt too long (382 lines)? Could Gemini be missing these instructions?
- Are tags marked as "MANDATORY" or "REQUIRED" strongly enough?

### Step 2: Examine Model Configuration
**Files:** 
- `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/config.py`
- `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/main.py`

Check:
- What model is active? (Should be `gemini-3-flash-preview`)
- Lines 376-386 in main.py: Does `automatic_function_calling` config interfere with text formatting?
- Lines 402-413 in main.py: Same check for non-cached path
- Are there any response filters that might strip tags?

### Step 3: Check Response Processing **[PRIORITY!]**
**File:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/main.py`

**Critical Questions:**
1. Lines 753-807: `parse_quick_replies()` - how does it extract tags from text?
   - Does it search for `[QUICK_REPLIES]...[/QUICK_REPLIES]` in response text?
   - Does it return a JSON array for the `quick_replies` field?
   - **Is this function being called and working?** (Screenshot proves it works in production!)

2. Lines 940-1050: `/chat` endpoint - where does `quick_replies` JSON come from?
   - Does it call `parse_quick_replies()` on the Gemini response?
   - Is the result stored in `response['quick_replies']`?

3. **TIP tags:** Are they in `response_text_geo` but my test didn't check the right place?

### Step 4: Research Google GenAI SDK
**Reference:** https://github.com/googleapis/python-genai

Research:
1. Does the new SDK support `response_schema` for structured output?
   - Can we force required fields like `tip` and `quick_replies`?
2. Does `automatic_function_calling` prevent proper text formatting?
3. Are there known issues with Gemini 3 Flash Preview instruction following?
4. What's the difference between `gemini-2.5-flash` vs `gemini-3-flash-preview` for instruction compliance?

### Step 5: Check Context Caching Impact
**File:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/app/cache/context_cache.py`

Check:
- Is the system prompt properly included in cached content?
- Could caching cause Gemini to "forget" tag instructions?

## 📤 Deliverable

Please provide a markdown analysis with:

### 1. Executive Summary (3-5 sentences)
- What's the root cause?
- What's your recommended solution?

### 2. Detailed Findings
For each file you reviewed:
- What did you find?
- Is this contributing to the problem?
- Evidence/code snippets

### 3. Root Cause Determination
- **Primary Cause:** [with evidence from code/config]
- **Contributing Factors:** [list with evidence]

### 4. Solution Proposals (Ranked by Priority)

For each solution provide:

**Solution 1: [Name]**
- **Approach:** What to change
- **Effort:** Low/Medium/High
- **Risk:** Low/Medium/High
- **Expected Impact:** % compliance improvement estimate
- **Files to Modify:** List
- **Implementation Steps:**
  1. Step 1
  2. Step 2
  3. ...
- **Pros:** 
- **Cons:**

Provide 3-5 ranked solutions.

### 5. Recommended Testing Plan
- How to verify the fix works
- Regression tests needed

## ⚠️ Important Constraints

**DO NOT:**
- ❌ Modify any code yet (analysis first!)
- ❌ Break Context Caching (we're saving 96% costs - $360→$15/mo)
- ❌ Remove function calling (critical for product search)

**DO:**
- ✅ Read all referenced files thoroughly
- ✅ Research the SDK documentation
- ✅ Propose low-risk, high-impact solutions first

## 📚 Additional Context

**Known Limitations (from README.md):**
- Gemini 3 Flash has "markdown formatting sometimes missing"
- We recently fixed function calling limit (10→30 calls)

**System Architecture:**
- Backend: FastAPI + Google GenAI SDK (`google-genai>=1.0.0`)
- Model: `gemini-3-flash-preview`
- Context caching: Active (85% token savings)
- Database: MongoDB

**Frontend Expectation:**
`scoop-vercel/src/components/chat-response.tsx` parses `[TIP]` and `[QUICK_REPLIES]` to render UI components.

## 🎯 Success Criteria

Analysis is complete when you can answer:
1. ✅ WHY are tags missing?
2. ✅ WHERE should we intervene?
3. ✅ WHAT are 3-5 viable solutions?
4. ✅ HOW would each solution work?

---

**Priority:** HIGH  
**Timeline:** ASAP  

Thank you! 🚀
