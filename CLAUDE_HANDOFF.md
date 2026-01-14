# 🔴 URGENT: Gemini Prompt Compliance Issue - Claude Code Investigation

**Date:** 2026-01-14  
**Severity:** HIGH - Critical UX Impact  
**Status:** Needs Analysis & Fix  

---

## 🎯 Overview

Scoop AI backend (Gemini 3 Flash Preview) has **0% compliance** with critical system prompt formatting tags:
- ❌ `[TIP]` tags - completely missing from all responses
- ❌ `[QUICK_REPLIES]` tags - completely missing from all responses

**Impact:** Frontend UI cannot display practical tips section or follow-up action buttons, resulting in degraded user experience.

---

## 📊 Test Results

### Test Environment
- **Backend:** Local (http://localhost:8080)
- **Model:** `gemini-3-flash-preview`
- **SDK:** `google-genai>=1.0.0` (new unified SDK)
- **Context Cache:** Active
- **Database:** MongoDB connected

### Test Cases Run

#### Test 1: Product Recommendation
**Query:** "მაჩვენე whey პროტეინები"

**Expected Response Format:**
```
[Product recommendations with strict markdown format]

[TIP]
პროტეინი მიიღეთ ვარჯიშის შემდეგ 30 წუთში მაქსიმალური ეფექტისთვის.
[/TIP]

[QUICK_REPLIES]
ამ პროტეინის შეძენა
100₾-მდე ალტერნატივები
როგორ მივიღო პროტეინი?
whey vs isolate განსხვავება
[/QUICK_REPLIES]
```

**Actual Response:**
```
გვაქვს რამდენიმე პოპულარული Whey პროტეინი:

1. **Mutant 2270 Whey Protein** - 253 ლარი...
2. **Critical Whey Protein** - 253-260 ლარი...

რომელიმე ბრენდი ან არომატი ხომ არ გაინტერესებთ უფრო დეტალურად?
```

**Result:** ❌ Both tags completely missing

---

#### Test 2: Educational Question
**Query:** "როგორ მივიღო კრეატინი?"

**Expected:** Educational content + `[TIP]` + `[QUICK_REPLIES]`  
**Actual:** Long educational text, no tags  
**Result:** ❌ Both tags completely missing

---

## 🔍 TASK FOR CLAUDE CODE

### Phase 1: ANALYSIS ONLY (Do NOT modify anything yet!)

Please conduct a thorough analysis to understand **WHY** Gemini is ignoring the tags:

#### 1.1 Review System Prompt Configuration

**File to check:**
```
/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/prompts/system_prompt.py
```

**What to look for:**
- Lines 103-113: [TIP] tag instructions
- Lines 277-345: [QUICK_REPLIES] tag instructions (2+2 strategy)
- Are instructions clear enough?
- Are they prominent enough in the prompt?
- Is there conflicting guidance?

**Questions to answer:**
1. How explicit are the tag requirements in the prompt?
2. Are tags described as "MANDATORY" or "REQUIRED"?
3. Could Gemini be missing these instructions due to prompt length (382 lines)?

---

#### 1.2 Examine Model Configuration

**Files to check:**
```
/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/config.py
/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/main.py
```

**What to look for:**

**In config.py:**
- Current model name (line ~32)
- `max_output_tokens` setting
- `temperature` setting
- Any settings that might affect instruction following

**In main.py:**
- Lines 376-386: Chat config for cached context (automatic_function_calling config)
- Lines 402-413: Chat config for non-cached context
- `GenerateContentConfig` parameters
- Are there any filters or post-processing that might strip tags?

**Questions to answer:**
1. What model is currently active? (`gemini-3-flash-preview` or `gemini-2.5-flash`)
2. Is `automatic_function_calling` interfering with text formatting?
3. Are there any response filters between Gemini output and final API response?

---

#### 1.3 Check Response Processing Pipeline

**Files to check:**
```
/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/main.py
```

**Focus on:**
- Lines 753-807: `parse_quick_replies()` function
- Lines 809-825: `clean_leaked_function_calls()` function
- Lines 940-1050: `/chat` endpoint logic
- Any text processing that happens AFTER Gemini responds

**Questions to answer:**
1. Is `parse_quick_replies()` finding the tags but API still returns empty?
2. Does the function modify/strip the response text before returning?
3. Are there regex patterns that might accidentally remove tags?

---

#### 1.4 Investigate Context Caching Impact

**Files to check:**
```
/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/app/cache/context_cache.py
```

**What to look for:**
- How is system prompt cached?
- Does caching cause Gemini to "forget" tag instructions?
- Are tag instructions in cached or non-cached part?

**Questions to answer:**
1. Is the system prompt included in the cached content?
2. Does cache impact instruction following?
3. Test suggestion: Does disabling cache improve compliance?

---

#### 1.5 Research Google GenAI SDK Documentation

**Reference:** https://github.com/googleapis/python-genai

**What to research:**
1. **Structured Output:** Does the new SDK support `response_schema` for forcing JSON output?
   - Check docs for `GenerateContentConfig.response_schema`
   - Can we enforce required fields like `tip` and `quick_replies`?

2. **Function Calling Mode:** Is `automatic_function_calling` preventing proper text formatting?
   - Check if there's a `disable` option or alternative mode
   - Research if AFC bypasses final text generation

3. **Best Practices:** What does Google recommend for:
   - Ensuring instruction compliance
   - Formatting output with special tags
   - Combining function calling with structured text output

4. **Known Issues:** Search for:
   - Gemini 3 Flash Preview limitations
   - Instruction following issues in preview models
   - Differences between 2.5 Flash vs 3.0 Flash Preview

---

### Phase 2: ROOT CAUSE HYPOTHESIS (After Analysis)

Based on your analysis, please provide:

1. **Primary Root Cause:** Most likely reason for 0% tag compliance
2. **Contributing Factors:** Other issues that may worsen the problem
3. **Evidence:** Specific code/config that supports your hypothesis

Example hypothesis format:
```
PRIMARY: Gemini 3 Flash Preview has known instruction-following issues
EVIDENCE: README.md line 149 mentions "markdown formatting sometimes missing"
CONTRIBUTING: System prompt is 382 lines, tags instructions may be lost
CONTRIBUTING: automatic_function_calling may bypass text generation phase
```

---

### Phase 3: SOLUTION PROPOSALS (Do NOT implement yet!)

After your analysis, propose 3-5 ranked solutions:

#### Solution Template

**Solution N: [Name]**
- **Approach:** [What to change]
- **Effort:** [Low/Medium/High]
- **Risk:** [Low/Medium/High]
- **Expected Impact:** [% improvement estimate]
- **Files to modify:** [List]
- **Steps:** [Numbered list]

---

## 📚 Context Documents

### Current System Architecture

**Backend Stack:**
- FastAPI server
- Google GenAI SDK (`google-genai>=1.0.0`)
- MongoDB for persistence
- Context caching enabled (85% token savings)

**Relevant Files:**
1. `/prompts/system_prompt.py` - System instructions (382 lines)
2. `main.py` - FastAPI app + session management (~1240 lines)
3. `config.py` - Environment config + model settings
4. `/app/cache/context_cache.py` - Context caching logic
5. `/app/tools/user_tools.py` - Function calling tools

**Frontend Parsing:**
- `scoop-vercel/src/components/chat-response.tsx` - Parses `[TIP]` and `[QUICK_REPLIES]`
- Expects tags in specific format to render UI components

---

### Known Limitations (per README.md)

From `/scoop-genai-project-2026/README.md` lines 146-151:

```markdown
## ⚠️ Known Limitations (Gemini 3 Flash)

1. **Markdown Formatting**: Sometimes returns plain text instead of proper markdown
2. **[TIP] Tags**: Occasionally missing from responses
3. **[QUICK_REPLIES]**: Not always included (system prompt compliance issue)
4. **Verbosity**: Gemini 3 tends to ask clarifying questions vs immediate product recs
```

**Note:** "Occasionally missing" is actually "always missing" based on our tests!

---

### Previous Gemini 3 Fixes

From `/scoop-genai-project-2026/README.md` lines 18-38:

**Gemini 3 Flash Function Calling Fix (2026-01-14):**
- **Problem:** Hitting `max_remote_calls=10` limit
- **Solution:** Increased to 30 via `AutomaticFunctionCallingConfig`
- **Files Changed:** `config.py`, `main.py`, `app/cache/context_cache.py`

**Context:** We already fixed one Gemini 3 limitation. Tags might be another.

---

## 🎯 Success Criteria

Your analysis is complete when you can answer:

1. ✅ **Why** are tags missing? (root cause identified)
2. ✅ **Where** in the code should we intervene?
3. ✅ **What** are 3-5 viable solutions ranked by effort/impact?
4. ✅ **How** would each solution work? (step-by-step)

---

## ⚠️ IMPORTANT CONSTRAINTS

### DO NOT:
- ❌ Modify any code yet (analysis only!)
- ❌ Test changes on production
- ❌ Break existing Context Caching setup ($15/mo ops depends on it!)
- ❌ Remove function calling (critical for product search)

### DO:
- ✅ Read all referenced files thoroughly
- ✅ Research Google GenAI SDK docs (https://github.com/googleapis/python-genai)
- ✅ Check for similar issues in SDK GitHub Issues
- ✅ Consider backward compatibility
- ✅ Propose low-risk, high-impact solutions first

---

## 📎 Supporting Documents

### Full Test Results
**Location:** `/Users/maqashable/.gemini/antigravity/brain/419a75f7-a647-4661-83cd-5c54938de69d/prompt_compliance_analysis.md`

Contains:
- Complete test output
- Expected vs actual responses
- Detailed impact analysis
- Initial fix proposals (review these!)

### Test Script
**Location:** `/Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026/test_prompt_compliance.sh`

Run to reproduce issue:
```bash
cd /Users/maqashable/Desktop/Claude/06-01-26/scoop-ai/scoop-genai-project-2026
./test_prompt_compliance.sh
```

---

## 🚀 Deliverable

Please provide a comprehensive analysis document (markdown format) with:

### 1. Executive Summary
- Problem statement (1 paragraph)
- Root cause (1-2 sentences)
- Recommended solution (1 sentence)

### 2. Detailed Findings
- System prompt analysis
- Model configuration analysis
- Response processing analysis
- Context caching impact
- SDK research findings

### 3. Root Cause Determination
- Primary cause (with evidence)
- Contributing factors (with evidence)

### 4. Solution Proposals (Ranked)
- Solution 1: [Recommended] - [Name]
- Solution 2: [Alternative] - [Name]
- Solution 3: [Fallback] - [Name]
- (Optional) Solutions 4-5

Each solution must include:
- Approach description
- Effort estimation
- Risk assessment
- Expected impact (% tag compliance improvement)
- Files to modify
- Step-by-step implementation plan
- Pros/cons

### 5. Testing Plan
- How to verify fix works
- Regression test plan
- Rollback strategy if fix breaks something

---

## 📞 Questions?

If you need clarification on:
- System architecture
- Business requirements
- Testing approach
- Access to additional files

Please ask before proceeding with analysis.

---

## 🎯 Next Steps After Analysis

1. You provide analysis + solution proposals
2. User reviews and selects preferred solution
3. You implement the fix
4. We test and verify compliance improves
5. Deploy to production if successful

---

**Priority:** HIGH  
**Timeline:** Analysis due ASAP (today)  
**Assignee:** Claude Code  
**Reporter:** Gemini (Antigravity)

---

Good luck! 🚀
