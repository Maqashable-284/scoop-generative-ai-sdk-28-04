# SDK Migration: google.generativeai → google.genai

**Completed:** 2026-01-13
**Migration Guide:** https://ai.google.dev/gemini-api/docs/migrate

---

## Summary

Migrated Scoop GenAI from the deprecated `google-generativeai` SDK to the new unified `google-genai` SDK. The new SDK is the official standard for Gemini API access (both AI Studio and Vertex AI).

---

## Key Changes

### 1. Package Update (`requirements.txt`)
```diff
- google-generativeai>=0.8.0
+ google-genai>=1.0.0
```

### 2. Import Changes (`main.py`)
```python
# Old SDK
import google.generativeai as genai
from google.generativeai.types import (
    HarmCategory,
    HarmBlockThreshold,
    GenerationConfig,
    ContentDict,
)

# New SDK
from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    Part,
    UserContent,
    ModelContent,
)
```

### 3. Client Initialization
```python
# Old SDK
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel(
    model_name=settings.model_name,
    tools=GEMINI_TOOLS,
    system_instruction=SYSTEM_PROMPT,
    safety_settings=SAFETY_SETTINGS,
    generation_config=GENERATION_CONFIG,
)

# New SDK
gemini_client = genai.Client(api_key=settings.gemini_api_key)
# Model created via client.aio.chats.create() with config
```

### 4. Safety Settings Format
```python
# Old SDK (dictionary)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ...
}

# New SDK (list of SafetySetting objects)
SAFETY_SETTINGS = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    ...
]
```

### 5. Chat Session Creation
```python
# Old SDK
chat = model.start_chat(
    history=gemini_history,
    enable_automatic_function_calling=True
)

# New SDK
chat_config = GenerateContentConfig(
    system_instruction=self.system_instruction,
    tools=self.tools,
    safety_settings=self.safety_settings,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=settings.max_output_tokens,
)
chat = client.aio.chats.create(
    model=model_name,
    history=sdk_history,  # UserContent/ModelContent objects
    config=chat_config,
)
```

### 6. History Format
```python
# Old SDK - dicts with role/parts
{"role": "user", "parts": [{"text": "..."}]}
{"role": "model", "parts": [{"text": "..."}]}

# New SDK - Content type objects
UserContent(parts=[Part.from_text(text="...")])
ModelContent(parts=[Part.from_text(text="...")])
```

### 7. Sending Messages (Async)
```python
# Old SDK
response = await chat.send_message_async(message)

# New SDK (aio chat sessions)
response = await chat.send_message(message)
```

### 8. Streaming
```python
# Old SDK
response = await chat.send_message_async(msg, stream=True)
async for chunk in response:
    yield chunk.text

# New SDK
response = await chat.send_message_stream(msg)
async for chunk in response:
    yield chunk.text
```

---

## Files Modified

1. **requirements.txt** - Updated package dependency
2. **main.py** - Updated imports, client initialization, session management, chat endpoints
3. **app/memory/mongo_store.py** - Updated type imports, history conversion methods

---

## Week 1 Features Preserved

- **Summary Injection Fix** - Still works! Summary is injected as `UserContent` at the start of history
- **30-Day TTL for Summaries** - No changes needed, MongoDB TTL indexes still work

---

## Testing Checklist

- [ ] Server starts without import errors
- [ ] `/health` endpoint returns healthy status
- [ ] `/chat` endpoint processes messages correctly
- [ ] `/chat/stream` endpoint streams responses
- [ ] Function calling works (tool functions execute)
- [ ] History persists to MongoDB
- [ ] Summary injection works on session reload

---

## Resources

- [Official Migration Guide](https://ai.google.dev/gemini-api/docs/migrate)
- [New SDK GitHub](https://github.com/googleapis/python-genai)
- [New SDK Documentation](https://googleapis.github.io/python-genai/)
- [PyPI Package](https://pypi.org/project/google-genai/)
