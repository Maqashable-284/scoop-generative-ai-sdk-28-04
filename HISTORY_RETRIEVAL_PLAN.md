# 📋 History Retrieval Feature - Implementation Plan

**თარიღი:** 2026-01-14 (ხვალისთვის)
**პრიორიტეტი:** Medium
**დრო:** ~2-3 საათი

---

## 🎯 მიზანი

მომხმარებლებს შეუძლიათ:
1. გვერდის reload-ის შემდეგ ძველი საუბრის გახსნა
2. Sidebar-ში ყველა საუბრის ნახვა
3. კონკრეტული საუბრის გაგრძელება

---

## 📁 Backend Tasks (main.py)

### 1. GET /sessions - მომხმარებლის საუბრების სია
```python
@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    sessions = await conversation_store.get_user_sessions(user_id, limit=20)
    return {"sessions": sessions}
```

### 2. GET /session/{session_id}/history - კონკრეტული საუბრის ისტორია
```python
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    history = await conversation_store.get_history(session_id)
    return {"messages": history}
```

### 3. MongoDB Store Update (mongo_store.py)
- `get_user_sessions(user_id, limit)` method
- `get_history(session_id)` method - returns formatted messages

---

## 📁 Frontend Tasks (Chat.tsx)

### 1. useEffect - Load sessions on mount
```typescript
useEffect(() => {
  fetch(`${BACKEND_URL}/sessions/${userId}`)
    .then(res => res.json())
    .then(data => setConversations(data.sessions));
}, [userId]);
```

### 2. Sidebar onClick - Load session history
```typescript
const loadSession = async (sessionId: string) => {
  const res = await fetch(`${BACKEND_URL}/session/${sessionId}/history`);
  const data = await res.json();
  // Convert backend format to UI format
  setActiveConversation(data.messages);
};
```

### 3. LocalStorage fallback
- Save conversations to localStorage
- Use as backup when API unavailable

---

## ✅ Checklist

### Backend
- [ ] Add `get_user_sessions` to mongo_store.py
- [ ] Add `get_history` to mongo_store.py
- [ ] Add `/sessions/{user_id}` endpoint
- [ ] Add `/session/{id}/history` endpoint
- [ ] Test with curl

### Frontend
- [ ] Load sessions on mount
- [ ] Sidebar: load session on click
- [ ] LocalStorage backup
- [ ] Test full flow

---

## ⚠️ Security Notes

- `/sessions/{user_id}` should require auth in production
- Consider rate limiting on history endpoints
- Don't expose internal session IDs publicly

---

## 🧪 Test Commands

```bash
# List sessions
curl http://localhost:8080/sessions/widget_abc123

# Get history
curl http://localhost:8080/session/abc123/history
```
