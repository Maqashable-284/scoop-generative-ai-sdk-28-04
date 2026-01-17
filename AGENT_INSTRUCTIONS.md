-----------------------------------------------
SCOOP AI - AGENT INSTRUCTIONS
-----------------------------------------------

შენ ხარ **Scoop AI Development Agent** - ქართული სპორტული კვების AI კონსულტანტის სისტემის მართვის ინჟინერი.

---

# 🎯 მთავარი თეზისები

## რა არის Scoop AI?
**ქართულენოვანი AI ჩატბოტი** scoop.ge-სთვის - სპორტული დანამატების ონლაინ მაღაზიის ჩეთბოტი.

## როგორ მუშაობს?
```
მომხმარებელი → Frontend (Next.js) → /chat/v2 API → Gemini 3 Flash → პასუხი
                                                        ↓
                                                   MongoDB Atlas (პროდუქტები + მეხსიერება)
```

**ტექნოლოგიური სტეკი:**
- **Backend:** Python 3.11+, FastAPI, Google GenAI SDK, MongoDB Motor
- **Frontend:** Next.js 16, React 19, TypeScript, Tailwind CSS
- **AI Model:** Gemini 3 Flash Preview + Automatic Function Calling
- **Database:** MongoDB Atlas

**ძირითადი ფუნქციონალი:**
1. **პროდუქტის ძებნა** - MongoDB text search ფასებით და სურათებით
2. **კონტექსტის შენახვა** - მახსოვს წინა საუბარი (3600 წამი TTL)
3. **სამედიცინო ლოგიკა** - ვითვალისწინებ ჯანმრთელობის პირობებს
4. **ეთიკური საზღვრები** - უსაფრთხო რჩევები, არა დიაგნოსტიკა

---

# 📦 რეპოზიტორიები & Production

| კომპონენტი | GitHub რეპო | Production URL |
|------------|-------------|----------------|
| **Backend** | [scoop-generative-ai-sdk-28-04](https://github.com/Maqashable-284/scoop-generative-ai-sdk-28-04) | [Cloud Run Console](https://console.cloud.google.com/run/detail/europe-west1/scoop-ai-sdk?project=gen-lang-client-0366926113) |
| **Frontend** | [scoop-vercel-fresh](https://github.com/Maqashable-284/scoop-vercel-fresh) | https://scoop-vercel-358331686110.europe-west1.run.app/ |

**🚨 DEPLOYMENT:** `main` branch-ში push → **ავტომატურად Cloud Run-ზე** (Cloud Build). ხელით deploy არ გვჭირდება!

---

# ⛔ მკაცრი აკრძალვები

### 1. არავითარი ხელით Deployment
- **აკრძალულია:** `gcloud run deploy` ან მსგავსი ბრძანებები
- **მიზეზი:** CI/CD ავტომატურად მუშაობს

### 2. არ შეეხო `.env` ფაილებს Git-ში
- არასდროს commit-ში `.env`
- არასდროს ლოგებში API Keys, Mongo URI

### 3. არ შეცვალო ფოლდერების სტრუქტურა
- არ გადაიტანო და არ შეუცვალო სახელი repos-ს

### 4. გამოიყენე Lean System Prompt
- ცვლილებები: `prompts/system_prompt_lean.py`
- `system_prompt.py` არის არქივი

---

# ✅ სავალდებულო ქცევები

### Deployment პროცედურა
```bash
git add . && git commit -m "description" && git push origin main
```
**ავტომატურად გადადის Cloud Run-ზე!**

### ტესტირება ცვლილებამდე
```bash
python3 -m evals.runner --set Simple
```

### Frontend სტილები
- მხოლოდ **Tailwind CSS**
- შეინარჩუნე `max-w-[1184px]` `Chat.tsx`-ში

---

# 🏗️ პროექტის სტრუქტურა

## Backend

```
├── main.py                      # 🔥 ENTRY POINT - FastAPI + /chat/v2
├── config.py                    # ⚙️  Settings, timeouts, model config
├── requirements.txt             # 📦 Dependencies
├── .env                         # 🔐 Local secrets
│
├── app/
│   ├── memory/mongo_store.py    # Conversation persistence
│   └── tools/tool_definitions.py # Gemini functions
│
├── prompts/
│   ├── system_prompt.py         # არქივი
│   └── system_prompt_lean.py    # ⭐ Production
│
└── evals/                       # 🧪 AI Evaluation (25 tests)
    ├── runner.py, judge.py, test_cases.yaml
```

## Frontend

```
├── package.json, next.config.ts
├── src/
│   ├── app/
│   │   ├── page.tsx             # 🔥 Main page
│   │   └── globals.css          # 🎨 Styles
│   └── components/
│       ├── Chat.tsx             # 🔥 Chat container
│       ├── chat-response.tsx    # Message rendering
│       └── thinking-steps-loader.tsx
```

---

# 📂 ფაილების ნავიგაცია

| ფუნქცია | ფაილი |
|---------|-------|
| API Logic | `main.py` |
| AI Personality | `prompts/system_prompt_lean.py` |
| Function Calling | `app/tools/tool_definitions.py` |
| Config | `config.py` |
| Chat UI | `src/components/Chat.tsx` |
| Message Render | `src/components/chat-response.tsx` |

---

# 🛠️ ხშირი ბრძანებები

### Backend
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
python3 -m evals.runner                    # Full evals
python3 -m evals.runner --test S1          # Single test
```

### Frontend
```bash
npm run dev
npm run build
```

### Health Check
```bash
curl http://localhost:8080/health
```

---

# 🐛 ხშირი პრობლემები

| პრობლემა | გამოსავალი |
|----------|------------|
| პროდუქტები არ ჩანს | გადატვირთე backend |
| Slow response (>15s) | `MAX_FUNCTION_CALLS=3` config.py |
| Layout shift | `max-w-[1184px]` Chat.tsx |
| CORS error | `ALLOWED_ORIGINS=*` .env |

---

# ✅ Checklist

- [ ] Backend :8080 ✓
- [ ] Frontend :3000 ✓
- [ ] `/health` healthy
- [ ] პროდუქტები ფასებით
- [ ] კონტექსტი 3+ turn
- [ ] Evals 80%+

---

**Version:** 5.0 | **Last Updated:** 2026-01-17
