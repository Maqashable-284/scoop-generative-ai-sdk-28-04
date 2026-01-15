"""
Scoop GenAI - Google Gemini SDK Implementation
==============================================

Production-ready FastAPI server with:
- Gemini 2.5 Flash integration
- MongoDB persistence
- SSE streaming
- Automatic function calling
- Comprehensive error handling

ANSWERS TO ALL QUESTIONS:

Question #4: Technical Implementation
-------------------------------------
- Async Support: send_message_async() is production-stable
- Streaming: Use send_message(..., stream=True) or generate_content_async(stream=True)
- Error Handling: See GEMINI_EXCEPTIONS below
- Auto Function Calling: Model retries with different params if tool returns error

Question #5: Production Considerations
--------------------------------------
- Cloud Run: Cold start ~2-3s (SDK import), use min_instances=1
- Observability: Google Cloud Trace integration shown below
- Rate Limits: 2000 RPM (paid), 15 RPM (free)
- Retry: Exponential backoff on 429/503

Question #6: Security
---------------------
- Prompt Injection: Use SafetySettings (shown below)
- PII: MongoDB encryption at rest recommended
"""
import os
import re
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import uuid

# Note: nest_asyncio removed due to uvloop conflict in Python 3.13
# Async loop warnings in user_tools.py are non-critical - app continues to work

# Prompt injection detection patterns (logging only, no blocking)
SUSPICIOUS_PATTERNS = [
    'ignore previous', 'forget instructions', 'disregard', 'override system',
    'system prompt', 'ignore above', 'new instructions', 'you are now'
]

# FastAPI
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# Google GenAI SDK (new unified SDK)
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

# Local imports
from config import settings, SYSTEM_PROMPT
from app.memory.mongo_store import (
    db_manager,
    ConversationStore,
    UserStore,
)
from app.catalog.loader import CatalogLoader
from app.tools.user_tools import (
    get_user_profile,
    update_user_profile,
    search_products,
    get_product_details,
    set_stores,
    GEMINI_TOOLS,
    _current_user_id,  # ContextVar for setting current user
)

# Week 4: Context Caching
from app.cache.context_cache import ContextCacheManager, CacheRefreshTask

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# =============================================================================
# LOGGING & OBSERVABILITY
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Question #5: Observability - Google Cloud Trace Integration
# Uncomment for Cloud Run deployment:
# from opentelemetry import trace
# from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
#
# trace.set_tracer_provider(TracerProvider())
# cloud_trace_exporter = CloudTraceSpanExporter()
# trace.get_tracer_provider().add_span_processor(
#     BatchSpanProcessor(cloud_trace_exporter)
# )
# tracer = trace.get_tracer(__name__)


# =============================================================================
# GEMINI CONFIGURATION (New SDK)
# =============================================================================

# Initialize Gemini client (new SDK uses client-based approach)
# API key can be passed directly or via GEMINI_API_KEY env var
gemini_client = genai.Client(api_key=settings.gemini_api_key)

# Question #6: Security - Safety Settings (new format: list of SafetySetting objects)
SAFETY_SETTINGS = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]

# Generation config - now part of GenerateContentConfig in new SDK
# Will be merged into chat config when creating sessions


# =============================================================================
# EXCEPTION HANDLING
# =============================================================================

"""
ANSWER TO QUESTION #4: Error Handling - Gemini SDK Exceptions

Common exceptions to catch:
1. google.api_core.exceptions.ResourceExhausted (429) - Rate limit
2. google.api_core.exceptions.ServiceUnavailable (503) - Service down
3. google.api_core.exceptions.InvalidArgument (400) - Bad request
4. google.generativeai.types.BlockedPromptException - Safety filter
5. google.generativeai.types.StopCandidateException - Generation stopped
"""

RETRY_EXCEPTIONS = (
    "ResourceExhausted",  # 429 - Rate limit
    "ServiceUnavailable",  # 503 - Temporary outage
    "DeadlineExceeded",  # Timeout
)


class GeminiTimeoutError(Exception):
    """Raised when Gemini API call times out"""
    pass


async def call_with_retry(
    func,
    *args,
    max_retries: int = 4,
    base_delay: float = 2.0,
    **kwargs
):
    """
    ANSWER TO QUESTION #5: Retry Logic for 429 errors

    Exponential backoff: 2s, 4s, 8s, 16s
    Now wrapped with timeout for Gemini 3 compatibility.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Wrap with timeout for Gemini 3 compatibility
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=settings.gemini_timeout_seconds
            )
        except asyncio.TimeoutError:
            raise GeminiTimeoutError(
                f"Gemini API timed out after {settings.gemini_timeout_seconds}s"
            )
        except Exception as e:
            error_type = type(e).__name__

            if error_type in RETRY_EXCEPTIONS:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {error_type}, "
                    f"waiting {delay}s"
                )
                await asyncio.sleep(delay)
                last_exception = e
            else:
                raise

    raise last_exception


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@dataclass
class Session:
    """Chat session with Gemini model (new SDK)"""
    user_id: str
    session_id: str
    chat: Any  # google.genai async chat session
    history: list = field(default_factory=list)  # Track history for MongoDB
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    def update_activity(self):
        self.last_activity = datetime.utcnow()


class SessionManager:
    """
    Manages chat sessions per user (New SDK version)

    ANSWER TO QUESTION #1: Multi-session Support

    - Each user gets persistent session
    - Sessions persist in MongoDB
    - In-memory cache for active sessions
    - TTL-based cleanup

    New SDK Migration Notes:
    - Uses client.aio.chats.create() for async chat sessions
    - History format uses UserContent/ModelContent types
    - Config includes system_instruction, tools, safety_settings

    Week 4 Update:
    - Supports context caching for 85% token savings
    - When cache_manager provided, uses cached_content instead of system_instruction
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        system_instruction: str,
        tools: list,
        conversation_store: ConversationStore,
        user_store: UserStore,
        safety_settings: list = None,
        ttl_seconds: int = 3600,
        cache_manager: Optional[ContextCacheManager] = None,  # Week 4
    ):
        self.client = client
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools
        self.safety_settings = safety_settings
        self.conversation_store = conversation_store
        self.user_store = user_store
        self.ttl = timedelta(seconds=ttl_seconds)
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.cache_manager = cache_manager  # Week 4: Context caching

    def _bson_to_sdk_history(self, bson_history: list) -> list:
        """Convert BSON history to new SDK Content format

        New SDK uses UserContent/ModelContent instead of dicts with 'role' key.
        """
        sdk_history = []
        for entry in bson_history:
            role = entry.get("role", "user")
            parts = []

            for part in entry.get("parts", []):
                if "text" in part:
                    parts.append(Part.from_text(text=part["text"]))
                elif "function_call" in part:
                    fc = part["function_call"]
                    parts.append(Part.from_function_call(
                        name=fc.get("name", ""),
                        args=fc.get("args", {})
                    ))
                elif "function_response" in part:
                    fr = part["function_response"]
                    parts.append(Part.from_function_response(
                        name=fr.get("name", ""),
                        response=fr.get("response", {})
                    ))

            if parts:
                if role == "user":
                    sdk_history.append(UserContent(parts=parts))
                else:  # model
                    sdk_history.append(ModelContent(parts=parts))

        return sdk_history

    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one

        Args:
            user_id: The user identifier
            session_id: Optional specific session ID for multi-conversation support
        """
        # Use session_id as cache key if provided, otherwise use user_id
        cache_key = session_id if session_id else user_id

        async with self._lock:
            # Check in-memory cache
            if cache_key in self._sessions:
                session = self._sessions[cache_key]
                session.update_activity()
                return session

            # Load from MongoDB
            history, loaded_session_id, summary = await self.conversation_store.load_history(
                user_id,
                session_id=session_id
            )

            # Use provided session_id or auto-generated one
            final_session_id = session_id if session_id else loaded_session_id

            # Convert BSON history to new SDK format
            sdk_history = self._bson_to_sdk_history(history)

            # WEEK 1 FIX: Inject summary as context if exists
            # Previously this was just logged but never actually added to history!
            if summary:
                # Prepend summary as first user message for context
                summary_content = UserContent(
                    parts=[Part.from_text(text=f"[წინა საუბრის კონტექსტი: {summary}]")]
                )
                sdk_history = [summary_content] + sdk_history
                logger.info(f"✅ Summary injected for {user_id}: {summary[:100]}...")

            # Week 4: Check if context caching is available
            use_cached_context = (
                self.cache_manager is not None and
                self.cache_manager.is_cache_valid
            )

            if use_cached_context:
                # Use cached context - 85% token savings
                cached_content_name = self.cache_manager.get_cached_content_name()
                logger.info(f"🚀 Using cached context: {cached_content_name}")

                # Config without system_instruction (it's in the cache)
                # FIX: Add automatic_function_calling with increased limit (default is 10)
                # Gemini 3 Flash Preview makes multiple function calls rapidly and exhausts the limit
                chat_config = GenerateContentConfig(
                    tools=self.tools,
                    safety_settings=self.safety_settings if settings.enable_safety_settings else None,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=settings.max_output_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=settings.max_function_calls
                    ),
                )

                # Create chat with cached content
                chat = self.client.aio.chats.create(
                    model=self.model_name,
                    history=sdk_history if sdk_history else None,
                    config=chat_config,
                )
            else:
                # Fallback: include full system instruction (no caching)
                if self.cache_manager:
                    self.cache_manager.record_cache_miss()
                    logger.warning("⚠️ Context cache unavailable, using full system instruction")

                # FIX: Add automatic_function_calling with increased limit (default is 10)
                # Gemini 3 Flash Preview makes multiple function calls rapidly and exhausts the limit
                chat_config = GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=self.tools,
                    safety_settings=self.safety_settings if settings.enable_safety_settings else None,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=settings.max_output_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=settings.max_function_calls
                    ),
                )

                # Create async chat session (new SDK)
                chat = self.client.aio.chats.create(
                    model=self.model_name,
                    history=sdk_history if sdk_history else None,
                    config=chat_config,
                )

            session = Session(
                user_id=user_id,
                session_id=final_session_id,
                chat=chat,
                history=history,  # Keep BSON history for MongoDB persistence
            )

            self._sessions[cache_key] = session
            return session

    async def save_session(self, session: Session) -> None:
        """Save session to MongoDB"""
        # Get user profile for metadata
        user = await self.user_store.get_user(session.user_id)

        metadata = {
            "language": "ka",
            "last_topic": None,
            "products_viewed": [],
            "products_recommended": []
        }

        # Get history from chat session (new SDK)
        try:
            chat_history = session.chat.get_history()
            # Convert SDK history back to BSON format for storage
            bson_history = self._sdk_history_to_bson(chat_history)
        except Exception as e:
            logger.warning(f"Could not get chat history: {e}, using tracked history")
            bson_history = session.history

        await self.conversation_store.save_history(
            user_id=session.user_id,
            session_id=session.session_id,
            history=bson_history,
            metadata=metadata
        )

    def _sdk_history_to_bson(self, sdk_history: list) -> list:
        """Convert SDK Content history back to BSON format for MongoDB storage"""
        bson_history = []
        for content in sdk_history:
            role = "model" if isinstance(content, ModelContent) else "user"
            # Handle both UserContent/ModelContent and generic Content
            if hasattr(content, 'role'):
                role = content.role

            entry = {"role": role, "parts": []}

            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    entry["parts"].append({"text": part.text})
                elif hasattr(part, 'function_call') and part.function_call:
                    entry["parts"].append({
                        "function_call": {
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args) if part.function_call.args else {}
                        }
                    })
                elif hasattr(part, 'function_response') and part.function_response:
                    entry["parts"].append({
                        "function_response": {
                            "name": part.function_response.name,
                            "response": part.function_response.response
                        }
                    })

            if entry["parts"]:
                bson_history.append(entry)

        return bson_history

    async def clear_session(self, user_id: str) -> bool:
        """Clear user session"""
        async with self._lock:
            if user_id in self._sessions:
                session = self._sessions.pop(user_id)
                await self.conversation_store.clear_session(session.session_id)
                return True
            return False

    async def cleanup_stale_sessions(self) -> int:
        """Remove expired sessions"""
        now = datetime.utcnow()
        expired = []

        async with self._lock:
            for user_id, session in self._sessions.items():
                if now - session.last_activity > self.ttl:
                    # Save before removing
                    await self.save_session(session)
                    expired.append(user_id)

            for user_id in expired:
                del self._sessions[user_id]

        logger.info(f"Cleaned up {len(expired)} stale sessions")
        return len(expired)


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

conversation_store = ConversationStore(
    max_messages=settings.max_history_messages,
    max_tokens=settings.max_history_tokens
)
user_store = UserStore()
catalog_loader: Optional[CatalogLoader] = None
session_manager: Optional[SessionManager] = None
context_cache_manager: Optional[ContextCacheManager] = None  # Week 4
cache_refresh_task: Optional[CacheRefreshTask] = None  # Week 4


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global catalog_loader, session_manager, context_cache_manager, cache_refresh_task

    # Startup
    logger.info("Starting Scoop GenAI server...")

    # Connect to MongoDB
    if settings.mongodb_uri:
        await db_manager.connect(
            settings.mongodb_uri,
            settings.mongodb_database
        )

    # Initialize catalog loader
    catalog_loader = CatalogLoader(
        db=db_manager.db if settings.mongodb_uri else None,
        cache_ttl_seconds=settings.catalog_cache_ttl_seconds
    )

    # Load catalog context - LEAN ARCHITECTURE: Use minimal summary
    catalog_context = await catalog_loader.get_catalog_context(lean=True)
    logger.info(f"Loaded lean catalog summary: ~{len(catalog_context)//4} tokens")

    # Prepare system instruction with catalog context
    full_system_instruction = SYSTEM_PROMPT + "\n\n" + catalog_context

    # Set up tool stores with sync client for avoiding async loop conflicts
    # FIX: Gemini function calling runs sync, so we need sync MongoDB client
    from pymongo import MongoClient
    sync_db = None
    if settings.mongodb_uri:
        sync_client = MongoClient(settings.mongodb_uri)
        sync_db = sync_client[settings.mongodb_database]
        logger.info("Initialized sync MongoDB client for tool functions")

        # WARM-UP: Execute a simple query to establish connection pool
        # This prevents "cold start" issues where first queries return 0 results
        try:
            warmup_count = sync_db.products.count_documents({"in_stock": True})
            logger.info(f"🔥 MongoDB warm-up complete: {warmup_count} in-stock products")
        except Exception as warmup_err:
            logger.warning(f"⚠️ MongoDB warm-up failed: {warmup_err}")

    set_stores(
        user_store=user_store,
        db=db_manager.db if settings.mongodb_uri else None,
        sync_db=sync_db
    )

    # Week 4: Initialize context caching for 85% token savings
    if settings.enable_context_caching:
        logger.info("🚀 Week 4: Initializing context caching...")
        context_cache_manager = ContextCacheManager(
            client=gemini_client,
            model_name=settings.model_name,
            cache_ttl_minutes=settings.context_cache_ttl_minutes,
        )

        # Create initial cache
        cache_success = await context_cache_manager.create_cache(
            system_instruction=SYSTEM_PROMPT,
            catalog_context=catalog_context,
            display_name="scoop-context-cache"
        )

        if cache_success:
            logger.info(
                f"✅ Context cache created successfully "
                f"(~{context_cache_manager.metrics.cached_token_count} tokens cached)"
            )

            # Start background cache refresh task
            cache_refresh_task = CacheRefreshTask(
                cache_manager=context_cache_manager,
                refresh_before_expiry_minutes=settings.cache_refresh_before_expiry_minutes,
                check_interval_minutes=settings.cache_check_interval_minutes,
            )
            await cache_refresh_task.start()
        else:
            logger.warning("⚠️ Context cache creation failed, running without caching")
            context_cache_manager = None
    else:
        logger.info("Context caching disabled via settings")
        context_cache_manager = None

    # Initialize session manager (New SDK)
    # ANSWER TO QUESTION #4: Automatic Function Calling Setup
    # New SDK passes tools via config when creating chat sessions
    session_manager = SessionManager(
        client=gemini_client,
        model_name=settings.model_name,
        system_instruction=full_system_instruction,
        tools=GEMINI_TOOLS,
        conversation_store=conversation_store,
        user_store=user_store,
        safety_settings=SAFETY_SETTINGS,
        ttl_seconds=settings.session_ttl_seconds,
        cache_manager=context_cache_manager,  # Week 4
    )

    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Stop cache refresh task
    if cache_refresh_task:
        await cache_refresh_task.stop()

    # Delete context cache
    if context_cache_manager:
        await context_cache_manager.delete_cache()

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Save all active sessions
    for user_id, session in session_manager._sessions.items():
        await session_manager.save_session(session)

    await db_manager.disconnect()


async def cleanup_loop():
    """Background task to clean up stale sessions"""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        if session_manager:
            await session_manager.cleanup_stale_sessions()


app = FastAPI(
    title="Scoop GenAI",
    description="Sports Nutrition AI Consultant (Gemini SDK)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Security: Validate configuration
cors_origins = settings.allowed_origins.split(",")
if "*" in cors_origins and not settings.debug:
    logger.warning(
        "⚠️ SECURITY: CORS allows all origins (*) in production mode! "
        "Set ALLOWED_ORIGINS env var to restrict access."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, max_length=128)  # For multi-conversation support

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid user_id format - only alphanumeric, underscore, and dash allowed')
        return v

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()


class ChatResponse(BaseModel):
    response_text_geo: str
    current_state: str = "CHAT"
    quick_replies: list = []
    picked_product_ids: list = []
    carousel: Optional[dict] = None
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# ADMIN AUTHENTICATION
# =============================================================================

async def verify_admin_token(x_admin_token: str = Header(None)) -> bool:
    """Verify admin token for protected endpoints"""
    if not settings.admin_token:
        # If no admin token configured, block access entirely
        raise HTTPException(status_code=403, detail="Admin access not configured")
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return True


# =============================================================================
# QUICK REPLIES PARSER
# =============================================================================

def parse_quick_replies(text: str) -> tuple[str, list]:
    """
    Extract quick replies from response text
    
    Primary format:
    [QUICK_REPLIES]
    Option 1
    Option 2
    [/QUICK_REPLIES]
    
    Fallback format (if primary not found):
    **შემდეგი ნაბიჯი:**
    - Option 1
    - Option 2
    """
    # First: Clean up any leaked function call XML/code
    # Gemini sometimes outputs function calls as text instead of proper API calls
    text = clean_leaked_function_calls(text)
    
    # Primary: Look for [QUICK_REPLIES] tag
    pattern = r'\[QUICK_REPLIES\](.*?)\[/QUICK_REPLIES\]'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        quick_text = match.group(1).strip()
        quick_replies = [
            {"title": line.strip(), "payload": line.strip()}
            for line in quick_text.split("\n")
            if line.strip()
        ]
        clean_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return clean_text, quick_replies
    
    # Fallback: Look for "შემდეგი ნაბიჯი:" section with bullet points
    fallback_pattern = r'\*?\*?შემდეგი ნაბიჯი:?\*?\*?\s*\n+((?:[-•*]\s*.+\n?)+)'
    fallback_match = re.search(fallback_pattern, text, re.IGNORECASE)
    
    if fallback_match:
        bullet_text = fallback_match.group(1).strip()
        quick_replies = []
        
        for line in bullet_text.split("\n"):
            # Remove bullet point prefix (-, •, *)
            clean_line = re.sub(r'^[-•*]\s*', '', line.strip())
            if clean_line:
                quick_replies.append({
                    "title": clean_line,
                    "payload": clean_line
                })
        
        if quick_replies:
            # Remove the "შემდეგი ნაბიჯი:" section from display text
            clean_text = re.sub(fallback_pattern, '', text, flags=re.IGNORECASE).strip()
            return clean_text, quick_replies
    
    # No quick replies found
    return text, []


def clean_leaked_function_calls(text: str) -> str:
    """
    Remove leaked function call XML/code from response text.
    Gemini sometimes outputs function calls as text instead of proper API calls.
    """
    # Safety check - if text is None, return empty string
    if text is None:
        return ""
    
    # Remove <execute_function> tags
    text = re.sub(r'<execute_function[^>]*/?>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove </execute_function> tags
    text = re.sub(r'</execute_function>', '', text, flags=re.IGNORECASE)
    
    # Remove any partial function call syntax
    text = re.sub(r'<\?xml[^>]*>', '', text)
    text = re.sub(r'<function_calls>.*?</function_calls>', '', text, flags=re.DOTALL)
    
    # Remove print(...) statements that look like function calls
    text = re.sub(r'print\([^)]+\)', '', text)
    
    # Remove any remaining XML-like function tags
    text = re.sub(r'</?[a-z_]+[^>]*>', '', text, flags=re.IGNORECASE)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# =============================================================================
# PRODUCT FORMAT INJECTION (Gemini 3 Flash Preview Fix)
# =============================================================================

def extract_search_products_results(response) -> list:
    """
    Extract products from search_products AFC results.

    With automatic function calling, the SDK handles function execution internally.
    We can access the history to find function call/response pairs.

    Returns list of product dicts with keys:
    - name, brand, price, servings, pricePerServing, description, buyLink
    """
    products = []

    try:
        # Access the chat history to find function responses
        # The response object has candidates with content parts
        for candidate in response.candidates:
            for part in candidate.content.parts:
                # Check for function response in part
                if hasattr(part, 'function_response') and part.function_response:
                    func_resp = part.function_response
                    if func_resp.name == 'search_products':
                        # Extract products from response
                        response_data = func_resp.response
                        if isinstance(response_data, dict) and 'products' in response_data:
                            products.extend(response_data['products'])
                        elif hasattr(response_data, 'get'):
                            prods = response_data.get('products', [])
                            if prods:
                                products.extend(prods)
    except Exception as e:
        logger.warning(f"Failed to extract search_products results from response: {e}")

    # Also try to get from chat history if available
    # This catches products from earlier in the conversation

    return products


def has_valid_product_markdown(text: str) -> bool:
    """
    Check if response already has properly formatted products.

    Looks for pattern:
    **Product Name**
    *Brand*
    **Price ₾** · Servings · Price/Serving

    Returns True if the format is already correct.
    """
    if not text:
        return False

    # Regex patterns from parseProducts.ts
    # Product name: **bold** text on its own line
    product_name_pattern = r'^\*\*[^*]+\*\*\s*$'
    # Brand: *italic* text on its own line
    brand_pattern = r'^\*[^*]+\*\s*$'
    # Price metadata: **XXX ₾** · XX პორცია · X.XX ₾/პორცია
    price_metadata_pattern = r'\*?\*?\d+(?:\.\d+)?\s*₾\*?\*?\s*·\s*\d+\s*პორცია'

    has_names = bool(re.search(product_name_pattern, text, re.MULTILINE))
    has_brands = bool(re.search(brand_pattern, text, re.MULTILINE))
    has_prices = bool(re.search(price_metadata_pattern, text))

    # Need at least name AND price to consider valid
    # Brand is optional but preferred
    is_valid = has_names and has_prices

    if is_valid:
        logger.info("✅ Response has valid product markdown format")

    return is_valid


def format_products_markdown(products: list, intro_text: str = "") -> str:
    """
    Generate properly formatted markdown for products.

    Format:
    **რეკომენდებული**
    **Product Name (Size)**
    *Brand*
    **Price ₾** · Servings პორცია · PricePerServing ₾/პორცია
    Description
    [ყიდვა →](buyLink)

    ---
    """
    if not products:
        return ""

    formatted = []
    ranks = ['რეკომენდებული', 'ალტერნატივა', 'ბიუჯეტური']

    for idx, product in enumerate(products[:3]):  # Max 3 products
        rank = ranks[min(idx, len(ranks)-1)]

        # Extract fields with fallbacks
        name = product.get('name') or product.get('name_ka') or 'უცნობი პროდუქტი'
        brand = product.get('brand') or ''
        price = product.get('price') or 0
        servings = product.get('servings') or 0

        # Calculate price per serving if not provided
        price_per_serving = 0
        if servings and price:
            price_per_serving = price / servings

        # Get URL - try different field names
        buy_link = (
            product.get('url') or
            product.get('product_url') or
            product.get('buyLink') or
            f"https://scoop.ge/search?q={name.replace(' ', '+')}"
        )

        # Generate description based on category if not provided
        description = product.get('description') or ''
        if not description:
            # Generate contextual description based on product name
            name_lower = name.lower()
            if 'whey' in name_lower or 'პროტეინ' in name_lower:
                description = 'მაღალი ხარისხის პროტეინი კუნთის ზრდისა და აღდგენისთვის.'
            elif 'creatine' in name_lower or 'კრეატინ' in name_lower:
                description = 'კრეატინი ძალისა და გამძლეობის გასაზრდელად.'
            elif 'bcaa' in name_lower or 'ამინო' in name_lower:
                description = 'ამინომჟავები კუნთის აღდგენისა და პროტეინის სინთეზისთვის.'
            elif 'pre' in name_lower or 'ენერგ' in name_lower:
                description = 'პრე-ვორკაუთი ენერგიისა და ფოკუსისთვის ვარჯიშის დროს.'
            elif 'gainer' in name_lower or 'მასა' in name_lower:
                description = 'გეინერი კალორიების და კუნთის მასის მოსაპოვებლად.'
            else:
                description = 'ხარისხიანი სპორტული დანამატი scoop.ge-დან.'

        # Format product markdown
        product_md = f"""**{rank}**
**{name}**
*{brand}*
**{price:.0f} ₾** · {servings} პორცია · {price_per_serving:.2f} ₾/პორცია
{description}
[ყიდვა →]({buy_link})

---"""

        formatted.append(product_md)

    return "\n\n".join(formatted)


def extract_products_from_text(text: str) -> list:
    """
    Fallback: Extract product info from unformatted text when Gemini
    doesn't call search_products but includes product info in response.

    Looks for patterns like:
    - "1. **Product Name** (XXX ლარი)"
    - "1. **Product Name** - XXX ლარი"
    - "1. **Product Name (Brand)** - XXX-YYY ლარი (ZZ პორცია)"

    Returns list of product dicts with extracted info.
    """
    products = []

    # Pattern: Numbered list with bold name, price can be in parens or after dash
    # Captures: 1=name, 2=price (first number found after name)
    # e.g., "1. **Critical Whey (Applied Nutrition)** - 253-260 ლარი (66 პორცია)"
    # e.g., "1. **Nitro Tech (Muscletech)** - 299 ლარი (60 პორცია)"
    # e.g., "3. **Mutant Whey** - 253 ლარი"
    patterns = [
        # === Numbered list patterns ===
        # Pattern: "1. **Name** (size) - **253 ლარი**" (price in bold after dash)
        r'\d+\.\s*\*\*([^*]+)\*\*[^*]*-\s*\*\*(\d+)(?:-\d+)?\s*(?:ლარი|₾)\*\*',
        # Pattern: "1. **Name** - ... ფასი: **253 ლარი**"
        r'\d+\.\s*\*\*([^*]+)\*\*[^*]*ფასი:\s*\*\*(\d+)(?:-\d+)?\s*(?:ლარი|₾)\*\*',
        # Pattern: price in bold anywhere after name on same entry
        r'\d+\.\s*\*\*([^*]+)\*\*.*?\*\*(\d+)(?:-\d+)?\s*(?:ლარი|₾)\*\*',
        # Pattern: non-bold price directly after dash
        r'\d+\.\s*\*\*([^*]+)\*\*\s*[-–—]\s*(\d+)(?:-\d+)?\s*(?:ლარი|₾)',
        # Pattern: non-bold price in parentheses
        r'\d+\.\s*\*\*([^*]+)\*\*\s*\((\d+(?:\.\d+)?)\s*(?:ლარი|₾)\)',

        # === Bullet point patterns (*, -, •) ===
        # Pattern: "* **Name** - XXX ლარი"
        r'[\*\-•]\s*\*\*([^*]+)\*\*\s*[-–—]\s*(\d+)(?:-\d+)?\s*(?:ლარი|₾)',
        # Pattern: "* **Name** (XXX ლარი)"
        r'[\*\-•]\s*\*\*([^*]+)\*\*\s*\((\d+(?:\.\d+)?)\s*(?:ლარი|₾)\)',
        # Pattern: "* **Name** - price in text" (looser match)
        r'[\*\-•]\s*\*\*([^*]+)\*\*[^0-9\n]*?(\d{2,3})(?:-\d+)?\s*(?:ლარი|₾)',
    ]

    seen_names = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            price = float(match.group(2))

            # Skip if we've already seen this product
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # Try to find servings info nearby
            servings = 0
            # Look for "XX პორცია" near the product name
            serving_pattern = rf'{re.escape(name)}[^0-9]*\((\d+)\s*პორცია\)'
            serving_match = re.search(serving_pattern, text, re.IGNORECASE)
            if serving_match:
                servings = int(serving_match.group(1))
            else:
                # Try simpler pattern
                simple_serving = re.search(rf'{re.escape(name[:20])}.*?(\d+)\s*პორცია', text)
                if simple_serving:
                    servings = int(simple_serving.group(1))

            # Extract brand - check for known brands in name
            brand = ""
            brands = ['Mutant', 'Applied Nutrition', 'Applied', 'Optimum', 'Muscletech',
                     'BioTech', 'Dymatize', 'MyProtein', 'BSN', 'Critical', 'Nitro']
            for b in brands:
                if b.lower() in name.lower():
                    brand = b
                    break

            # Clean up product name - remove brand in parentheses
            clean_name = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()

            products.append({
                'name': clean_name,
                'brand': brand,
                'price': price,
                'servings': servings,
                'url': f'https://scoop.ge/search?q={clean_name.replace(" ", "+")}'
            })

    if products:
        logger.info(f"📝 Extracted {len(products)} products from text: {[p['name'] for p in products[:3]]}")

    return products[:3]  # Max 3 products


def ensure_product_format(response_text: str, products_data: list) -> str:
    """
    Inject properly formatted products if Gemini didn't use markdown format.
    Similar to ensure_tip_tag().

    This function:
    1. Checks if response already has proper product markdown
    2. If not, tries to extract products from text or uses products_data
    3. Injects formatted markdown, preserving intro text
    """
    # Check if already formatted correctly
    if has_valid_product_markdown(response_text):
        logger.info("✅ Products already in correct markdown format")
        return response_text

    # Try to get products from function call results first
    final_products = products_data if products_data else []

    # If no products from function calls, try to extract from text
    if not final_products:
        final_products = extract_products_from_text(response_text)

    if not final_products:
        logger.info("📦 No products to format")
        return response_text

    # Generate formatted markdown
    formatted_products = format_products_markdown(final_products)

    if not formatted_products:
        return response_text

    logger.warning(f"⚠️ Product markdown format missing - injecting {len(final_products)} formatted products")

    # Strategy: Find intro text and insert products after it
    # Look for first paragraph/sentence before any list or numbered content

    # Split at first double newline or numbered list
    intro_patterns = [
        r'^(.+?)(?=\n\n|\n\d+\.|\n-\s)',  # Text before double newline or list
        r'^(.+?\.)(?=\s)',  # First sentence
    ]

    intro = ""
    rest = response_text

    for pattern in intro_patterns:
        match = re.match(pattern, response_text, re.DOTALL)
        if match:
            intro = match.group(1).strip()
            rest = response_text[len(match.group(0)):].strip()
            break

    # If we couldn't find a good split point, use first 200 chars as intro
    if not intro and len(response_text) > 200:
        # Find a sentence break
        period_pos = response_text.find('.', 50)
        if period_pos > 0 and period_pos < 300:
            intro = response_text[:period_pos + 1].strip()
            rest = response_text[period_pos + 1:].strip()
        else:
            intro = ""
            rest = response_text

    # Build final response
    if intro:
        injected = f"{intro}\n\n{formatted_products}"
    else:
        injected = formatted_products

    # Add any remaining content if it looks like a conclusion (not just more product text)
    # Skip content that looks like plain product descriptions
    if rest:
        # Check if 'rest' is just more unformatted product text
        has_product_indicators = any(x in rest.lower() for x in ['პროტეინ', 'კრეატინ', 'გეინერ', 'whey', 'bcaa'])
        has_list_format = rest.strip().startswith(('1.', '2.', '-', '•'))

        if not has_list_format and not (has_product_indicators and len(rest) > 100):
            injected = f"{injected}\n\n{rest}"

    logger.info(f"💉 Injected formatted products into response")

    return injected


# =============================================================================
# TIP TAG INJECTION (Gemini 3 Flash Preview Fix)
# =============================================================================

def generate_contextual_tip(text: str) -> str:
    """
    Generate contextual tip based on response content.

    Gemini 3 Flash Preview doesn't reliably generate [TIP] tags despite
    system prompt instructions. This function generates appropriate tips
    based on response content keywords.

    Args:
        text: The response text to analyze

    Returns:
        Contextual tip string (1-2 sentences in Georgian)
    """
    text_lower = text.lower()

    # Product-specific tips mapped to keywords
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

        # Fat burners / Weight
        'fat burn': 'fat burner-ების ეფექტურობისთვის აუცილებელია კალორიული დეფიციტი.',
        'წონის კლება': 'წონის კლებისთვის მთავარია კალორიული დეფიციტი - დანამატები დამხმარე საშუალებაა.',
        'წონა': 'წონის ცვლილებისთვის მთავარია კალორიების ბალანსი - დანამატები დამხმარე საშუალებაა.',
        'მასა': 'კუნთოვანი მასის მოსაპოვებლად საჭიროა კალორიული სუფიციტი და საკმარისი პროტეინი.',
        'კუნთ': 'კუნთის ზრდისთვის საჭიროა რეგულარული ვარჯიში, საკმარისი პროტეინი და დასვენება.',

        # Hydration
        'წყალი': 'დღეში მინიმუმ 2-3 ლიტრი წყალი მიიღეთ, განსაკუთრებით კრეატინის მიღებისას.',
    }

    # Find matching tip based on keywords
    for keyword, tip in contextual_tips.items():
        if keyword in text_lower:
            logger.info(f"💡 Generated contextual tip for keyword: '{keyword}'")
            return tip

    # Default fallback tip
    logger.info("💡 Using default generic tip (no keyword match)")
    return 'რეკომენდაციებთან დაკავშირებით კითხვების შემთხვევაში მოგვწერეთ support@scoop.ge'


def ensure_tip_tag(response_text: str) -> str:
    """
    Ensure response has [TIP] tag. If missing, inject contextual tip.

    This is a safety net for Gemini 3 Flash Preview which doesn't reliably
    generate [TIP] tags despite explicit system prompt instructions.
    The frontend (parseProducts.ts) expects [TIP]...[/TIP] tags to render
    the yellow "პრაქტიკული რჩევა" box.

    Args:
        response_text: The model's response text

    Returns:
        Response text with guaranteed [TIP] tag
    """
    # Safety check
    if not response_text:
        return response_text

    # Check if TIP tag already exists
    if '[TIP]' in response_text and '[/TIP]' in response_text:
        logger.info("✅ [TIP] tag already present in response")
        return response_text

    logger.warning("⚠️ [TIP] tag missing from Gemini response - injecting contextual tip")

    # Generate contextual tip based on response content
    tip = generate_contextual_tip(response_text)

    # Determine injection point
    # CRITICAL: Inject BEFORE [QUICK_REPLIES] if it exists
    if '[QUICK_REPLIES]' in response_text:
        # Split at QUICK_REPLIES and insert TIP before it
        parts = response_text.split('[QUICK_REPLIES]', 1)
        injected = f"{parts[0].rstrip()}\n\n[TIP]\n{tip}\n[/TIP]\n\n[QUICK_REPLIES]{parts[1]}"
        logger.info(f"💉 Injected TIP before [QUICK_REPLIES]: {tip[:60]}...")
    else:
        # Append TIP at the very end
        injected = f"{response_text.rstrip()}\n\n[TIP]\n{tip}\n[/TIP]"
        logger.info(f"💉 Appended TIP at end: {tip[:60]}...")

    return injected


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Scoop GenAI",
        "version": "1.0.0",
        "model": settings.model_name,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    db_status = await db_manager.ping() if settings.mongodb_uri else True

    # Week 4: Include cache status
    cache_status = "disabled"
    if context_cache_manager:
        cache_status = "active" if context_cache_manager.is_cache_valid else "expired"

    return {
        "status": "healthy" if db_status else "degraded",
        "database": "connected" if db_status else "disconnected",
        "model": settings.model_name,
        "context_cache": cache_status,  # Week 4
    }


@app.get("/cache/metrics")
async def cache_metrics(authorized: bool = Depends(verify_admin_token)):
    """
    Week 4: Get context cache metrics and cost savings.

    Returns:
        - Cache status (active/expired/disabled)
        - Cached token count
        - Cache hits/misses
        - Estimated token savings
        - Estimated cost savings in USD
    """
    if not context_cache_manager:
        return {
            "enabled": False,
            "message": "Context caching is disabled"
        }

    metrics = await context_cache_manager.get_cache_info()
    metrics["enabled"] = True

    # Add cache hit rate
    total_requests = metrics.get("cache_hits", 0) + metrics.get("cache_misses", 0)
    if total_requests > 0:
        metrics["cache_hit_rate"] = round(metrics["cache_hits"] / total_requests * 100, 2)
    else:
        metrics["cache_hit_rate"] = 0

    return metrics


@app.post("/cache/refresh")
async def refresh_cache(authorized: bool = Depends(verify_admin_token)):
    """
    Week 4: Manually refresh the context cache.

    Use when:
    - Product catalog has been updated
    - Cache needs to be regenerated
    """
    if not context_cache_manager:
        raise HTTPException(status_code=400, detail="Context caching is disabled")

    if not catalog_loader:
        raise HTTPException(status_code=500, detail="Catalog loader not initialized")

    # Get fresh catalog context
    catalog_context = await catalog_loader.get_catalog_context(force_refresh=True)

    # Refresh cache
    success = await context_cache_manager.create_cache(
        system_instruction=SYSTEM_PROMPT,
        catalog_context=catalog_context,
        display_name=f"scoop-context-cache-manual-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    )

    if success:
        return {
            "success": True,
            "message": "Cache refreshed successfully",
            "cached_tokens": context_cache_manager.metrics.cached_token_count,
            "expires_at": context_cache_manager.metrics.cache_expires_at.isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh cache")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """
    Main chat endpoint (New SDK)

    ANSWER TO QUESTION #4: Async Support
    - New SDK uses send_message() on async chat sessions (client.aio.chats)
    - Automatic function calling handles tool execution via config
    """
    try:
        # Get or create session (with optional session_id for multi-conversation support)
        session = await session_manager.get_or_create_session(
            chat_request.user_id,
            session_id=chat_request.session_id
        )

        # CRITICAL: Set user_id in context for all tool functions
        # This prevents AI from hallucinating wrong user_id when calling get_user_profile()
        _current_user_id.set(chat_request.user_id)
        logger.info(f"🔍 Set context: user_id={chat_request.user_id}")

        # Prompt injection detection (logging only, no blocking)
        message_lower = chat_request.message.lower()
        if any(pattern in message_lower for pattern in SUSPICIOUS_PATTERNS):
            logger.warning(f"Possible prompt injection detected: {chat_request.message[:100]}")

        # Send message with retry (New SDK)
        # New SDK: session.chat.send_message() is async for aio chat sessions
        response = await call_with_retry(
            session.chat.send_message,
            chat_request.message
        )

        # Extract text safely (Gemini 3 may return empty response)
        try:
            response_text = response.text
        except ValueError as e:
            # Gemini returned empty response - provide fallback
            logger.warning(f"Empty Gemini response for user {chat_request.user_id}: {e}")
            return ChatResponse(
                response_text_geo=(
                    "სამწუხაროდ, პასუხის გენერირება ვერ მოხერხდა. "
                    "გთხოვთ სცადოთ მარტივი კითხვა."
                ),
                quick_replies=[
                    {"title": "რა არის კრეატინი?", "payload": "რა არის კრეატინი?"},
                    {"title": "რომელი პროტეინი ჯობია?", "payload": "რომელი პროტეინი ჯობია?"},
                ],
                success=False,
                error="empty_response"
            )

        # CRITICAL FIX: Extract products from function calls and ensure format
        # Gemini 3 Flash Preview doesn't reliably format products as markdown
        # This must be called BEFORE ensure_tip_tag() to inject products first
        search_products_results = extract_search_products_results(response)
        if search_products_results:
            logger.info(f"📦 Extracted {len(search_products_results)} products from search_products calls")
        # Always try to ensure product format (fallback extracts from text)
        response_text = ensure_product_format(response_text, search_products_results)

        # CRITICAL FIX: Ensure [TIP] tag is present (inject if missing)
        # Gemini 3 Flash Preview doesn't reliably generate [TIP] tags
        # This must be called BEFORE parse_quick_replies() so TIP is in the text
        response_text = ensure_tip_tag(response_text)

        # Parse quick replies
        clean_text, quick_replies = parse_quick_replies(response_text)

        # Save session
        await session_manager.save_session(session)

        # Update user stats
        await user_store.increment_stats(chat_request.user_id)

        return ChatResponse(
            response_text_geo=clean_text,
            quick_replies=quick_replies,
            success=True
        )

    except GeminiTimeoutError as e:
        # Gemini 3 compatibility: Handle timeout gracefully
        logger.warning(f"Gemini timeout for user {chat_request.user_id}: {e}")
        return ChatResponse(
            response_text_geo=(
                "კითხვა ძალიან რთულია და დამუშავებას დიდი დრო სჭირდება. "
                "გთხოვთ დაყავით რამდენიმე მარტივ კითხვად. "
                "მაგალითად: ჯერ იკითხეთ პროტეინზე, შემდეგ კრეატინზე."
            ),
            quick_replies=[
                {"title": "რომელი პროტეინი ჯობია ვეგეტარიანელისთვის?", "payload": "რომელი პროტეინი ჯობია ვეგეტარიანელისთვის?"},
                {"title": "რა ღირს კრეატინი?", "payload": "რა ღირს კრეატინი?"},
            ],
            success=False,
            error="timeout"
        )

    except Exception as e:
        # Generate error ID for log correlation
        error_id = uuid.uuid4().hex[:8]
        logger.error(f"Chat error [{error_id}]: {type(e).__name__}: {e}", exc_info=True)
        logger.error(f"Full traceback for error [{error_id}]:", exc_info=True)

        # Check for safety block
        error_type = type(e).__name__
        if "Blocked" in error_type:
            return ChatResponse(
                response_text_geo="ბოდიში, ეს კითხვა ვერ დამუშავდა. სცადეთ სხვანაირად.",
                success=False,
                error="content_blocked"
            )

        return ChatResponse(
            response_text_geo="დაფიქსირდა შეცდომა. გთხოვთ სცადოთ თავიდან.",
            success=False,
            error=f"internal_error:{error_id}"  # Safe: only error ID, not details
        )


@app.post("/chat/stream")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_stream(request: Request, stream_request: ChatRequest):
    """
    SSE Streaming endpoint with Manual Function Calling.

    CRITICAL: google-genai SDK streaming is INCOMPATIBLE with automatic function calling.
    This endpoint creates a NEW chat session WITHOUT AFC to enable manual function handling.

    Flow:
    1. Create chat session without AFC (automatic_function_calling disabled)
    2. Send user message, check for function_call in response
    3. If function_call found, execute locally and send result back
    4. Repeat until we get final text response
    5. Stream text chunks to frontend progressively
    """
    import json

    async def generate():
        try:
            import time
            stream_start = time.time()
            logger.info(f"🚀 Stream: user={stream_request.user_id}, session={stream_request.session_id}")

            # CRITICAL: Create a NEW chat session WITHOUT AFC for streaming
            # The regular session_manager creates sessions WITH AFC which conflicts with manual handling

            # Load existing history for context continuity
            logger.info(f"📚 Stream: Loading history...")
            history, loaded_session_id, summary = await conversation_store.load_history(
                stream_request.user_id,
                session_id=stream_request.session_id
            )
            logger.info(f"📚 Stream: Loaded {len(history)} history entries")

            # Convert BSON history to SDK format
            sdk_history = []
            for entry in history:
                role = entry.get("role", "user")
                parts = []
                for part in entry.get("parts", []):
                    if "text" in part:
                        parts.append(Part.from_text(text=part["text"]))
                if parts:
                    if role == "user":
                        sdk_history.append(UserContent(parts=parts))
                    else:
                        sdk_history.append(ModelContent(parts=parts))

            # Inject summary if exists
            if summary:
                summary_content = UserContent(
                    parts=[Part.from_text(text=f"[წინა საუბრის კონტექსტი: {summary}]")]
                )
                sdk_history = [summary_content] + sdk_history

            logger.info(f"📚 Stream: Converted {len(sdk_history)} SDK history entries")

            # Create chat config WITHOUT automatic_function_calling
            # This allows us to handle function calls manually
            stream_chat_config = GenerateContentConfig(
                system_instruction=session_manager.system_instruction,
                tools=GEMINI_TOOLS,
                safety_settings=SAFETY_SETTINGS if settings.enable_safety_settings else None,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=settings.max_output_tokens,
                # NO automatic_function_calling - this is the key difference!
            )

            # Create new chat session for streaming (without AFC)
            # NOTE: gemini_client.aio.chats.create() is SYNC - it returns AsyncChat directly
            logger.info(f"🔧 Stream: Creating chat session with model={settings.model_name}")
            stream_chat = gemini_client.aio.chats.create(
                model=settings.model_name,
                history=sdk_history if sdk_history else None,
                config=stream_chat_config,
            )
            setup_time = time.time() - stream_start
            logger.info(f"✅ Stream: Chat session created (setup: {setup_time:.2f}s)")

            # Set user context for tool functions
            _current_user_id.set(stream_request.user_id)

            accumulated_text = ""
            search_products_results = []
            max_function_rounds = 3  # Reduced from 5 - 3 rounds is enough for most queries
            search_products_calls = 0  # Limit search_products to 1 call per request
            gemini_start = time.time()

            # Initial message
            current_message = stream_request.message

            for round_num in range(max_function_rounds):
                round_start = time.time()
                logger.info(f"🔄 Stream round {round_num + 1}")

                try:
                    response = await asyncio.wait_for(
                        stream_chat.send_message(current_message),
                        timeout=settings.gemini_timeout_seconds
                    )
                    round_time = time.time() - round_start
                    logger.info(f"⏱️ Round {round_num + 1} Gemini response: {round_time:.2f}s")
                except asyncio.TimeoutError:
                    logger.error(f"⏰ Stream: Timeout after {settings.gemini_timeout_seconds}s")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Timeout - გთხოვთ სცადოთ უფრო მარტივი კითხვა'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                except Exception as send_err:
                    logger.error(f"❌ Stream: send_message error: {send_err}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'content': f'API error: {str(send_err)[:100]}'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                # Check response for function calls or text
                if not hasattr(response, 'candidates') or not response.candidates:
                    logger.warning(f"⚠️ Stream: Empty response")
                    break

                candidate = response.candidates[0]
                if not hasattr(candidate, 'content') or not candidate.content.parts:
                    break

                # Process all parts
                function_calls_to_execute = []
                text_parts = []

                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls_to_execute.append(part.function_call)
                    elif hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                # Stream any text immediately
                for text_chunk in text_parts:
                    accumulated_text += text_chunk
                    yield f"data: {json.dumps({'type': 'text', 'content': text_chunk})}\n\n"

                # If no function calls, we're done
                if not function_calls_to_execute:
                    break

                # Execute function calls and prepare responses
                function_responses = []
                # OPTIMIZATION: Only process FIRST search_products call per batch
                # Gemini sometimes sends multiple search_products in one response
                search_calls_in_batch = [fc for fc in function_calls_to_execute if fc.name == "search_products"]
                if len(search_calls_in_batch) > 1:
                    logger.warning(f"⚠️ Stream: Gemini sent {len(search_calls_in_batch)} search_products calls - using only first")
                    # Keep only first search_products, remove others
                    first_search_found = False
                    filtered_calls = []
                    for fc in function_calls_to_execute:
                        if fc.name == "search_products":
                            if not first_search_found:
                                filtered_calls.append(fc)
                                first_search_found = True
                            # Skip subsequent search_products
                        else:
                            filtered_calls.append(fc)
                    function_calls_to_execute = filtered_calls

                logger.info(f"🔧 Stream: Processing {len(function_calls_to_execute)} function calls, search_products_calls={search_products_calls}")
                for fc in function_calls_to_execute:
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}
                    logger.info(f"🔧 Stream: {func_name}({func_args})")

                    # Execute the function
                    if func_name == "search_products":
                        # LIMIT: Only execute search_products once per request
                        # Increment counter BEFORE execution to block ALL subsequent calls
                        search_products_calls += 1
                        if search_products_calls > 1:
                            logger.warning(f"⚠️ Stream: SKIPPING search_products (call #{search_products_calls}) - returning {len(search_products_results)} cached products")
                            result = {"products": search_products_results, "count": len(search_products_results), "note": "Already searched - using cached results"}
                        else:
                            logger.info(f"✅ Stream: EXECUTING search_products (call #{search_products_calls})")
                            result = search_products(**func_args)
                            # Track for product formatting (even if 0 results)
                            if result.get("products"):
                                search_products_results.extend(result["products"])
                            logger.info(f"✅ Stream: Got {len(result.get('products', []))} products")
                    elif func_name == "get_user_profile":
                        result = get_user_profile()
                    elif func_name == "update_user_profile":
                        result = update_user_profile(**func_args)
                    elif func_name == "get_product_details":
                        result = get_product_details(**func_args)
                    else:
                        result = {"error": f"Unknown function: {func_name}"}

                    # Build function response part
                    function_responses.append(
                        Part.from_function_response(
                            name=func_name,
                            response=result
                        )
                    )

                # Send all function responses back as next message
                current_message = function_responses

            # Post-processing: Ensure product format
            if search_products_results and not has_valid_product_markdown(accumulated_text):
                formatted_products = format_products_markdown(search_products_results)
                if formatted_products:
                    yield f"data: {json.dumps({'type': 'products', 'content': formatted_products})}\n\n"
                    accumulated_text += "\n\n" + formatted_products

            # Post-processing: Ensure TIP tag
            if "[TIP]" not in accumulated_text and "[/TIP]" not in accumulated_text:
                tip = generate_contextual_tip(accumulated_text)
                if tip:
                    tip_block = f"\n\n[TIP]\n{tip}\n[/TIP]"
                    yield f"data: {json.dumps({'type': 'tip', 'content': tip_block})}\n\n"
                    accumulated_text += tip_block

            # Parse and send quick replies
            clean_text, quick_replies = parse_quick_replies(accumulated_text)
            if quick_replies:
                yield f"data: {json.dumps({'type': 'quick_replies', 'content': quick_replies})}\n\n"

            # Save conversation history from stream_chat
            try:
                stream_history = stream_chat.get_history()
                # Convert SDK history to BSON for MongoDB storage
                bson_history = []
                for content in stream_history:
                    role = "model" if isinstance(content, ModelContent) else "user"
                    if hasattr(content, 'role'):
                        role = content.role
                    entry = {"role": role, "parts": []}
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            entry["parts"].append({"text": part.text})
                        elif hasattr(part, 'function_call') and part.function_call:
                            entry["parts"].append({
                                "function_call": {
                                    "name": part.function_call.name,
                                    "args": dict(part.function_call.args) if part.function_call.args else {}
                                }
                            })
                        elif hasattr(part, 'function_response') and part.function_response:
                            entry["parts"].append({
                                "function_response": {
                                    "name": part.function_response.name,
                                    "response": part.function_response.response
                                }
                            })
                    if entry["parts"]:
                        bson_history.append(entry)

                # Save to MongoDB
                final_session_id = stream_request.session_id if stream_request.session_id else loaded_session_id
                await conversation_store.save_history(
                    user_id=stream_request.user_id,
                    session_id=final_session_id,
                    history=bson_history,
                    metadata={"language": "ka"}
                )
            except Exception as save_err:
                logger.warning(f"⚠️ Stream: History save failed: {save_err}")

            # Update user stats
            await user_store.increment_stats(stream_request.user_id)

            total_time = time.time() - stream_start
            logger.info(f"⏱️ Stream TOTAL: {total_time:.2f}s")

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"❌ Stream error [{error_id}]: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': f'internal_error:{error_id}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/session/clear")
async def clear_session(user_id: str, authorized: bool = Depends(verify_admin_token)):
    """Clear user session (admin only)"""
    success = await session_manager.clear_session(user_id)
    return {"success": success, "user_id": user_id}


@app.get("/sessions")
async def list_sessions(authorized: bool = Depends(verify_admin_token)):
    """List active sessions (admin only)"""
    sessions = []
    for user_id, session in session_manager._sessions.items():
        sessions.append({
            "user_id": user_id,
            "session_id": session.session_id,
            "message_count": len(session.chat.history),
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        })
    return {"sessions": sessions, "count": len(sessions)}


# =============================================================================
# HISTORY RETRIEVAL ENDPOINTS (Public - for frontend sidebar)
# =============================================================================

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """
    Get user's conversation sessions for sidebar display.
    
    Returns list of sessions with title and metadata.
    """
    # Validate user_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    
    sessions = await conversation_store.get_user_sessions(user_id, limit=20)
    return {"sessions": sessions}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get message history for a specific session.
    
    Returns formatted messages ready for frontend rendering.
    """
    # Validate session_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    messages = await conversation_store.get_session_history(session_id)
    return {"messages": messages, "session_id": session_id}


# =============================================================================
# PRIVACY CONTROLS (GDPR Compliance)
# =============================================================================

@app.delete("/user/{user_id}/data")
async def delete_user_data(user_id: str):
    """
    Delete all user data (GDPR Right to Erasure).
    
    Removes:
    - All conversation history
    - User profile and preferences
    - Active sessions from memory
    """
    # Validate user_id format
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    
    try:
        # Delete all conversations
        deleted_sessions = await conversation_store.clear_user_sessions(user_id)
        
        # Delete user profile
        deleted_user = await user_store.delete_user(user_id)
        
        # Clear from memory cache
        if session_manager:
            # CRITICAL FIX: Cache keys use session_id, not user_id
            # Must check Session.user_id property, not cache key pattern
            keys_to_remove = []
            for cache_key, session in list(session_manager._sessions.items()):
                if session.user_id == user_id:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                session_manager._sessions.pop(key, None)
            
            logger.info(f"Cleared {len(keys_to_remove)} cached sessions for user {user_id}")
        
        logger.info(f"Deleted data for user {user_id}: {deleted_sessions} sessions")
        
        return {
            "success": True,
            "deleted_sessions": deleted_sessions,
            "deleted_profile": deleted_user,
            "message": "ყველა მონაცემი წაიშალა"
        }
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user data")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Question #5: Cloud Run Compatibility
    # - Set PORT env var for Cloud Run
    # - Use 0.0.0.0 host
    # - Consider min_instances=1 to avoid cold starts

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=int(os.environ.get("PORT", settings.port)),
        reload=settings.debug
    )
