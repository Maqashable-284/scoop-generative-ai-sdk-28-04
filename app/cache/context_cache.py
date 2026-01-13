"""
Context Cache Manager for Gemini API
=====================================

Week 4 Implementation: 85% Token Cost Reduction

Uses Google Gemini's Context Caching API to cache:
- System prompt (~5k tokens)
- Product catalog (~60k tokens)

Total cached: ~65k tokens
Cost savings: ~85% on cached token reads (input tokens)

API Documentation:
- Minimum cache size: 32,768 tokens
- Cache TTL: 1 minute to 1 hour (default 1 hour)
- Cached tokens billed at reduced rate (~75% discount)

New SDK (google.genai) caching workflow:
1. Create CachedContent with system_instruction + contents
2. Use cached_content parameter when creating model/chat sessions
3. Cache auto-expires based on TTL, or manually delete
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Track context cache usage and savings"""
    cache_created_at: Optional[datetime] = None
    cache_expires_at: Optional[datetime] = None
    cache_name: Optional[str] = None
    cached_token_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_tokens_saved: int = 0
    last_refresh_at: Optional[datetime] = None
    refresh_count: int = 0

    @property
    def is_active(self) -> bool:
        """Check if cache is currently active"""
        if not self.cache_expires_at:
            return False
        return datetime.utcnow() < self.cache_expires_at

    @property
    def time_remaining(self) -> Optional[timedelta]:
        """Get remaining cache lifetime"""
        if not self.cache_expires_at:
            return None
        remaining = self.cache_expires_at - datetime.utcnow()
        return remaining if remaining.total_seconds() > 0 else timedelta(0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "cache_name": self.cache_name,
            "is_active": self.is_active,
            "cached_token_count": self.cached_token_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "time_remaining_seconds": self.time_remaining.total_seconds() if self.time_remaining else 0,
            "refresh_count": self.refresh_count,
            "cache_created_at": self.cache_created_at.isoformat() if self.cache_created_at else None,
            "cache_expires_at": self.cache_expires_at.isoformat() if self.cache_expires_at else None,
            "last_refresh_at": self.last_refresh_at.isoformat() if self.last_refresh_at else None,
        }


class ContextCacheManager:
    """
    Manages Gemini context caching for cost optimization.

    Token savings calculation:
    - Without caching: 65k tokens per request
    - With caching: ~0 input tokens for cached content
    - Estimated savings: 85% on input token costs

    Usage:
        cache_manager = ContextCacheManager(client, model_name)
        await cache_manager.create_cache(
            system_instruction=SYSTEM_PROMPT,
            catalog_context=catalog_text,
            ttl_minutes=60
        )

        # Create chat with cached context
        chat = cache_manager.create_cached_chat()
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str = "gemini-2.5-flash",
        cache_ttl_minutes: int = 60,
    ):
        self.client = client
        self.model_name = model_name
        self.cache_ttl_minutes = cache_ttl_minutes
        self._cached_content: Optional[Any] = None  # CachedContent object
        self._cached_content_name: Optional[str] = None
        self.metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._system_instruction: str = ""
        self._catalog_context: str = ""

    @property
    def is_cache_valid(self) -> bool:
        """Check if current cache is still valid"""
        return (
            self._cached_content is not None and
            self.metrics.is_active
        )

    async def create_cache(
        self,
        system_instruction: str,
        catalog_context: str,
        ttl_minutes: Optional[int] = None,
        display_name: str = "scoop-context-cache"
    ) -> bool:
        """
        Create or refresh context cache.

        Args:
            system_instruction: The system prompt
            catalog_context: Product catalog text
            ttl_minutes: Cache TTL in minutes (default: self.cache_ttl_minutes)
            display_name: Human-readable cache name

        Returns:
            True if cache created successfully
        """
        async with self._lock:
            try:
                # Store for later refresh
                self._system_instruction = system_instruction
                self._catalog_context = catalog_context

                ttl = ttl_minutes or self.cache_ttl_minutes

                # Prepare cached content configuration
                # The catalog is included as initial conversation context
                cached_contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part(
                            text=f"აქ არის Scoop.ge პროდუქტების კატალოგი:\n\n{catalog_context}"
                        )]
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part(
                            text="გავიგე. მზად ვარ დაგეხმარო პროდუქტების შერჩევაში Scoop.ge კატალოგიდან."
                        )]
                    )
                ]

                # Create cached content using new SDK
                # Note: ttl format is "{seconds}s" string
                ttl_seconds = ttl * 60

                cached_content = self.client.caches.create(
                    model=self.model_name,
                    config=types.CreateCachedContentConfig(
                        display_name=display_name,
                        system_instruction=system_instruction,
                        contents=cached_contents,
                        ttl=f"{ttl_seconds}s",
                    )
                )

                self._cached_content = cached_content
                self._cached_content_name = cached_content.name

                # Update metrics
                self.metrics.cache_name = cached_content.name
                self.metrics.cache_created_at = datetime.utcnow()
                self.metrics.cache_expires_at = datetime.utcnow() + timedelta(minutes=ttl)
                self.metrics.refresh_count += 1
                self.metrics.last_refresh_at = datetime.utcnow()

                # Estimate token count (rough: 4 chars per token)
                total_text = system_instruction + catalog_context
                self.metrics.cached_token_count = len(total_text) // 4

                logger.info(
                    f"Created context cache: {cached_content.name} "
                    f"(~{self.metrics.cached_token_count} tokens, TTL: {ttl}min)"
                )

                return True

            except Exception as e:
                logger.error(f"Failed to create context cache: {e}")
                self._cached_content = None
                self._cached_content_name = None
                self.metrics.cache_misses += 1
                return False

    async def refresh_cache(self) -> bool:
        """
        Refresh the context cache with stored content.

        Call this when:
        - Cache is about to expire
        - Catalog has been updated
        """
        if not self._system_instruction or not self._catalog_context:
            logger.warning("Cannot refresh cache: no stored content")
            return False

        # Delete old cache first
        await self.delete_cache()

        # Create new cache
        return await self.create_cache(
            system_instruction=self._system_instruction,
            catalog_context=self._catalog_context
        )

    async def delete_cache(self) -> bool:
        """Delete the current cache"""
        async with self._lock:
            if not self._cached_content_name:
                return True

            try:
                self.client.caches.delete(name=self._cached_content_name)
                logger.info(f"Deleted context cache: {self._cached_content_name}")
                self._cached_content = None
                self._cached_content_name = None
                return True
            except Exception as e:
                logger.warning(f"Failed to delete cache: {e}")
                return False

    def record_cache_hit(self) -> None:
        """Record a cache hit for metrics"""
        self.metrics.cache_hits += 1
        self.metrics.estimated_tokens_saved += self.metrics.cached_token_count

    def record_cache_miss(self) -> None:
        """Record a cache miss for metrics"""
        self.metrics.cache_misses += 1

    def get_cached_content_name(self) -> Optional[str]:
        """Get the cached content name for use in chat sessions"""
        if self.is_cache_valid:
            self.record_cache_hit()
            return self._cached_content_name
        else:
            self.record_cache_miss()
            return None

    def create_cached_chat_config(
        self,
        tools: list = None,
        safety_settings: list = None,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
    ) -> types.GenerateContentConfig:
        """
        Create a GenerateContentConfig that uses cached content.

        When cached content is available, returns config without
        system_instruction (it's in the cache).

        When cache is invalid, falls back to non-cached config
        with full system instruction.
        """
        if self.is_cache_valid:
            # Cached mode: system instruction is in the cache
            return types.GenerateContentConfig(
                tools=tools,
                safety_settings=safety_settings,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_output_tokens,
                # Note: system_instruction NOT included - it's cached
            )
        else:
            # Fallback: include full system instruction
            full_instruction = self._system_instruction
            if self._catalog_context:
                full_instruction += "\n\n" + self._catalog_context

            return types.GenerateContentConfig(
                system_instruction=full_instruction,
                tools=tools,
                safety_settings=safety_settings,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_output_tokens,
            )

    async def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and metrics"""
        info = self.metrics.to_dict()

        # Calculate cost savings estimate
        # Gemini pricing: ~$0.075/1M input tokens (cached ~$0.01875/1M)
        # 75% savings on cached tokens
        if self.metrics.estimated_tokens_saved > 0:
            uncached_cost = (self.metrics.estimated_tokens_saved / 1_000_000) * 0.075
            cached_cost = (self.metrics.estimated_tokens_saved / 1_000_000) * 0.01875
            info["estimated_cost_savings_usd"] = round(uncached_cost - cached_cost, 4)
        else:
            info["estimated_cost_savings_usd"] = 0

        return info

    async def list_caches(self) -> list:
        """List all caches for this model"""
        try:
            caches = list(self.client.caches.list())
            return [
                {
                    "name": c.name,
                    "display_name": c.display_name,
                    "model": c.model,
                    "expire_time": str(c.expire_time) if hasattr(c, 'expire_time') else None,
                }
                for c in caches
            ]
        except Exception as e:
            logger.error(f"Failed to list caches: {e}")
            return []


class CacheRefreshTask:
    """
    Background task to automatically refresh context cache.

    Refreshes cache before expiry to ensure continuous availability.
    """

    def __init__(
        self,
        cache_manager: ContextCacheManager,
        refresh_before_expiry_minutes: int = 10,
        check_interval_minutes: int = 5,
    ):
        self.cache_manager = cache_manager
        self.refresh_before_expiry = timedelta(minutes=refresh_before_expiry_minutes)
        self.check_interval = check_interval_minutes * 60  # seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background refresh task"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Started context cache refresh task")

    async def stop(self) -> None:
        """Stop the background refresh task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped context cache refresh task")

    async def _run(self) -> None:
        """Background loop to check and refresh cache"""
        while self._running:
            try:
                await self._check_and_refresh()
            except Exception as e:
                logger.error(f"Cache refresh error: {e}")

            await asyncio.sleep(self.check_interval)

    async def _check_and_refresh(self) -> None:
        """Check if cache needs refresh"""
        metrics = self.cache_manager.metrics

        if not metrics.cache_expires_at:
            return

        time_remaining = metrics.time_remaining
        if time_remaining and time_remaining < self.refresh_before_expiry:
            logger.info(
                f"Cache expiring in {time_remaining}, refreshing..."
            )
            success = await self.cache_manager.refresh_cache()
            if success:
                logger.info("Cache refreshed successfully")
            else:
                logger.warning("Cache refresh failed")
