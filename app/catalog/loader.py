"""
Catalog Loader with Context Caching
====================================

ANSWER TO QUESTION #3: Catalog Context Strategy

315 products ~= 60,000 tokens in context

Strategies implemented:
1. Context Caching (Google's Caching API) - Week 4
2. In-memory cache with TTL
3. Dynamic refresh on product changes
4. Fallback to tool-based search if context too large

Week 4 Update:
- Integrated with ContextCacheManager for 85% token savings
- Uses new google.genai SDK caching API
- Automatic cache refresh on catalog updates
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import logging
import hashlib

from motor.motor_asyncio import AsyncIOMotorDatabase

if TYPE_CHECKING:
    from app.cache.context_cache import ContextCacheManager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    data: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class CatalogLoader:
    """
    Loads product catalog for Gemini context

    ANSWER TO QUESTION #3: Context Caching

    Google's Context Caching API:
    - Available in Gemini 1.5+ models
    - Caches system instructions and static context
    - Reduces cost by ~75% for cached tokens
    - Minimum 32K tokens to cache
    - Cache TTL: 1 hour default, max 1 week

    Implementation:
    ```python
    import google.generativeai as genai

    # Create cached content
    cache = genai.caching.CachedContent.create(
        model="gemini-2.5-flash",
        display_name="scoop-catalog",
        system_instruction=SYSTEM_PROMPT,
        contents=[{"role": "user", "parts": [CATALOG_TEXT]}],
        ttl=timedelta(hours=1)
    )

    # Use cached model
    model = genai.GenerativeModel.from_cached_content(cache)
    ```
    """

    def __init__(
        self,
        db: Optional[AsyncIOMotorDatabase] = None,
        cache_ttl_seconds: int = 3600
    ):
        self.db = db
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: Optional[CacheEntry] = None
        self._catalog_hash: Optional[str] = None
        self._gemini_cache: Optional[Any] = None  # CachedContent object

    # -------------------------------------------------------------------------
    # Product Loading
    # -------------------------------------------------------------------------

    async def load_products(self) -> List[Dict[str, Any]]:
        """Load all products from MongoDB"""
        if self.db is None:
            return self._get_mock_products()

        cursor = self.db.products.find(
            {},
            {
                "id": 1,
                "name": 1,
                "name_ka": 1,
                "category": 1,
                "brand": 1,
                "price": 1,
                "servings": 1,
                "in_stock": 1,
                "product_url": 1,
                "description": 1,
                "_id": 0
            }
        )

        products = await cursor.to_list(length=500)

        if not products:
            logger.warning("No products in database, using mock data")
            return self._get_mock_products()

        return products

    def _get_mock_products(self) -> List[Dict[str, Any]]:
        """Mock products for testing"""
        return [
            {
                "id": "prod_001",
                "name": "Optimum Nutrition Gold Standard Whey",
                "name_ka": "გოლდ სტანდარტ ვეი პროტეინი",
                "category": "protein",
                "brand": "Optimum Nutrition",
                "price": 159.99,
                "servings": 74,
                "in_stock": True,
                "product_url": "https://scoop.ge/product/on-whey",
                "description": "მსოფლიოში #1 გაყიდვადი ვეი პროტეინი"
            },
            {
                "id": "prod_002",
                "name": "Creatine Monohydrate",
                "name_ka": "კრეატინ მონოჰიდრატი",
                "category": "creatine",
                "brand": "MyProtein",
                "price": 45.99,
                "servings": 200,
                "in_stock": True,
                "product_url": "https://scoop.ge/product/creatine",
                "description": "სუფთა კრეატინ მონოჰიდრატი 5გ პორციაში"
            },
            {
                "id": "prod_003",
                "name": "C4 Original Pre-Workout",
                "name_ka": "C4 პრე-ვორქაუთი",
                "category": "pre_workout",
                "brand": "Cellucor",
                "price": 89.99,
                "servings": 60,
                "in_stock": True,
                "product_url": "https://scoop.ge/product/c4",
                "description": "ენერგია და ფოკუსი ვარჯიშის წინ"
            },
            {
                "id": "prod_004",
                "name": "BCAA 2:1:1",
                "name_ka": "BCAA ამინომჟავები",
                "category": "bcaa",
                "brand": "Scivation Xtend",
                "price": 75.00,
                "servings": 90,
                "in_stock": True,
                "product_url": "https://scoop.ge/product/bcaa",
                "description": "ტოტალური აღდგენა ვარჯიშის შემდეგ"
            }
        ]

    # -------------------------------------------------------------------------
    # Context Formatting
    # -------------------------------------------------------------------------

    def format_catalog_context(self, products: List[Dict[str, Any]]) -> str:
        """
        Format products as context for Gemini

        Output format optimized for token efficiency:
        - Markdown tables for structured data
        - Georgian names for user-facing
        - Essential info only
        """
        if not products:
            return "პროდუქტების კატალოგი ცარიელია."

        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for p in products:
            cat = p.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(p)

        # Format output
        lines = ["# Scoop.ge პროდუქტების კატალოგი\n"]

        category_names = {
            "protein": "🥛 პროტეინი",
            "creatine": "⚡ კრეატინი",
            "bcaa": "💪 BCAA / ამინომჟავები",
            "pre_workout": "🔥 პრე-ვორქაუთი",
            "vitamin": "💊 ვიტამინები",
            "gainer": "📈 გეინერი",
            "fat_burner": "🔥 ცხიმის მწვავი",
            "other": "📦 სხვა"
        }

        for category, prods in sorted(by_category.items()):
            cat_name = category_names.get(category, f"📦 {category}")
            lines.append(f"\n## {cat_name}\n")

            for p in prods:
                stock = "✅" if p.get("in_stock") else "❌"
                lines.append(
                    f"- **{p.get('name_ka', p.get('name'))}** ({p.get('brand')})\n"
                    f"  - ფასი: {p.get('price', 0):.2f}₾ | პორციები: {p.get('servings', '-')} | {stock}\n"
                    f"  - ID: `{p.get('id')}` | [ყიდვა]({p.get('product_url', '#')})\n"
                )

        lines.append(f"\n---\nსულ: {len(products)} პროდუქტი | განახლდა: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    async def get_catalog_context(self, force_refresh: bool = False) -> str:
        """
        Get catalog context with caching

        ANSWER TO QUESTION #3: Dynamic vs Static

        Strategy:
        1. Check in-memory cache first
        2. Check if catalog hash changed (products updated)
        3. Refresh if TTL expired or products changed
        """
        # Check cache
        if not force_refresh and self._cache and not self._cache.is_expired:
            return self._cache.data

        # Load fresh products
        products = await self.load_products()

        # Check if products changed
        new_hash = self._compute_hash(products)
        if new_hash != self._catalog_hash:
            logger.info("Catalog changed, refreshing context")
            self._catalog_hash = new_hash

        # Format context
        context = self.format_catalog_context(products)

        # Update cache
        self._cache = CacheEntry(
            data=context,
            expires_at=datetime.utcnow() + self.cache_ttl
        )

        return context

    def _compute_hash(self, products: List[Dict[str, Any]]) -> str:
        """Compute hash of product data for change detection"""
        import json
        data = json.dumps(products, sort_keys=True, default=str)
        return hashlib.md5(data.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Week 4: Context Cache Integration
    # -------------------------------------------------------------------------

    async def initialize_context_cache(
        self,
        cache_manager: "ContextCacheManager",
        system_prompt: str
    ) -> bool:
        """
        Initialize context caching with the ContextCacheManager.

        Week 4 Implementation:
        - Creates cached content with system prompt + catalog
        - Achieves ~85% token cost reduction
        - Cache auto-refreshes via background task

        Args:
            cache_manager: The ContextCacheManager instance
            system_prompt: The system instruction to cache

        Returns:
            True if cache created successfully
        """
        try:
            # Get fresh catalog context
            catalog_context = await self.get_catalog_context(force_refresh=True)

            # Create cache via manager
            success = await cache_manager.create_cache(
                system_instruction=system_prompt,
                catalog_context=catalog_context,
                display_name=f"scoop-catalog-{self._catalog_hash[:8] if self._catalog_hash else 'initial'}"
            )

            if success:
                logger.info(
                    f"Context cache initialized with ~{len(catalog_context)//4} catalog tokens"
                )
            return success

        except Exception as e:
            logger.error(f"Failed to initialize context cache: {e}")
            return False

    async def refresh_context_cache(
        self,
        cache_manager: "ContextCacheManager",
        system_prompt: str
    ) -> bool:
        """
        Refresh context cache when catalog changes.

        Called when:
        - Product prices update
        - New products added
        - Stock status changes

        Returns:
            True if cache refreshed successfully
        """
        # Check if catalog actually changed
        products = await self.load_products()
        new_hash = self._compute_hash(products)

        if new_hash == self._catalog_hash:
            logger.debug("Catalog unchanged, skipping cache refresh")
            return True

        logger.info(f"Catalog changed (old: {self._catalog_hash}, new: {new_hash})")
        self._catalog_hash = new_hash

        # Force refresh local cache
        self._cache = None
        catalog_context = await self.get_catalog_context(force_refresh=True)

        # Update cache manager
        return await cache_manager.create_cache(
            system_instruction=system_prompt,
            catalog_context=catalog_context,
            display_name=f"scoop-catalog-{new_hash[:8]}"
        )

    def get_catalog_hash(self) -> Optional[str]:
        """Get current catalog hash for change detection"""
        return self._catalog_hash

    # -------------------------------------------------------------------------
    # Fallback Strategy
    # -------------------------------------------------------------------------

    def should_use_fallback(self, context: str) -> bool:
        """
        ANSWER TO QUESTION #3: Fallback if context too large

        If catalog > 100k tokens, switch to tool-based search
        instead of embedding in context
        """
        estimated_tokens = len(context) // 4
        return estimated_tokens > 100000

    async def get_context_or_tools(self) -> tuple[Optional[str], bool]:
        """
        Returns (context, use_tools)

        If context is manageable: (catalog_context, False)
        If too large: (None, True) - use tool-based search
        """
        context = await self.get_catalog_context()

        if self.should_use_fallback(context):
            logger.warning("Catalog too large, using tool-based search")
            return None, True

        return context, False
