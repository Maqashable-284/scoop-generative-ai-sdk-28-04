"""
User Tools for Gemini Function Calling
=======================================

ANSWER TO QUESTION #4: Technical Implementation - Automatic Function Calling

Gemini SDK Function Calling:
- Define functions as Python callables
- Pass to GenerativeModel(tools=[...])
- Enable automatic calling: enable_automatic_function_calling=True
- Model calls function, gets result, generates response

Error Handling in Automatic Function Calling:
- If function raises exception: Model receives error message
- Model CAN retry with different parameters
- You can also return {"error": "message"} from function

Function Schema Requirements:
- Type hints required for parameters
- Docstring becomes function description
- Parameter descriptions from docstring

FIX: Using sync PyMongo client instead of async Motor to avoid event loop conflicts
FIX: Using ContextVar for user_id to prevent AI hallucination of wrong user_id
"""
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextvars import ContextVar
import logging

logger = logging.getLogger(__name__)

# Global context variable for current user_id (async-safe)
# This is set in the chat endpoint and automatically available to all tool functions
_current_user_id: ContextVar[Optional[str]] = ContextVar('current_user_id', default=None)

# Store references (set by main.py on startup)
_user_store = None
_product_service = None
_db = None
# Sync MongoDB client for tool functions (avoids async loop conflicts)
_sync_db = None


def proto_to_native(obj: Any) -> Any:
    """
    Recursively convert Gemini protobuf types to native Python types.
    Fixes RepeatedComposite serialization errors when saving to MongoDB.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, 'items'):  # dict-like (MapComposite)
        return {k: proto_to_native(v) for k, v in obj.items()}
    if hasattr(obj, '__iter__'):  # list-like (RepeatedComposite)
        return [proto_to_native(item) for item in obj]
    # Fallback: convert to string
    return str(obj)


def set_stores(user_store=None, product_service=None, db=None, sync_db=None):
    """Set store references for tools to use"""
    global _user_store, _product_service, _db, _sync_db
    _user_store = user_store
    _product_service = product_service
    _db = db
    _sync_db = sync_db


# =============================================================================
# USER PROFILE TOOLS
# =============================================================================

def get_user_profile() -> dict:
    """
    Retrieve the CURRENT user's profile including name, allergies, and preferences.
    
    This automatically gets the profile for whoever is currently chatting.
    No need to specify user_id - it's automatically determined from the active session.

    Call this when you need to:
    - Check user's allergies before recommending products
    - Get user's name for personalized responses
    - See user's fitness goals
    - Check purchase history

    Returns:
        dict with keys: error, name, allergies, goals, preferences, stats
    """
    # Get user_id from context (set by chat endpoint)
    user_id = _current_user_id.get(None)
    
    # DIAGNOSTIC: Log function call
    logger.info(f"🔍 get_user_profile CALLED (auto user_id={user_id})")
    
    if not user_id:
        logger.error("🔍 ERROR: No user_id in context! This should never happen.")
        return {
            "error": "No user context - cannot retrieve profile",
            "name": None,
            "allergies": [],
            "goals": [],
            "preferences": {},
            "stats": {"total_messages": 0}
        }
    
    # Use sync MongoDB client to avoid async loop conflicts
    if _sync_db is None:
        logger.warning("🔍 No DB connection - returning empty profile")
        return {
            "error": None,
            "name": None,
            "allergies": [],
            "goals": [],
            "preferences": {},
            "stats": {"total_messages": 0}
        }

    try:
        logger.info(f"🔍 Querying MongoDB for user_id={user_id}")
        user = _sync_db.users.find_one({"user_id": user_id})
        
        if not user:
            logger.info(f"🔍 No user found in DB for {user_id} - returning empty profile")
            return {
                "error": None,
                "name": None,
                "allergies": [],
                "goals": [],
                "preferences": {},
                "stats": {"total_messages": 0}
            }

        profile = user.get("profile", {})
        name = profile.get("name")
        allergies = proto_to_native(profile.get("allergies", []))
        
        logger.info(f"🔍 Found user in DB: name={name}, allergies={allergies}")
        
        result = {
            "error": None,
            "name": name,
            "allergies": allergies,
            "goals": proto_to_native(profile.get("goals", [])),
            "preferences": proto_to_native(profile.get("preferences", {})),
            "stats": proto_to_native(user.get("stats", {}))
        }
        
        logger.info(f"🔍 Returning profile: {result}")
        return result
    except Exception as e:
        logger.error(f"🔍 Error getting user profile: {e}")
        return {"error": str(e)}


def update_user_profile(
    name: Optional[str] = None,
    allergies: Optional[List[str]] = None,
    goals: Optional[List[str]] = None,
    fitness_level: Optional[str] = None
) -> dict:
    """
    Update the CURRENT user's profile information.
    
    This automatically updates the profile for whoever is currently chatting.

    Call this when user provides:
    - Their name ("მე ვარ გიორგი")
    - Allergy information ("ლაქტოზის აუტანლობა მაქვს")
    - Fitness goals ("მასის მომატება მინდა")
    - Experience level ("დამწყები ვარ")

    Args:
        name: User's name (optional)
        allergies: List of allergies like ["lactose", "gluten"] (optional)
        goals: List of goals like ["muscle_gain", "weight_loss"] (optional)
        fitness_level: One of "beginner", "intermediate", "advanced" (optional)

    Returns:
        dict with success status and updated profile
    """
    # Get user_id from context
    user_id = _current_user_id.get(None)
    
    if not user_id:
        logger.error("🔍 ERROR: No user_id in context for update_user_profile")
        return {"success": False, "error": "No user context"}
    
    logger.info(f"🔍 update_user_profile CALLED (auto user_id={user_id})")
    
    # Use sync MongoDB client to avoid async loop conflicts
    if _sync_db is None:
        return {"success": False, "error": "Database not connected"}

    try:
        profile_updates = {}
        if name is not None:
            profile_updates["name"] = proto_to_native(name)
        if allergies is not None:
            # Convert Gemini protobuf list to native Python list
            profile_updates["allergies"] = proto_to_native(allergies)
        if goals is not None:
            # Convert Gemini protobuf list to native Python list
            profile_updates["goals"] = proto_to_native(goals)
        if fitness_level is not None:
            profile_updates["fitness_level"] = proto_to_native(fitness_level)

        if not profile_updates:
            return {"success": False, "error": "No updates provided"}

        # Upsert user profile
        from datetime import datetime, timezone
        _sync_db.users.update_one(
            {"user_id": user_id},
            {
                "$set": {f"profile.{k}": v for k, v in profile_updates.items()},
                "$setOnInsert": {"user_id": user_id, "created_at": datetime.now(timezone.utc)}
            },
            upsert=True
        )

        return {
            "success": True,
            "message": "პროფილი განახლდა",
            "updated_fields": list(profile_updates.keys())
        }
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PRODUCT SEARCH TOOLS
# =============================================================================

def search_products(
    query: str = "",  # Default empty string for Gemini 3 sporadic bug
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = False
) -> dict:
    """
    Search for products in the Scoop.ge catalog.

    Call this when user asks about:
    - Specific products ("გინდა პროტეინი?")
    - Categories ("რა კრეატინები გაქვთ?")
    - Price-based queries ("100 ლარამდე რა არის?")
    - Brand searches ("Optimum Nutrition")

    Args:
        query: Search query in Georgian or English
        category: Filter by category (protein, creatine, bcaa, pre_workout, vitamin, gainer)
        max_price: Maximum price in Georgian Lari
        in_stock_only: Only show in-stock products (default True)

    Returns:
        dict with products list and count
    """
    # Defensive check: Gemini 3 sometimes sends empty query
    if not query or not query.strip():
        return {
            "error": "საძიებო ტექსტი საჭიროა",
            "products": [],
            "count": 0
        }

    # Use sync MongoDB client to avoid async loop conflicts
    if _sync_db is None:
        # Return mock data
        return {
            "products": [
                {
                    "id": "prod_001",
                    "name": "Gold Standard Whey",
                    "price": 159.99,
                    "brand": "Optimum Nutrition",
                    "in_stock": True
                }
            ],
            "count": 1,
            "query": query
        }

    try:
        products = []
        logger.info(f"🔎 search_products called with query='{query}'")

        # === STEP 0: Translate Georgian to English FIRST ===
        # This translation is used for BOTH $text and $regex searches
        query_map = {
            # Proteins
            "პროტეინ": ["protein", "whey", "isolate", "casein"],
            "ვეი": ["whey"],
            "იზოლატ": ["isolate"],
            "კაზეინ": ["casein"],
            "ცილა": ["protein", "whey"],
            # Creatine
            "კრეატინ": ["creatine"],
            # Vitamins & Minerals
            "ვიტამინ": ["vitamin"],
            "მინერალ": ["mineral", "magnesium", "zinc", "calcium"],
            "ომეგა": ["omega"],
            "მაგნიუმ": ["magnesium"],
            "თუთია": ["zinc"],
            # Amino Acids
            "ამინო": ["amino", "bcaa", "eaa"],
            "bcaa": ["bcaa"],
            "eaa": ["eaa"],
            # Pre-workout & Energy
            "პრევორკაუთ": ["pre-workout", "pre workout", "preworkout"],
            "პრე-ვორკაუტ": ["pre-workout", "pre workout", "preworkout"],
            "ენერგ": ["energy", "caffeine"],
            "კოფეინ": ["caffeine"],
            # Mass & Weight Gainers
            "გეინერ": ["gainer", "mass", "weight gainer", "serious mass"],
            "მასა": ["mass", "gainer", "weight"],
            "წონა": ["weight", "mass"],
            # Recovery
            "აღდგენ": ["recovery", "glutamine"],
            "გლუტამინ": ["glutamine"],
            # Fat Burners
            "ცხიმ": ["fat", "burn", "l-carnitine", "carnitine", "hydroxycut"],
            "კარნიტინ": ["carnitine", "l-carnitine"],
            "ცხიმისმწველ": ["fat burner", "hydroxycut", "carnitine"],
            # Collagen
            "კოლაგენ": ["collagen"],
            # Sugar-free / Low-carb
            "შაქარ": ["zero sugar", "sugar free", "low carb", "zero", "isolate"],
            "უშაქრო": ["zero sugar", "sugar free", "zero"],
            "ნახშირწყალ": ["low carb", "zero carb", "isolate"],
            # Brands
            "optimum": ["optimum", "optimum nutrition", "gold standard"],
            "მუსლტექ": ["muscletech"],
            "დაიმატაიზ": ["dymatize"],
            "მიუტანტ": ["mutant"],
        }

        # Check if query is Georgian and translate
        search_terms = [query.lower()]  # Default: use original query
        text_search_query = query  # For $text search
        for geo, eng_list in query_map.items():
            if geo in query.lower():
                search_terms = eng_list
                # For $text search, use the FIRST English term (most specific)
                text_search_query = eng_list[0]
                logger.info(f"🔄 Translated '{query}' → $text: '{text_search_query}', $regex terms: {eng_list}")
                break

        # === Phase 1: Try $text search first (indexed, ~10x faster) ===
        # Requires text index: db.products.createIndex({name: "text", name_ka: "text", brand: "text", category: "text"}, {default_language: "none"})
        # Skip $text entirely and go straight to $regex - more reliable
        # NOTE: $text search has cold start issues (returns 0 on first call)
        # $regex is slower but consistent
        products = []  # Skip $text, let $regex handle it

        # === Phase 2: Fallback to $regex if $text found nothing ===
        if not products:
            # Build MongoDB $or conditions for EACH search term
            or_conditions = []
            for term in search_terms:
                safe_term = re.escape(term)
                # English fields - case-insensitive
                or_conditions.append({"name": {"$regex": safe_term, "$options": "i"}})
                or_conditions.append({"brand": {"$regex": safe_term, "$options": "i"}})

            # Also search Georgian fields with original query (no translation needed)
            safe_query = re.escape(query)
            or_conditions.append({"name_ka": {"$regex": safe_query}})
            or_conditions.append({"category": {"$regex": safe_query}})

            mongo_query: Dict[str, Any] = {"$or": or_conditions}
            logger.info(f"📝 Regex search: {len(search_terms)} terms, {len(or_conditions)} conditions")

            # DISABLED: Category filter causes 0 results when Gemini passes wrong category format
            # The $or conditions already include category search via regex
            if category:
                logger.info(f"⚠️ Category filter ignored: '{category}' (using regex instead)")

            if max_price:
                mongo_query["price"] = {"$lte": max_price}

            if in_stock_only:
                mongo_query["in_stock"] = True

            # Sync query
            logger.info(f"🔍 MongoDB query: {mongo_query}")
            products = list(_sync_db.products.find(mongo_query).limit(10))
            logger.info(f"📝 $regex found {len(products)} products for '{query}'")
            if products:
                logger.info(f"📦 First product: name='{products[0].get('name')}', brand='{products[0].get('brand')}'")

        # Format results
        results = []
        for p in products:
            results.append({
                "id": p.get("id"),
                "name": p.get("name_ka", p.get("name")),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "servings": p.get("servings"),
                "in_stock": p.get("in_stock"),
                "url": p.get("product_url")
            })

        return {
            "products": results,
            "count": len(results),
            "query": query
        }
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return {"error": str(e), "products": [], "count": 0}


def get_product_details(product_id: str) -> dict:
    """
    Get detailed information about a specific product.

    Call this when user wants more details about a product
    they've seen or been recommended.

    Args:
        product_id: The product ID (e.g., "prod_001")

    Returns:
        dict with full product details including description
    """
    # Use sync MongoDB client to avoid async loop conflicts
    if _sync_db is None:
        return {
            "error": "Database not connected",
            "product": None
        }

    try:
        product = _sync_db.products.find_one({"id": product_id})

        if not product:
            return {
                "error": f"პროდუქტი '{product_id}' ვერ მოიძებნა",
                "product": None
            }

        return {
            "error": None,
            "product": {
                "id": product.get("id"),
                "name": product.get("name"),
                "name_ka": product.get("name_ka"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price": product.get("price"),
                "servings": product.get("servings"),
                "in_stock": product.get("in_stock"),
                "description": product.get("description"),
                "url": product.get("product_url")
            }
        }
    except Exception as e:
        logger.error(f"Error getting product details: {e}")
        return {"error": str(e), "product": None}


# =============================================================================
# TOOL LIST FOR GEMINI
# =============================================================================

# List of all tools to pass to Gemini
GEMINI_TOOLS = [
    get_user_profile,
    update_user_profile,
    search_products,
    get_product_details,
]


# =============================================================================
# ASYNC VERSIONS (For use with Gemini's async API)
# =============================================================================

async def async_get_user_profile(user_id: str) -> dict:
    """Async version of get_user_profile"""
    if _user_store is None:
        return {
            "error": None,
            "name": None,
            "allergies": [],
            "goals": [],
            "preferences": {},
            "stats": {"total_messages": 0}
        }

    user = await _user_store.get_user(user_id)
    if not user:
        return {
            "error": None,
            "name": None,
            "allergies": [],
            "goals": [],
            "preferences": {},
            "stats": {"total_messages": 0}
        }

    return {
        "error": None,
        "name": user.get("profile", {}).get("name"),
        "allergies": user.get("profile", {}).get("allergies", []),
        "goals": user.get("profile", {}).get("goals", []),
        "preferences": user.get("profile", {}).get("preferences", {}),
        "stats": user.get("stats", {})
    }


async def async_search_products(
    query: str = "",  # Default empty string for Gemini 3 sporadic bug
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = True
) -> dict:
    """Async version of search_products"""
    # Defensive check: Gemini 3 sometimes sends empty query
    if not query or not query.strip():
        return {
            "error": "საძიებო ტექსტი საჭიროა",
            "products": [],
            "count": 0
        }

    if _db is None:
        return {"products": [], "count": 0, "query": query}

    query_map = {
        "პროტეინ": "protein",
        "კრეატინ": "creatine",
        "ვიტამინ": "vitamin",
    }

    search_term = query.lower()
    for geo, eng in query_map.items():
        if geo in search_term:
            search_term = eng
            break

    # SECURITY: escape regex special chars
    safe_term = re.escape(search_term)
    safe_query = re.escape(query)
    mongo_query = {
        "$or": [
            {"name": {"$regex": safe_term, "$options": "i"}},
            {"name_ka": {"$regex": safe_query, "$options": "i"}},
            {"brand": {"$regex": safe_term, "$options": "i"}},
            {"category": {"$regex": safe_term, "$options": "i"}},
        ]
    }

    if in_stock_only:
        mongo_query["in_stock"] = True
    if max_price:
        mongo_query["price"] = {"$lte": max_price}
    if category:
        mongo_query["category"] = category

    cursor = _db.products.find(mongo_query).limit(10)
    products = await cursor.to_list(length=10)

    results = [{
        "id": p.get("id"),
        "name": p.get("name_ka", p.get("name")),
        "brand": p.get("brand"),
        "price": p.get("price"),
        "in_stock": p.get("in_stock"),
        "url": p.get("product_url")
    } for p in products]

    return {"products": results, "count": len(results), "query": query}
