"""
Scoop AI Evals - API Client
Calls the Scoop AI backend for evaluation
"""
import os
import httpx
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Response from Scoop AI backend"""
    text: str
    quick_replies: List[str]
    success: bool
    error: Optional[str] = None


class ScoopClient:
    """Client for Scoop AI backend API"""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize client with backend URL"""
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8080")
        self.timeout = 60.0  # 60 second timeout for AI responses
        
    async def chat(
        self,
        message: str,
        user_id: str = "eval_test",
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Send a chat message to Scoop AI
        
        Args:
            message: User message
            user_id: User identifier
            session_id: Session for multi-turn conversations
            
        Returns:
            ChatResponse with AI response text
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json={
                        "user_id": user_id,
                        "message": message,
                        "session_id": session_id
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                return ChatResponse(
                    text=data.get("response_text_geo", ""),
                    quick_replies=[qr.get("title", "") for qr in data.get("quick_replies", [])],
                    success=data.get("success", True)
                )
                
        except httpx.TimeoutException:
            logger.error(f"Timeout calling backend for: {message[:50]}")
            return ChatResponse(
                text="",
                quick_replies=[],
                success=False,
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"Backend call failed: {e}")
            return ChatResponse(
                text="",
                quick_replies=[],
                success=False,
                error=str(e)
            )
    
    def chat_sync(
        self,
        message: str,
        user_id: str = "eval_test",
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """Synchronous version of chat()"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat",
                    json={
                        "user_id": user_id,
                        "message": message,
                        "session_id": session_id
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                return ChatResponse(
                    text=data.get("response_text_geo", ""),
                    quick_replies=[qr.get("title", "") for qr in data.get("quick_replies", [])],
                    success=data.get("success", True)
                )
                
        except Exception as e:
            logger.error(f"Backend call failed: {e}")
            return ChatResponse(
                text="",
                quick_replies=[],
                success=False,
                error=str(e)
            )


def create_client() -> ScoopClient:
    """Factory function to create Scoop client"""
    return ScoopClient()
