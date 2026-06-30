"""
LLM service for text generation.
Supports Ollama (local), OpenAI, and other LLM providers.
"""

from typing import Dict, Any, Optional, AsyncIterator
import os
import httpx
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    Service for LLM text generation.
    
    Supports multiple LLM providers with a unified interface.
    Default provider is Ollama (local).
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM service.
        
        Args:
            provider: LLM provider (ollama, openai, huggingface)
            model: Model name (defaults to settings)
        """
        self.provider = provider or getattr(settings, 'default_provider', 'ollama')
        
        if self.provider == 'ollama':
            self.model = model or settings.ollama_default_model
            self.base_url = settings.ollama_host
            self.client = httpx.AsyncClient(timeout=settings.ollama_timeout)
        elif self.provider == 'openai':
            from openai import AsyncOpenAI
            self.model = model or getattr(settings, 'llm_model', 'gpt-3.5-turbo')
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("No OpenAI API key found.")
                self.client = None
            else:
                self.client = AsyncOpenAI(api_key=api_key)
        else:
            logger.warning(f"Unsupported provider: {self.provider}")
            self.client = None
        
        logger.info(f"Initialized LLMService with provider: {self.provider}, model: {self.model}")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Generation temperature (0-2)
            max_tokens: Maximum tokens to generate
            system_message: Optional system message
            
        Returns:
            Dictionary with generated text and metadata
        """
        if self.provider == 'ollama':
            return await self._generate_ollama(prompt, temperature, max_tokens, system_message)
        elif self.provider == 'openai':
            return await self._generate_openai(prompt, temperature, max_tokens, system_message)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> Dict[str, Any]:
        """Generate using Ollama."""
        try:
            # Build prompt with system message
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            # Call Ollama API
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "options": {
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
            )
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get("response", "")
            
            # Debug logging
            if not generated_text:
                logger.warning(f"Ollama returned empty response. Full data: {data}")
            else:
                logger.info(f"Ollama generated {len(generated_text)} characters")
            
            result = {
                "text": generated_text,
                "model": self.model,
                "tokens_used": data.get("eval_count", 0),
                "finish_reason": "stop"
            }
            
            logger.info(f"Generated {result['tokens_used']} tokens with Ollama")
            return result
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> Dict[str, Any]:
        """Generate using OpenAI."""
        if not self.client:
            raise ValueError("OpenAI client not initialized.")
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            generated_text = response.choices[0].message.content
            
            result = {
                "text": generated_text,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason
            }
            
            logger.info(f"Generated {result['tokens_used']} tokens with OpenAI")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_message: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate text with streaming.
        
        Args:
            prompt: Input prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_message: Optional system message
            
        Yields:
            Response chunks
        """
        if self.provider == 'ollama':
            async for chunk in self._generate_stream_ollama(prompt, temperature, max_tokens, system_message):
                yield chunk
        elif self.provider == 'openai':
            async for chunk in self._generate_stream_openai(prompt, temperature, max_tokens, system_message):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_stream_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream using Ollama."""
        try:
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "options": {"num_predict": max_tokens},
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield {
                                "type": "token",
                                "content": data["response"]
                            }
                        if data.get("done"):
                            yield {"type": "done", "model": self.model}
                            break
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    async def _generate_stream_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream using OpenAI."""
        if not self.client:
            raise ValueError("OpenAI client not initialized.")
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "token",
                        "content": chunk.choices[0].delta.content
                    }
            
            yield {"type": "done", "model": self.model}
            
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def generate_with_history(
        self,
        message: str,
        history: list,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate response with conversation history.
        
        Args:
            message: Current user message
            history: List of previous messages
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated response
        """
        if not self.client:
            raise ValueError("LLM client not initialized. Please set OPENAI_API_KEY.")
        
        try:
            # Build messages from history
            messages = []
            
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            generated_text = response.choices[0].message.content
            
            result = {
                "text": generated_text,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason
            }
            
            return result
            
        except Exception as e:
            logger.error(f"LLM generation with history failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "available": self.client is not None
        }


# Made with Bob