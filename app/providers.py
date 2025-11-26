from typing import Iterable, Dict, Any, List
from .settings import get_settings
import time

# OpenAI
from openai import OpenAI
# Gemini
import google.generativeai as genai

# Models that are "reasoning" models and need special handling
# These models don't support temperature and may return content differently
OPENAI_REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini",
                           "gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini", "gpt-5"}

def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model that needs special handling."""
    model_lower = model.lower()
    # Check exact matches
    if model_lower in OPENAI_REASONING_MODELS:
        return True
    # Check prefixes for model families
    for prefix in ["o1", "o3", "o4", "gpt-5", "gpt-4.1"]:
        if model_lower.startswith(prefix):
            return True
    return False

class ProviderBase:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def stream_chat(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIProvider(ProviderBase):
    def __init__(self, model: str, embed_model: str | None = None, api_key: str | None = None):
        super().__init__(model, api_key)
        # Use provided API key, fall back to settings if not provided
        key = api_key or get_settings().OPENAI_API_KEY
        self.client = OpenAI(api_key=key)
        self.embed_model = embed_model or "text-embedding-3-small"

    def stream_chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for evt in resp:
            if evt.choices and evt.choices[0].delta and evt.choices[0].delta.content:
                yield evt.choices[0].delta.content

    def embed(self, texts: List[str]):
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]
    
    def _extract_content(self, resp) -> str:
        """Extract content from OpenAI response, handling different model response formats."""
        if not resp.choices:
            return ""
        
        message = resp.choices[0].message
        
        # Standard content field (gpt-4o-mini, gpt-4o, etc.)
        if message.content:
            return message.content
        
        # Some reasoning models may use different output formats
        # Check for reasoning_content or other fields
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            return message.reasoning_content
        
        # Check if there's a refusal
        if hasattr(message, 'refusal') and message.refusal:
            print(f"[OpenAIProvider] Model refused: {message.refusal}")
            return ""
        
        # Try to get any text from the message
        if hasattr(message, 'text') and message.text:
            return message.text
            
        return ""

    def complete(self, messages):
        """
        Non-streaming call used for summarization.
        Returns the FULL summary as a single string.
        
        Supports:
        - OpenAI: gpt-4.1-nano, gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-5, o1, o3, etc.
        """
        is_reasoning = _is_reasoning_model(self.model)
        
        # For reasoning models, we need to use a simpler, non-reasoning model for summarization
        # because they have restrictions and may not return content properly
        if is_reasoning:
            print(f"[OpenAIProvider.complete] Model {self.model} is a reasoning model, using gpt-4o-mini for summarization")
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Use a reliable model for summarization
                    messages=messages,
                    temperature=0.3,
                    max_tokens=300,
                )
                content = self._extract_content(resp)
                print(f"[OpenAIProvider.complete] Response: {content[:200] if content else 'None/Empty'}")
                return content or ""
            except Exception as e:
                print(f"[OpenAIProvider.complete ERROR with gpt-4o-mini] {e}")
                return ""
        
        # For standard models (gpt-4o-mini, gpt-4o, etc.)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=300,
            )
            content = self._extract_content(resp)
            print(f"[OpenAIProvider.complete] Response: {content[:200] if content else 'None/Empty'}")
            return content or ""
        except Exception as e:
            error_str = str(e)
            # Handle parameter compatibility issues
            if "max_completion_tokens" in error_str:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=300,
                    )
                    content = self._extract_content(resp)
                    return content or ""
                except Exception as e2:
                    print(f"[OpenAIProvider.complete ERROR] {e2}")
                    return ""
            elif "temperature" in error_str:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=300,
                    )
                    content = self._extract_content(resp)
                    return content or ""
                except Exception as e2:
                    print(f"[OpenAIProvider.complete ERROR] {e2}")
                    return ""
            print(f"[OpenAIProvider.complete ERROR] {e}")
            return ""

class GeminiProvider(ProviderBase):
    def __init__(self, model: str, embed_model: str | None = None, api_key: str | None = None):
        super().__init__(model, api_key)
        # Use provided API key, fall back to settings if not provided
        key = api_key or get_settings().GEMINI_API_KEY
        genai.configure(api_key=key)
        self.gen_model = genai.GenerativeModel(model)
        self.embed_model = embed_model or "text-embedding-004"

    def complete(self, messages):
        """
        Non-streaming call used for summarization.
        Returns the FULL summary as a single string.
        
        Supports:
        - Gemini: gemini-2.0-flash-lite, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro, etc.
        """
        try:
            # Convert OpenAI-style messages to single prompt w/ roles
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role.upper()}: {content}")
            prompt = "\n\n".join(parts)
            
            # Configure generation settings for summarization
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.3,
            )
            
            response = self.gen_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text from response - handle different response formats
            content = ""
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'parts') and response.parts:
                content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            
            print(f"[GeminiProvider.complete] Response: {content[:200] if content else 'None/Empty'}")
            return content or ""
        except Exception as e:
            print(f"[GeminiProvider.complete ERROR] {e}")
            # Try without generation config as fallback
            try:
                parts = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role.upper()}: {content}")
                prompt = "\n\n".join(parts)
                
                response = self.gen_model.generate_content(prompt)
                content = response.text if hasattr(response, 'text') else ""
                print(f"[GeminiProvider.complete fallback] Response: {content[:200] if content else 'None/Empty'}")
                return content or ""
            except Exception as e2:
                print(f"[GeminiProvider.complete fallback ERROR] {e2}")
                return ""

    def stream_chat(self, messages):
        # Convert OpenAI-style messages to single prompt w/ roles
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        prompt = "\n\n".join(parts)
        # stream = self.gen_model.generate_content(prompt, stream=True)
        # for chunk in stream:
        #     if getattr(chunk, "text", None):
        #         yield chunk.text
        #         time.sleep(0.2)
        stream = self.gen_model.generate_content(prompt, stream=True)
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                for char in text:
                    yield char  # yield character by character
                    time.sleep(0.000005)

    def embed(self, texts: List[str]):
        out = genai.embed_content(model=self.embed_model, content=texts)
        # API returns dict with 'embedding' or list type depending on batch support
        if isinstance(out, dict) and "embedding" in out:
            return [out["embedding"]]
        if isinstance(out, list):
            return [d["embedding"] for d in out]
        raise RuntimeError("Unexpected Gemini embedding response")
