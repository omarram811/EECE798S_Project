from typing import Iterable, Dict, Any, List, Tuple, Optional
from .settings import get_settings
import time

# OpenAI
from openai import OpenAI
# Gemini
import google.generativeai as genai


# ============ API KEY VALIDATION ============

def validate_openai_api_key(api_key: str, embed_model: str = "text-embedding-3-small") -> Tuple[bool, Optional[str]]:
    """
    Validate an OpenAI API key by making a minimal API call.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
        - (True, None) if the key is valid
        - (False, error_message) if invalid
    """
    try:
        client = OpenAI(api_key=api_key)
        # Use embeddings API as it's the cheapest to test
        # We embed a single short text
        resp = client.embeddings.create(
            model=embed_model,
            input=["test"]
        )
        # If we get here, the key is valid
        if resp.data and len(resp.data) > 0:
            return True, None
        return False, "Unexpected empty response from OpenAI API"
    except Exception as e:
        error_str = str(e).lower()
        
        # Parse specific error types
        if "invalid api key" in error_str or "incorrect api key" in error_str:
            return False, "Invalid API Key: The OpenAI API key you provided is not valid. Please check your key and try again."
        
        if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
            return False, "Authentication Error: Your OpenAI API key was rejected. Please verify it's correct and active."
        
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return False, "Rate Limit Exceeded: Your OpenAI API key has hit its rate limit. Please wait a moment and try again."
        
        if "quota" in error_str or "exceeded" in error_str or "insufficient" in error_str or "billing" in error_str:
            return False, "Quota Exceeded: Your OpenAI API key has exceeded its usage limit or has billing issues. Please check your OpenAI account."
        
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return False, f"Model Error: The embedding model '{embed_model}' is not available. Please select a different model."
        
        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return False, "Network Error: Unable to connect to OpenAI. Please check your internet connection and try again."
        
        # Generic error
        return False, f"OpenAI API Error: {str(e)[:200]}"


def validate_gemini_api_key(api_key: str, embed_model: str = "models/text-embedding-004") -> Tuple[bool, Optional[str]]:
    """
    Validate a Gemini API key by making a minimal API call.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
        - (True, None) if the key is valid
        - (False, error_message) if invalid
    """
    try:
        genai.configure(api_key=api_key)
        # Use embeddings API as it's the cheapest to test
        resp = genai.embed_content(
            model=embed_model,
            content=["test"]
        )
        # If we get here, the key is valid
        if resp and "embedding" in resp:
            return True, None
        return False, "Unexpected empty response from Gemini API"
    except Exception as e:
        error_str = str(e).lower()
        
        # Parse specific error types
        if "api key not valid" in error_str or "invalid api key" in error_str or "api_key_invalid" in error_str:
            return False, "Invalid API Key: The Gemini API key you provided is not valid. Please check your key and try again."
        
        if "401" in error_str or "unauthorized" in error_str or "permission denied" in error_str or "403" in error_str:
            return False, "Authentication Error: Your Gemini API key was rejected. Please verify it's correct and has the required permissions."
        
        if "429" in error_str or "resource exhausted" in error_str or "quota" in error_str:
            return False, "Quota Exceeded: Your Gemini API key has exceeded its usage limit. Please check your Google Cloud quota or try again later."
        
        if "rate limit" in error_str or "too many requests" in error_str:
            return False, "Rate Limit Exceeded: Your Gemini API key has hit its rate limit. Please wait a moment and try again."
        
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str or "not supported" in error_str):
            return False, f"Model Error: The embedding model '{embed_model}' is not available. Please select a different model."
        
        if "billing" in error_str or "payment" in error_str:
            return False, "Billing Error: There's a billing issue with your Google Cloud account. Please check your payment settings."
        
        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return False, "Network Error: Unable to connect to Gemini. Please check your internet connection and try again."
        
        # Generic error
        return False, f"Gemini API Error: {str(e)[:200]}"


def validate_api_key_with_provider(provider: str, api_key: str, embed_model: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an API key by making a test call to the provider.
    
    Args:
        provider: "openai" or "gemini"
        api_key: The API key to validate
        embed_model: The embedding model to use for the test
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if provider == "openai":
        return validate_openai_api_key(api_key, embed_model)
    elif provider == "gemini":
        return validate_gemini_api_key(api_key, embed_model)
    else:
        return False, f"Unknown provider: {provider}"


# ============ PROVIDER CLASSES ============


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
        key = api_key or get_settings().OPENAI_API_KEY
        self.client = OpenAI(api_key=key)
        self.embed_model = embed_model or "text-embedding-3-small"

    def stream_chat(self, messages):
        """Streaming chat completion."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            for evt in resp:
                if evt.choices and evt.choices[0].delta and evt.choices[0].delta.content:
                    yield evt.choices[0].delta.content
        except Exception as e:
            print(f"[OpenAIProvider.stream_chat ERROR] {e}")
            yield f"Error: {str(e)}"

    def embed(self, texts: List[str]):
        """Generate embeddings for texts."""
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def complete(self, messages):
        """Non-streaming call used for summarization."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[OpenAIProvider.complete ERROR] {e}")
            return ""


class GeminiProvider(ProviderBase):
    def __init__(self, model: str, embed_model: str | None = None, api_key: str | None = None):
        super().__init__(model, api_key)
        key = api_key or get_settings().GEMINI_API_KEY
        genai.configure(api_key=key)
        self.gen_model = genai.GenerativeModel(model)
        self.embed_model = embed_model or "models/text-embedding-004"

    def _convert_messages_to_prompt(self, messages) -> str:
        """Convert OpenAI-style messages to a single prompt string."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(parts)

    def complete(self, messages):
        """Non-streaming call used for summarization."""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            response = self.gen_model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text
            return ""
        except Exception as e:
            print(f"[GeminiProvider.complete ERROR] {e}")
            return ""

    def stream_chat(self, messages):
        """Streaming chat completion."""
        prompt = self._convert_messages_to_prompt(messages)
        try:
            stream = self.gen_model.generate_content(prompt, stream=True)
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    for char in text:
                        yield char
                        time.sleep(0.000005)
        except Exception as e:
            print(f"[GeminiProvider.stream_chat ERROR] {e}")
            yield f"Error: {str(e)}"

    def embed(self, texts: List[str]):
        """Generate embeddings for texts."""
        out = genai.embed_content(model=self.embed_model, content=texts)
        if isinstance(out, dict) and "embedding" in out:
            return [out["embedding"]]
        if isinstance(out, list):
            return [d["embedding"] for d in out]
        raise RuntimeError("Unexpected Gemini embedding response")
