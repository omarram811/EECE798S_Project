from typing import Iterable, Dict, Any, List
from .settings import get_settings
import time

# OpenAI
from openai import OpenAI
# Gemini
import google.generativeai as genai

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
    


    def complete(self, messages):
        """
        Non-streaming call used for summarization.
        Returns the FULL summary as a single string.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            return resp.choices[0].message.content
        except Exception as e:
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
