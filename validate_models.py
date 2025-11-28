#!/usr/bin/env python3
"""
Model Combination Validator

Tests all supported model + embedding combinations for OpenAI and Gemini providers.
This script validates that each combination can successfully:
1. Send a chat completion request
2. Generate embeddings

Usage:
    python validate_models.py --openai-key YOUR_OPENAI_KEY --gemini-key YOUR_GEMINI_KEY
    python validate_models.py --openai-key YOUR_OPENAI_KEY  # Test only OpenAI
    python validate_models.py --gemini-key YOUR_GEMINI_KEY  # Test only Gemini

The script will output a summary table showing which combinations pass/fail.
"""

import argparse
import sys
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import providers from the app
sys.path.insert(0, '.')
from app.providers import OpenAIProvider, GeminiProvider


@dataclass
class TestResult:
    provider: str
    chat_model: str
    embed_model: str
    chat_success: bool
    embed_success: bool
    chat_error: Optional[str]
    embed_error: Optional[str]
    duration_ms: float


# Define supported models (same as in agents.py)
SUPPORTED_MODELS = {
    "openai": {
        "chat": ["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5"],
        "embed": ["text-embedding-3-small", "text-embedding-ada-002"],
    },
    "gemini": {
        "chat": ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "embed": ["models/text-embedding-004", "models/gemini-embedding-001"],
    },
}


def test_chat(provider, timeout: int = 30) -> Tuple[bool, Optional[str]]:
    """Test chat completion with a simple prompt."""
    try:
        # Use a simple test prompt
        messages = [{"role": "user", "content": "Reply with just the word 'OK'."}]
        response = provider.complete(messages)
        
        if response and isinstance(response, str) and len(response) > 0:
            return True, None
        else:
            return False, "Empty or invalid response"
    except Exception as e:
        return False, str(e)[:200]


def test_embed(provider, timeout: int = 30) -> Tuple[bool, Optional[str]]:
    """Test embedding generation with a simple text."""
    try:
        texts = ["This is a test sentence for embedding validation."]
        embeddings = provider.embed(texts)
        
        if embeddings and len(embeddings) > 0:
            # Check if we got valid embeddings (list of floats)
            first_emb = embeddings[0]
            if isinstance(first_emb, (list, tuple)) and len(first_emb) > 0:
                return True, None
            else:
                return False, f"Invalid embedding format: {type(first_emb)}"
        else:
            return False, "Empty embeddings returned"
    except Exception as e:
        return False, str(e)[:200]


def test_combination(provider_name: str, chat_model: str, embed_model: str, api_key: str) -> TestResult:
    """Test a single chat + embed model combination."""
    start_time = time.time()
    
    # Create provider instance
    if provider_name == "openai":
        provider = OpenAIProvider(chat_model, embed_model, api_key)
    else:
        provider = GeminiProvider(chat_model, embed_model, api_key)
    
    # Test chat
    chat_success, chat_error = test_chat(provider)
    
    # Test embedding
    embed_success, embed_error = test_embed(provider)
    
    duration_ms = (time.time() - start_time) * 1000
    
    return TestResult(
        provider=provider_name,
        chat_model=chat_model,
        embed_model=embed_model,
        chat_success=chat_success,
        embed_success=embed_success,
        chat_error=chat_error,
        embed_error=embed_error,
        duration_ms=duration_ms
    )


def print_results_table(results: List[TestResult]):
    """Print a formatted results table."""
    print("\n" + "=" * 120)
    print("MODEL VALIDATION RESULTS")
    print("=" * 120)
    
    # Group by provider
    for provider_name in ["openai", "gemini"]:
        provider_results = [r for r in results if r.provider == provider_name]
        if not provider_results:
            continue
        
        print(f"\n{'─' * 60}")
        print(f"Provider: {provider_name.upper()}")
        print(f"{'─' * 60}")
        
        # Header
        print(f"{'Chat Model':<25} {'Embed Model':<30} {'Chat':<8} {'Embed':<8} {'Time (ms)':<10}")
        print(f"{'─' * 25} {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 10}")
        
        for r in provider_results:
            chat_status = "✅ PASS" if r.chat_success else "❌ FAIL"
            embed_status = "✅ PASS" if r.embed_success else "❌ FAIL"
            
            print(f"{r.chat_model:<25} {r.embed_model:<30} {chat_status:<8} {embed_status:<8} {r.duration_ms:>8.0f}")
            
            # Print errors if any
            if r.chat_error:
                print(f"  └─ Chat Error: {r.chat_error[:80]}...")
            if r.embed_error:
                print(f"  └─ Embed Error: {r.embed_error[:80]}...")
    
    # Summary
    print("\n" + "=" * 120)
    total = len(results)
    passed = sum(1 for r in results if r.chat_success and r.embed_success)
    failed = total - passed
    
    print(f"SUMMARY: {passed}/{total} combinations passed ({failed} failed)")
    
    if failed > 0:
        print("\nFailed combinations:")
        for r in results:
            if not r.chat_success or not r.embed_success:
                issues = []
                if not r.chat_success:
                    issues.append("chat")
                if not r.embed_success:
                    issues.append("embed")
                print(f"  - {r.provider}/{r.chat_model} + {r.embed_model}: {', '.join(issues)} failed")
    
    print("=" * 120)


def run_validation(openai_key: Optional[str], gemini_key: Optional[str], verbose: bool = False):
    """Run validation for all model combinations."""
    results: List[TestResult] = []
    
    # Calculate total combinations
    openai_combos = 0
    gemini_combos = 0
    
    if openai_key:
        openai_combos = len(SUPPORTED_MODELS["openai"]["chat"]) * len(SUPPORTED_MODELS["openai"]["embed"])
    if gemini_key:
        gemini_combos = len(SUPPORTED_MODELS["gemini"]["chat"]) * len(SUPPORTED_MODELS["gemini"]["embed"])
    
    total_combos = openai_combos + gemini_combos
    
    if total_combos == 0:
        print("Error: No API keys provided. Use --openai-key and/or --gemini-key")
        return
    
    print(f"\nValidating {total_combos} model combinations...")
    print(f"  - OpenAI: {openai_combos} combinations")
    print(f"  - Gemini: {gemini_combos} combinations")
    print()
    
    current = 0
    
    # Test OpenAI combinations
    if openai_key:
        for chat_model in SUPPORTED_MODELS["openai"]["chat"]:
            for embed_model in SUPPORTED_MODELS["openai"]["embed"]:
                current += 1
                print(f"[{current}/{total_combos}] Testing OpenAI: {chat_model} + {embed_model}...", end=" ", flush=True)
                
                result = test_combination("openai", chat_model, embed_model, openai_key)
                results.append(result)
                
                status = "✅" if result.chat_success and result.embed_success else "❌"
                print(f"{status} ({result.duration_ms:.0f}ms)")
                
                if verbose and (result.chat_error or result.embed_error):
                    if result.chat_error:
                        print(f"    Chat error: {result.chat_error}")
                    if result.embed_error:
                        print(f"    Embed error: {result.embed_error}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
    
    # Test Gemini combinations
    if gemini_key:
        for chat_model in SUPPORTED_MODELS["gemini"]["chat"]:
            for embed_model in SUPPORTED_MODELS["gemini"]["embed"]:
                current += 1
                print(f"[{current}/{total_combos}] Testing Gemini: {chat_model} + {embed_model}...", end=" ", flush=True)
                
                result = test_combination("gemini", chat_model, embed_model, gemini_key)
                results.append(result)
                
                status = "✅" if result.chat_success and result.embed_success else "❌"
                print(f"{status} ({result.duration_ms:.0f}ms)")
                
                if verbose and (result.chat_error or result.embed_error):
                    if result.chat_error:
                        print(f"    Chat error: {result.chat_error}")
                    if result.embed_error:
                        print(f"    Embed error: {result.embed_error}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
    
    # Print results table
    print_results_table(results)
    
    # Return exit code based on results
    all_passed = all(r.chat_success and r.embed_success for r in results)
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate all supported model + embedding combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_models.py --openai-key sk-... --gemini-key AI...
  python validate_models.py --openai-key sk-...  # Test only OpenAI
  python validate_models.py --gemini-key AI...   # Test only Gemini
  python validate_models.py --openai-key sk-... --verbose  # Show detailed errors
        """
    )
    
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (starts with sk-)",
        default=None
    )
    
    parser.add_argument(
        "--gemini-key",
        help="Gemini API key (starts with AI)",
        default=None
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed error messages"
    )
    
    args = parser.parse_args()
    
    if not args.openai_key and not args.gemini_key:
        parser.error("At least one API key is required. Use --openai-key and/or --gemini-key")
    
    exit_code = run_validation(args.openai_key, args.gemini_key, args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
