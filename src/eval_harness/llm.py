"""Thin multi-provider LLM client (Phase 4).

Wraps litellm so one interface reaches OpenAI, Anthropic, or local Ollama.
- Token + cost accounting from the provider response (cost = 0 for Ollama).
- On-disk response cache (diskcache) keyed by hash(model, prompt, temperature, messages).
"""
