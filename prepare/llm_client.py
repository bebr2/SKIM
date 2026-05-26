import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

env_file = os.getenv("ENV_FILE")
if env_file:
	# Allow selecting a specific dotenv file, e.g. ENV_FILE=.env.q8
	load_dotenv(dotenv_path=env_file, override=True)
else:
	load_dotenv()
    

def _env(*keys: str) -> Optional[str]:
	for key in keys:
		value = os.getenv(key)
		if value:
			return value
	return None


def _extract_message_text(content: Any) -> str:
	"""Normalize model message content to plain text."""
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		texts: List[str] = []
		for part in content:
			if isinstance(part, dict):
				if part.get("type") == "text":
					texts.append(str(part.get("text", "")))
				elif "text" in part:
					texts.append(str(part["text"]))
			else:
				texts.append(str(part))
		return "\n".join([t for t in texts if t])
	return str(content)


def _extract_json_block(text: str) -> str:
	"""Extract JSON payload from raw model output (handles markdown code fences)."""
	cleaned = text.strip()
	fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
	if fence_match:
		return fence_match.group(1).strip()
	return cleaned


@dataclass
class LLMConfig:
	provider: str
	model: str
	max_tokens: int = 4096
	temperature: float = 0.0


class LLMClient:
	"""
	A simple wrapper that supports:
	1) OpenAI API
	2) vLLM API (OpenAI-compatible endpoint)
	3) Azure OpenAI API
	"""

	def __init__(self, api_client: Any, config: LLMConfig):
		self.api_client = api_client
		self.config = config

	@classmethod
	def from_env(
		cls,
		model: Optional[str] = None,
		max_tokens: int = 4096,
		temperature: float = 0.0,
	) -> "LLMClient":
		# Azure mode has higher priority when required env vars exist.
		azure_api_key = _env("AZURE_OPENAI_API_KEY")
		azure_endpoint = _env("AZURE_OPENAI_ENDPOINT")
		azure_api_version = _env("AZURE_OPENAI_API_VERSION")

		if azure_api_key and azure_endpoint and azure_api_version:
			azure_model = model or _env("AZURE_OPENAI_MODEL", "LLM_MODEL")
			if not azure_model:
				raise ValueError(
					"Azure mode detected, but model is missing. "
					"Set AZURE_OPENAI_MODEL or LLM_MODEL, or pass model=..."
				)
			client = AzureOpenAI(
				api_key=azure_api_key,
				azure_endpoint=azure_endpoint,
				api_version=azure_api_version,
			)
			return cls(
				api_client=client,
				config=LLMConfig(
					provider="azure",
					model=azure_model,
					max_tokens=max_tokens,
					temperature=temperature,
				),
			)

		# OpenAI-compatible mode (OpenAI official API or vLLM endpoint).
		base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL")
		api_key = _env("LLM_API_KEY", "OPENAI_API_KEY")
		if (not api_key) and isinstance(base_url, str):
			base_url_lower = base_url.lower()
			if ("localhost" in base_url_lower) or ("127.0.0.1" in base_url_lower):
				# Local vLLM often ignores API key but OpenAI client expects one.
				api_key = "EMPTY"

		if not api_key:
			raise ValueError(
				"OpenAI-compatible mode requires LLM_API_KEY or OPENAI_API_KEY."
			)

		openai_model = model or _env("LLM_MODEL", "OPENAI_MODEL")
		if not openai_model:
			raise ValueError(
				"OpenAI-compatible mode requires model. "
				"Set LLM_MODEL or OPENAI_MODEL, or pass model=..."
			)

		kwargs: Dict[str, Any] = {"api_key": api_key}
		if base_url:
			kwargs["base_url"] = base_url

		client = OpenAI(**kwargs)
		return cls(
			api_client=client,
			config=LLMConfig(
				provider="openai_compatible",
				model=openai_model,
				max_tokens=max_tokens,
				temperature=temperature,
			),
		)

	@staticmethod
	def build_text_message(role: str, text: str) -> Dict[str, Any]:
		"""Build a message in the explicit content-part format."""
		return {
			"role": role,
			"content": [
				{
					"type": "text",
					"text": text,
				}
			],
		}

	def chat_completion(
		self,
		messages: List[Dict[str, Any]],
		model: Optional[str] = None,
		max_tokens: Optional[int] = None,
		temperature: Optional[float] = None,
		stream: bool = False,
		timeout: Optional[float] = None,
		extra_kwargs: Optional[Dict[str, Any]] = None,
	) -> Tuple[str, Any]:
		"""Return (raw_text, full_response)."""
		payload: Dict[str, Any] = {
			"model": model or self.config.model,
			"messages": messages,
			"max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
			"temperature": temperature if temperature is not None else self.config.temperature,
			"stream": stream,
		}
		if timeout is not None:
			payload["timeout"] = timeout
		if extra_kwargs:
			payload.update(extra_kwargs)

		response = self.api_client.chat.completions.create(**payload)
		content = response.choices[0].message.content
		text = _extract_message_text(content)
		return text, response

	def chat_json(
		self,
		messages: List[Dict[str, Any]],
		model: Optional[str] = None,
		max_tokens: Optional[int] = None,
		temperature: Optional[float] = None,
		stream: bool = False,
		timeout: Optional[float] = None,
		max_retries: int = 3,
		retry_sleep_seconds: float = 1.5,
	) -> Dict[str, Any]:
		"""Call chat completion and parse strict JSON from model output."""
		last_error: Optional[Exception] = None

		for attempt in range(1, max_retries + 1):
			try:
				text, _ = self.chat_completion(
					messages=messages,
					model=model,
					max_tokens=max_tokens,
					temperature=temperature,
					stream=stream,
					timeout=timeout,
				)
				json_text = _extract_json_block(text)
				return json.loads(json_text)
			except Exception as exc:
				last_error = exc
				if attempt < max_retries:
					time.sleep(retry_sleep_seconds * attempt)

		raise RuntimeError(f"Failed to get valid JSON after retries: {last_error}")
