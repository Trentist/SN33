from typing import List
import json
from openai import OpenAI

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib, model_override
from conversationgenome.utils.Utils import Utils


class LlmOpenRouter(LlmLib):
    def __init__(self):
        api_key = c.get('env', "OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it in the .env file or as an environment variable.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = c.get('env', "OPENROUTER_MODEL", "deepseek/deepseek-chat")
        pref = c.get('env', "OPENROUTER_PROVIDER_PREFERENCE", "chutes")
        self.provider_preference = None if (not pref or str(pref).lower() in ("none", "null", "off")) else pref
        self.embedding_model = c.get('env', "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.max_tokens = Utils._int(c.get("env", "OPENROUTER_MAX_TOKENS", 1024), default=1024)
        # Mercury-2 / other OpenRouter reasoning models: none|low|medium|high
        # Instance attr (reasoning_effort) wins over env when set via override.
        env_effort = (c.get("env", "OPENROUTER_REASONING_EFFORT", "") or "").strip().lower()
        self.reasoning_effort = env_effort if env_effort and env_effort not in ("none", "off", "null", "") else None


    ###############################################################################################
    ################################## Abstract methods override ##################################
    ###############################################################################################
    def basic_prompt(self, prompt: str, response_format: str = "text") -> str|None:
        api_format = {"type": "json_object"} if response_format == "json" else None
        
        extra_body = {}
        if self.provider_preference:
            extra_body["provider"] = {"order": [self.provider_preference]}
        effort = (self.reasoning_effort or "").strip().lower()
        if effort == "none":
            # Claude Haiku thinks for ~4k tokens unless this is sent explicitly.
            extra_body["reasoning"] = {"effort": "none"}
        elif effort and effort not in ("off", "null"):
            # OpenRouter reasoning format (Mercury-2: low|medium|high)
            extra_body["reasoning"] = {"effort": effort}

        try:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": api_format,
                "max_tokens": self.max_tokens,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            content = message.content or ""
            # Some reasoning models put the answer only in reasoning when truncated
            if not content.strip():
                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
                if isinstance(reasoning, str) and reasoning.strip():
                    content = reasoning
            return content
        except Exception as e:
            print(f"OpenRouter Completion Error")
            # Uncomment the line below for debugging purposes
            # print(e)
            return None

    def get_vector_embeddings(self, tag: str, dimensions=1536) -> List[float]|None:
        tag = tag.replace("\n", " ")
        # Note: OpenRouter might not support embeddings for all models.
        # We use a default embedding model (likely OpenAI's) or a configured one.
        try:
            response = self.client.embeddings.create(
                input=tag,
                model=self.embedding_model,
                # dimensions=dimensions # Some models don't support dimensions param
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenRouter Embedding Error")
            # Uncomment the line below for debugging purposes
            # print(e)
            return None
