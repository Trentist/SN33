import requests
from typing import List
from openai import OpenAI

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib
from conversationgenome.utils.Utils import Utils


class LlmChutes(LlmLib):
    def __init__(self):
        self.api_key = c.get('env', "CHUTES_API_KEY")
        if not self.api_key:
            raise ValueError("CHUTES_API_KEY environment variable not set. Please set it in the .env file or as an environment variable.")
        
        self.client = OpenAI(
            base_url="https://llm.chutes.ai/v1",
            api_key=self.api_key,
            timeout=120.0,
        )
        self.model = c.get('env', "CHUTES_MODEL", "deepseek-ai/DeepSeek-V3")
        self.embedding_url = c.get('env', "CHUTES_EMBEDDING_URL", "https://chutes-qwen-qwen3-embedding-8b.chutes.ai/v1/embeddings")


    ###############################################################################################
    ################################## Abstract methods override ##################################
    ###############################################################################################
    def basic_prompt(self, prompt: str, response_format: str = "text") -> str|None:
        # GLM / reasoning models on Chutes often need a high ceiling so the
        # final answer is not truncated into reasoning_content only.
        max_tokens = Utils._int(c.get("env", "CHUTES_MAX_TOKENS", 2048), default=2048)

        try:
            completion_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if response_format == "json":
                completion_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**completion_params)
            message = response.choices[0].message
            content = (message.content or "").strip()
            # Fallback: some TEE models put the answer only in reasoning when truncated
            if not content:
                reasoning = getattr(message, "reasoning_content", None) or ""
                if reasoning:
                    lines = [ln.strip().strip('"') for ln in reasoning.splitlines() if ln.strip()]
                    # Prefer a comma-delimited tag-looking line
                    for ln in reversed(lines):
                        if "," in ln and len(ln) < 400 and not ln.startswith("*"):
                            content = ln
                            break
                    if not content and lines:
                        content = lines[-1]
            return content or None
        except Exception as e:
            print(f"Chutes Completion Error: {type(e).__name__}: {e}")
            return None

    def get_vector_embeddings(self, tag: str, dimensions=1536) -> List[float]|None:
        tag = tag.replace("\n", " ")
        try:
            headers = {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json"
            }
            
            body = {
                "input": tag,
                "model": None
            }

            response = requests.post(
                self.embedding_url,
                headers=headers,
                json=body
            )

            if response.status_code == 200:
                result = response.json()
                return result['data'][0]['embedding']
            else:
                print(f"Chutes Embedding Error: Status {response.status_code}")
                # print(response.text)
                return None
        except Exception as e:
            print(f"Chutes Embedding Error")
            # print(e)
            return None
