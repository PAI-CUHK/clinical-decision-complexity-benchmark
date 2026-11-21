from __future__ import annotations
import json
import os
import re
import time

import requests


def extract_root_block(text: str) -> str | None:
    match = re.search(
        r'<root>[\s\S]*?</root>',
        text,
        flags=re.IGNORECASE
    )
    return match.group(0).strip() if match else None


class LLM_Models:
    def __init__(self):
        self.secrete_file = os.environ.get("MEDNP_SECRETS_FILE", "secrets.txt")
        self.responses_base_url = os.environ.get("RESPONSES_BASE_URL", "http://localhost:8000/v1")
        self.responses_secret_file = self.secrete_file
        try:
            self.open_ai_key = self._load_secret("open_ai_key")
        except (OSError, KeyError):
            self.open_ai_key = None


    def _load_secret(self, key_name, secret_file=None):
        """Read a named key from a comma-separated secret file.
        
        Args:
            key_name: Secret name, for example QWEN_API_KEY.
            secret_file: Optional path; defaults to self.secrete_file.
        
        Returns:
            The secret string."""
        secret_file = secret_file or self.secrete_file
        with open(secret_file) as f:
            for line in f:
                parts = line.split(',', 1)
                if len(parts) == 2 and parts[0].strip() == key_name:
                    return parts[1].strip()
        raise KeyError(f"Secret '{key_name}' not found in {secret_file}")

    def run_model(
            self,
            messages,
            backend="vllm",  # "vllm" | "gpt"
            temperature=0.0,
            max_tokens=2048,
            **kwargs
    ):
        """Call the selected model backend."""

        if backend == "vllm":
            return self.run_vllm(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                vllm_port=kwargs.get("vllm_port", 8000),
                vllm_model=kwargs.get(
                    "vllm_model",
                    "mistralai/Mistral-7B-Instruct-v0.3"
                )
            )

        elif backend == "gpt":
            return self.run_gpt(
                messages=messages,
                model=kwargs.get("gpt_model", "gpt-5"),
                retry_interval=kwargs.get("retry_interval", 10),
                query_usage=kwargs.get("query_usage", True),
                max_tokens=max_tokens,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")


    def _query_recent_cost(self, model='gpt-5', verbose=False):
        """Query the total cost over the preceding hour.
        
        Args:
            model: Model name.
            verbose: Whether to print diagnostic details.
        
        Returns:
            The total cost, or -1.0 on failure; zero also represents no data."""
        url = "https://api.chatanywhere.tech/v1/query/usage_details"
        headers = {
            "Authorization": self.open_ai_key,
            "Content-Type": "application/json",
        }
        model_pattern = model if model.endswith('%') else f"{model}%"
        payload = {"model": model_pattern, "hours": 1}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if verbose:
                print(f"  [cost query] model={model_pattern}, status={resp.status_code}, data_len={len(data) if isinstance(data, list) else 'not-list'}")
            if isinstance(data, list) and len(data) > 0:
                total_cost = sum(d.get("cost", 0) for d in data)
                if verbose:
                    print(f"  [cost query] total_cost={total_cost}, entries={len(data)}")
                return round(total_cost, 6)
            return 0.0  # API succeeded but returned no data (free tier or delayed reporting)
        except Exception as e:
            if verbose:
                print(f"  [cost query] FAILED: {e}")
            return -1.0  # Query failed


    def run_gpt(self, messages, model='gpt-5', retry_interval=10, query_usage=True,
                 max_tokens=8192, temperature=0.0):
        """Call the GPT API, retrying failures until a response is received.
        
        Args:
            messages: Message list.
            model: Model identifier.
            retry_interval: Delay between retries in seconds.
            query_usage: Whether to query recent costs after a successful call.
            max_tokens: Maximum generated tokens.
            temperature: Sampling temperature.
        
        Returns:
            A dictionary containing content and usage data."""
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.chatanywhere.tech/v1",
            api_key=self.open_ai_key
        )

        def _extract_usage_data(response):
            """Extract a usage dictionary from the response."""
            usage_data = {}
            if hasattr(response, 'usage') and response.usage:
                u = response.usage
                usage_data = {
                    "prompt_tokens": getattr(u, 'prompt_tokens', 0),
                    "completion_tokens": getattr(u, 'completion_tokens', 0),
                    "total_tokens": getattr(u, 'total_tokens', 0),
                }
                for extra_field in ['cached_tokens', 'reasoning_tokens', 'cost',
                                    'prompt_cost', 'completion_cost', 'total_cost',
                                    'billable_tokens', 'gpa_tokens',
                                    'preflight_tokens']:
                    val = getattr(u, extra_field, None)
                    if val is not None and val != 0:
                        usage_data[extra_field] = val

                # Extract reasoning_tokens from completion_tokens_details
                ctd = getattr(u, 'completion_tokens_details', None)
                if ctd is not None:
                    rt = getattr(ctd, 'reasoning_tokens', None)
                    if rt is not None and rt != 0:
                        usage_data['reasoning_tokens'] = rt
                    for sub in ['accepted_prediction_tokens', 'rejected_prediction_tokens',
                                'audio_tokens']:
                        sv = getattr(ctd, sub, None)
                        if sv is not None and sv != 0:
                            usage_data[f'completion_{sub}'] = sv

            if hasattr(response, 'headers') and response.headers:
                for hdr_key in ['x-ratelimit-cost', 'x-usage-cost', 'x-billing-cost']:
                    val = response.headers.get(hdr_key)
                    if val:
                        usage_data[hdr_key.replace('-', '_')] = float(val)

            try:
                if hasattr(response, 'model_dump'):
                    resp_dict = response.model_dump()
                    if isinstance(resp_dict, dict):
                        raw_usage = resp_dict.get('usage', {})
                        if isinstance(raw_usage, dict):
                            for k, v in raw_usage.items():
                                if k not in usage_data and v not in (None, 0, ''):
                                    usage_data[k] = v
                elif hasattr(response, 'to_dict'):
                    resp_dict = response.to_dict()
                    if isinstance(resp_dict, dict):
                        raw_usage = resp_dict.get('usage', {})
                        if isinstance(raw_usage, dict):
                            for k, v in raw_usage.items():
                                if k not in usage_data and v not in (None, 0, ''):
                                    usage_data[k] = v
            except Exception:
                pass

            return usage_data

        def _is_reasoning_starving(usage_data):
            """Check whether reasoning tokens consumed most of the completion budget."""
            ct = usage_data.get('completion_tokens', 0)
            rt = usage_data.get('reasoning_tokens', 0)
            return ct > 0 and rt > 0 and rt >= ct * 0.8

        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                usage_data = _extract_usage_data(response)

                # Diagnose empty content and return the response immediately
                if not content.strip():
                    comp_tokens = usage_data.get('completion_tokens', 0)
                    prompt_tokens = usage_data.get('prompt_tokens', 0)
                    rt = usage_data.get('reasoning_tokens', 0)
                    if _is_reasoning_starving(usage_data):
                        print(f"⚠️ run_gpt: 空 content, reasoning tokens 占满预算 "
                              f"(reasoning={rt}, completion={comp_tokens}). "
                              f"调大 --max_tokens 或降低推理强度可避免。")
                    elif comp_tokens >= max_tokens * 0.95:
                        print(f"⚠️ run_gpt: 空 content, max_tokens 耗尽 "
                              f"(completion={comp_tokens}, limit={max_tokens})")
                    else:
                        print(f"⚠️ run_gpt: 空 content, API 偶发 "
                              f"(prompt={prompt_tokens}, completion={comp_tokens}, reasoning={rt})")
                    # Return immediately; usage already includes token consumption
                    if query_usage:
                        hour_cost = self._query_recent_cost(model=model)
                        if hour_cost >= 0:
                            usage_data["estimated_cost"] = hour_cost
                    return {"content": content, "usage": usage_data}

                # Return the response when content is present
                if query_usage:
                    hour_cost = self._query_recent_cost(model=model)
                    if hour_cost >= 0:
                        usage_data["estimated_cost"] = hour_cost

                return {"content": content, "usage": usage_data}

            except Exception as e:
                print("Request failed, retrying in", retry_interval, "seconds. Error:", e)
                time.sleep(retry_interval)

    def _resolve_model_id(self, model, base_url, api_key):
        """Resolve a case-insensitive model name to the gateway model ID.
        
        Query /v1/models once per base URL and cache IDs. If resolution fails,
        return the original name."""
        cache = getattr(self, "_model_id_cache", None)
        if cache is None:
            cache = self._model_id_cache = {}
        if base_url not in cache:
            ids = []
            try:
                resp = requests.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                if resp.ok:
                    ids = [d.get("id") for d in (resp.json().get("data") or [])
                           if isinstance(d, dict) and d.get("id")]
            except Exception as e:
                print(f"⚠️ 解析 model id 失败，按原值使用: {e}")
            cache[base_url] = ids
        for mid in cache[base_url]:
            if mid.lower() == model.lower():
                return mid
        return model

    def _call_responses_api(self, messages, model, temperature=1.0, max_tokens=16384,
                            reasoning_effort="xhigh", base_url=None, api_key=None):
        """Call the configured Responses API gateway using a streaming request.
        
        Args:
            reasoning_effort: Gateway reasoning level, or None to omit it.
        
        Returns:
            A dictionary containing content and usage data."""
        base_url = (base_url or self.responses_base_url).rstrip("/")
        if api_key is None:
            api_key = self._load_secret("QWEN_API_KEY", self.responses_secret_file)
        url = f"{base_url}/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        def _extract_usage(u):
            u = u or {}
            input_details = u.get("input_tokens_details") or {}
            output_details = u.get("output_tokens_details") or {}
            return {
                "prompt_tokens": u.get("input_tokens", 0),
                "completion_tokens": u.get("output_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
                "cached_tokens": input_details.get("cached_tokens", 0),
                "reasoning_tokens": output_details.get("reasoning_tokens", 0),
            }

        def _make_result(content, usage=None):
            return {"content": content, "usage": usage or {}}

        def _stream_once(model_name):
            """Send one streaming Responses request.
            
            Return response text, reasoning text, usage, and an error (None on success).
            Output and reasoning deltas are accumulated separately."""
            data = {
                "model": model_name,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "stream": True,
            }
            if reasoning_effort:
                data["reasoning"] = {"effort": reasoning_effort}

            response_text = ""
            reasoning_text = ""
            usage = {}
            # read_timeout=600 permits up to 600 seconds between chunks.
            with requests.post(url, headers=headers, json=data, stream=True,
                               timeout=(30, 600)) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "response.output_text.delta":
                        response_text += event.get("delta") or ""
                    elif etype == "response.reasoning_text.delta":
                        reasoning_text += event.get("delta") or ""
                    elif etype == "response.completed":
                        usage = _extract_usage((event.get("response") or {}).get("usage"))
                    elif etype in ("response.failed", "error"):
                        raise RuntimeError(json.dumps(event.get("response") or event, ensure_ascii=False))
            return response_text, reasoning_text, usage, None

        # Resolve model names to the case-sensitive IDs registered by the gateway
        model_name = self._resolve_model_id(model, base_url, api_key)

        try:
            response_text, reasoning_text, usage, err = _stream_once(model_name)
            if err is not None:
                raise err if isinstance(err, Exception) else RuntimeError(str(err))
        except Exception as e:
            print(f"⚠️ Responses API 调用异常 (model={model_name}): {e}")
            return _make_result("Error: No valid response after multiple attempts")

        comp_tokens = usage.get("completion_tokens", 0)
        # Empty content with reasoning may indicate insufficient max_output_tokens,
        # causing truncation before the final answer starts.
        if not response_text and reasoning_text:
            print(f"⚠️ content 为空但思考内容有 {len(reasoning_text)} 字符 "
                  f"(completion_tokens={comp_tokens})。")
            print("   ⚠️ 这几乎肯定是 max_output_tokens 不足导致思考阶段被截断！")
            # Wrap reasoning content in tags so subsequent cleanup can handle it
            response_text = f"<thinking>{reasoning_text}</thinking>"
        elif not response_text and not reasoning_text:
            print("⚠️ 流式调用返回完全空内容（content 和 reasoning 都为空）")
            return _make_result("Error: No valid response after multiple attempts", usage)

        # Remove reasoning tags, including fallback thinking tags
        cleaned_output = re.sub(
            r'(<thinking>[\s\S]*?</thinking>|<think>[\s\S]*?</think>)',
            '', response_text, flags=re.DOTALL
        ).strip()
        json_str = None

        # 1. Final Response heading format
        match = re.search(r'## Final Response\s*({.*})', cleaned_output, flags=re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        # 2. Direct JSON/dict output
        elif cleaned_output.startswith("{") and cleaned_output.endswith("}"):
            json_str = cleaned_output.strip()
        # 3. Root tag format
        root_str = extract_root_block(cleaned_output)
        if root_str:
            json_str = root_str
        # 4. Otherwise return the full cleaned output
        if not json_str and cleaned_output:
            print("⚠️ 未找到标准格式，返回清理后的输出")
            return _make_result(cleaned_output, usage)

        if json_str:  # Return only a valid result
            return _make_result(json_str, usage)

        print(f"⚠️ 结果不符合要求 (model={model_name})")
        return _make_result("Error: No valid response after multiple attempts", usage)

    def run_vllm(self, messages=None, temperature=1.0, max_tokens=16384,
                 vllm_port=8000, vllm_model="qwen3.8-27b", reasoning_effort="xhigh"):
        """Call the configured Responses API gateway.
        
        The API key is read from QWEN_API_KEY in the configured secret file.
        
        Args:
            vllm_port: Legacy compatibility parameter; unused by the gateway.
            vllm_model: Model ID on the gateway.
            reasoning_effort: Gateway reasoning level.
        
        Returns:
            A dictionary containing content and usage data."""
        return self._call_responses_api(
            messages,
            model=vllm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
