"""
Modular ReAct Agent with three separate temperature/seed controlled modules:
1. Summarization - Summarizes search results after tool responses
2. Reasoning - Generates inference/analysis based on context
3. Query - Issues search queries or provides final answers

Each module can have independent temperature and seed settings for 
studying stochasticity at the component level.
"""

import copy
import json
import json5
import os
import re
import difflib
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import AutoTokenizer
from datetime import datetime
import time
import asyncio
import httpx
from openai import OpenAI

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.tools import BaseTool

from prompt import SYSTEM_PROMPT
from tool_search import Search

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 10))

TOOL_CLASS = [Search()]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def today_date():
    return datetime.today().strftime("%Y-%m-%d")


def strip_answer_tags(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_finding(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def similarity_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_finding(a), normalize_finding(b)).ratio()


def extract_json_list(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = strip_answer_tags(text)
    cleaned = cleaned.strip()
    # Strip code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        cleaned = cleaned.rstrip("```").strip()
    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Try to parse the first JSON array in the text
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


# Module-specific prompts
SUMMARIZATION_PROMPT = """You are a research assistant. Your task is to summarize the search results concisely.

Given the search results below, extract and summarize the most relevant information that helps answer the research question.

## Research Question
{question}

## Search Results
{search_results}

## Instructions
1. Identify the most relevant facts and findings from the search results
2. Organize the information logically
3. Include source URLs for key facts
4. Be concise but comprehensive
5. Highlight any conflicting information if present

Provide your summary in a clear, structured format."""


REASONING_PROMPT = """Based on the current research context, analyze what you know and what you still need to find.

## Research Question
{question}

## Current Knowledge
{context}

## Instructions
Think step by step:
1. What key facts have you established so far?
2. What gaps remain in your understanding?
3. What additional information would help answer the question definitively?
4. Are there any contradictions that need resolution?

Wrap your analysis in <reasoning></reasoning> tags."""


QUERY_PROMPT = """Based on your reasoning, decide the next action.

## Research Question
{question}

## Your Reasoning
{reasoning}

## Current Context
{context}

## Remaining Search Budget
You have **{remaining_trials} search attempts remaining** out of {max_trials} total.
{budget_warning}

## Instructions
You have two options:

1. **If you need more information**: Generate search queries wrapped in <tool_call></tool_call> tags.
   Format: <tool_call>{{"name": "search", "arguments": {{"query": ["query 1", "query 2", "query 3"]}}}}</tool_call>
   Requirements:
   - Use broader queries when unsure, and include multiple complementary queries.
   - Avoid repeating the same queries across rounds.
   - Incorporate new information from the last round's search results to refine or expand the next queries.

2. **If you have enough information**: Provide the final answer wrapped in <answer></answer> tags. Do NOT respond with "I don't know", "can't identify" or similar non-answers.

Choose ONE action only.

"""

FINAL_REPORT_FORMAT_GUIDANCE = """Regarding the final report format, please create a detailed report to the overall research question that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language.
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
"""

FINAL_SYNTHESIS_PROMPT = """You have exhausted your search budget. Based on ALL the information gathered during your research, you MUST now provide your best possible answer.

## Research Question
{question}

## All Research Context Gathered
{context}

## Instructions
1. Review all the information you have gathered above
2. Synthesize the findings to answer the research question
3. If you found the answer, state it clearly
4. If the information is incomplete, provide your best inference based on available evidence
5. Do NOT say "No answer found" - always attempt to answer based on what you learned

You MUST wrap your final answer in <answer></answer> tags. This is mandatory."""


class ModularReactAgent(FnCallAgent):
    """
    A modular ReAct agent with three separate LLM-powered modules:
    - Summarization: Processes tool responses
    - Reasoning: Generates analysis and inference
    - Query: Issues search queries or final answers
    
    Each module can have independent temperature and seed settings.
    """
    
    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        module_configs: Optional[Dict] = None,
        mitigation_config: Optional[Dict[str, bool]] = None,
        step_temp_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        
        # Local OpenAI-compatible client
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:30000/v1")
        self.client = self._create_openai_client(self.llm_base_url)
        
        # Module-specific configurations (temperature, seed)
        # Defaults: temperature=0, seed=42 for all modules
        default_module_config = {"temperature": 0.0, "seed": 42}
        self.module_configs = {
            "summarization": default_module_config.copy(),
            "reasoning": default_module_config.copy(),
            "query": default_module_config.copy(),
        }
        
        # Override with provided configs
        if module_configs:
            for module_name, config in module_configs.items():
                if module_name in self.module_configs:
                    self.module_configs[module_name].update(config)

        # Mitigation toggles
        self.mitigation_config = {
            "use_ensemble": False,
            "use_consistency": False,
            "use_structure": False,
        }
        if mitigation_config:
            self.mitigation_config.update(mitigation_config)

        # Step-level temperature overrides (same for all runs)
        # Format: {"module": {"steps": {1,2,3}, "temp": 0.5}}
        self.step_temp_config = step_temp_config or {}
        
        # Tokenizer for token counting
        self._tokenizer = None
    
    def _get_tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            except:
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        return self._tokenizer

    def _create_openai_client(self, base_url: str) -> OpenAI:
        api_key = os.getenv("LLM_API_KEY", "EMPTY")
        try:
            return OpenAI(api_key=api_key, base_url=base_url)
        except (PermissionError, OSError) as exc:
            print(f"Warning: OpenAI client SSL setup failed ({exc}); retrying with verify=False.")
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(verify=False)
            )
    
    def _strip_tool_response(self, content: str) -> str:
        return re.sub(r"<tool_response>.*?</tool_response>", "<tool_response></tool_response>", content, flags=re.DOTALL)

    def count_tokens(self, messages, exclude_tool_response: bool = True):
        """Count tokens in a message list."""
        tokenizer = self._get_tokenizer()
        if exclude_tool_response:
            sanitized = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = self._strip_tool_response(content)
                sanitized.append({**msg, "content": content})
            messages = sanitized
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(full_prompt)
        return len(tokens)

    def _ensure_min_temperature(self, temperature: float, min_temp: float = 0.2) -> float:
        if temperature <= 0:
            return min_temp
        return temperature

    def _structured_response_format(self) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "findings_list",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding": {"type": "string"},
                            "confidence": {"type": "number"},
                            "is_new": {"type": "boolean"}
                        },
                        "required": ["finding", "confidence", "is_new"],
                        "additionalProperties": False
                    }
                },
                "strict": True
            }
        }

    def _parse_json_list(self, content: str) -> List[str]:
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else []
        except Exception:
            match = re.search(r"\[[\s\S]*\]", content)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, list) else []
            except Exception:
                return []

    def _llm_intersection_queries(
        self,
        query_sets: List[List[str]],
        run_configs: Dict,
        round_num: Optional[int] = None
    ) -> List[str]:
        if len(query_sets) < 2:
            return query_sets[0] if query_sets else []
        local_configs = copy.deepcopy(run_configs)
        local_configs["query"]["temperature"] = 0.0
        prompt = (
            "You are given multiple sets of search queries from independent proposals.\n"
            "Find the intersection across ALL sets using semantic equivalence.\n"
            "Only return queries that are effectively the same across every set.\n"
            "Return ONLY a JSON array of strings. If none, return [].\n\n"
            f"Query sets:\n{json.dumps(query_sets, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": "You are a strict evaluator of query overlap."},
            {"role": "user", "content": prompt}
        ]
        response = self._call_module("query", messages, local_configs, stop_sequences=[], round_num=round_num)
        return [q for q in self._parse_json_list(response) if isinstance(q, str) and q.strip()]

    def _llm_rank_queries(
        self,
        queries: List[str],
        run_configs: Dict,
        round_num: Optional[int] = None
    ) -> List[str]:
        if not queries:
            return []
        local_configs = copy.deepcopy(run_configs)
        local_configs["query"]["temperature"] = 0.0
        prompt = (
            "Rank the following search queries by usefulness for answering the question.\n"
            "Return ONLY a JSON array of strings ordered from best to worst.\n\n"
            f"Queries:\n{json.dumps(queries, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": "You are a strict query ranker."},
            {"role": "user", "content": prompt}
        ]
        response = self._call_module("query", messages, local_configs, stop_sequences=[], round_num=round_num)
        ranked = [q for q in self._parse_json_list(response) if isinstance(q, str) and q.strip()]
        if not ranked:
            return []
        seen = set()
        ordered = []
        for q in ranked:
            if q in queries and q not in seen:
                seen.add(q)
                ordered.append(q)
        for q in queries:
            if q not in seen:
                ordered.append(q)
        return ordered

    def _get_step_temperature(self, module_name: str, round_num: int, base_temp: float) -> float:
        config = self.step_temp_config.get(module_name)
        if not config:
            return base_temp
        steps = config.get("steps", set())
        temp = config.get("temp", base_temp)
        return temp if round_num in steps else base_temp

    def _parse_findings(self, text: str) -> List[Dict[str, Any]]:
        parsed = extract_json_list(text)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict) and "finding" in parsed[0]:
                return parsed
            if parsed and isinstance(parsed[0], str):
                return [{"finding": f, "confidence": 0.6, "is_new": True} for f in parsed if f.strip()]
            if not parsed:
                return []
        # Fallback: split lines/sentences
        cleaned = strip_answer_tags(text)
        lines = [line.strip("-* \t") for line in cleaned.splitlines() if line.strip()]
        if not lines:
            sentences = re.split(r"\.\s+", cleaned)
            lines = [s.strip() for s in sentences if s.strip()]
        return [{"finding": line, "confidence": 0.6, "is_new": True} for line in lines]

    def _majority_vote_findings(
        self,
        findings_runs: List[List[Dict[str, Any]]],
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for run_idx, findings in enumerate(findings_runs):
            for finding in findings:
                text = finding.get("finding", "").strip()
                if not text:
                    continue
                matched = False
                for cluster in clusters:
                    if similarity_ratio(text, cluster["finding"]) >= similarity_threshold:
                        cluster["runs"].add(run_idx)
                        cluster["confidences"].append(float(finding.get("confidence", 0.6)))
                        cluster["is_new"] = cluster["is_new"] or bool(finding.get("is_new", True))
                        matched = True
                        break
                if not matched:
                    clusters.append({
                        "finding": text,
                        "runs": {run_idx},
                        "confidences": [float(finding.get("confidence", 0.6))],
                        "is_new": bool(finding.get("is_new", True)),
                    })
        majority = []
        for cluster in clusters:
            if len(cluster["runs"]) >= 2:
                avg_conf = sum(cluster["confidences"]) / max(len(cluster["confidences"]), 1)
                majority.append({
                    "finding": cluster["finding"],
                    "confidence": round(avg_conf, 3),
                    "is_new": cluster["is_new"],
                })
        return majority

    def _build_context(
        self,
        summaries: str,
        previous_query: Optional[str] = None,
        previous_reasoning: Optional[str] = None
    ) -> str:
        parts: List[str] = []
        if previous_query:
            parts.append("## Previous Query\n" + previous_query.strip())
        if previous_reasoning:
            parts.append("## Previous Reasoning\n" + previous_reasoning.strip())
        if summaries:
            parts.append("## Summaries\n" + summaries.strip())
        return "\n\n".join(parts).strip()
    
    def _call_module(
        self, 
        module_name: str, 
        messages: List[Dict], 
        run_configs: Dict,
        max_tries: int = 10,
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        round_num: Optional[int] = None
    ) -> str:
        """
        Call a specific module with its configured temperature and seed.
        
        Args:
            module_name: One of 'summarization', 'reasoning', 'query'
            messages: The messages to send
            run_configs: The module configs for this specific run (thread-safe)
            max_tries: Maximum retry attempts
            stop_sequences: Optional stop sequences
        
        Returns:
            The model's response content
        """
        config = run_configs[module_name]
        temperature = config["temperature"]
        if round_num is not None:
            temperature = self._get_step_temperature(module_name, round_num, temperature)
        seed = config["seed"]
        
        base_sleep_time = 1
        
        for attempt in range(max_tries):
            try:
                print(f"--- [{module_name.upper()}] Attempt {attempt + 1}/{max_tries}, temp={temperature}, seed={seed} ---")
                
                api_params = {
                    "model": self.llm_local_path,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 10000,
                    "seed": seed
                }
                
                if stop_sequences:
                    api_params["stop"] = stop_sequences
                if response_format:
                    api_params["response_format"] = response_format
                
                chat_response = self.client.chat.completions.create(**api_params)
                content = chat_response.choices[0].message.content
                
                if content and content.strip():
                    print(f"--- [{module_name.upper()}] Success ---")
                    return content.strip()
                else:
                    print(f"Warning: [{module_name}] Attempt {attempt + 1} received empty response.")
            
            except Exception as e:
                print(f"Error: [{module_name}] Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt), 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        return f"[{module_name}] Server error after {max_tries} attempts"
    
    def call_summarization(self, question: str, search_results: str, run_configs: Dict, round_num: int) -> str:
        """
        Summarization module: Processes and summarizes search results.
        """
        use_structure = self.mitigation_config.get("use_structure", False)
        use_consistency = self.mitigation_config.get("use_consistency", False)
        local_configs = copy.deepcopy(run_configs)
        if use_consistency:
            local_configs["summarization"]["temperature"] = self._ensure_min_temperature(
                local_configs["summarization"]["temperature"]
            )

        if use_structure:
            prompt = (
                "Extract concise factual findings from the search results. "
                "Return ONLY a JSON array of objects with keys: "
                "finding (string), confidence (float 0-1), is_new (bool).\n\n"
                f"Research Question: {question}\n\n"
                f"Search Results:\n{search_results}"
            )
        else:
            prompt = SUMMARIZATION_PROMPT.format(
                question=question,
                search_results=search_results
            )

        def run_once() -> str:
            messages = [
                {"role": "system", "content": "You are a research assistant specialized in summarizing search results."},
                {"role": "user", "content": prompt}
            ]
            response_format = self._structured_response_format() if use_structure else None
            response = self._call_module(
                "summarization",
                messages,
                local_configs,
                response_format=response_format,
                round_num=round_num
            )
            if use_structure and "Server error" in response:
                response = self._call_module("summarization", messages, local_configs, round_num=round_num)
            return response

        if use_consistency:
            responses = [run_once() for _ in range(3)]
            findings_runs = [self._parse_findings(r) for r in responses]
            majority = self._majority_vote_findings(findings_runs)
            if use_structure:
                summary = json.dumps(majority, ensure_ascii=False)
            else:
                summary = "\n".join(f"- {item['finding']}" for item in majority)
            response = summary
        else:
            response = run_once()
            if use_structure:
                parsed = self._parse_findings(response)
                response = json.dumps(parsed, ensure_ascii=False)

        if "<summarize>" not in response:
            response = f"<summarize>\n{response}\n</summarize>"

        return response
    
    def call_reasoning(self, question: str, context: str, run_configs: Dict, round_num: int) -> str:
        """
        Reasoning module: Generates analysis and inference.
        """
        prompt = REASONING_PROMPT.format(
            question=question,
            context=context if context else "No information gathered yet."
        )
        
        messages = [
            {"role": "system", "content": "You are a research analyst. Think deeply about the research question."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_module("reasoning", messages, run_configs, round_num=round_num)
        
        # Ensure response has reasoning tags
        if "<reasoning>" not in response:
            response = f"<reasoning>\n{response}\n</reasoning>"
        
        return response
    
    def call_query(
        self,
        question: str,
        reasoning: str,
        context: str,
        run_configs: Dict,
        remaining_trials: int = None,
        max_trials: int = None,
        force_action: Optional[str] = None,
        round_num: Optional[int] = None,
        proposed_queries: Optional[List[str]] = None
    ) -> str:
        """
        Query module: Issues search queries or provides final answers.
        
        Args:
            question: The research question
            reasoning: Output from reasoning module
            context: Accumulated context from previous searches
            run_configs: Module configurations for this run
            remaining_trials: Number of search attempts remaining
            max_trials: Total number of search attempts allowed
        """
        # Generate budget warning based on remaining trials
        if remaining_trials is not None and max_trials is not None:
            if remaining_trials <= 1:
                budget_warning = "⚠️ **CRITICAL**: This is your last search attempt! Consider providing your final answer if you have sufficient information."
            elif remaining_trials <= 2:
                budget_warning = "⚠️ **Warning**: You are running low on search attempts. Be strategic with your remaining queries."
            elif remaining_trials <= max_trials // 2:
                budget_warning = "You have used more than half of your search budget."
            else:
                budget_warning = ""
        else:
            remaining_trials = "unknown"
            max_trials = "unknown"
            budget_warning = ""
        
        action_guidance = ""
        if force_action == "tool_call":
            action_guidance = "\nYou MUST output a <tool_call> and you are FORBIDDEN from answering."
        elif force_action == "answer":
            action_guidance = "\nYou MUST output a final answer in <answer></answer> tags and you are FORBIDDEN from searching."

        prompt = QUERY_PROMPT.format(
            question=question,
            reasoning=reasoning,
            context=context if context else "No information gathered yet.",
            remaining_trials=remaining_trials,
            max_trials=max_trials,
            budget_warning=budget_warning
        )
        if proposed_queries:
            proposed_block = "\n".join(f"- {q}" for q in proposed_queries)
            prompt += (
                "\n\n## Queries Already Proposed This Round\n"
                f"{proposed_block}\n"
                "Do NOT repeat these. Propose complementary or refined queries only."
            )
        prompt += action_guidance
        
        messages = [
            {"role": "system", "content": "You are a research assistant. Decide the next action based on your analysis."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_module(
            "query", 
            messages,
            run_configs,
            stop_sequences=["\n<tool_response>", "<tool_response>"]
            ,
            round_num=round_num
        )

    def _extract_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if "<tool_call>" not in content or "</tool_call>" not in content:
            return None
        parts = content.split("<tool_call>")
        tool_call_str = parts[-1].split("</tool_call>")[0].strip()
        try:
            tool_call = json5.loads(tool_call_str)
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            return tool_name, tool_args
        except Exception:
            return None

    def generate_queries(
        self,
        question: str,
        reasoning: str,
        context: str,
        run_configs: Dict,
        remaining_trials: int,
        max_trials: int,
        base_response: Optional[str] = None,
        initial_k: int = 4,
        min_k: int = 4,
        round_num: Optional[int] = None
    ) -> List[str]:
        # Start with more samples, then decay as rounds progress.
        if round_num is None:
            k = initial_k
        else:
            k = max(min_k, initial_k - max(0, round_num - 1))
        local_configs = copy.deepcopy(run_configs)
        local_configs["query"]["temperature"] = self._ensure_min_temperature(
            local_configs["query"]["temperature"]
        )
        responses = []
        if base_response:
            responses.append(base_response)
        while len(responses) < k:
            responses.append(
                self.call_query(
                    question,
                    reasoning,
                    context,
                    local_configs,
                    remaining_trials=remaining_trials,
                    max_trials=max_trials,
                    force_action="tool_call",
                    round_num=round_num
                )
            )

        query_sets = []
        for resp in responses:
            parsed = self._extract_tool_call(resp)
            if not parsed:
                continue
            tool_name, tool_args = parsed
            if tool_name != "search":
                continue
            queries = [q.strip() for q in tool_args.get("query", []) if q and q.strip()]
            if queries:
                query_sets.append(queries)

        if not query_sets:
            return []

        intersected = self._llm_intersection_queries(query_sets, run_configs, round_num=round_num)
        return intersected or query_sets[0]

    def call_final_report(self, question: str, context: str, run_configs: Dict, round_num: int) -> str:
        use_structure = self.mitigation_config.get("use_structure", False)
        if use_structure:
            prompt = (
                "You must produce the final report as a JSON array of objects with keys: "
                "finding (string), confidence (float 0-1), is_new (bool). "
                "Return ONLY valid JSON, no extra text.\n\n"
                f"Research Question: {question}\n\n"
                f"All Research Context:\n{context}"
            )
            messages = [
                {"role": "system", "content": "You are a research assistant. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_module(
                "query",
                messages,
                run_configs,
                response_format=self._structured_response_format(),
                round_num=round_num
            )
            if "Server error" in response:
                response = self._call_module("query", messages, run_configs, round_num=round_num)
            parsed = self._parse_findings(response)
            return json.dumps(parsed, ensure_ascii=False)

        prompt = (
            "Provide the final report now. Follow the required report format. "
            "Wrap the report in <answer></answer> tags.\n\n"
            f"{FINAL_REPORT_FORMAT_GUIDANCE}\n"
            f"Research Question: {question}\n\n"
            f"All Research Context:\n{context}"
        )
        messages = [
            {"role": "system", "content": "You are a research assistant. Provide the final report."},
            {"role": "user", "content": prompt}
        ]
        response = self._call_module("query", messages, run_configs, round_num=round_num)
        if "<answer>" in response and "</answer>" in response:
            return response.split("<answer>")[1].split("</answer>")[0].strip()
        return response.strip()

    def _ensure_non_empty_report(self, report: Optional[str], question: str, context: str, run_configs: Dict, round_num: int) -> str:
        if report is None:
            return self.call_final_report(question, context, run_configs, round_num=round_num)
        stripped = str(report).strip()
        if not stripped or stripped.lower() == "none":
            return self.call_final_report(question, context, run_configs, round_num=round_num)
        return stripped
    
    def call_final_synthesis(self, question: str, context: str, run_configs: Dict, round_num: int) -> str:
        """
        Final synthesis module: Forces an answer when call limit is exceeded.
        
        Uses the accumulated context to synthesize the best possible answer.
        """
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            question=question,
            context=context if context else "No information was gathered during research."
        )
        
        messages = [
            {"role": "system", "content": "You are a research assistant. You must synthesize all available information into a final answer."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_module(
            "query",
            messages,
            run_configs,
            stop_sequences=[],
            round_num=round_num
        )
    
    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        """Execute a tool call."""
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            return raw_result
        else:
            return f"Error: Tool {tool_name} not found"
    
    def _run(self, data: Dict, model: str, **kwargs) -> Dict:
        """
        Main execution loop using modular approach.
        
        Flow:
        1. Reasoning module analyzes current state
        2. Query module decides action (search or answer)
        3. If search: Execute tool, then Summarization module processes results
        4. Repeat until answer or limit reached
        """
        self.model = model
        
        # Extract question
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
        
        start_time = time.time()
        answer = data['item']['answer']
        
        # Create a LOCAL copy of module configs for this run (thread-safe)
        run_module_configs = copy.deepcopy(self.module_configs)
        
        # Override module seeds if provided in task data
        if 'module_seeds' in data:
            for module_name, seed in data['module_seeds'].items():
                if module_name in run_module_configs:
                    run_module_configs[module_name]['seed'] = seed
        
        # Override module temperatures per run if provided
        if 'module_temps' in data:
            for module_name, temp in data['module_temps'].items():
                if module_name in run_module_configs:
                    run_module_configs[module_name]['temperature'] = temp
        
        # Initialize tracking
        system_prompt = SYSTEM_PROMPT + str(today_date())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Track accumulated context (summaries only; no raw search results)
        accumulated_context = ""
        num_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0
        previous_query = ""
        previous_reasoning = ""
        
        while num_calls_available > 0:
            # Check time limit (150 minutes)
            if time.time() - start_time > 150 * 60:
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "No answer found after 2h30mins",
                    "termination": "timeout"
                }
            
            round_num += 1
            num_calls_available -= 1
            
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}")
            print(f"{'='*60}")
            
            # Step 1: Reasoning module
            print("\n[STEP 1: REASONING]")
            reasoning_context = self._build_context(
                summaries=accumulated_context,
                previous_query=previous_query,
                previous_reasoning=previous_reasoning
            )
            reasoning_response = self.call_reasoning(question, reasoning_context, run_module_configs, round_num=round_num)
            print(f"Reasoning: {reasoning_response[:500]}...")
            
            # Extract reasoning content
            reasoning_content = reasoning_response
            if "<reasoning>" in reasoning_response and "</reasoning>" in reasoning_response:
                reasoning_content = reasoning_response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            previous_reasoning = reasoning_content
            
            # Step 2: Query module (with trial awareness)
            print("\n[STEP 2: QUERY/ACTION]")

            # If this is the absolute last turn, force final report generation
            if num_calls_available == 0:
                final_report_context = self._build_context(
                    summaries=accumulated_context,
                    previous_query=previous_query,
                    previous_reasoning=previous_reasoning
                )
                final_report = self.call_final_report(question, final_report_context, run_module_configs, round_num=round_num)
                final_report = self._ensure_non_empty_report(final_report, question, final_report_context, run_module_configs, round_num=round_num)
                combined_response = f"{reasoning_response}\n<answer>{final_report}</answer>"
                messages.append({"role": "assistant", "content": combined_response})
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": final_report,
                    "termination": "forced_final_report"
                }
            # Normal behavior
            query_context = self._build_context(
                summaries=accumulated_context,
                previous_query=previous_query
            )
            query_response = self.call_query(
                question, 
                reasoning_content, 
                query_context, 
                run_module_configs,
                remaining_trials=num_calls_available,
                max_trials=MAX_LLM_CALL_PER_RUN,
                round_num=round_num
            )

            print(f"Query response: {query_response[:500]}...")
            
            # Combine for message history
            combined_response = f"{reasoning_response}\n{query_response}"
            messages.append({"role": "assistant", "content": combined_response})
            previous_query = query_response
            
            # Check for final answer
            if '<answer>' in query_response and '</answer>' in query_response:
                # Always generate a full final report for consistency
                final_report_context = self._build_context(
                    summaries=accumulated_context,
                    previous_query=previous_query,
                    previous_reasoning=previous_reasoning
                )
                final_report = self.call_final_report(question, final_report_context, run_module_configs, round_num=round_num)
                final_report = self._ensure_non_empty_report(final_report, question, final_report_context, run_module_configs, round_num=round_num)
                messages.append({"role": "assistant", "content": f"<answer>{final_report}</answer>"})
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": final_report,
                    "termination": "answer"
                }
            
            # Process tool call
            if '<tool_call>' in query_response and '</tool_call>' in query_response:
                try:
                    parsed = self._extract_tool_call(query_response)
                    if not parsed:
                        raise ValueError("Invalid tool call format.")
                    tool_name, tool_args = parsed
                    if tool_name == "search" and self.mitigation_config.get("use_ensemble", False):
                        queries = self.generate_queries(
                            question,
                            reasoning_content,
                            accumulated_context,
                            run_module_configs,
                            remaining_trials=num_calls_available,
                            max_trials=MAX_LLM_CALL_PER_RUN,
                            base_response=query_response,
                            round_num=round_num
                        )
                        if queries:
                            tool_args = {"query": queries}
                    
                    print(f"\n[EXECUTING TOOL: {tool_name}]")
                    raw_result = self.custom_call_tool(tool_name, tool_args)
                    
                    # Step 3: Summarization module
                    print("\n[STEP 3: SUMMARIZATION]")
                    summary = self.call_summarization(question, raw_result, run_module_configs, round_num=round_num)
                    print(f"Summary: {summary[:500]}...")
                    
                    # Update accumulated context with the summary (strip tags for context)
                    summary_content = summary
                    if "<summarize>" in summary and "</summarize>" in summary:
                        summary_content = summary.split("<summarize>")[1].split("</summarize>")[0].strip()
                    accumulated_context += f"\n\n## Search Round {round_num}\n{summary_content}"
                    
                    # Add tool response to messages (summary already wrapped in <summarize> tags)
                    tool_response = f"<tool_response>\n{raw_result}\n</tool_response>\n\n{summary}"
                    messages.append({"role": "user", "content": tool_response})
                    
                except Exception as e:
                    error_msg = f"Error executing tool call: {e}"
                    print(error_msg)
                    messages.append({"role": "user", "content": f"<tool_response>\n{error_msg}\n</tool_response>"})
            
            # Check token limit
            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages, exclude_tool_response=True)
            print(f"Round {round_num}, Token count: {token_count}")
            
            if token_count > max_tokens:
                print(f"Token limit exceeded: {token_count} > {max_tokens}")
                final_report_context = self._build_context(
                    summaries=accumulated_context,
                    previous_query=previous_query,
                    previous_reasoning=previous_reasoning
                )
                final_report = self.call_final_report(question, final_report_context, run_module_configs, round_num=round_num)
                final_report = self._ensure_non_empty_report(final_report, question, final_report_context, run_module_configs, round_num=round_num)
                messages.append({"role": "assistant", "content": f"<answer>{final_report}</answer>"})
                prediction = final_report
                termination = "token_limit_final_report"
                
                return {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
        
        # Exceeded call limit - synthesize or force structured final report
        if self.mitigation_config.get("use_structure", False):
            final_report_context = self._build_context(
                summaries=accumulated_context,
                previous_query=previous_query,
                previous_reasoning=previous_reasoning
            )
            final_report = self.call_final_report(question, final_report_context, run_module_configs, round_num=round_num)
            final_report = self._ensure_non_empty_report(final_report, question, final_report_context, run_module_configs, round_num=round_num)
            messages.append({"role": "assistant", "content": f"<answer>{final_report}</answer>"})
            prediction = final_report
            termination = "call_limit_final_report"
        else:
            print("\n[FINAL SYNTHESIS: Extracting answer from accumulated context]")
            synthesis_response = self.call_final_synthesis(
                question, 
                accumulated_context, 
                run_module_configs,
                round_num=round_num
            )
            print(f"Synthesis response: {synthesis_response[:500]}...")
            messages.append({"role": "assistant", "content": f"[FINAL SYNTHESIS]\n{synthesis_response}"})
            
            # Extract answer from synthesis
            if '<answer>' in synthesis_response and '</answer>' in synthesis_response:
                prediction = synthesis_response.split('<answer>')[1].split('</answer>')[0].strip()
                termination = "call_limit_synthesized"
            else:
                # Try to use the whole response as the answer
                prediction = synthesis_response.strip()
                termination = "call_limit_synthesized_no_tags"
        
        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }


# Legacy compatibility: wrapper that mimics original agent interface
class MultiTurnReactAgent(ModularReactAgent):
    """
    Backward-compatible wrapper for the modular agent.
    Can be used as a drop-in replacement with the same interface.
    """
    pass
