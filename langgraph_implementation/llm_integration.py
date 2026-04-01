"""
LLM integration for the Elysia LangGraph system.
Uses LangChain when available and falls back to the project's OpenAI client.
"""

import asyncio
import json
import os
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DecisionOutput(BaseModel):
    """Structured output for decision-making."""
    action: str = Field(description="The selected action/tool name")
    reasoning: str = Field(description="Detailed reasoning for the choice")
    confidence: float = Field(description="Confidence score 0-1")
    parameters: dict = Field(default_factory=dict, description="Parameters for the action")
    end_actions: bool = Field(default=False, description="Whether to stop after this action")
    impossible: bool = Field(default=False, description="Whether the task is judged impossible")


class DecisionMaker:
    """
    Handles LLM-based decision-making for the Elysia tree.
    Mirrors the DSPy signatures and ChainOfThought logic in Elysia.
    """
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview", llm_client=None):
        """
        Args:
            model_name: The LLM model to use
        """
        self.model_name = model_name
        self._external_client = llm_client
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM, preferring an injected client over LangChain."""
        self.llm = None
        self.decision_parser = None
        self.raw_client = None

        if self._external_client is not None:
            if hasattr(self._external_client, "chat") and hasattr(self._external_client.chat, "completions"):
                self.raw_client = self._external_client
                return
            self.llm = self._external_client
            return

        try:
            from langchain.chat_models import ChatOpenAI
            from langchain.output_parsers import PydanticOutputParser
            
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1000,
            )
            
            self.decision_parser = PydanticOutputParser(pydantic_object=DecisionOutput)
            
        except ImportError:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                try:
                    from openai import OpenAI
                    self.raw_client = OpenAI(api_key=api_key)
                    return
                except Exception:
                    pass
            print("WARNING: LangChain not available, using mock LLM")
    
    async def decide(
        self,
        context: dict[str, Any],
        previous_failures: Optional[list[dict]] = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        Make a decision using the LLM.
        Mirrors the DecisionPrompt in Elysia with internal retries for validation.
        
        Args:
            context: Decision context (available actions, user prompt, etc.)
            previous_failures: Previous failed attempts to learn from
            max_retries: Maximum number of internal retries before falling back
        
        Returns:
            Decision with action, reasoning, and parameters
        """
        
        if self.llm is None and self.raw_client is None:
            return self._mock_decision(context, previous_failures)

        if self.raw_client is not None:
            return await self._decide_with_openai(context, previous_failures, max_retries)

        from langchain.schema import SystemMessage, HumanMessage
        
        current_failures = list(previous_failures) if previous_failures else []
        
        for attempt in range(max_retries):
            try:
                # Build the prompt
                prompt = self._build_decision_prompt(context, current_failures)
                
                messages = [
                    SystemMessage(content="""You are a routing agent (Elly) for an agentic retrieval system.
Your role is to choose the best action to take next based on available tools and the user's query.

Available tools:
- query: Search collections for specific information
- aggregate: Perform statistical operations
- summarize: Summarize retrieved information
- text_response: Generate a text answer to the user

Analyze the user's query, what information you already have, and what tools are available.
Choose the action that best progresses toward answering the user's query.
Explain your reasoning clearly.
MUST ONLY return an action from the Available Actions list.

You MUST also decide:
- end_actions: set to True if you believe all useful actions are exhausted AFTER this action completes.
- impossible: set to True ONLY if the task is clearly impossible given the available tools."""),
                    HumanMessage(content=prompt),
                ]
                
                response = await self.llm.apredict_messages(messages)
                
                # Parse the response strictly
                decision = self._parse_decision_response(
                    response.content,
                    context,
                    strict=True,
                )
                
                return decision
                
            except Exception as e:
                print(f"Error in LLM decision (attempt {attempt+1}/{max_retries}): {e}")
                current_failures.append({
                    "error_message": f"Previous attempt failed validation: {str(e)}."
                })
        
        print("Max retries reached during decision making, falling back to mock/default decision")
        return self._mock_decision(context, previous_failures)
    
    async def generate_response(
        self,
        user_prompt: str,
        environment: dict[str, Any],
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a response to the user using retrieved information.
        Mirrors the summarization/text generation in Elysia.
        
        Args:
            user_prompt: The original user query
            environment: Retrieved objects from tools
            conversation_history: Conversation context
        
        Returns:
            Generated response text
        """
        
        if self.llm is None and self.raw_client is None:
            return self._mock_response(user_prompt, environment)

        if self.raw_client is not None:
            return await self._generate_response_with_openai(
                user_prompt,
                environment,
                conversation_history,
            )

        try:
            # Format retrieved information
            retrieved_info = self._format_retrieved_info(environment)
            
            # Build the response prompt
            response_prompt = f"""Based on the following retrieved information, provide a comprehensive answer to the user's query.

User Query: {user_prompt}

Retrieved Information:
{retrieved_info}

Please provide a clear, concise, and helpful answer."""
            
            from langchain.schema import HumanMessage
            
            response = await self.llm.apredict(response_prompt)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._mock_response(user_prompt, environment)

    async def _decide_with_openai(
        self,
        context: dict[str, Any],
        previous_failures: Optional[list[dict]] = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Use the raw OpenAI client with retry loop mirroring AssertedModule."""
        available_actions = list(context.get("available_actions", {}).keys())
        if not available_actions:
            return self._mock_decision(context, previous_failures)

        current_failures = list(previous_failures) if previous_failures else []

        for attempt in range(max_retries):
            prompt = self._build_decision_prompt(context, current_failures)
            schema_hint = {
                "action": "<one of: " + ", ".join(available_actions) + ">",
                "reasoning": "Short explanation tied to available tools and current context.",
                "confidence": 0.7,
                "parameters": {},
                "end_actions": False,
                "impossible": False,
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an Elysia-style routing agent. "
                        f"You must choose exactly one action from: {', '.join(available_actions)}. "
                        "Set end_actions to true if you believe all useful actions are exhausted after this one. "
                        "Set impossible to true ONLY if the task cannot be done with available tools. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "Respond with a JSON object using this shape:\n"
                        f"{json.dumps(schema_hint, ensure_ascii=False)}"
                    ),
                },
            ]

            try:
                response = await asyncio.to_thread(
                    self.raw_client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
            except Exception as e:
                print(f"Error in raw OpenAI decision (attempt {attempt+1}/{max_retries}): {e}")
                current_failures.append({"error_message": f"API call failed: {e}"})
                continue

            action = parsed.get("action")

            # Assertion: action must be in available_actions (mirrors _tool_assertion)
            if action not in available_actions:
                feedback = (
                    f"You picked '{action}' — that is NOT in available_actions! "
                    f"Your output MUST be one of: {available_actions}"
                )
                print(f"AssertedModule feedback (attempt {attempt+1}): {feedback}")
                current_failures.append({"error_message": feedback})
                continue

            # Guard: don't text_response/summarize when nothing retrieved
            retrieved_summary = context.get("retrieved_objects_summary", "")
            no_results_yet = (
                "No objects retrieved yet." in retrieved_summary
                or "No retrieved objects yet." in retrieved_summary
            )
            if no_results_yet:
                if "search" in available_actions and action in {"text_response", "summarize"}:
                    action = "search"
                elif "query" in available_actions and action in {"text_response", "summarize"}:
                    action = "query"

            confidence = parsed.get("confidence", 0.7)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.7

            return {
                "action": action,
                "reasoning": parsed.get("reasoning", f"Selected {action}"),
                "confidence": max(0.0, min(1.0, confidence)),
                "parameters": parsed.get("parameters", {}) or {},
                "end_actions": bool(parsed.get("end_actions", False)),
                "impossible": bool(parsed.get("impossible", False)),
            }

        # All retries exhausted
        print("Max retries reached in _decide_with_openai, falling back to mock")
        return self._mock_decision(context, previous_failures)

    async def _generate_response_with_openai(
        self,
        user_prompt: str,
        environment: dict[str, Any],
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a grounded response from the Elysia environment."""
        retrieved_info = self._format_retrieved_info(environment)
        history_text = ""
        if conversation_history:
            last_turns = conversation_history[-4:]
            history_text = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}"
                for item in last_turns
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise RAG assistant. "
                    "Answer using only the retrieved information provided. "
                    "If the answer is missing from the retrieved information, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation context:\n{history_text or 'None'}\n\n"
                    f"User query: {user_prompt}\n\n"
                    f"Retrieved information:\n{retrieved_info}"
                ),
            },
        ]

        try:
            response = await asyncio.to_thread(
                self.raw_client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1200,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"Error generating raw OpenAI response: {e}")
            return self._mock_response(user_prompt, environment)
    
    def _build_decision_prompt(
        self,
        context: dict[str, Any],
        previous_failures: Optional[list[dict]],
    ) -> str:
        """Build the decision prompt for the LLM."""
        
        prompt = f"""
User Query: {context.get('user_prompt', '')}

Current Status:
- Tree depth: {context.get('tree_count', '0/5')}
- Retrieved objects: {context.get('retrieved_objects_summary', 'None yet')}
"""
        tasks_completed = context.get('tasks_completed_summary')
        if tasks_completed and tasks_completed != "No tasks completed yet.":
            prompt += f"\nTasks Completed So Far:\n{tasks_completed}\n"
            
        prompt += "\nAvailable Actions:\n"
        
        for action_name, action_info in context.get('available_actions', {}).items():
            prompt += f"\n- {action_name}: {action_info.get('description', '')}"
        
        if previous_failures:
            prompt += "\n\nPrevious Errors (learn from these):\n"
            for error in previous_failures[-3:]:  # Last 3 errors
                prompt += f"- {error.get('error_message', '')}\n"
        
        prompt += "\n\nWhat action should be taken next? Explain your reasoning."
        
        return prompt
    
    def _format_retrieved_info(self, environment: dict[str, Any]) -> str:
        """Format retrieved information for readability."""
        if not environment:
            return "No information retrieved yet."
        
        formatted = ""
        for tool_name, collections_dict in environment.items():
            for collection_name, results_list in collections_dict.items():
                for result in results_list:
                    formatted += f"\n**From {tool_name} ({collection_name}):**\n"
                    for obj in result.objects:
                        props = obj.properties
                        if "page_content" in props:
                            source = props.get("source_name") or props.get("source") or collection_name
                            page = props.get("page_idx")
                            page_txt = f" page {page + 1}" if isinstance(page, int) else ""
                            formatted += f"- Source: {source}{page_txt}\n"
                            title = props.get("title_path")
                            if title:
                                formatted += f"  Title: {title}\n"
                            formatted += f"  Content: {props.get('page_content', '')}\n"
                        else:
                            for key, value in props.items():
                                formatted += f"- {key}: {value}\n"
        
        return formatted

    def _parse_decision_response(
        self,
        response_text: str,
        context: dict[str, Any],
        strict: bool = False,
    ) -> dict[str, Any]:
        """Parse the LLM response into a structured decision."""
        
        available_actions = list(context.get('available_actions', {}).keys())
        
        action = None
        for candidate_action in available_actions:
            if candidate_action.lower() in response_text.lower():
                action = candidate_action
                break
                
        if strict and not action:
            raise ValueError(f"Could not find any of the available actions {available_actions} in the model response text.")
        
        if not action and available_actions:
            action = available_actions[0]
        
        return {
            "action": action or "text_response",
            "reasoning": response_text,
            "confidence": 0.7,
            "parameters": {},
            "end_actions": "end_actions" in response_text.lower() and "true" in response_text.lower(),
            "impossible": "impossible" in response_text.lower() and "true" in response_text.lower(),
        }

        
    def _mock_decision(
        self,
        context: dict[str, Any],
        previous_failures: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Create a mock decision mirroring Elysia's decision logic for demo/testing."""

        available_actions = list(context.get('available_actions', {}).keys())

        retrieved_summary = context.get("retrieved_objects_summary", "")
        no_results_yet = (
            "No objects retrieved yet." in retrieved_summary
            or "No retrieved objects yet." in retrieved_summary
        )
        has_results = not no_results_yet

        # Decision logic:
        # 1. If no results → search (or query)
        # 2. If results exist → text_response with end_actions=True
        if no_results_yet and "search" in available_actions:
            action = "search"
            end_actions = False
        elif no_results_yet and "query" in available_actions:
            action = "query"
            end_actions = False
        elif has_results and "text_response" in available_actions:
            action = "text_response"
            end_actions = True
        elif has_results and "summarize" in available_actions:
            action = "summarize"
            end_actions = True
        elif available_actions:
            action = available_actions[0]
            end_actions = False
        else:
            action = "text_response"
            end_actions = True

        return {
            "action": action,
            "reasoning": f"Mock decision: chose {action} (has_results={has_results})",
            "confidence": 0.5,
            "parameters": {},
            "end_actions": end_actions,
            "impossible": False,
        }

    def _mock_response(
        self,
        user_prompt: str,
        environment: dict[str, Any],
    ) -> str:
        """Create a mock response (for demo/testing)."""
        
        num_objects = sum(
            len(result.objects)
            for tool_results in environment.values()
            for results in tool_results.values()
            for result in results
        ) if environment else 0
        
        if num_objects == 0:
            return f"I found no results for: {user_prompt}"
        else:
            return f"Based on {num_objects} retrieved objects, here's what I found about '{user_prompt}': [detailed information would be shown here]"


class DSPyCompatibleModule:
    """
    A wrapper to make LangGraph compatible with DSPy-style signatures.
    Useful for gradual migration from Elysia to LangGraph.
    """
    
    def __init__(self, signature_class, llm_client=None):
        """
        Args:
            signature_class: A DSPy-like signature (Pydantic BaseModel)
            llm_client: LLM client to use
        """
        self.signature = signature_class
        self.llm_client = llm_client
        self.decision_maker = DecisionMaker()
    
    async def aforward(self, **kwargs) -> dict[str, Any]:
        """
        Forward pass using LLM (async).
        Maps to DSPy's aforward semantics.
        """
        # Extract inputs based on signature
        input_fields = {
            k: v for k, v in kwargs.items()
            if not k.startswith("_")
        }
        
        # Call LLM
        result = await self.decision_maker.decide(
            context=input_fields,
        )
        
        return result


# Example DSPy-like signatures that can be used with the LangGraph system
class QueryCreatorSignature(BaseModel):
    """Similar to elysia/tools/retrieval/prompt_templates.py::QueryCreatorPrompt"""
    user_prompt: str
    available_collections: list[str]
    collection_schemas: dict[str, Any]
    previous_queries: list[str]
    
    # Output fields
    reasoning: str
    query_output: dict[str, Any]


class AggregationSignature(BaseModel):
    """Similar to elysia/tools/retrieval/prompt_templates.py::AggregationPrompt"""
    user_prompt: str
    available_collections: list[str]
    collection_schemas: dict[str, Any]
    
    # Output fields
    reasoning: str
    aggregation_output: dict[str, Any]
