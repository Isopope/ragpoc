"""Couche d'intégration LLM du package rag_agent.

Contient :
- make_llm_caller()    : factory retournant un appel LLM avec timeout (threading)
- make_embedder()      : factory retournant un embedding avec timeout
- parse_json_llm()     : parser JSON robuste à 4 stratégies de fallback
- PlanningOutput       : modèle Pydantic pour le noeud analyze_and_plan
- DecisionOutput       : modèle Pydantic pour les décisions de l'arbre (impl B)
- DecisionMaker        : orchestrateur de décision sans dépendance LangChain
"""
from __future__ import annotations

import ast
import json
import re
import threading
from typing import Any, Callable, Optional
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# ── Helpers timeout ────────────────────────────────────────────────────────────

def make_llm_caller(client, model: str, timeout: float) -> Callable:
    """Retourne une fonction d'appel LLM via LiteLLM avec timeout."""
    def _call(messages: list, **kwargs) -> Any:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from llm import get_llm_completion
        
        # LITELLM automatically manages providers
        api_key = getattr(client, "api_key", None) if client else None
        
        # Extraire `response_format` si nécessaire pour la compatibilité avec certains modèles (comme OpenAI)
        response_format = kwargs.pop("response_format", None)
        
        # Pour les modèles qui permettent `response_format`
        if response_format and model.startswith("gpt"):
           kwargs["response_format"] = response_format
           
        resp = get_llm_completion(
            model=model,
            messages=messages,
            timeout=timeout,
            api_key=api_key,
            **kwargs
        )
        return resp

    return _call


def make_embedder(client, model: str, timeout: float) -> Callable:
    """Retourne une fonction d'embedding via LiteLLM."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm import make_embedder as get_embedder_factory
    
    return get_embedder_factory(client, model, timeout)

def _strip_fences(text: str) -> str:
    """Retire les balises Markdown ``` éventuellement ajoutées par le LLM."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_json_llm(text: str) -> object:
    """Parse du JSON potentiellement malformé produit par un LLM.

    Stratégies (dans l'ordre) :
    1. json.loads standard après nettoyage des fences
    2. ast.literal_eval (accepte guillemets simples, tuples Python)
    3. Extraction du premier bloc JSON par regex ({} ou [])
    4. Nettoyage des virgules trailing puis retry json.loads
    """
    if not text:
        raise ValueError("Réponse LLM vide")

    cleaned = _strip_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                pass

    no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(no_trailing)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Impossible de parser la réponse LLM : {cleaned[:200]!r}")


# ── Modèles Pydantic ───────────────────────────────────────────────────────────

class PlanningOutput(BaseModel):
    """Sortie structurée du nœud analyze_and_plan."""

    targets: list[str] = Field(
        default_factory=list,
        description="Noms de fichiers explicitement mentionnés et pertinents pour la question",
    )
    reason: str = Field(
        default="",
        description="Courte explication de la décision de planification",
    )
    sub_queries: list[str] = Field(
        description="1 à 3 sous-requêtes de récupération optimisées",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confiance dans le plan",
    )

    @field_validator("sub_queries")
    @classmethod
    def at_least_one_query(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("sub_queries doit contenir au moins une requête")
        return v[:3]  # max 3 sous-requêtes

    @field_validator("targets")
    @classmethod
    def normalize_targets(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in v:
            text = str(item).strip()
            if text and text.lower() != "null" and text not in cleaned:
                cleaned.append(text)
        return cleaned


class DecisionOutput(BaseModel):
    """Sortie structurée pour les décisions de l'arbre (porté de langgraph_implementation)."""

    action: str = Field(description="Nom de l'action à exécuter")
    reasoning: str = Field(default="", description="Raisonnement du LLM")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── DecisionMaker ──────────────────────────────────────────────────────────────

class DecisionMaker:
    """Orchestrateur de décision pour l'arbre de navigation.

    Port de langgraph_implementation/llm_integration.py sans dépendance LangChain.
    Utilise le client OpenAI natif via make_llm_caller().
    """

    def __init__(self, llm_call: Optional[Callable] = None) -> None:
        self._llm_call = llm_call

    def decide(
        self,
        context: dict,
        previous_failures: Optional[list] = None,
        max_retries: int = 3,
    ) -> DecisionOutput:
        """Choisit l'action suivante à partir du contexte de l'arbre.

        Retourne DecisionOutput. En cas d'echec, utilise _mock_decision().
        """
        if self._llm_call is None:
            return self._mock_decision(context)

        available = context.get("available_actions", [])
        prompt = self._build_decision_prompt(context, previous_failures or [])

        for attempt in range(max_retries):
            try:
                resp = self._llm_call(
                    messages=[
                        {"role": "system", "content": "Tu es un agent de décision. Réponds en JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                parsed = parse_json_llm(raw)
                if isinstance(parsed, dict):
                    action = parsed.get("action", "")
                    if action and (not available or action in available):
                        return DecisionOutput(
                            action=action,
                            reasoning=parsed.get("reasoning", ""),
                            confidence=float(parsed.get("confidence", 0.8)),
                            parameters=parsed.get("parameters", {}),
                        )
            except Exception as exc:
                logger.warning("DecisionMaker attempt {}/{}: {}", attempt + 1, max_retries, exc)

        return self._mock_decision(context)

    def generate_response(
        self,
        user_prompt: str,
        environment: dict,
        conversation_history: Optional[list] = None,
    ) -> str:
        """Génère une réponse finale à partir des informations récupérées."""
        if self._llm_call is None:
            return self._mock_response(user_prompt, environment)

        info = self._format_retrieved_info(environment)
        history_ctx = ""
        if conversation_history:
            history_ctx = "Historique de conversation :\n" + "\n".join(
                f"- {m.get('role', '?')}: {str(m.get('content', ''))[:200]}"
                for m in (conversation_history or [])[-5:]
            ) + "\n\n"

        try:
            resp = self._llm_call(
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant expert. Réponds en français de façon structurée.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{history_ctx}"
                            f"Question : {user_prompt}\n\n"
                            f"Informations récupérées :\n{info}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("DecisionMaker.generate_response: {}", exc)
            return self._mock_response(user_prompt, environment)

    # ── Helpers privés ─────────────────────────────────────────────────────────

    def _build_decision_prompt(self, context: dict, failures: list) -> str:
        instruction = context.get("instruction", "")
        user_prompt = context.get("user_prompt", "")
        available   = context.get("available_actions", [])
        successive  = context.get("successive_actions", {})
        tasks_done  = context.get("tasks_completed", "")
        env_summary = context.get("environment_summary", "")
        error_ctx   = ""
        if failures:
            error_ctx = f"\nTentatives échouées : {failures}\n"

        return (
            f"Instruction : {instruction}\n"
            f"Question utilisateur : {user_prompt}\n"
            f"Actions disponibles : {available}\n"
            f"Actions successives possibles : {successive}\n"
            f"Tâches complétées : {tasks_done}\n"
            f"Environnement actuel : {env_summary}\n"
            f"{error_ctx}\n"
            "Réponds en JSON : {\"action\": \"<nom>\", \"reasoning\": \"<why>\", \"confidence\": 0.9, \"parameters\": {}}"
        )

    def _format_retrieved_info(self, environment: dict) -> str:
        if not environment:
            return "Aucune information récupérée."
        lines = []
        for tool, collections in environment.items():
            for coll, items in collections.items():
                lines.append(f"[{tool} / {coll}]")
                for item in items[:3]:
                    lines.append(f"  - {str(item)[:200]}")
        return "\n".join(lines)

    def _mock_decision(self, context: dict) -> DecisionOutput:
        """Décision heuristique sans LLM (mode mock)."""
        available = context.get("available_actions", [])
        action = available[0] if available else "text_response"
        return DecisionOutput(
            action=action,
            reasoning="Mock decision — no LLM available",
            confidence=0.5,
        )

    def _mock_response(self, user_prompt: str, environment: dict) -> str:
        """Réponse mock sans LLM."""
        n = sum(len(v) for c in environment.values() for v in c.values())
        return f"[Mock] {n} résultat(s) trouvé(s) pour : {user_prompt[:100]}"


# ── DSPy compatibility shim ────────────────────────────────────────────────────

class DSPyCompatibleModule:
    """Wrapper minimaliste pour accepter des signatures DSPy (Pydantic BaseModels)."""

    def __init__(self, llm_call: Optional[Callable] = None) -> None:
        self._maker = DecisionMaker(llm_call)

    def forward(self, signature: type[BaseModel], **kwargs) -> DecisionOutput:
        """Exécute une décision en utilisant la signature comme contexte."""
        context = {
            "instruction": signature.__doc__ or "",
            "user_prompt": str(kwargs),
            "available_actions": list(signature.model_fields.keys()),
        }
        return self._maker.decide(context)
