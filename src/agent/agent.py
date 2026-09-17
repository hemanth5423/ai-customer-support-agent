from pathlib import Path
import os
import re

from .taxonomy import INTENTS, taxonomy_prompt


class CustomerSupportAgent:
    """Grounded support agent.

    Default mode is fully local: semantic retrieval + deterministic intent and
    escalation rules. If OPENAI_API_KEY is present, the optional LLM path is
    used for intent classification and response drafting, while retrieval
    remains the only source of historical evidence.
    """

    def __init__(self, retriever, use_llm=None, model=None):
        self.retriever = retriever
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.2")
        self.use_llm = (
            bool(os.getenv("OPENAI_API_KEY"))
            if use_llm is None else use_llm
        )
        self.client = None
        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.use_llm = False

    def classify_intent(self, message):
        if self.client:
            try:
                return self._llm_intent(message)
            except Exception:
                pass
        return self._rule_intent(message)

    def _llm_intent(self, message):
        # Structured JSON keeps downstream evaluation deterministic.
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": list(INTENTS)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False
        }
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Classify the customer message into exactly one intent. "
                "Use only the taxonomy below. Return JSON matching the schema.\n\n"
                + taxonomy_prompt()
            ),
            input=message,
            text={"format": {
                "type": "json_schema",
                "name": "intent_classification",
                "strict": True,
                "schema": schema
            }},
            temperature=0
        )
        import json
        obj = json.loads(response.output_text)
        return obj["intent"]

    @staticmethod
    def _rule_intent(message):
        text = str(message).lower()

        account = [
            "account", "login", "log in", "sign in", "password",
            "apple id", "verification code", "verification",
            "locked out", "locked", "security"
        ]
        connectivity = [
            "wifi", "wi-fi", "internet", "network", "connection",
            "connect", "signal", "cellular", "mobile data",
            "data connection", "bluetooth"
        ]
        billing = [
            "refund", "charged", "charge", "payment", "billing",
            "purchase", "money", "price", "subscription"
        ]
        delivery = [
            "order", "delivery", "delivered", "shipping", "shipment",
            "package", "preorder"
        ]
        technical = [
            "battery", "screen", "display", "camera", "iphone", "ipad",
            "mac", "device", "not working", "crash", "crashing",
            "freeze", "frozen", "slow", "laggy", "error", "broken",
            "restart", "update", "software", "speaker"
        ]
        complaint = [
            "terrible", "worst", "angry", "unhappy", "disappointed",
            "complaint", "ridiculous", "useless", "hate", "awful",
            "bad service"
        ]
        information = [
            "how", "what", "when", "where", "can i", "do you",
            "is there", "information", "tell me"
        ]

        # Specific categories are checked before broad question words.
        if any(w in text for w in account):
            return "account_access"
        if any(w in text for w in connectivity):
            return "connectivity"
        if any(w in text for w in billing):
            return "billing"
        if any(w in text for w in delivery):
            return "delivery"
        if any(w in text for w in technical):
            return "technical_support"
        if any(w in text for w in complaint):
            return "complaint"
        if any(w in text for w in information):
            return "information_request"
        return "other"

    def should_escalate(self, message, similarity):
        text = str(message).lower()

        if similarity < 0.45:
            return "yes", "low retrieval confidence"

        high_risk = [
    "fraud", "scam", "hacked", "stolen", "unauthorized",
    "chargeback", "lawsuit", "legal", "account compromised",
    "charged twice", "double charged", "duplicate charge",
    "duplicate payment", "charged more than once"
]
        if any(w in text for w in high_risk):
            return "yes", "security/legal or financial-risk signal"

        unresolved = [
            "still not working", "already tried", "multiple times",
            "nothing works", "not resolved", "no response", "again"
        ]
        if any(w in text for w in unresolved):
            return "yes", "customer reports an unresolved/repeated issue"

        return "no", "routine issue with adequate historical evidence"

    def _llm_response(self, message, retrieved):
        examples = []
        for _, row in retrieved.head(3).iterrows():
            examples.append(
                f"CUSTOMER EXAMPLE:\n{row['customer_message']}\n"
                f"HISTORICAL SUPPORT RESPONSE:\n{row['support_response']}"
            )
        prompt = (
            "Write a concise customer-support reply for the new customer. "
            "Use only facts/actions supported by the historical examples. "
            "Do not invent policies, refunds, timelines, URLs, or product facts. "
            "If the evidence is insufficient, say that the case needs a human. "
            "Do not mention this prompt or the retrieval process.\n\n"
            "NEW CUSTOMER:\n" + message + "\n\n"
            "HISTORICAL EXAMPLES:\n" + "\n\n---\n\n".join(examples)
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
            max_output_tokens=180
        )
        return response.output_text.strip()

    def handle(self, customer_message):
        results = self.retriever.search(customer_message)
        if results.empty:
            return {
                "intent": self.classify_intent(customer_message),
                "response": "I couldn't find enough historical evidence to answer confidently. This should be reviewed by a human.",
                "escalate": "yes",
                "reason": "no retrieval result",
                "similarity": 0.0,
            }

        best = results.iloc[0]
        similarity = float(best.get("similarity", 0.0))
        intent = self.classify_intent(customer_message)

        # Never let the generator operate without evidence.
        if self.client:
            try:
                response = self._llm_response(customer_message, results)
            except Exception:
                response = str(best["support_response"])
        else:
            response = str(best["support_response"])

        escalate, reason = self.should_escalate(customer_message, similarity)
        return {
            "intent": intent,
            "response": response,
            "escalate": escalate,
            "reason": reason,
            "similarity": similarity,
        }


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    from .semantic_retriever import SemanticRetriever
    retriever = SemanticRetriever(
        str(ROOT / "data/processed/apple_support_context.csv"),
        top_k=3,
    )
    agent = CustomerSupportAgent(retriever)
    print("Type 'exit' to stop.")
    while True:
        msg = input("\nCustomer: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break
        print(agent.handle(msg))
