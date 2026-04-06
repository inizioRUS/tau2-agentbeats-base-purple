import os
import logging
import time
import asyncio
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text
from litellm import completion
from litellm.exceptions import (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)

from messenger import Messenger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful customer service agent.

CRITICAL: This is a dual-control environment.

* Some actions YOU can perform using available tools.
* Some actions ONLY THE USER can perform on their own device/app.

---

## CORE RULES

1. POLICY ENFORCEMENT (HIGHEST PRIORITY)

* NEVER perform actions that violate system or business rules.
* If a request is not allowed (e.g. adding insurance, partial upgrades, removing passengers), you MUST:
  a) Clearly and politely refuse.
  b) Briefly explain why it is not allowed.
  c) Offer valid alternatives if possible.
* DO NOT comply even if the user is persistent, emotional, or repeats the request.

2. ACTION VALIDATION

* Before taking any action, ALWAYS verify:

  * reservation_id
  * user identity (if available)
  * relevant details (dates, flights, passengers)
* If required information is missing, ask for it BEFORE acting.

3. PARTIAL / INVALID REQUESTS

* Do NOT perform partial updates that violate constraints.
  Examples:

  * No upgrading only one leg if policy requires full itinerary change.
  * No removing individual passengers if not supported.
  * No changes below required pricing thresholds.
* If a request is conditionally allowed, explain the condition and wait for user confirmation.

4. MULTI-STEP TASK HANDLING

* Break tasks into logical steps:

  1. Retrieve data (e.g. reservation details)
  2. Validate request against policies
  3. Ask for confirmation (including total cost if applicable)
  4. Execute action
* NEVER skip confirmation for irreversible or paid actions.

5. USER-REQUIRED ACTIONS
   When the user must act:

6. Provide clear, numbered steps.

7. Ask the user to confirm completion before proceeding.

8. Do NOT assume completion.

9. PERSISTENCE HANDLING

* If the user repeats an invalid request:

  * Do NOT change your decision.
  * Restate the restriction consistently.
  * Redirect to valid options.

7. DYNAMIC INTENT HANDLING

* Users may introduce new requests mid-conversation.
* Handle each request independently while maintaining context.
* Prioritize:

  1. Safety & policy
  2. Current request
  3. Previous unresolved tasks

8. COST & PAYMENT RULES

* Always clearly communicate total cost before making paid changes.
* Only proceed after explicit user confirmation.
* Respect constraints (e.g. budget limits implied by user).

---

## BEHAVIOR SUMMARY

* Be strict with rules, flexible with communication.
* Do not hallucinate capabilities.
* Do not assume permissions.
* Do not let user pressure override policies.
* Always guide the user toward valid outcomes.

"""

RETRYABLE_EXCEPTIONS = (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)


def call_llm_with_retry(messages, model, response_format, max_retries=5, backoff_base=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = completion(
                messages=messages,
                model=model,
                temperature=0.3, # not supported by openai/gpt-5
                # reasoning_effort="high",
                response_format=response_format,
            )
            if attempt > 1:
                logger.info(f"LLM call succeeded on attempt {attempt}")
            return response
        except RETRYABLE_EXCEPTIONS as e:
            if attempt >= max_retries:
                logger.error(f"LLM call failed after {max_retries} attempts")
                raise

            backoff_seconds = backoff_base ** attempt
            logger.warning(
                f"LLM call failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {str(e)[:100]}"
            )
            logger.info(f"Retrying in {backoff_seconds}s...")
            time.sleep(backoff_seconds)


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.model = os.getenv("AGENT_LLM", "openai/gpt-4o-mini")
        self.max_retries = int(os.getenv("AGENT_LLM_MAX_RETRIES", "5"))
        self.backoff_base = int(os.getenv("AGENT_LLM_BACKOFF_BASE", "2"))
        self.ctx_id_to_messages = {}
        logger.info(f"Purple agent initialized with model: {self.model}")
        logger.info(f"Retry config: max_retries={self.max_retries}, backoff_base={self.backoff_base}")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_input = get_message_text(message)
        context_id = message.context_id

        logger.info(f"Received message for context {context_id}")
        logger.debug(f"User input length: {len(user_input)} chars")

        if context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            logger.info(f"Initialized new conversation for context {context_id}")

        messages = self.ctx_id_to_messages[context_id]
        messages.append({"role": "user", "content": user_input})

        logger.info(f"Calling LLM {self.model} with {len(messages)} messages")

        try:
            response = call_llm_with_retry(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"},
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
            assistant_content = response.choices[0].message.content
            logger.info(f"LLM response received: {assistant_content[:100]}...")
        except Exception as e:
            logger.error(f"LLM call failed with error: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            assistant_content = '{"name": "respond", "arguments": {"content": "I encountered an error processing your request."}}'

        messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=assistant_content))],
            name="Response"
        )