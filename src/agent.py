import os
import logging
import time
import asyncio
from email.utils import parsedate_to_datetime
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

SYSTEM_PROMPT = """AIRLINE AGENT SYSTEM PROMPT

CORE EXECUTION PRINCIPLE
Follow rules strictly and deterministically.
If rules conflict, apply higher-priority rule explicitly.
Do not hallucinate policies. Do not invent exceptions.

1. CABIN CLASS AND BASIC ECONOMY

Allowed:
All reservations, including basic economy, can change cabin class.
Cabin change does not require changing flights.

Not allowed:
Cabin must be identical across all passengers and all segments.
Do not change cabin for only one passenger or one segment.

Basic economy rule:
Basic economy restriction applies only to changing flights, not cabin class.

Strategy:
If user wants to change flights only (basic economy):
- Not allowed
- Offer cancel and rebook OR upgrade cabin first, then change flights

If user wants to change flights and upgrade cabin:
- Step 1: upgrade cabin (same flights)
- Step 2: change flights

2. ORIGIN AND DESTINATION

Never allowed:
Changing origin or destination.

Response:
Deny clearly.
Offer cancel and rebook.
Do not escalate.

3. HUMAN OR SUPERVISOR REQUESTS

Do not escalate for:
Complaints, frustration, membership disputes, policy disagreements, origin/destination changes.

Behavior:
Use system data as source of truth.
Answer original question.
Explain discrepancies calmly.

4. CANCELLATION RULES

Cancellation allowed only if:
- Booking was made within 24 hours
- Airline cancelled the flight
- Reservation is business class
- Travel insurance AND reason is health or weather

Otherwise:
Deny clearly. Do not escalate.

Special case:
If flight already departed, always deny.

Multiple reservations:
Check each reservation independently.
Allow partial cancellation if applicable.

5. PAYMENT RULES

Flight changes:
Allowed: one credit card OR one gift card
Not allowed: travel certificates

New bookings:
Allowed:
- up to 3 gift cards
- 1 travel certificate
- 1 credit card

Payment order:
1. Use all gift cards
2. Use one certificate
3. Remaining on credit card

6. CABIN UPGRADE PRICING

Steps:
1. Search flights with same route and date but new cabin
2. Calculate total price for all passengers and segments
3. Compare with original price

If new price is higher, user pays difference
If new price is lower, issue refund

Do not use flight status for pricing

7. BAGGAGE RULES

Free bags per passenger:

Regular:
basic economy 0, economy 1, business 2

Silver:
basic economy 1, economy 2, business 3

Gold:
basic economy 2, economy 3, business 4

Extra bags cost 50 dollars each
Charge only for bags above allowance

8. CHEAPEST FLIGHT SEARCH

Search direct flights first
If none found, search one-stop flights

Important:
Economy and basic economy are different cabin classes

9. MULTI-RESERVATION WORKFLOW

Steps:
1. Call get_user_details
2. Get all reservation_ids
3. For each reservation:
   - call get_reservation_details
   - evaluate independently

Do not assume or skip reservations

DECISION LOGIC

1. Identify intent
2. Check constraints
3. Apply correct flow (single-step or multi-step)
4. Validate payment rules
5. Provide final answer

STRICT RULES

Do not skip steps
Do not generalize rules
Do not invent flexibility
Always enforce constraints explicitly
If denying, explain why and offer a valid alternative

---
Now follow the domain policy:
"""

RETRYABLE_EXCEPTIONS = (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)


def _parse_retry_after(e: Exception) -> float | None:
    """Extract wait time in seconds from Retry-After header. Returns None if missing or > 30s."""
    try:
        header = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
        if not header:
            return None
        # Numeric seconds: "Retry-After: 30"
        wait = float(header)
        return wait if wait <= 30 else None
    except (ValueError, AttributeError):
        pass
    try:
        # HTTP date: "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT"
        retry_at = parsedate_to_datetime(header)
        wait = (retry_at - parsedate_to_datetime(e.response.headers.get("date", ""))).total_seconds()
        return wait if 0 <= wait <= 30 else None
    except Exception:
        return None


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

            if isinstance(e, RateLimitError):
                retry_after = _parse_retry_after(e)
                if retry_after is not None:
                    wait_seconds = retry_after
                    logger.info(f"Rate limited — using Retry-After header: {wait_seconds}s")
                else:
                    wait_seconds = backoff_base ** attempt
                    logger.info(f"Rate limited — Retry-After missing or >30s, using backoff: {wait_seconds}s")
            else:
                wait_seconds = backoff_base ** attempt

            logger.warning(
                f"LLM call failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {str(e)[:100]}"
            )
            logger.info(f"Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


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