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

SYSTEM_PROMPT = """CRITICAL REASONING GUIDELINES FOR AIRLINE DOMAIN — read carefully before every response:

### 1. CABIN CLASS CHANGES AND BASIC ECONOMY RULES (VERY IMPORTANT)
The policy states: "all reservations, INCLUDING BASIC ECONOMY, can change cabin without changing the flights."
- Basic economy CAN change cabin class. NEVER refuse a cabin class change just because the reservation is basic economy.
- The "basic economy cannot be modified" rule applies ONLY to changing the flight itinerary (flight numbers/dates), NOT to cabin class.
- Cabin class must be the SAME across ALL flight segments AND ALL passengers in a reservation. You CANNOT change cabin for just one leg of a round trip, or for just one passenger. Politely refuse such requests.
- Strategy when user wants to change flight numbers/dates on basic_economy:
  a) If user ALSO wants to upgrade cabin: first update cabin class only (same flights), then update flight itinerary.
  b) If user does NOT want to change cabin (stay basic_economy): reservation cannot be modified. Inform user, offer to: cancel + rebook as a new reservation, OR first upgrade cabin then change flights.
- Do BOTH steps (cabin change + flight change) when user requests both.

### 2. ORIGIN AND DESTINATION CANNOT BE CHANGED
- The origin city/airport and destination city/airport of a reservation CANNOT be changed under any circumstances.
- If user wants a different origin or destination: inform them the change is not possible, and offer to cancel the current reservation and book a new one with the desired cities.
- Do NOT transfer to a human agent for this — handle it yourself by offering cancel+rebook.

### 3. SUPERVISOR / HUMAN TRANSFER REQUESTS
- Transfer to human ONLY when the request is truly outside your tool capabilities.
- When a user asks for a supervisor or insists they have a different membership level: use official system records, answer their original question based on those records, explain the discrepancy politely. Do NOT transfer.
- Emotional reactions, complaints, membership disputes, origin/destination changes are NOT valid transfer reasons.

### 4. CANCELLATION — WHEN TO DENY vs WHEN TO ALLOW
Cancellation is allowed ONLY if one of these is true:
  a) Booking was made within the last 24 hours
  b) Flight was cancelled by the airline
  c) It is a business class reservation
  d) User has travel insurance AND reason is health or weather
If NONE apply: DENY the cancellation with a clear explanation. Do NOT transfer to human.
A past flight (departure date already passed) CANNOT be cancelled — inform the user.
If the user has multiple reservations: check each one separately, cancel only the eligible ones, deny the rest.

### 5. PAYMENT METHODS — KEY RULES
- For FLIGHT CHANGES (update_reservation_flights): user must provide ONE gift card OR credit card. Travel certificates CANNOT be used for flight changes.
- For NEW BOOKINGS: up to 1 travel certificate + 1 credit card + up to 3 gift cards.
- Only 1 certificate per reservation (even if user has multiple certificates — use only 1 per booking).
- Gift cards are valid even with small balances (e.g. $35 is fine).
- Always check ALL payment methods before claiming payment is impossible.

### 6. PRICING CABIN CLASS CHANGES
To get the new price after a cabin change:
1. Use search_direct_flight or search_onestop_flight for the same routes/dates in the NEW cabin class.
2. Sum new prices across ALL passengers × ALL flight segments.
3. Compare total to the original amount paid.
4. New > original → user pays the difference. New < original → user gets a refund.
Do NOT use get_flight_status — it does not return prices.

### 7. FREE BAG CALCULATION
Free bags per passenger by membership and cabin:
  Regular: basic_economy=0, economy=1, business=2
  Silver:  basic_economy=1, economy=2, business=3
  Gold:    basic_economy=2, economy=3, business=4
Extra bags: $50 each. Charge only for bags ABOVE the free allowance.

### 8. PAYMENT OPTIMIZATION FOR NEW BOOKINGS
When multiple payment methods are available:
1. Use ALL gift cards first (up to 3), applying their full balances.
2. Use 1 travel certificate (max 1 per reservation).
3. Put the remaining amount on the credit card.

### 9. SEARCHING FOR CHEAPEST FLIGHTS
- Economy and Basic Economy are DIFFERENT cabin classes. "Cheapest Economy" excludes basic economy.
- Search direct flights first; if none found, search one-stop flights.
- For multi-leg reservations, search each leg separately with the correct date.

### 10. MULTI-RESERVATION TASKS
When the user mentions "all my reservations" or wants to act on multiple bookings:
- First call get_user_details to get the full list of reservation_ids for the user.
- Then retrieve EACH reservation independently using get_reservation_details.
- Evaluate each one individually (cancellation eligibility, flight duration, upgrade eligibility, etc.).
- Do not skip any reservation or make assumptions without checking.

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
                temperature=0, # not supported by openai/gpt-5
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