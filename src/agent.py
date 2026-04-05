import os
import json
import copy
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
DB_PATH = Path(__file__).resolve().parent / "db.json"
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

SYSTEM_PROMPT = """
# System rules
You are an airline customer service agent.

You must strictly follow the airline policy.

Available read tasks:
- find user by user_id
- find reservation by reservation_id
- find all reservations for a user
- search available flights by origin, destination, and date
- get flight status by flight_number and date
- get user payment methods
- get saved passengers for a user

Rules:
1. Always return valid JSON.
2. Return exactly one of the following shapes:

For a tool call:
{
  "type": "tool_call",
  "name": "<tool_name>",
  "arguments": { ... }
}

Use only these exact tool names:
get_user
get_user_payment_methods
get_saved_passengers
find_user_reservations
get_reservation
get_flight_status
search_available_flights

For a user response:
{
  "type": "respond",
  "content": "<message to user>"
}

3. Only make one tool call at a time.
4. If you make a tool call, do not also respond to the user in the same message.
5. Never invent data. Use only user-provided information or tool results.
6. If required identifiers are missing, ask the user for them.
7. Be concise and factual.
8. You will receive prior database context and tool history inside the conversation context. Use it.


# Airline Agent Policy

The current time is 2024-05-15 15:00:00 EST.

As an airline agent, you can help users **book**, **modify**, or **cancel** flight reservations. You also handle **refunds and compensation**.

Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.

You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.

You should deny user requests that are against this policy.

You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.

## Domain Basic

### User
Each user has a profile containing:
- user id
- email
- addresses
- date of birth
- payment methods
- membership level
- reservation numbers

There are three types of payment methods: **credit card**, **gift card**, **travel certificate**.

There are three membership levels: **regular**, **silver**, **gold**.

### Flight
Each flight has the following attributes:
- flight number
- origin
- destination
- scheduled departure and arrival time (local time)

A flight can be available at multiple dates. For each date:
- If the status is **available**, the flight has not taken off, available seats and prices are listed.
- If the status is **delayed** or **on time**, the flight has not taken off, cannot be booked.
- If the status is **flying**, the flight has taken off but not landed, cannot be booked.

There are three cabin classes: **basic economy**, **economy**, **business**. **basic economy** is its own class, completely distinct from **economy**.

Seat availability and prices are listed for each cabin class.

### Reservation
Each reservation specifies the following:
- reservation id
- user id
- trip type
- flights
- passengers
- payment methods
- created time
- baggages
- travel insurance information

There are two types of trip: **one way** and **round trip**.

## Book flight

The agent must first obtain the user id from the user. 

The agent should then ask for the trip type, origin, destination.

Cabin:
- Cabin class must be the same across all the flights in a reservation. 

Passengers: 
- Each reservation can have at most five passengers. 
- The agent needs to collect the first name, last name, and date of birth for each passenger. 
- All passengers must fly the same flights in the same cabin.

Payment: 
- Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards. 
- The remaining amount of a travel certificate is not refundable. 
- All payment methods must already be in user profile for safety reasons.

Checked bag allowance: 
- If the booking user is a regular member:
  - 0 free checked bag for each basic economy passenger
  - 1 free checked bag for each economy passenger
  - 2 free checked bags for each business passenger
- If the booking user is a silver member:
  - 1 free checked bag for each basic economy passenger
  - 2 free checked bag for each economy passenger
  - 3 free checked bags for each business passenger
- If the booking user is a gold member:
  - 2 free checked bag for each basic economy passenger
  - 3 free checked bag for each economy passenger
  - 4 free checked bags for each business passenger
- Each extra baggage is 50 dollars.

Do not add checked bags that the user does not need.

Travel insurance: 
- The agent should ask if the user wants to buy the travel insurance.
- The travel insurance is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.

## Modify flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

Change flights: 
- Basic economy flights cannot be modified.
- Other reservations can be modified without changing the origin, destination, and trip type.
- Some flight segments can be kept, but their prices will not be updated based on the current price.
- The API does not check these for the agent, so the agent must make sure the rules apply before calling the API!

Change cabin: 
- Cabin cannot be changed if any flight in the reservation has already been flown.
- In other cases, all reservations, including basic economy, can change cabin without changing the flights.
- Cabin class must remain the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.
- If the price after cabin change is higher than the original price, the user is required to pay for the difference.
- If the price after cabin change is lower than the original price, the user is should be refunded the difference.

Change baggage and insurance: 
- The user can add but not remove checked bags.
- The user cannot add insurance after initial booking.

Change passengers:
- The user can modify passengers but cannot modify the number of passengers.
- Even a human agent cannot modify the number of passengers.

Payment: 
- If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons.

## Cancel flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

The agent must also obtain the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

If any portion of the flight has already been flown, the agent cannot help and transfer is needed.

Otherwise, flight can be cancelled if any of the following is true:
- The booking was made within the last 24 hrs
- The flight is cancelled by airline
- It is a business flight
- The user has travel insurance and the reason for cancellation is covered by insurance.

The API does not check that cancellation rules are met, so the agent must make sure the rules apply before calling the API!

Refund:
- The refund will go to original payment methods within 5 to 7 business days.

## Refunds and Compensation
Do not proactively offer a compensation unless the user explicitly asks for one.

Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.

Always confirms the facts before offering compensation.

Only compensate if the user is a silver/gold member or has travel insurance or flies business.

- If the user complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.

- If the user complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.

Do not offer compensation for any other reason than the ones listed above.
"""

RETRYABLE_EXCEPTIONS = (
ServiceUnavailableError,
RateLimitError,
Timeout,
APIConnectionError,
)

def call_llm_with_retry(
        messages: List[Dict[str, Any]],
        model: str,
        response_format: Dict[str, Any],
        max_retries: int = 5,
        backoff_base: int = 2,
):
    for attempt in range(1, max_retries + 1):
        try:
            response = completion(
                messages=messages,
                model=model,
                temperature=0.3,
                response_format=response_format,
            )
            if attempt > 1:
                logger.info("LLM call succeeded on attempt %s", attempt)
            return response
        except RETRYABLE_EXCEPTIONS as e:
            if attempt >= max_retries:
                logger.error("LLM call failed after %s attempts", max_retries)
                raise

            backoff_seconds = backoff_base ** attempt
            logger.warning(
                "LLM call failed (attempt %s/%s): %s: %s",
                attempt,
                max_retries,
                type(e).__name__,
                str(e)[:200],
            )
            logger.info("Retrying in %ss...", backoff_seconds)
            time.sleep(backoff_seconds)


class AirlineReadOnlyDB:
    """
    Read-only in-memory database loaded from db.json.

    Expected db.json structure:
    {
      "flights": [...],
      "users": [...],
      "reservations": [...]
    }
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.raw: Dict[str, Any] = {}
        self.users: List[Dict[str, Any]] = []
        self.flights: List[Dict[str, Any]] = []
        self.reservations: List[Dict[str, Any]] = []

        # Indexes
        self.users_by_id: Dict[str, Dict[str, Any]] = {}
        self.reservations_by_id: Dict[str, Dict[str, Any]] = {}
        self.reservations_by_user_id: Dict[str, List[Dict[str, Any]]] = {}
        self.flights_by_number: Dict[str, Dict[str, Any]] = {}

        self._load()

    def _load(self) -> None:
        logger.info("Loading DB from %s", self.db_path)
        with open(self.db_path) as f:
            self.raw = json.load(f)

        self.users = self.raw.get("users", [])
        self.flights = self.raw.get("flights", [])
        self.reservations = self.raw.get("reservations", [])

        self.users_by_id = {
            u: self.users[u]
            for u in self.users
        }
        self.reservations_by_id = {
            r: self.reservations[r]
            for r in self.reservations
        }
        self.reservations_by_user_id = {}
        for r in self.reservations:
            user_id = self.reservations[r]['user_id']
            if not user_id:
                continue
            self.reservations_by_user_id.setdefault(user_id, []).append(r)

        self.flights_by_number = {
            f: self.flights[f]
            for f in self.flights
        }
        logger.info(
            "DB loaded: %s users, %s flights, %s reservations",
            len(self.users),
            len(self.flights),
            len(self.reservations),
        )

    @staticmethod
    def _deepcopy(data: Any) -> Any:
        return copy.deepcopy(data)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        user = self.users_by_id.get(user_id)
        if not user:
            return {"found": False, "user_id": user_id}

        return {
            "found": True,
            "user": self._deepcopy(user),
        }

    def get_user_payment_methods(self, user_id: str) -> Dict[str, Any]:
        user = self.users_by_id.get(user_id)
        if not user:
            return {"found": False, "user_id": user_id}

        payment_methods = list((user.get("payment_methods") or {}).values())
        return {
            "found": True,
            "user_id": user_id,
            "payment_methods": self._deepcopy(payment_methods),
        }

    def get_saved_passengers(self, user_id: str) -> Dict[str, Any]:
        user = self.users_by_id.get(user_id)
        if not user:
            return {"found": False, "user_id": user_id}

        return {
            "found": True,
            "user_id": user_id,
            "saved_passengers": self._deepcopy(user.get("saved_passengers", [])),
        }

    def find_user_reservations(self, user_id: str) -> Dict[str, Any]:
        user = self.users_by_id.get(user_id)
        if not user:
            return {"found": False, "user_id": user_id}

        reservations = self.reservations_by_user_id.get(user_id, [])
        return {
            "found": True,
            "user_id": user_id,
            "reservations": self._deepcopy(reservations),
        }

    def get_reservation(
            self,
            reservation_id: str,
            user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        reservation = self.reservations_by_id.get(reservation_id)
        if not reservation:
            return {
                "found": False,
                "reservation_id": reservation_id,
            }

        if user_id is not None and reservation.get("user_id") != user_id:
            return {
                "found": False,
                "reservation_id": reservation_id,
                "user_id": user_id,
                "reason": "reservation does not belong to provided user_id",
            }

        return {
            "found": True,
            "reservation": self._deepcopy(reservation),
        }

    def get_flight_status(self, flight_number: str, flight_date: str) -> Dict[str, Any]:
        flight = self.flights_by_number.get(flight_number)
        if not flight:
            return {
                "found": False,
                "flight_number": flight_number,
                "flight_date": flight_date,
            }

        date_info = (flight.get("dates") or {}).get(flight_date)
        if not date_info:
            return {
                "found": False,
                "flight_number": flight_number,
                "flight_date": flight_date,
                "reason": "flight date not found",
            }

        return {
            "found": True,
            "flight_number": flight_number,
            "flight_date": flight_date,
            "flight": self._deepcopy({
                "origin": flight.get("origin"),
                "destination": flight.get("destination"),
                "flight_number": flight.get("flight_number"),
                "scheduled_departure_time_est": flight.get("scheduled_departure_time_est"),
                "scheduled_arrival_time_est": flight.get("scheduled_arrival_time_est"),
                "date_info": date_info,
            }),
        }

    def search_available_flights(
            self,
            origin: str,
            destination: str,
            flight_date: str,
    ) -> Dict[str, Any]:
        matches = []

        for flight in self.flights:
            if flight.get("origin") != origin:
                continue
            if flight.get("destination") != destination:
                continue

            date_info = (flight.get("dates") or {}).get(flight_date)
            if not date_info:
                continue
            if date_info.get("status") != "available":
                continue

            matches.append({
                "flight_number": flight.get("flight_number"),
                "origin": flight.get("origin"),
                "destination": flight.get("destination"),
                "scheduled_departure_time_est": flight.get("scheduled_departure_time_est"),
                "scheduled_arrival_time_est": flight.get("scheduled_arrival_time_est"),
                "flight_date": flight_date,
                "status": date_info.get("status"),
                "available_seats": date_info.get("available_seats", {}),
                "prices": date_info.get("prices", {}),
            })

        def sort_key(x: Dict[str, Any]) -> Tuple:
            prices = x.get("prices", {})
            min_price = min(
                [
                    v for v in prices.values()
                    if isinstance(v, (int, float))
                ] or [10 ** 9]
            )
            return (min_price, x.get("flight_number", ""))

        matches.sort(key=sort_key)

        return {
            "found": True,
            "origin": origin,
            "destination": destination,
            "flight_date": flight_date,
            "flights": self._deepcopy(matches),
        }


class AirlineReadTools:
    def __init__(self, db: AirlineReadOnlyDB):
        self.db = db
        self.handlers = {
            "get_user": self.db.get_user,
            "get_user_payment_methods": self.db.get_user_payment_methods,
            "get_saved_passengers": self.db.get_saved_passengers,
            "find_user_reservations": self.db.find_user_reservations,
            "get_reservation": self.db.get_reservation,
            "get_flight_status": self.db.get_flight_status,
            "search_available_flights": self.db.search_available_flights,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_user",
                "description": "Get a user profile by user_id",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "get_user_payment_methods",
                "description": "Get payment methods stored in a user profile",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "get_saved_passengers",
                "description": "Get saved passengers for a user",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "find_user_reservations",
                "description": "Find all reservations belonging to a user",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    },
                    "required": ["user_id"]
                }
            },
            {
                "name": "get_reservation",
                "description": "Get a reservation by reservation_id, optionally verifying user_id ownership",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {"type": "string"},
                        "user_id": {"type": "string"}
                    },
                    "required": ["reservation_id"]
                }
            },
            {
                "name": "get_flight_status",
                "description": "Get the status of a flight on a given date",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string"},
                        "flight_date": {"type": "string"}
                    },
                    "required": ["flight_number", "flight_date"]
                }
            },
            {
                "name": "search_available_flights",
                "description": "Search available flights by origin, destination, and date",
                "arguments_schema": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "flight_date": {"type": "string"}
                    },
                    "required": ["origin", "destination", "flight_date"]
                }
            },
        ]

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.handlers:
            return {"error": f"Unknown tool: {name}"}

        try:
            return self.handlers[name](**arguments)
        except TypeError as e:
            return {
                "error": f"Invalid arguments for tool {name}: {str(e)}"
            }
        except Exception as e:
            logger.exception("Tool call failed")
            return {
                "error": f"Tool {name} failed: {type(e).__name__}: {str(e)}"
            }


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.model = os.getenv("AGENT_LLM", "openai/gpt-4o-mini")
        self.max_retries = int(os.getenv("AGENT_LLM_MAX_RETRIES", "5"))
        self.backoff_base = int(os.getenv("AGENT_LLM_BACKOFF_BASE", "2"))
        self.db_path = os.getenv("AIRLINE_DB_PATH", DB_PATH)

        self.db = AirlineReadOnlyDB(self.db_path)
        self.tools = AirlineReadTools(self.db)

        # Structure:
        # {
        #   context_id: {
        #       "messages": [...],
        #       "db_context": [...],
        #       "tool_history": [...]
        #   }
        # }
        self.ctx_id_to_messages: Dict[str, Dict[str, Any]] = {}

        logger.info("Purple agent initialized with model: %s", self.model)
        logger.info(
            "Retry config: max_retries=%s, backoff_base=%s",
            self.max_retries,
            self.backoff_base,
        )
        logger.info("DB path: %s", self.db_path)

    def _ensure_context(self, context_id: str) -> None:
        if context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context_id] = {
                "messages": [],
                "db_context": [],
                "tool_history": [],
            }
            logger.info("Initialized new conversation for context %s", context_id)

    def _append_user_message(self, context_id: str, user_input: str) -> None:
        self.ctx_id_to_messages[context_id]["messages"].append({
            "role": "user",
            "content": user_input,
        })

    def _append_assistant_message(self, context_id: str, assistant_content: str) -> None:
        self.ctx_id_to_messages[context_id]["messages"].append({
            "role": "assistant",
            "content": assistant_content,
        })

    def _append_tool_result(
            self,
            context_id: str,
            tool_name: str,
            arguments: Dict[str, Any],
            result: Dict[str, Any],
    ) -> None:
        tool_event = {
            "tool_name": tool_name,
            "arguments": copy.deepcopy(arguments),
            "result": copy.deepcopy(result),
        }

        self.ctx_id_to_messages[context_id]["tool_history"].append(tool_event)

    def _build_llm_messages(self, context_id: str) -> List[Dict[str, str]]:
        ctx = self.ctx_id_to_messages[context_id]

        db_context_payload = {
            "available_tools": self.tools.list_tools(),
            "tool_history": ctx["tool_history"],
            "database_context": ctx["db_context"],
        }

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                        "Here is persistent context collected from prior database lookups in this conversation.\n"
                        "Use it as working memory.\n"
                        + json.dumps(db_context_payload, ensure_ascii=False)
                ),
            },
        ]

        messages.extend(ctx["messages"])
        return messages

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_input = get_message_text(message)
        context_id = message.context_id

        logger.info("Received message for context %s", context_id)
        logger.debug("User input length: %s chars", len(user_input))

        self._ensure_context(context_id)
        self._append_user_message(context_id, user_input)

        final_response_json = None
        max_steps = 16

        for step in range(max_steps):
            llm_messages = self._build_llm_messages(context_id)

            logger.info(
                "Calling LLM %s with %s messages at step %s",
                self.model,
                len(llm_messages),
                step + 1,
            )

            try:
                response = call_llm_with_retry(
                    messages=llm_messages,
                    model=self.model,
                    response_format={"type": "json_object"},
                    max_retries=self.max_retries,
                    backoff_base=self.backoff_base,
                )
                assistant_content = response.choices[0].message.content
                logger.info("LLM response received: %s...", assistant_content[:300])
            except Exception as e:
                logger.error("LLM call failed with error: %s: %s", type(e).__name__, e)
                logger.exception("Full traceback:")
                final_response_json = {
                    "type": "respond",
                    "content": "I encountered an error processing your request."
                }
                break

            self._append_assistant_message(context_id, assistant_content)

            try:
                assistant_obj = json.loads(assistant_content)
            except json.JSONDecodeError:
                logger.error("Model returned invalid JSON: %s", assistant_content)
                final_response_json = {
                    "type": "respond",
                    "content": "Internal error: invalid JSON returned by the model."
                }
                break
            if "type" not in assistant_obj:
                if "name" in assistant_obj and "arguments" in assistant_obj:
                    if assistant_obj["name"] != "respond":
                        assistant_obj = {
                            "type": "tool_call",
                            "name": assistant_obj["name"],
                            "arguments": assistant_obj.get("arguments", {}),
                        }
                    else:
                        assistant_obj = {
                            "type": "respond",
                            "content": assistant_obj["arguments"]["content"],
                        }
            response_type = assistant_obj.get("type")

            if response_type == "respond":
                final_response_json = assistant_obj
                break

            if response_type == "tool_call":
                tool_name = assistant_obj.get("name")
                arguments = assistant_obj.get("arguments", {})

                if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                    final_response_json = {
                        "type": "respond",
                        "content": "Internal error: malformed tool call."
                    }
                    break

                logger.info("Executing tool: %s with args: %s", tool_name, arguments)
                tool_result = self.tools.call(tool_name, arguments)

                self._append_tool_result(
                    context_id=context_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=tool_result,
                )

                # Tool results are not directly sent to user.
                # They are added to persistent db context and will be passed back to the model.
                continue

            final_response_json = {
                "type": "respond",
                "content": "Internal error: unknown response type."
            }
            break

        if final_response_json is None:
            final_response_json = {
                "type": "respond",
                "content": "I could not complete the request in the allowed number of steps."
            }

        final_text = json.dumps(final_response_json, ensure_ascii=False)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=final_text))],
            name="Response"
        )
