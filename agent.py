import streamlit as st
import json
import redis
import os
from typing import TypedDict, List, Union
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
REDIS_HOST      = st.secrets["REDIS_HOST"]
REDIS_PORT      = int(st.secrets["REDIS_PORT"])
REDIS_PASSWORD  = st.secrets["REDIS_PASSWORD"]


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = st.secrets["langchain_api_key"]
os.environ["LANGCHAIN_PROJECT"]    = st.secrets["project_name"]


st.set_page_config(page_title="Hotel Booking Agent", page_icon="🏨")
st.title("🏨 Hotel Booking AI Agent")

SEARCH_DATA = {
    "city": "Jaipur",
    "hotels": [
        {"hotel_id": "HTL001", "name": "Maharani Palace",  "stars": 5, "location": "MI Road",     "base_price": 4500},
        {"hotel_id": "HTL002", "name": "Royal Haveli",     "stars": 4, "location": "Pink City",   "base_price": 2800},
        {"hotel_id": "HTL003", "name": "Budget Stay",      "stars": 2, "location": "Sindhi Camp", "base_price": 900},
    ],
}
AVAILABILITY_DATA = {
    "hotel_id": "HTL001",
    "hotel_name": "Maharani Palace",
    "pricingByRoom": [
        {"roomType": "Deluxe Room", "maxAvailableRoom": 3, "data": [
            {"category": "Room Only",           "totalPrice": 4725,  "basePrice": 4500},
            {"category": "Room With Breakfast", "totalPrice": 5512,  "basePrice": 5250},
            {"category": "Room With All Meals", "totalPrice": 6562,  "basePrice": 6250},
        ]},
        {"roomType": "Suite", "maxAvailableRoom": 1, "data": [
            {"category": "Room Only",           "totalPrice": 8925,  "basePrice": 8500},
            {"category": "Room With Breakfast", "totalPrice": 10237, "basePrice": 9750},
        ]},
    ],
}
HOTEL_DETAILS = {
    "hotel_id": "HTL001",
    "name": "Maharani Palace",
    "address": "MI Road, Jaipur, Rajasthan 302001",
    "amenities": ["Pool", "Spa", "Free WiFi", "Restaurant", "Valet Parking"],
    "policies": {
        "checkin": "14:00", "checkout": "11:00",
        "cancellation": "Free cancellation up to 24 hours before check-in",
        "pets": "Not allowed",
    },
    "nearby": ["Hawa Mahal – 2km", "Amber Fort – 11km", "Jaipur Junction – 3km"],
}

SESSION_ID         = "hotel_user_1"
TTL_SECONDS        = 300
SUMMARY_TRIGGER    = 5
KEEP_AFTER_SUMMARY = 2


@st.cache_resource
def get_redis():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD or None,
        decode_responses=True
    )


rc = get_redis()


def get_turns(sid):
    return [json.loads(t) for t in rc.lrange(f"chat:{sid}:turns", 0, -1)]


def get_summary(sid):
    return rc.get(f"chat:{sid}:summary") or ""


def store_turn(sid, turn):
    key = f"chat:{sid}:turns"
    rc.rpush(key, json.dumps(turn))
    rc.expire(key, TTL_SECONDS)
    rc.expire(f"chat:{sid}:summary", TTL_SECONDS)


def maybe_summarize(sid, llm):
    turns = get_turns(sid)
    if len(turns) <= SUMMARY_TRIGGER:
        return
    n     = len(turns) - KEEP_AFTER_SUMMARY
    lines = []
    for t in turns[:n]:
        lines += [f"User: {t['user']}", f"Assistant: {t['assistant']}"]
    prompt = f"""Merge existing summary with new turns into ONE concise summary.
Existing summary:\n{get_summary(sid)}\n\nNew turns:\n{chr(10).join(lines)}\n\nReturn updated summary only."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    rc.set(f"chat:{sid}:summary", resp.content.strip())
    rc.ltrim(f"chat:{sid}:turns", n, -1)


def get_cache(key):
    data = rc.get(key)
    return json.loads(data) if data else None


def set_cache(key, value, ttl):
    rc.setex(key, ttl, json.dumps(value))


@tool
def search_hotels(city: str, checkin: str, checkout: str, guests: int) -> str:
    """Search hotels by city and dates."""
    key    = f"hotel:search:{city}:{checkin}:{checkout}"
    cached = get_cache(key)
    if cached:
        return json.dumps({"source": "cache", "data": cached})
    set_cache(key, SEARCH_DATA, 600)
    return json.dumps({"source": "api", "data": SEARCH_DATA})


@tool
def check_availability(hotel_id: str, checkin: str, checkout: str) -> str:
    """Check room availability and pricing for a specific hotel."""
    key    = f"hotel:availability:{hotel_id}:{checkin}:{checkout}"
    cached = get_cache(key)
    if cached:
        return json.dumps({"source": "cache", "data": cached})
    set_cache(key, AVAILABILITY_DATA, 300)
    return json.dumps({"source": "api", "data": AVAILABILITY_DATA})


@tool
def get_hotel_details(hotel_id: str) -> str:
    """Get amenities, policies, and location details for a specific hotel."""
    key    = f"hotel:details:{hotel_id}"
    cached = get_cache(key)
    if cached:
        return json.dumps({"source": "cache", "data": cached})
    set_cache(key, HOTEL_DETAILS, 1800)
    return json.dumps({"source": "api", "data": HOTEL_DETAILS})


tools = [search_hotels, check_availability, get_hotel_details]

SYSTEM_PROMPT = """You are a Hotel Booking AI Agent. Help users search hotels, check availability, and get hotel details.

Tools available:
- search_hotels(city, checkin, checkout, guests): find hotels in a city
- check_availability(hotel_id, checkin, checkout): room types & pricing
- get_hotel_details(hotel_id): amenities, policies, nearby landmarks

Always pick the right tool. Be concise and helpful."""


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]


@st.cache_resource
def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    ).bind_tools(tools)

    def model_node(state):
        return {"messages": state["messages"] + [llm.invoke(state["messages"])]}

    def router(state):
        last = state["messages"][-1]
        return "call_tool" if isinstance(last, AIMessage) and last.tool_calls else END

    def tool_node(state):
        last     = state["messages"][-1]
        new_msgs = list(state["messages"])
        for tc in last.tool_calls:
            fn     = next((t for t in tools if t.name == tc["name"]), None)
            result = fn.invoke(tc["args"]) if fn else f"Tool {tc['name']} not found"
            new_msgs.append(ToolMessage(content=str(result), name=tc["name"], tool_call_id=tc["id"]))
        return {"messages": new_msgs}

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("call_tool", tool_node)
    g.set_entry_point("model")
    g.add_conditional_edges("model", router)
    g.add_edge("call_tool", "model")
    return g.compile()


agent = build_agent()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input := st.chat_input("Ask about hotels… e.g. Find hotels in Jaipur for 2 guests Oct 10–12"):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                summary  = get_summary(SESSION_ID)
                if summary:
                    messages.append(SystemMessage(content=f"Conversation summary:\n{summary}"))
                for t in get_turns(SESSION_ID):
                    messages += [HumanMessage(content=t["user"]), AIMessage(content=t["assistant"])]
                messages.append(HumanMessage(content=user_input))

                final_state = agent.invoke({"messages": messages})

                ai_response = ""
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_response = msg.content
                        break

                st.markdown(ai_response)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

                store_turn(SESSION_ID, {"user": user_input, "assistant": ai_response})
                base_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
                maybe_summarize(SESSION_ID, base_llm)

            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})
