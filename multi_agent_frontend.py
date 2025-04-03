import streamlit as st
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Live Cricket Match Agent -> Agent - 1
match_agent = Agent(
    name="Live Match Agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key),
    tools=[DuckDuckGo()],
    instructions=[
        "Search for live cricket match scores.",
        "Summarize the score, top players, and match situation.",
        "Use markdown tables for clarity."
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# Player Stats Agent -> Agent - 2
player_agent = Agent(
    name="Player Stats Agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key),
    tools=[DuckDuckGo()],
    instructions=[
        "Find recent cricket player statistics.",
        "Include batting and bowling stats for the last 5 matches.",
        "Use tables for formatting."
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# Cricket News Agent -> Agent - 3
news_agent = Agent(
    name="Cricket News Agent",
    model=OpenAIChat(id="gpt-4o", api_key=api_key),
    tools=[DuckDuckGo()],
    instructions=[
        "Find and summarize the latest cricket news.",
        "Highlight upcoming matches, injuries, and tournament updates.",
        "List headlines with sources."
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# Main Cricket Analysis Team -> Agent - 4
cricket_team = Agent(
    name="Cricket Analysis Team",
    model=OpenAIChat(id="gpt-4o", api_key=api_key),
    team=[match_agent, player_agent, news_agent],
    instructions=[
        "Provide live match scores, player statistics, and news updates.",
        "Use structured formatting and markdown tables.",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# Streamlit App
st.set_page_config(page_title="🏏 Live Cricket Updates", layout="wide")
st.title("🏏 Cricket AI - Live Scores, Player Stats & News")

col1, col2 = st.columns(2)
with col1:
    match_query = st.text_input("Enter Match (e.g., 'India vs Australia'):")
with col2:
    player_query = st.text_input("Enter Player (e.g., 'Virat Kohli'):")

if st.button("Get Cricket Updates"):
    query = "Get the latest score"
    if match_query:
        query += f" of {match_query}"
    if player_query:
        query += f", recent stats for {player_query}"
    query += ", and latest cricket news."

    with st.spinner("Fetching Cricket Data..."):
        response = cricket_team.run(query) 
        st.markdown(response)  
