from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Live Cricket Agent -> Agent - 1

match_agent = Agent(
    name = "Live Cricket Match Agent",
    model = OpenAIChat(id = 'gpt-4o', api_key = api_key),
    tools = [DuckDuckGo()],
    instructions = [
        'Search for live cricket match score',
        'Summarize the score, top players, and match situation',
        'use markdown tables for better clarity and readability'
    ],
    show_tool_calls = True,
    markdown = True,
    debug_mode = True
)

# Player Stats Agent -> Agent - 2

player_stats_agent = Agent(
    name = "Player Stats Agent",
    model = OpenAIChat(id = 'gpt-4o', api_key = api_key),
    tools = [DuckDuckGo()],
    instructions = [
        'Search for player statistics',
        'Summarize the player batting and bowling stats for last 5 matches',
        'use markdown tables for better clarity and readability'
    ],
    show_tool_calls = True,
    markdown = True,
    debug_mode = True
)

# Cricket News Agent -> Agent - 3

cricket_news_agent = Agent(
    name = "Cricket News Agent",
    model = OpenAIChat(id = 'gpt-4o', api_key = api_key),
    tools = [DuckDuckGo()],
    instructions = [
        'Search the latest cricket news',
        'Highlight upcoming matches, tournaments, and player updates',
        'list headlines and key insights in markdown format for more readability and clarity'
    ],
    show_tool_calls = True,
    markdown = True,
    debug_mode = True
)

# Main Cricket News Agent -> Agent - 4 ( Combining all together)

main_cricket_team = Agent(
    name = "Cricket Analysis Team",
    model = OpenAIChat(id = 'gpt-4o', api_key = api_key),
    tools = [match_agent, player_stats_agent, cricket_news_agent],
    instructions = [
        'Gather live cricket match score, player statistics, and cricket news',
        'Use structured markdown format for better readability and clarity',
    ],
    show_tool_calls = True,
    markdown = True,
    debug_mode = True
)

main_cricket_team.print_response(
    "Get the latest score of SRH vs KKR, recent stats for pat cummins, and cricket news",
    stream = True
)