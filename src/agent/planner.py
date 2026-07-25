"""LLM Intent Parser and Orchestrator using OpenRouter."""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "planner_prompt.txt"

def plan_execution(user_query: str) -> dict:
    """Parse query intent and build a dynamic JSON tool execution plan."""
    
    # Initialize OpenRouter Client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001", # High-speed free model on OpenRouter
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    
    return json.loads(response.choices[0].message.content)