"""LLM Intent Parser and Orchestrator using OpenRouter."""
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "planner_prompt.txt"
def plan_execution(user_query: str) -> dict:
    """Parse query intent and build a dynamic JSON tool execution plan."""
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    augmented_system_prompt = (
        system_prompt + 
        "\n\nCRITICAL REQUIREMENT: You must output ONLY a valid JSON object with keys: "
        "'intent', 'extracted_entities', 'tools_to_invoke', and 'skipped_tools'. "
        "Do not wrap it in markdown code blocks like ```json ... ```. Output raw JSON only."
    )

    response = client.chat.completions.create(
        model="openrouter/free", 
        messages=[
            {"role": "system", "content": augmented_system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError(f"Failed to parse LLM response into JSON: {content}")
