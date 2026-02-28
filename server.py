"""
Grok MCP Server - Universal Research & Analysis Interface
Provides access to Grok's agentic capabilities: web search, X platform, code execution
"""

from fastmcp import FastMCP
from openai import AsyncOpenAI
import os
import time
import uuid
import asyncio
import re
from typing import Optional

# Initialize MCP server
mcp = FastMCP("Grok Research Platform")

# Configuration
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Initialize OpenAI client configured for Grok
grok_client = AsyncOpenAI(
    api_key=GROK_API_KEY,
    base_url=GROK_API_URL
)

# Task storage for async mode
tasks = {}
MAX_TASKS = 100
TASK_EXPIRY_HOURS = 24

async def call_grok_api(prompt: str) -> str:
    """Call Grok API with streaming and return complete response"""
    
    if not GROK_API_KEY:
        return "Error: GROK_API_KEY environment variable not set"
    
    try:
        stream = await grok_client.chat.completions.create(
            model="grok-4.1-fast",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=True  # Enable streaming
        )
        
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        # Remove all <think>...</think> tags from anywhere in response
        full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
        
        return full_response
        
    except Exception as e:
        return f"Error calling Grok API: {str(e)}"

def cleanup_old_tasks():
    """Remove tasks older than TASK_EXPIRY_HOURS or exceed MAX_TASKS"""
    current_time = time.time()
    expiry_threshold = current_time - (TASK_EXPIRY_HOURS * 3600)
    
    # Remove expired tasks
    expired = [tid for tid, task in tasks.items() 
               if task["created_at"] < expiry_threshold]
    for tid in expired:
        del tasks[tid]
    
    # If still too many, remove oldest
    if len(tasks) > MAX_TASKS:
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1]["created_at"])
        to_remove = len(tasks) - MAX_TASKS
        for tid, _ in sorted_tasks[:to_remove]:
            del tasks[tid]

async def execute_task_background(task_id: str, prompt: str):
    """Execute Grok research in background and update task status"""
    try:
        tasks[task_id]["status"] = "running"
        result = await call_grok_api(prompt)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        tasks[task_id]["completed_at"] = time.time()
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = time.time()

@mcp.tool()
async def grok(prompt: str) -> str:
    """
    Autonomous Agentic Researcher: web and X search equipped.

    Stateless—restate context each call.
    
    PROVIDE:
    - Objective
    - Context/background
    - Key questions to answer
    - Constraints (time, sources, depth)
    - Deliverable/format
    - A very rich high quality detailed prompt
    
    Grok follows your prompt. Prompt quality = output quality. Clear, thorough task descriptions produce comprehensive analysis. 

    Autonomous AI research platform with web search, X platform access, and code 
    execution. You describe goals, Grok intelligently determines which tools to use 
    and orchestrates execution strategy.
    
    ---
    Args:
        prompt: Research request/question/task (brief to extensively detailed multi-phase)
    
    Returns:
        Task ID for async retrieval. Research tasks run in background (1-3 minutes).
        Use grok_check_task tool to retrieve results.
    """
    
    full_prompt = prompt
    
    # Research trail feature - kept for potential future use
    # if include_research_trail:
    #     full_prompt += """
    #
    # ---
    # IMPORTANT - Document your research process at END:
    #
    # ## Research Process (Step-by-Step)
    #
    # For EVERY action, document chronologically:
    #
    # **Step N: [Action]**
    # - Reasoning: Why this approach
    # - Tool: Exact name (web_search, browse_page, x_keyword_search, x_semantic_search, 
    #   code_execution, etc.)
    # - Parameters: Complete query/URL/code/filters
    # - Found: Summary of results
    # - Next: Decision based on results
    #
    # Document failures/issues and adaptations.
    #
    # ## Research Summary
    # - Total steps: [count]
    # - Tools used: [tool_name (N times), ...]
    # - Sources accessed: [All URLs, X posts, etc.]
    # - Key sources: [Top 2-3]
    # - Search time ranges: [Date ranges if applicable]
    # - Code executed: [Yes/No, purpose]
    # - Overall approach: [2-3 sentence strategy summary]
    # """
    
    # Always use async mode
    cleanup_old_tasks()
    task_id = f"grok_{uuid.uuid4().hex[:8]}"
    tasks[task_id] = {
        "status": "pending",
        "created_at": time.time(),
        "completed_at": None,
        "result": None,
        "error": None,
        "prompt": prompt[:200]
    }
    asyncio.create_task(execute_task_background(task_id, full_prompt))
    
    return f"""Task started: {task_id}

Research task initiated. Expected completion time: 1-3 minutes.

CRITICAL INSTRUCTION: Return control to the user immediately. Inform them that the research task is running in the background and will take approximately 1-3 minutes to complete. Wait for their next message before checking the results.
 
When the user responds again, use grok_check_task tool with '{task_id}' as task_id to retrieve the results."""

@mcp.tool()
async def grok_check_task(task_id: str) -> str:
    """
    Check status and retrieve results of an async Grok task.
    
    When grok() is called, it returns a task_id. Use this tool to check if the 
    research is complete and get the results.
    
    Returns one of:
    - "Status: running" → Task in progress, check again in 15-30s
    - Full research results → Task completed (no "Status:" prefix)
    - "Status: failed - [error]" → Task failed, see error message
    - "Status: not_found" → Invalid/expired task_id (kept 24hrs, max 100 tasks)
    
    Workflow:
    1. grok(prompt) → Returns task_id
    2. Wait 15-60s depending on complexity
    3. grok_check_task(task_id) → Check status
    4. If completed, full results returned
    5. If running, wait and check again
    
    Args:
        task_id: Task identifier from grok()
        
    Returns:
        Status update or full research results when completed
    """
    
    cleanup_old_tasks()
    
    if task_id not in tasks:
        return f"Status: not_found - Task '{task_id}' does not exist or has expired (tasks kept 24 hours, max 100 tasks stored)."
    
    task = tasks[task_id]
    status = task["status"]
    
    if status == "pending":
        elapsed = int(time.time() - task["created_at"])
        return f"""Task pending ({elapsed} seconds elapsed).

The task is queued and will begin processing shortly. Research tasks typically complete in 1-3 minutes total. Return control to the user and inform them the task is initializing. Check status again when the user next interacts with you."""
    
    elif status == "running":
        elapsed = int(time.time() - task["created_at"])
        return f"""Task still running ({elapsed} seconds elapsed).

Research tasks typically complete in 1-3 minutes total. Return control to the user and inform them the task is still in progress. Check status again when the user next interacts with you."""
    
    elif status == "completed":
        return task["result"]  # Return results directly, no prefix
    
    elif status == "failed":
        return f"Status: failed - {task['error']}"
    
    else:
        return f"Status: unknown - Unexpected task state: {status}"


def main():
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
