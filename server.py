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
            model="grok-4-fast",
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
        
        # Remove <think>...</think> from start if present
        if full_response.startswith('<think>'):
            end_tag = full_response.find('</think>')
            if end_tag != -1:
                full_response = full_response[end_tag + 8:].strip()  # +8 for len('</think>')
        
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
    
    CAPABILITIES
    
    WEB (3 modes):
    • web_search - Broad queries, top results (5-10s)
    • web_search_with_snippets - Quick fact-check with excerpts, no page load (3-8s)
    • browse_page - Full content extraction, specific sections (5-15s/page)
    
    X PLATFORM (5 tools - real-time info, 6-48hr ahead of web):
    
    • x_keyword_search - Advanced syntax:
      from:user to:user @mention | since:YYYY-MM-DD until:YYYY-MM-DD within_time:Nh 
      since_time:unix | min_faves:N min_retweets:N min_replies:N | filter:media|links|
      replies|verified url:domain | (A OR B) AND C -exclude "exact phrase"
      
    • x_semantic_search - AI relevance search, understands meaning beyond keywords
    • x_user_search - Find accounts by expertise/description  
    • x_thread_fetch - Complete conversation with replies (use when you need full 
      discussion context, not just individual posts)
    • x_video_view - Analyze via frames/subtitles
    
    When to use: X for real-time (<48h), expert opinions, sentiment, breaking news, 
    discussions. Web for documentation, historical info, official resources.
    
    CODE (Python 3.12 sandboxed offline):
    Data: numpy, pandas, scipy, statsmodels | Viz: matplotlib (data only) | ML: torch, 
    networkx | Math: sympy, mpmath | Finance: polygon, coingecko (pre-configured) | 
    Scientific: astropy, qutip, control | Other: PuLP, biopython, rdkit
    
    Use for: calculations, analysis, verification, financial metrics, structured output, 
    processing web/X data. Can combine with searches in same query.
    
    Limitations: No pip, no external network (except polygon/coingecko), no image outputs
    
    ---
    OPERATION
    
    AGENTIC: Autonomously selects tools and sequences. Don't specify "use web_search 
    then x_keyword_search" - describe your goal, Grok plans approach.
    
    Example: "What do AI researchers think about o3?" → Grok autonomously searches 
    AI researcher X accounts, checks web announcements, browses discussions, possibly 
    analyzes sentiment, synthesizes with sources.
    
    STATELESS: Zero memory between calls. Include all context in each prompt.
    ✓ Good: "Earlier research found o3 scores 75.7% on ARC-AGI at $20/task. Now 
       investigate if this cost is prohibitive - user experiences, alternatives, 
       calculate cost at scale."
    ✗ Bad: "What about the cost?" (Cost of what?)
    
    INTELLIGENT: Understands nuanced intent, domain context, implied priorities. Rich 
    context significantly improves results.
    
    SPEED: Simple 6-10s | Medium 15-30s | Complex 30-60s | Extensive 60-120s
    
    CAPACITY: Handles 10-20 item research, dozens of pages, multi-day tracking, 50+ 
    post analysis, complex calculations. Organization enables scale - structure requests 
    clearly for volume work. Response length adapts from brief answers to multi-thousand 
    word reports based on your prompt.
    
    ---
    PROMPTING
    
    CONTEXT (dramatically improves results):
    Include: Why asking, background/expertise, what you'll do with info, what you 
    already know, conversation continuity
    
    Example: "I'm evaluating ML frameworks for production. Python experience but new 
    to ML infrastructure. Research..." vs just "Research ML frameworks"
    
    ADAPTIVE STYLE:
    Simple → "Bitcoin price" | Medium → "Research [topic] checking X experts and web 
    docs" | Complex → Numbered phases, specific sources, synthesis needs, output format
    
    DEPTH SIGNALS:
    "Quick check:" / "Brief:" = concise | "Research:" / "Investigate:" = standard | 
    Multi-phase structure = comprehensive | "Check for updates:" = monitoring
    
    X PATTERNS:
    Expert opinions: from:expert1 OR from:expert2 topic
    Time-sensitive: topic since:2024-01-01 within_time:24h  
    High quality: topic min_faves:100 min_retweets:50
    Combined: (from:user1 OR from:user2) topic since:DATE min_faves:50 filter:verified -noise
    
    COMPLEX PROMPT STRUCTURE:
    [Goal] → [Context/why] → [Phase 1: action + sources] → [Phase 2: action + extraction] 
    → [Phase N: synthesis] → [Output: format/length/focus]
    
    CODE REQUESTS:
    Be specific: "Calculate CAGR using [data]" | Specify sources: "Use coingecko for 
    Bitcoin prices [dates]" or provide data | Request verification: "Verify claim [X] 
    using [method]" | Format: "Return as table/JSON/formatted"
    
    SYNTHESIS REQUESTS:
    Be explicit about what matters:
    "Synthesize: What's the consensus among experts?"
    "Analyze: Is this trend real or hype?"
    "Compare: Which option is better for [use case]?"
    "Evaluate: What are the risks vs benefits?"
    "Explain: Why do researchers disagree about this?"
    
    Specify perspectives: "Give both sides" | "Focus on practical not theory" | 
    "Decision-making insight not just facts"
    
    HANDLING UNCERTAINTY/CONFLICTS:
    When information might conflict:
    "Find claims about [X] and verify which are data-supported"
    "I've heard both [A] and [B] - investigate which is accurate"
    "Search for [topic] and note any disagreement among experts"
    
    Grok will: Search multiple perspectives, note conflicts, verify numerically when 
    possible, be explicit about confidence levels.
    
    OUTPUT FORMATTING:
    Control structure and presentation:
    "Structure as: Executive summary, detailed findings, conclusion"
    "Format as: Table comparing [items] by [attributes]"
    "Organize by: time period / category / source credibility"
    "Keep under 500 words but include all key points"
    "Be comprehensive - I want full detail on each item"
    
    Defaults: Clear headings, bullets, section separation, summaries when appropriate
    
    WHEN TO USE:
    ✓ Current info (<48h), expert opinions, sentiment analysis, verification, multi-source 
      synthesis, calculations, systematic research
    ✗ Trained knowledge, pure reasoning, creative writing, clarifying questions
    
    AVOID:
    ✗ Over-specifying tools (let Grok choose)
    ✗ Assuming memory (include context)  
    ✗ Vague scope (be specific)
    ✗ Missing time context for current events
    
    ---
    EXAMPLES (adapt, don't copy)
    
    Simple: "Bitcoin price"
    
    Brief: "AI developments last 24h: X from @OpenAI @AnthropicAI @GoogleAI min_faves:50, 
    web headlines. 5 bullets <300w."
    
    Complex: "Due diligence [company]: 1) Site: products/team/model 2) Funding news/
    valuations 3) Code: revenue growth, margins, burn vs benchmarks 4) Competitors: 
    identify 3-5, compare 5) X: user sentiment (genuine not marketing) 6) X: analyst 
    opinions 7) Risks: regulatory/competitive/tech. Synthesize: strengths, concerns, 
    financial health, recommendation. Code for comparison matrices."
    
    Specialized: "Timeline [situation]: Origins (spark, dates) → primary sources (browse 
    docs) → evolution (X time-bounded for discourse shifts) → key players/positions → 
    current state. Code to organize chronologically. Full narrative with context for 
    why developments mattered."
    
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

When the user responds again, use grok_check_task('{task_id}') to retrieve the results."""

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
