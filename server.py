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
async def grok(
    prompt: str,
    include_research_trail: bool = False,
    async_mode: bool = True
) -> str:
    """
    Universal interface to Grok - an agentic AI research and analysis platform.
    
    ═══════════════════════════════════════════════════════════════════════════
    WHAT GROK IS
    ═══════════════════════════════════════════════════════════════════════════
    
    Grok is an intelligent research platform that autonomously orchestrates 
    multiple tools to accomplish your goals. Think of it as a research analyst 
    with deep expertise and access to comprehensive information sources, not a 
    simple search engine.
    
    Grok decides which of its integrated capabilities to use based on your 
    prompt. You describe what you want to accomplish; Grok determines how to 
    achieve it.
    
    ═══════════════════════════════════════════════════════════════════════════
    CORE CAPABILITIES
    ═══════════════════════════════════════════════════════════════════════════
    
    WEB ACCESS (3 complementary modes):
    
      • web_search
        General queries returning top results across the internet
        Use for: Broad information gathering, discovering sources
        Speed: 5-10 seconds
        
      • web_search_with_snippets  
        Quick fact-checking with result excerpts, no full page loading
        Use for: Rapid verification of specific facts, grabbing quick data
        Speed: 3-8 seconds
        
      • browse_page
        Reads complete webpage content with intelligent extraction
        Use for: Detailed information needs, extracting specific sections,
                 reading documentation, getting full context
        Speed: 5-15 seconds per page
        Special: Can follow instructions to extract specific information
    
    
    X PLATFORM ACCESS (5 specialized tools):
    
    X.com provides real-time information that often precedes traditional web 
    sources by 6-48 hours. Expert opinions, breaking developments, and community
    discussions happen here first.
    
      • X keyword search with advanced operators:
        Powerful syntax for precise queries
        - User targeting: from:username, to:username, @mentions
        - Time filtering: since:YYYY-MM-DD, until:YYYY-MM-DD, since_time:unix
        - Engagement filters: min_faves:N, min_retweets:N, min_replies:N
        - Content filters: filter:media, filter:links, filter:replies, filter:verified
        - URL filtering: url:domain
        - Boolean logic: (A OR B) AND C, -exclude_term
        - Phrases: "exact phrase match"
        
        Use for: Posts from specific people, timebound searches, high-quality
                 content (engagement filtered), specific topics
        Speed: 5-10 seconds
        
      • X semantic search
        AI-powered relevance search that understands meaning beyond keywords
        Can filter by: date ranges, specific users, exclude users, relevancy threshold
        Use for: Finding posts by concept/meaning, discovering related discussions,
                 sentiment analysis, when you don't know exact keywords
        Speed: 5-10 seconds
        
      • X user search
        Find accounts by description or domain expertise
        Use for: Discovering experts, finding relevant accounts to follow/monitor
        
      • X thread fetch
        Retrieves complete conversation context including parent posts and replies
        Use for: Understanding full discussions, getting conversation context,
                 following debate threads
        
      • View X video
        Analyzes video content through frames and subtitle extraction
        Use for: Understanding video content without watching, extracting claims
                 from video posts
    
    X PLATFORM GUIDANCE:
    X excels for real-time information (last 24-48 hours), expert opinions, 
    community sentiment, breaking news, and primary source accounts. Use X 
    searches to complement web searches when you want human reactions and 
    discussions, not just established facts. For current discourse, X is 
    unmatched. For established documentation or historical information, web 
    search is typically more efficient.
    
    
    CODE EXECUTION (Python 3.12 with scientific computing stack):
    
    Full Python environment with extensive libraries:
    - Data analysis: numpy, pandas, scipy, statsmodels
    - Visualization: matplotlib
    - Mathematics: sympy (symbolic), mpmath
    - Machine learning: torch, networkx
    - Finance: polygon (stocks), coingecko (crypto) - APIs pre-configured
    - Optimization: PuLP
    - Scientific: astropy, qutip, control
    - And more: biopython, rdkit, chess, pygame, etc.
    
    Use for: Calculations, data analysis, statistical verification, financial 
    metrics, creating comparisons, processing extracted data, verifying claims 
    with math, generating structured output from unstructured data
    
    Note: Code runs in sandboxed offline environment. No pip install or external
    network access from code. No visualization image outputs (matplotlib generates
    data only). Focus on computation and analysis.
    
    ═══════════════════════════════════════════════════════════════════════════
    HOW GROK OPERATES
    ═══════════════════════════════════════════════════════════════════════════
    
    AGENTIC BEHAVIOR:
    Grok autonomously chooses which tools to use and in what sequence. You do 
    not specify "use web_search then x_keyword_search then code_execution" - 
    you describe your goal and Grok plans the approach.
    
    Example: "What do AI researchers think about o3?"
    Grok will autonomously:
    - Recognize this needs expert opinions (X is ideal)
    - Search from AI researcher accounts
    - Maybe search web for official announcements
    - Possibly browse linked technical discussions
    - Could use code to analyze sentiment patterns
    - Synthesize findings with sources
    
    You didn't specify these steps. Grok determined them based on understanding
    your underlying need.
    
    STATELESS EXECUTION:
    Each call to Grok is completely independent with no memory of previous calls.
    
    Implications:
    - Grok does NOT remember earlier conversations or research
    - If building on previous work, include that context in your new prompt
    - Reference specific URLs or findings from earlier research when relevant
    - Don't assume continuity - each prompt is a fresh start
    
    Example of good follow-up:
    "Earlier research found o3 scores 75.7% on ARC-AGI at $20/task. Now 
    investigate whether this cost is prohibitive for practical use - search 
    for user experiences, compare to alternatives, calculate cost at scale."
    
    Bad follow-up:
    "What about the cost?" (Grok: "Cost of what?")
    
    INTELLIGENCE & CONTEXT UNDERSTANDING:
    Grok is powered by an advanced language model, so rich context significantly
    improves results. Grok understands:
    - Nuanced intent beyond literal words
    - Implied priorities from how you phrase questions
    - Domain context and technical terminology
    - What matters based on why you're asking
    
    Providing context helps Grok:
    - Prioritize relevant sources over tangential information
    - Adjust depth appropriately (overview vs deep analysis)
    - Tailor synthesis to your actual needs
    - Make better decisions about tool selection
    
    VARIABLE SPEED BY TASK COMPLEXITY:
    Response time scales with workload:
    - Simple queries: 6-10 seconds ("Bitcoin price", "Latest from @OpenAI")
    - Medium research: 15-30 seconds (multi-source with synthesis)
    - Complex investigation: 30-60 seconds (deep multi-phase research)
    - Extensive systematic: 60-120 seconds (10+ items researched thoroughly)
    
    Your prompt determines complexity. Use "quick" or "brief" for speed, or 
    structure detailed investigations when thoroughness matters more than speed.
    
    ═══════════════════════════════════════════════════════════════════════════
    WORKLOAD CAPACITY
    ═══════════════════════════════════════════════════════════════════════════
    
    Grok can handle substantial, systematic work. Don't hesitate to assign 
    extensive research tasks:
    
    Examples of workload Grok handles well:
    - Systematically researching 10-20 items (products, papers, companies, episodes)
    - Browsing dozens of pages with specific extraction criteria
    - Multi-day event tracking with daily synthesis
    - Processing 50+ X posts into sentiment analysis
    - Complex financial calculations across multiple companies/years
    - Timeline reconstruction from multiple sources
    - Comprehensive comparative analysis with code-generated matrices
    
    Key principle: Organization enables scale
    The clearer your structure, the better Grok executes volume work. Break 
    large tasks into numbered phases, specify extraction criteria, indicate 
    output structure preferences. Grok excels at systematic execution of 
    well-organized requests.
    
    Response length adapts to task scope. Grok can generate responses from a 
    few words to tens of thousands of words depending on what you request. How 
    you prompt determines depth and length.
    
    ═══════════════════════════════════════════════════════════════════════════
    RESEARCH TRAIL PARAMETER
    ═══════════════════════════════════════════════════════════════════════════
    
    include_research_trail: Optional[bool] = False
    
    When set to True, Grok documents its complete research methodology:
    - Every search query executed (exact strings and operators used)
    - Every URL browsed (full links)
    - Every tool invoked (with parameters and counts)
    - Reasoning at each decision point
    - What was found and how it informed next steps
    - Summary of overall approach and sources
    
    USE RESEARCH TRAIL (True) WHEN:
    - Complex multi-source research where you want to verify methodology
    - You anticipate follow-up questions and want to build on the work
    - Important decisions requiring full transparency and source verification
    - You want to understand Grok's research strategy for learning
    - User explicitly asks "how did you find this?" or "show your work"
    
    SKIP RESEARCH TRAIL (False) WHEN:
    - Simple, quick lookups where overhead isn't valuable
    - Speed is the priority and you trust the result
    - One-off queries with no expected follow-up
    - The findings themselves are sufficient without methodology
    
    Research trails add minimal time (1-3 seconds) but significantly increase
    response length. Use judiciously based on whether methodology matters.
    
    ═══════════════════════════════════════════════════════════════════════════
    ASYNC MODE (RECOMMENDED)
    ═══════════════════════════════════════════════════════════════════════════
    
    async_mode: Optional[bool] = True (default)
    
    Grok queries typically take 15-60+ seconds, even for simple lookups, because
    they involve real-time web/X searches and analysis. Most MCP clients will
    timeout on connections longer than 10-30 seconds.
    
    ASYNC MODE (default=True, recommended):
    - Returns task_id immediately
    - Research executes in background
    - Retrieve results with grok_check_task(task_id)
    - Prevents timeout failures
    
    SYNC MODE (async_mode=False):
    - Waits for complete response before returning
    - Only use if your MCP client has no timeout limits
    - Simpler for one-off testing
    
    Workflow:
    1. grok(prompt) → Returns task_id instantly (async is default)
    2. grok_check_task(task_id) → Check status after 20-30s
    3. If completed, retrieve full results
    4. If still running, check again
    
    Tasks stored for 24 hours (max 100 tasks).
    
    ═══════════════════════════════════════════════════════════════════════════
    ADVANCED PROMPTING GUIDELINES
    ═══════════════════════════════════════════════════════════════════════════
    
    This section is critical for effective use. Read carefully.
    
    
    ADAPTIVE PROMPTING PHILOSOPHY:
    
    Grok responds to your communication style, not rigid templates. Your prompts
    should adapt to the task at hand:
    
    Simple question → Simple prompt:
      "Bitcoin price"
      "Latest post from @AnthropicAI"
      "What is the weather in Tokyo?"
      
    Complex investigation → Structured prompt:
      Multiple numbered phases
      Specific sources to check
      Synthesis requirements
      Output format preferences
      
    Massive systematic task → Highly detailed organization:
      Clear step-by-step breakdown
      Explicit extraction criteria per source
      Code analysis requirements
      Structured output specification
    
    DO NOT feel constrained by any examples shown in this documentation. They 
    are inspirations demonstrating range, not templates to copy. Your prompts 
    can be completely different in structure, style, length, and approach.
    
    Grok understands:
    - Natural language variations and informal phrasing
    - Technical jargon and domain-specific terminology
    - Implied intent from context
    - Your underlying goals even when not explicitly stated
    - Different levels of specificity and detail
    
    
    CONTEXT-AWARE PROMPTING:
    
    Grok performs significantly better when you provide relevant context. Consider
    including:
    
    Why you're asking:
      "I'm deciding between Product A and B for [use case]..."
      "I'm writing an article about [topic] and need to verify..."
      "I'm researching for an investment decision in [sector]..."
      
    Your background (when relevant):
      "I'm a developer familiar with Python but new to ML..."
      "I have a background in finance but not cryptocurrency..."
      "I'm a casual viewer trying to decide if this show is for me..."
      
    What you'll do with the information:
      "I need to present findings to executives..."
      "This is for personal decision-making..."
      "I'm fact-checking claims in a debate..."
      
    What you already know:
      "I know the basic premise but need deeper analysis..."
      "I've heard conflicting claims about [X] and [Y]..."
      "Previous research found [fact], now I need..."
      
    Conversation context:
      "We discussed [topic] earlier - building on that..."
      "This relates to the [previous question] about..."
      
    Context helps Grok:
    - Prioritize information by relevance to your actual needs
    - Adjust technical depth appropriately
    - Focus on actionable insights vs academic completeness
    - Recognize which sources are authoritative for your purpose
    - Synthesize in ways that directly address your decision
    
    
    PROMPTING FOR DIFFERENT RESEARCH DEPTHS:
    
    Grok's depth adapts to your signals. Be explicit about what you need:
    
    Quick lookups:
      Start with "Quick check:", "Brief summary:", "What's the latest:"
      Grok will prioritize speed and conciseness
      
    Standard research:
      "Research [topic]", "Investigate [question]", "Find information about:"
      Grok will do thorough multi-source research with synthesis
      
    Comprehensive investigations:
      Structure in phases/steps, request detailed analysis, specify many sources
      Grok will do extensive systematic research with deep synthesis
      
    Monitoring/tracking:
      "Check for updates:", "Any news from:", "What happened today with:"
      Grok will focus on recent developments and changes
    
    
    STRUCTURING COMPLEX PROMPTS:
    
    For involved research, this structure works consistently well:
    
    [Clear goal statement]
      What you want to accomplish in one sentence
      
    [Context - why this matters]
      Optional but helps Grok prioritize
      
    [Phase-by-phase breakdown]
      1. First, do X (with specifics about sources/approach)
      2. Then do Y (with extraction criteria)
      3. Finally do Z (with synthesis requirements)
      
    [Output preferences]
      "Structure as: [format]"
      "Keep it under [length]" or "Be comprehensive"
      "Focus on [aspects]"
      
    [Research trail request if needed]
      Include if you want methodology documented
    
    
    X SEARCH BEST PRACTICES:
    
    The X platform is extraordinarily powerful when used effectively.
    
    For expert opinions:
      from:username1 OR from:username2 OR from:username3
      List specific experts relevant to your topic
      
    For time-sensitive information:
      since:YYYY-MM-DD or since:24_hours_ago or within_time:12h
      until:YYYY-MM-DD for bounded searches
      
    For high-quality content:
      min_faves:100 or min_retweets:50
      Filters to engagement ensure quality and visibility
      
    For specific content types:
      filter:links (posts with URLs)
      filter:media (posts with images/videos)
      filter:replies (posts that are replies)
      filter:verified (from verified accounts)
      
    For URL filtering:
      url:domain
      
    For focused topics:
      "exact phrase" for precision
      (keyword1 OR keyword2) AND keyword3 for combinations
      -excluded_term to remove noise
      
    For discovering accounts:
      Use X user search with descriptions of expertise
      Then search from:those_users for their takes
      
    Combine operators freely:
      (from:user1 OR from:user2) (topic OR "related phrase") since:2024-01-01 min_faves:50
    
    Remember: X is conversational and temporal. People discuss things in threads
    over time. Use thread fetch when you need full context of a discussion, not
    just individual posts.
    
    
    CODE EXECUTION GUIDANCE:
    
    When requesting calculations or analysis:
    
    Be specific about what to calculate:
      "Calculate compound annual growth rate using: [data/formula]"
      "Analyze these numbers for: mean, median, std dev, outliers"
      "Create comparison matrix of: [items] by [attributes]"
      
    Specify data sources:
      "Use coingecko API to get Bitcoin price history for [dates]"
      "Use polygon API for stock data on [ticker]"
      "Process these numbers: [provide data]"
      
    Request verification:
      "Verify the claim that [X] using: [method]"
      "Check if this math is correct: [calculation]"
      "Calculate to confirm whether [statement] is accurate"
      
    Indicate output format:
      "Return as: table / JSON / formatted numbers / chart data"
      "Show calculation steps so I can verify logic"
    
    Remember: Code runs offline. It can calculate, analyze, and process but 
    cannot fetch data except via pre-configured APIs (coingecko, polygon) or 
    data you provide in the prompt.
    
    
    HANDLING UNCERTAINTY AND CONFLICTS:
    
    When you know information might be conflicting or unclear:
    
      "Find claims about [X] and verify which are supported by data"
      "I've heard both [A] and [B] - investigate which is accurate"
      "Search for [topic] and note any disagreement among experts"
      
    Grok will:
    - Actively search for multiple perspectives
    - Note when sources conflict
    - Attempt verification through additional sources
    - Use code to check numerical claims when possible
    - Be explicit about confidence levels
    
    
    BUILDING ON PREVIOUS RESEARCH:
    
    Since Grok is stateless, effective follow-ups require context:
    
    Good follow-up pattern:
      "Previous research found [key findings]. Now dig deeper into [aspect]:
       - Specifically investigate [sub-topic]
       - Focus on [sources] not yet checked
       - Compare with [new angle]"
       
    Include enough context that Grok can:
    - Understand what's already known (avoid duplication)
    - Recognize what's new about this query
    - Build logically on previous findings
    
    If research trail was used previously, reference specific URLs or sources
    found to help Grok understand what has been covered.
    
    
    SYNTHESIS AND ANALYSIS REQUESTS:
    
    Grok excels at synthesis when you're clear about what matters:
    
      "Synthesize: What's the consensus among experts?"
      "Analyze: Is this trend real or hype?"
      "Compare: Which option is better for [use case]?"
      "Evaluate: What are the risks and benefits?"
      "Explain: Why do researchers disagree about this?"
      
    Be explicit about perspectives you want:
    - "Give me both sides of the argument"
    - "Focus on practical implications not theory"
    - "I need decision-making insight not just facts"
    - "Explain like I'm [your background level]"
    
    
    OUTPUT FORMATTING:
    
    Grok structures responses intelligently but respects your preferences:
    
      "Structure as: Executive summary, detailed findings, conclusion"
      "Format as: Table comparing [items] by [attributes]"
      "Organize by: Time period / Category / Source credibility"
      "Keep response under 500 words but include all key points"
      "Be comprehensive - I want full detail on each item"
      
    For long systematic research, Grok automatically uses:
    - Clear headings for major sections
    - Bullet points for lists and key points
    - Separation between analyzed items
    - Summary sections when appropriate
    
    You can request different structures, but defaults are designed for
    scannability even in extensive responses.
    
    
    COMMON MISTAKES TO AVOID:
    
    Don't over-specify tool usage:
      Bad: "Use x_keyword_search then browse_page then code_execution on..."
      Good: "Research [topic] by checking X, official docs, and analyzing data"
      Why: Grok's agency is its strength - let it choose the path
      
    Don't assume memory:
      Bad: "Tell me more about that"
      Good: "Building on the previous finding that [X], now investigate [Y]"
      Why: Stateless execution requires explicit context
      
    Don't use Grok for things you know:
      Bad: "What's the capital of France?" (you know this, or trained data knows)
      Good: "What's the current political situation in France?" (needs research)
      Why: Grok is for information gathering, not retrieving trained knowledge
      
    Don't be vague on scope:
      Bad: "Tell me about AI"
      Good: "Research the current state of multimodal AI models in 2024-2025"
      Why: Specificity enables focused, useful research
      
    Don't forget time context for current events:
      Bad: "What do people think about [recent event]?"
      Good: "Search X for reactions to [recent event] from [time period]"
      Why: Explicitly requesting X search + time bounds gets current discourse
    
    
    WHEN NOT TO USE GROK:
    
    Reserve Grok for situations requiring external information. Skip Grok when:
    
    - You're asking about trained knowledge easily accessible without research
    - Pure reasoning tasks with no factual dependencies
    - Creative writing, brainstorming, or ideation
    - Clarifying questions in conversation
    - Explaining concepts already understood
    - Responding to user feedback or questions
    
    Grok adds value when:
    - Current/recent information needed (especially last 24-48 hours)
    - Expert opinions and discourse matter
    - Claims need verification or fact-checking
    - Multiple sources need synthesis
    - Numerical analysis or calculations required
    - Systematic research of multiple items necessary
    
    
    EXPERIMENTATION AND LEARNING:
    
    Grok's flexibility means you should experiment:
    
    - Try different prompt styles and see what works for your needs
    - Vary levels of detail and structure
    - Test different ways of requesting the same information
    - Notice what works and iterate
    
    You'll develop intuition for:
    - When to be brief vs detailed
    - What context matters for different queries
    - How to structure complex research effectively
    - When X vs web is more appropriate
    - When code execution adds value
    
    There's no single "correct" way to prompt Grok. Effective prompting is 
    about clear communication of your goals and adaptation to the task at hand.
    
    ═══════════════════════════════════════════════════════════════════════════
    EXAMPLE APPROACHES
    ═══════════════════════════════════════════════════════════════════════════
    
    The following examples demonstrate the range and depth of Grok's capabilities.
    They are NOT templates to copy - they're inspiration showing different styles
    and approaches. Your prompts should adapt to your specific needs and can be
    completely different in structure.
    
    
    EXAMPLE 1: Comprehensive Financial Analysis
    (Shows: Heavy systematic workload, code execution, multi-year data)
    
    "Perform investment due diligence on [company/startup]:
    
    1. Company fundamentals:
       - Browse official website for products, team, business model
       - Search for recent funding announcements and valuations
       - Find technical documentation or whitepapers if available
    
    2. Financial analysis (if public):
       - Search for financial reports or disclosed metrics
       - Use code to calculate key ratios:
         * Revenue growth rates (YoY, QoQ)
         * Profit margins
         * Burn rate if startup
         * Compare to industry benchmarks
       - Track financial trends over available periods
    
    3. Market positioning:
       - Identify 3-5 direct competitors
       - Browse competitor sites for comparison
       - Search X for industry analyst opinions on this space
       - Note market size and company's share if available
    
    4. Sentiment analysis:
       - Search X for customer experiences (filter for genuine users not marketing)
       - Look for both enthusiastic supporters and critical voices
       - Search for any controversies, red flags, or concerns
       - Check employee sentiment if available
    
    5. Expert opinions:
       - Search X from relevant industry analysts and investors
       - Find any thought leaders discussing this company/sector
       - Note consensus view if one exists
    
    6. Risk assessment:
       - Regulatory concerns in this space
       - Competitive threats
       - Technology risks or limitations
       - Market timing considerations
    
    Synthesize into: Investment thesis with clear sections on strengths, 
    concerns, financial health, market position, and recommendation with 
    confidence level. Use code to create comparison matrices where helpful.
    
    Include research trail: True"
    
    
    EXAMPLE 2: Deep Content Analysis
    (Shows: Systematic multi-item research, historical X searches, long-form output)
    
    "Analyze the complete first season of [TV series] (all episodes):
    
    For each episode systematically:
    1. Browse episode wiki or IMDB page for:
       - Plot summary and key developments
       - Air date and episode number
       - Any notable production information
    
    2. Search X for viewer reactions when it originally aired:
       - Use episode air date +/- 1 day for time bound
       - Look for both enthusiastic and critical takes
       - Note any specific moments that generated discussion
    
    3. Check for any critical reception or reviews
    
    Then synthesize across the full season:
    - How does the narrative arc develop episode by episode?
    - What major themes emerge and how do they evolve?
    - Where is pacing effective or problematic?
    - How did audience reception change over the season?
    - Any standout episodes or particularly weak ones?
    - If this is not the first season, how does it compare?
    
    Structure response as:
    - Brief per-episode breakdown (3-5 bullets each with air date)
    - Comprehensive season-level analysis
    - Final assessment of season quality and evolution
    
    This will be a long response (likely 3000-5000 words) - that's expected
    for systematic analysis of 10+ episodes."
    
    
    EXAMPLE 3: Timeline Catchup Research
    (Shows: Temporal research, context building, narrative synthesis)
    
    "Get me fully caught up on [major ongoing situation/conflict/policy change]:
    
    I need the complete timeline from beginning to present:
    
    1. Origins: What sparked this initially?
       - Web search for background on how it started
       - Identify key dates and triggering events
    
    2. Primary sources:
       - Browse official statements, legislation, or policy documents
       - Find announcements from key organizations/governments involved
    
    3. Evolution over time:
       - Search X for how discourse evolved (use time-bounded searches)
       - Identify major turning points and why they mattered
       - Note shifts in public opinion or expert consensus
    
    4. Key players and positions:
       - Who are the major stakeholders?
       - What are their stated positions and actual actions?
       - Search official accounts and statements
    
    5. Current state:
       - What's the situation as of today?
       - What's the active debate about?
       - What are the near-term implications?
    
    Use code if helpful to organize timeline chronologically with dates.
    
    Provide complete chronological narrative from start to present, with context
    for why each development mattered and how we got to the current state. Help
    me understand not just what happened but why it progressed this way and what
    the competing perspectives are.
    
    Structure with: Timeline summary (bullet points with dates), then detailed
    narrative analysis, then current state assessment."
    
    
    EXAMPLE 4: Daily Intelligence Brief
    (Shows: Quick but sophisticated monitoring, efficient synthesis)
    
    "Daily AI development briefing:
    
    Quick checks across multiple sources:
    - X: Posts from major AI labs (OpenAI, Anthropic, Google, Meta, xAI) 
         from last 24 hours with min_faves:50 filter
    - X: Semantic search for 'AI breakthrough' OR 'AI announcement' OR 'new model'
         from last 24 hours, verified accounts
    - Web: Quick check of major AI news sites for headlines
    - X: Any trending discussions in AI community (high engagement posts)
    
    Synthesize into: 5-7 bullet points maximum covering what actually matters today.
    For each item: one-sentence summary + why it's significant.
    
    Keep total response under 300 words - prioritize signal over noise. Skip minor
    updates and focus on substantive developments only."
    
    
    EXAMPLE 5: Simple Factual Query
    (Shows: Grok handles simple queries efficiently too)
    
    "What is the current weather in Tokyo?"
    
    
    ═══════════════════════════════════════════════════════════════════════════
    GETTING STARTED
    ═══════════════════════════════════════════════════════════════════════════
    
    Start simple and experiment:
    
    - Begin with straightforward queries to understand Grok's style
    - Try a moderately complex research task to see multi-source synthesis
    - Experiment with different prompt structures to find what works for you
    - Pay attention to what level of detail and context produces useful results
    - Gradually increase complexity as you build intuition
    
    Grok adapts to you. There's no need to master everything immediately. Each
    interaction will help you understand how to communicate your needs effectively.
    
    The key is clear communication of what you want to accomplish. Grok will
    handle the complexity of figuring out how to achieve it.
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        prompt: Your research request, question, or task description. Can range
                from a few words to extensively detailed multi-phase instructions.
                
        include_research_trail: Whether to include detailed methodology 
                documentation showing Grok's research process step-by-step.
                Default False.
                
        async_mode: If True (default), returns a task_id immediately. Research
                executes in background; retrieve with grok_check_task(task_id).
                If False, waits for completion (only use if your client has no
                timeout limits). Default True.
    
    Returns:
        Research findings, analysis, or answer to your query. Length varies from
        brief responses to extensive multi-thousand word research reports based
        on your prompt. If async_mode=True, returns a task_id instead - retrieve
        results with grok_check_task().
    """
    
    # Build the full prompt with research trail request if needed
    full_prompt = prompt
    
    if include_research_trail:
        full_prompt += """

---

IMPORTANT - Document your research process:

At the END of your response, include a section called:
"## Research Process (Step-by-Step)"

For EVERY action you take, document in chronological order:

**Step N: [Brief action description]**
- Reasoning: Why you chose this approach
- Tool: Exact tool name (web_search, browse_page, x_keyword_search, x_semantic_search, code_execution, etc.)
- Parameters:
  * If web_search: Complete query string
  * If browse_page: Full URL + instructions given
  * If x_keyword_search: Query + limit + mode + any filters
  * If x_semantic_search: Query + date range + other parameters
  * If code_execution: The complete code block
- Found: Summary of what this step returned
- Next: What you decided to do next based on these results

If any step failed or had issues (access errors, no results, etc.), 
document that and explain how you adapted.

Also include at the end:

## Research Summary
- Total steps: [count]
- Tools used: [tool_name (N times), tool_name (N times), ...]
- Sources accessed: [All URLs browsed, X posts found, etc.]
- Key sources: [The 2-3 most important sources]
- Search time ranges: [Date ranges if X searches were used]
- Code executed: [Yes/No, if yes what for]
- Overall approach: [2-3 sentence summary of research strategy]

Be thorough and honest about your methodology.
"""
    
    # Handle async mode
    if async_mode:
        cleanup_old_tasks()
        task_id = f"grok_{uuid.uuid4().hex[:8]}"
        tasks[task_id] = {
            "status": "pending",
            "created_at": time.time(),
            "completed_at": None,
            "result": None,
            "error": None,
            "prompt": prompt[:200]  # Store snippet for debugging
        }
        
        # Start background task
        asyncio.create_task(execute_task_background(task_id, full_prompt))
        
        return f"Task started: {task_id}\n\nUse grok_check_task('{task_id}') to retrieve results when ready."
    
    # Synchronous execution
    result = await call_grok_api(full_prompt)
    return result


@mcp.tool()
async def grok_check_task(task_id: str) -> str:
    """
    Check status and retrieve results of an async Grok task.
    
    When grok() is called with async_mode=True, it returns a task_id. Use this
    tool to check if the research is complete and get the results.
    
    Returns one of:
    - "Status: running" → Task in progress, check again in 15-30s
    - Full research results → Task completed (no "Status:" prefix)
    - "Status: failed - [error]" → Task failed, see error message
    - "Status: not_found" → Invalid/expired task_id (kept 24hrs, max 100 tasks)
    
    Workflow:
    1. grok(prompt, async_mode=True) → Returns task_id
    2. Wait 15-60s depending on complexity
    3. grok_check_task(task_id) → Check status
    4. If completed, full results returned
    5. If running, wait and check again
    
    Args:
        task_id: Task identifier from grok() with async_mode=True
        
    Returns:
        Status update or full research results when completed
    """
    
    cleanup_old_tasks()
    
    if task_id not in tasks:
        return f"Status: not_found - Task '{task_id}' does not exist or has expired (tasks kept 24 hours, max 100 tasks stored)."
    
    task = tasks[task_id]
    status = task["status"]
    
    if status == "pending":
        return "Status: pending - Task queued but not yet started. Check again shortly."
    
    elif status == "running":
        elapsed = int(time.time() - task["created_at"])
        return f"Status: running - Task in progress (elapsed: {elapsed}s). Check again in 15-30 seconds."
    
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
