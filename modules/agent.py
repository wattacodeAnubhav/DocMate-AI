import json
import os
from groq import Groq
from dotenv import load_dotenv
from pydantic import ValidationError
from .structures import GraphData, MultiChartDashboard

# ==========================================
# 1. ENVIRONMENT & CLIENT INITIALIZATION
# ==========================================

# Load and sanitize the API key from the .env file. 
# The sanitization step removes any extraneous quotes or whitespace that might have been accidentally included when copying
load_dotenv(override=True)
raw_key = os.environ.get("GROQ_API_KEY", "")
sanitized_key = raw_key.replace('"', '').replace("'", "").strip()
client = Groq(api_key=sanitized_key)
HEAVY_MODEL = "llama-3.3-70b-versatile" # For deep reasoning and extraction
FAST_MODEL = "llama3-8b-8192"           # For lightning-fast classification

# ==========================================
# 2. PERSONA & AUDIT REGISTRY
# ==========================================

# The PERSONA_REGISTRY is a critical component that allows us to inject specific cognitive focuses and anti-hallucination guardrails into the LLM's reasoning process. 
# By selecting a persona, we can steer the LLM to prioritize certain types of information and analysis, which is especially important when dealing with complex documents that contain a mix of quantitative data, qualitative insights, and strategic implications. 
# Each persona is designed to excel in a particular domain of analysis, ensuring that the LLM's output is not only accurate but also relevant and actionable for the user's specific needs.
PERSONA_REGISTRY = {
    "Researcher": """You are an elite Academic Researcher. Your primary directive is exhaustive synthesis and strict factual fidelity.
COGNITIVE FOCUS: Cross-reference all provided context. Identify nuances, edge cases, and source discrepancies. Never summarize when detail is available.
ANTI-HALLUCINATION: You are strictly bound to the provided context. If a detail is absent, state explicitly: "Insufficient data to determine [X]." """,
    
    "Analyst": """You are a Senior Data Analyst. Your primary directive is statistical rigor and pattern extraction.
COGNITIVE FOCUS: Strip away narrative fluff; focus strictly on numbers, trends, and correlations.
ANTI-HALLUCINATION: Do not infer trends that are not mathematically supported by the text. Treat the context as an immutable database. """,
    
    "BI Expert": """You are a Strategic Business Intelligence Expert. Your primary directive is translating raw data into executive action.
COGNITIVE FOCUS: Focus on the 'So What?'. Identify process bottlenecks, efficiency gains, and ROI indicators.
ANTI-HALLUCINATION: Base all strategic recommendations strictly on the operational data provided. Do not invent hypothetical business scenarios. """,
    
    "Teacher": """You are an Expert Educator. Your primary directive is making complex information universally accessible.
COGNITIVE FOCUS: Break down complex mechanisms into sequential, step-by-step logic. Anticipate where a beginner might get confused.
ANTI-HALLUCINATION: While analogies are encouraged, the core mechanics being explained must strictly match the provided text. """
}

# Universal audit instructions that are appended to every prompt to enforce a rigorous self-checking process. 
# This is designed to mitigate hallucinations and ensure that the LLM's output is grounded in the provided context, while also adhering to the specified output format.
UNIVERSAL_AUDIT = """
OUTPUT PIPELINE & CRITICAL RENDERING INSTRUCTION:
1. <thinking> Map out facts and plan the structure. </thinking>
2. <draft> Write the initial response or formulate the JSON payload. </draft>
3. <audit> Answer these 4 questions internally: 
   - Fidelity: Did I strictly use the provided context?
   - Intent Match: Does my output format exactly match the requested schema/intent?
   - Data Binding: Do the exact strings in my 'x_axis' and 'y_axes' arrays PERFECTLY match the keys I created in the 'dataset' array?
   - Syntax Check: Are all Markdown tables closed? Is the JSON valid?
   Correct any column name mismatches internally before rendering the final output. </audit>
4. Provide the final, un-tagged output below the </audit> tag.
"""

# ==========================================
# 2. THE GATEKEEPER: FAST INTENT ROUTER
# ==========================================

# The Gatekeeper is a critical component that serves as the first line of defense against misrouted queries and ensures that each user request is handled by the most appropriate agent. 
# By performing a rapid classification of the query's intent, it prevents complex, unstructured questions from being sent to strict SQL agents that are not designed to handle them, thereby reducing the likelihood of errors and improving overall system efficiency. 
# This function uses a lightweight prompt and a fast model to achieve near-instantaneous routing decisions, which is essential for maintaining a responsive user experience in an enterprise-grade application.
def classify_query_intent(query: str) -> str:
    """
    Ultra-fast pre-processing step to route the query to the correct data engine.
    Mitigates Chokepoint A by preventing fuzzy questions from hitting strict SQL databases.
    """
    router_prompt = f"""You are a high-speed intent classifier for a data application.
    Analyze the query and determine the correct data retrieval route:
    1. 'SQL' - The user is asking for strict tabular aggregations, math, or tabular filtering (e.g., 'group by region', 'sum of revenue', 'average cost').
    2. 'VECTOR' - The user is asking for concepts, summaries, processes, standard QA, or dashboard generation from unstructured text.
    
    Output ONLY valid JSON matching this schema: {{"route": "SQL" or "VECTOR"}}
    
    Query: {query}
    """
    
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0,
            max_tokens=20
        )
        raw_output = response.choices[0].message.content.strip()
        
        # Parse the JSON safely
        clean_output = raw_output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_output)
        return parsed.get("route", "VECTOR").upper()
    except Exception as e:
        print(f"Gatekeeper classification failed, defaulting to VECTOR: {e}")
        return "VECTOR"

# ==========================================
# 3. TABULAR / SQL AGENT
# ==========================================

# This agent is designed to handle queries that require structured data manipulation and retrieval. 
# It uses the Gatekeeper's classification to ensure that only appropriate queries are routed here, and it generates SQL code based on the user's natural language request and a provided database schema. 
def generate_tabular_response(query: str, db_schema: str = "No schema provided"):
    """
    Dedicated handler for SQL/Tabular queries (Routed here by The Gatekeeper).
    Designed to convert natural language into strict SQL execution plans.
    """
    # Note: In Phase 2, this will execute against actual DB engines via SQLAlchemy/Pandas.
    # For now, it prepares the SQL syntax payload.
    prompt = f"""You are an elite SQL Data Analyst.
    Given the following database schema, write a highly optimized SQL query to answer the user's request.
    
    SCHEMA:
    {db_schema}
    
    USER QUERY: {query}
    
    Respond strictly with raw SQL code. Do not include explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model=HEAVY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        sql_payload = response.choices[0].message.content.strip()
        
        return {
            "intent": "TABULAR_SQL",
            "payload": sql_payload,
            "traced_context": []
        }
    except Exception as e:
        return {"intent": "ERROR", "payload": f"Tabular Agent Error: {str(e)}", "traced_context": []}

# ==========================================
# 4. CORE VECTOR AGENT ROUTER
# ==========================================

# This is the main agent function that handles all queries routed to the vector-based processing pipeline.
# It takes the user's query, the retrieved context from the document vector space, and the conversation memory to generate a response. 
# The function is designed to be flexible and can enforce specific intents that dictate the format and structure of the response, such as generating dashboards, extracting metrics, or creating visual graphs.
def generate_agent_response(query, context, memory, enforced_intent=None, active_persona="Analyst"):
    """
    Standard single-pass routing agent upgraded with Persona and Audit logic.
    """
    formatted_context = "NO DOCUMENTS FOUND." if not context else "".join(
        [f"Source: {chunk['metadata'].get('source_file')} (Page {chunk['metadata'].get('page_number')})\nText: {chunk.get('text', '')}\n\n" for chunk in context]
    )

    # Fetch the active persona
    persona_instruction = PERSONA_REGISTRY.get(active_persona, PERSONA_REGISTRY["Analyst"])

    base_instruction = f"""{persona_instruction}
Evaluate the user's query strictly against the knowledge base.

KNOWLEDGE BASE:
{formatted_context}
"""

    # --- DYNAMIC PROMPT ISOLATION ---
    if enforced_intent == "TABLE_GENERATION":
        schema_instruction = """
ROUTING & FORMATTING INSTRUCTIONS:
You MUST output ONLY a well-formatted Markdown Table containing the requested data. 
DO NOT include any conversational filler, introductory text, or concluding text. 
Start immediately with the markdown table header (e.g., | Column 1 | Column 2 |).
Include a final column in the table called 'Source' and cite the filename and page number.
"""
    elif enforced_intent == "STRICT_EXTRACTION":
        schema_instruction = """
ROUTING & FORMATTING INSTRUCTIONS:
Your ONLY job is to extract direct quotes, exact numbers, and raw facts.
Output a bulleted list. DO NOT add conversational filler, summaries, or explanations. 
Every single bullet point MUST end with a strict citation [filename, page].
"""
    elif enforced_intent == "SUMMARIZATION":
        schema_instruction = """
ROUTING & FORMATTING INSTRUCTIONS:
Provide a rapid, high-level summary using EXACTLY 3 to 4 bullet points. 
Do not perform deep analysis. Be brief and concise. Cite sources [filename, page].
"""
    elif enforced_intent == "EXPLANATION":
        schema_instruction = """
ROUTING & FORMATTING INSTRUCTIONS:
Act as an expert tutor. Break down the requested concept into simple, easy-to-understand terms.
Structure your response clearly using an Executive Summary followed by bullet points. 
Cite sources [filename, page].
"""
    elif enforced_intent in ["DASH_BASIC", "DASH_MULTI"]:
        chart_count = "EXACTLY 1 chart" if enforced_intent == "DASH_BASIC" else "2 to 4 distinct charts"
        schema_instruction = f"""
ROUTING & FORMATTING INSTRUCTIONS:
You MUST output ONLY a single, valid JSON object. DO NOT include any text outside the JSON block. Your entire response must start with {{ and end with }}.
You must generate {chart_count} in the 'charts' array.

CRITICAL PERSONA LINK: 
The metrics you extract and the charts you configure MUST strictly reflect your active Persona's cognitive focus.

CRITICAL DATA BINDING RULES:
1. INDEPENDENT DATA: Every chart MUST contain its OWN 'dataset' array. Do not try to combine different metrics into one global dataset.
2. EXACT MATCHING: The string you provide in 'x_axis' MUST EXACTLY MATCH a key used in THAT chart's 'dataset' (Case-Sensitive).
3. EXACT MATCHING: Every string you provide in 'y_axes' MUST EXACTLY MATCH numerical keys used in THAT chart's 'dataset' (Case-Sensitive).
4. NO PLACEHOLDERS: Do NOT use generic hallucinated axes like "Metric" or "Value". You must use the real column names (e.g., "Month", "Revenue").

The JSON must strictly match this schema:
{{
    "dashboard_title": "Title String",
    "executive_summary": "In-depth summary text analyzing findings from the document",
    "metrics": [
        {{"label": "Metric Name", "value": 123.4, "unit": "%", "trend": "up", "source_citation": "document.pdf"}}
    ],
    "charts": [
        {{
            "chart_title": "Revenue vs Cost Trend",
            "reasoning": "A bar chart effectively compares Revenue and Cost.",
            "chart_type": "bar", 
            "x_axis": "Category",
            "y_axes": ["Revenue", "Cost"],
            "dataset": [
                {{"Category": "Q1", "Revenue": 1000, "Cost": 800}},
                {{"Category": "Q2",  "Revenue": 1500, "Cost": 900}}
            ]
        }}
    ]
}}
Note: chart_type MUST be exactly 'bar', 'line', or 'scatter'.
"""
    elif enforced_intent == "VISUAL_GRAPH":
        schema_instruction = """
ROUTING & FORMATTING INSTRUCTIONS:
You MUST output ONLY a single, valid JSON object mapping a dense, highly interconnected entity network. DO NOT include any text outside the JSON block.

NETWORK TOPOLOGY & GRAPH DENSITY RULES:
1. STRICTLY AVOID LINEAR PATHS: Do not just generate a single straight line of sequential steps (e.g., A -> B -> C -> D).
2. EXTRACT COMPLEX RELATIONSHIPS: Actively look for cross-functional dependencies, lateral connections, feedback loops, and multi-node intersections within the text.
3. BE GRANULAR: Break major concepts down into their contributing sub-components. Map how secondary or minor nodes influence, require, or feed back into core nodes to create a dense, highly clustered "web" layout.

CRITICAL PERSONA LINK: 
The nodes and relationships you choose to map MUST strictly reflect your active Persona's cognitive focus (e.g., map ROI/bottlenecks for BI Expert, map edge-cases/risks for Researcher, map statistical pipelines for Analyst). Do not just extract generic process steps.

The JSON must strictly match this schema:
{
    "edges": [
        {"source": "Entity A", "target": "Entity B", "relationship": "Action/Connection"}
    ]
}
"""
    else:
        schema_instruction = """
ROUTING & FORMATTING INSTRUCTIONS:
Follow this structure:
- Begin with a <thinking> block to reason through the answer.
- Follow with an "### Executive Summary" (1-2 sentences).
- Follow with an "### In-Depth Analysis" using bullet points.
- Always cite sources [filename, page].
"""

    system_instruction = base_instruction + schema_instruction + UNIVERSAL_AUDIT

    messages = [{"role": "system", "content": system_instruction}]
    for msg in memory[-3:]: 
        messages.append(msg)
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model=HEAVY_MODEL,
            messages=messages,
            temperature=0.1
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        return {"intent": "ERROR", "payload": f"API Error: {str(e)}", "traced_context": context}

    # ==========================================
    # 5. DETERMINISTIC PAYLOAD PARSING
    # ==========================================
    if enforced_intent in ["DASH_BASIC", "DASH_MULTI", "VISUAL_GRAPH"]:
        try:
            clean_output = raw_output
            if "</audit>" in clean_output:
                clean_output = clean_output.split("</audit>")[-1].strip()
                
            clean_output = clean_output.replace("```json", "").replace("```", "").strip()
            json_start = clean_output.find("{")
            json_end = clean_output.rfind("}") + 1
            json_str = clean_output[json_start:json_end]
            parsed_dict = json.loads(json_str)

            if enforced_intent in ["DASH_BASIC", "DASH_MULTI"]:
                validated = MultiChartDashboard.model_validate(parsed_dict)
                return {"intent": "METRICS_DASHBOARD", "payload": validated.model_dump(), "traced_context": context}
            
            if enforced_intent == "VISUAL_GRAPH":
                json_str_graph = json_str.replace('"from":', '"source":').replace('"to":', '"target":')
                validated = GraphData.model_validate(json.loads(json_str_graph))
                return {"intent": "VISUAL_GRAPH", "payload": validated.model_dump(), "traced_context": context}

        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {"intent": "ERROR", "payload": "Failed to generate structured data. Try adjusting your prompt.", "traced_context": context}

    # Pass-through intent for the UI renderer
    if enforced_intent:
        return {"intent": enforced_intent, "payload": raw_output, "traced_context": context}

    return {"intent": "STANDARD_QA", "payload": raw_output, "traced_context": context}

# ==========================================
# 6. MULTI-STEP PROMPT CHAINING
# ==========================================

# This function demonstrates how to implement a complex, multi-step prompt chain that guides the LLM through a structured reasoning process. 
# It is designed to generate a comprehensive report based on the provided context, ensuring that the output is grounded in the source material and follows a logical structure. Each step of the chain has a specific purpose, from fact extraction to outline creation to final synthesis, and the function includes error handling to manage any issues that arise during the process.
def generate_chained_report(query, context):
    """
    A 3-step prompt chain for generating high-quality, hallucination-free 
    reports based strictly on local document context.
    """
    formatted_context = ""
    if not context:
        formatted_context = "NO DOCUMENTS FOUND."
    else:
        for chunk in context:
            meta = chunk["metadata"]
            formatted_context += f"Source: {meta.get('source_file', 'Unknown')} (Page {meta.get('page_number', 'N/A')})\nText: {chunk.get('text', '')}\n\n"

    try:
        # LINK 1: Fact Extraction
        prompt_1 = f"""
        You are an exact data extractor. Your ONLY job is to read the KNOWLEDGE BASE and extract every single fact, number, and concept related to: "{query}".
        Do not write an essay. Use bullet points. If it is not in the text, do not include it.
        
        KNOWLEDGE BASE:
        {formatted_context}
        """
        response_1 = client.chat.completions.create(
            model=HEAVY_MODEL, messages=[{"role": "user", "content": prompt_1}], temperature=0.0
        )
        extracted_facts = response_1.choices[0].message.content

        # LINK 2: Structural Outline
        prompt_2 = f"""
        You are a curriculum architect. Based ONLY on the facts below, create a logical, highly structured outline for a comprehensive report.
        Do not write the full report yet. Just write the headings and sub-headings.
        
        FACTS:
        {extracted_facts}
        """
        response_2 = client.chat.completions.create(
            model=HEAVY_MODEL, messages=[{"role": "user", "content": prompt_2}], temperature=0.1
        )
        outline = response_2.choices[0].message.content

        # LINK 3: Final Synthesis
        prompt_3 = f"""
        You are an elite technical writer. Your task is to write the final, comprehensive document.
        
        RULES:
        1. You MUST follow the EXACT structure of the OUTLINE provided below.
        2. You MUST strictly use the information from the RAW FACTS provided below. 
        3. Do not invent any new information.
        4. Write in a highly professional, educational, and insightful tone.
        5. Cite your sources using [filename, page].
        
        OUTLINE:
        {outline}
        
        RAW FACTS:
        {extracted_facts}
        """
        response_3 = client.chat.completions.create(
            model=HEAVY_MODEL, messages=[{"role": "user", "content": prompt_3}], temperature=0.3
        )
        final_report = response_3.choices[0].message.content

        return {
            "intent": "STANDARD_QA",
            "payload": final_report,
            "traced_context": context
        }

    except Exception as e:
        return {
            "intent": "ERROR",
            "payload": f"Chain Generation Failed: {str(e)}. Please try again.",
            "traced_context": context
        }