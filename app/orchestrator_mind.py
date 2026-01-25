"""
Clinical Orchestrator (LangGraph) - REASONING ENGINE
----------------------------------------------------
Responsibility:
- Handle USER QUERIES (Online)
- Intent detection -> Retrieval -> Answer
- Wraps the LangGraph logic in a callable API
"""

import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Optional, Annotated

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Agent Imports
from agents.retrieval_agent import RetrieverAgent
from agents.drug_intelligence_agent import fetch_drug_intelligence
from agents.literature_search_agent import literature_search
from agents.web_search_agent import web_search

# =====================================================
# 1. SETUP
# =====================================================

def _init_llm():
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found in environment. "
            "Please set GEMINI_API_KEY in your .env file."
        )

    # Expose for Google SDKs
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,   # deterministic
        max_tokens=None,
        max_retries=2,
    )

try:
    llm = _init_llm()
except Exception:
    llm = None 

# =====================================================
# 2. STATE DEFINITION
# =====================================================

class OrchestratorState(TypedDict):
    user_query: str
    intent: Optional[str]
    # NEW: Generic slot to hold the specific thing we are searching for 
    # (e.g., drug name, search keywords, or specific question)
    tool_query: Optional[str] 
    
    entities: Optional[List[Dict]]
    retrieved_context: List[Dict]
    tool_outputs: List[Dict]
    compressed_context: str
    final_answer: str
    sources: List[Dict] 
    messages: Annotated[List[BaseMessage], add_messages]

# =====================================================
# 3. NODES
# =====================================================

def detect_intent(state: OrchestratorState) -> OrchestratorState:
    # We update the prompt to ask for JSON output
    prompt = f"""
    You are a clinical AI ORCHESTRATOR.
    Analyze the query and extract the INTENT and the specific PARAMETER (tool_query).

    AGENTS:
    1. dashboard_summary
   - Use when the user wants an OVERVIEW or SUMMARY
   - Examples:
     • "Give me a summary of this patient's condition"
     • "Show patient clinical overview"
     • "Summarize the medical record"

2. qa
   - Use when the user asks a SPECIFIC clinical question
   - Requires precise answer from patient data
   - Examples:
     • The clinical impression is that the patient's unconjugated bilirubin level is elevated at what level?
     • "Is elevated troponin indicative of MI?"
     • "What does this lab value suggest?"
     • "Why was this medication stopped?"

3. drug_intelligence
   - Use when the user asks about a DRUG
   - Includes: mechanism, properties, contraindications, safety
   - Examples:
     • "Is aspirin safe in pregnancy?"

4. literature_search
   - Use when the user wants RESEARCH PAPERS, GUIDELINES, or STUDIES
   - Query-based (not patient-specific)
   - Examples:
     • "What are the side effects of metformin?"
     • "Recent studies on Imatinib resistance"
     • "WHO guidelines for dengue"
     • "Latest research on CKD treatment"

5. web_search
   - Use ONLY when information is:
     • very recent
     • not likely indexed in PubMed
     • policy / announcement / breaking update
   - Examples:
     • "New FDA warning released yesterday"
     • "Recent medical policy update"

    EXAMPLES:
    - "Side effects of metformin?" -> {{ "intent": "drug_intelligence", "tool_query": "metformin" }}
    - "Latest dengue guidelines?" -> {{ "intent": "literature_search", "tool_query": "dengue guidelines 2025" }}
    - "Is troponin elevated?" -> {{ "intent": "qa", "tool_query": "Is troponin elevated?" }}

    User Query: "{state['user_query']}"

    Output valid JSON only: {{ "intent": "...", "tool_query": "..." }}
    """
    
    try:
        response = llm.invoke(prompt).content.strip()
        # Clean potential markdown
        response = response.replace("```json", "").replace("```", "")
        parsed = json.loads(response)
        
        state["intent"] = parsed["intent"]
        state["tool_query"] = parsed["tool_query"]
    except Exception:
        # Fallback if JSON fails
        state["intent"] = "qa"
        state["tool_query"] = state["user_query"]
        
    state["sources"] = [] 
    return state

async def retrieval_node(state: OrchestratorState) -> OrchestratorState:
    retriever = RetrieverAgent()
    
    # Use the tool_query we extracted (which handles the QA question logic)
    query_text = state.get("tool_query") or state["user_query"]

    if state["intent"] == "dashboard_summary":
        results = retriever.retrieve_for_summary()
    else:
        results = retriever.retrieve_for_qa(query=query_text)

    state["retrieved_context"] = results
    
    # Source Extraction
    doc_sources = []
    seen = set()
    for r in results:
        src = r.get("metadata", {}).get("source", "Unknown Document")
        if src not in seen:
            doc_sources.append({
                "type": "internal_document", 
                "title": src, 
                "agent": "Vector Retrieval"
            })
            seen.add(src)
            
    state["sources"] = doc_sources
    return state

def drug_intelligence_node(state: OrchestratorState) -> OrchestratorState:
    # 1. First check if we extracted a drug name from the query
    extracted_drug = state.get("tool_query")
    
    drugs = []
    if extracted_drug:
        # If the LLM found a drug name in the question, use it
        drugs = [extracted_drug]
    else:
        # Fallback: Check if we have entities from ingestion (legacy support)
        # (This helps if you ever wire up the offline pipeline to this node)
        entities = state.get("entities", [])
        for e in entities:
            if e.get("type") == "medication":
                drugs.append(e["entity"])

    if not drugs:
        state["tool_outputs"] = []
        return state

    # 2. Fetch Data
    outputs = fetch_drug_intelligence(drugs)
    state["tool_outputs"] = outputs
    
    state["sources"] = [{
        "type": "database",
        "title": "PubChem / OpenFDA",
        "url": "https://pubchem.ncbi.nlm.nih.gov",
        "agent": "Drug Intelligence"
    }]
    return state

def literature_node(state: OrchestratorState) -> OrchestratorState:
    query = state.get("tool_query") or state["user_query"]
    output = literature_search(query) 
    state["tool_outputs"] = [output]
    
    papers = []
    if isinstance(output, dict) and "results" in output:
        for item in output["results"]:
            papers.append({
                "type": "academic_paper",
                "title": item.get("title", "Unknown Paper"),
                "url": item.get("link", item.get("url")),
                "agent": "PubMed/Arxiv"
            })
            
    state["sources"] = papers
    return state

def web_search_node(state: OrchestratorState) -> OrchestratorState:
    query = state.get("tool_query") or state["user_query"]
    results = web_search(query)
    state["tool_outputs"] = results
    
    web_sources = []
    for r in results:
        web_sources.append({
            "type": "web",
            "title": r.get("title"),
            "url": r.get("url"),
            "agent": "Tavily Search"
        })
        
    state["sources"] = web_sources
    return state

def compress_context(state: OrchestratorState) -> OrchestratorState:
    raw_context = state.get("retrieved_context", []) + state.get("tool_outputs", [])

    if not raw_context:
        state["compressed_context"] = ""
        return state

    text_blob = "\n\n".join(str(c) for c in raw_context)

    if len(text_blob) < 6000:
        state["compressed_context"] = text_blob
        return state

    prompt = f"""
    Condense this medical context, preserving facts and numbers:
    {text_blob}
    """
    condensed = llm.invoke(prompt).content
    state["compressed_context"] = condensed
    return state

def answer_node(state: OrchestratorState) -> OrchestratorState:
    context = state.get("compressed_context", "")
    sources = state.get("sources", [])
    
    # 1. Format sources with Index, Title, and URL for the LLM
    source_list = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "Unknown Source")
        url = s.get("url", "N/A")
        # Format: [1] Title (URL: ...)
        source_list.append(f"[{i}] {title} (URL: {url})")

    source_text = "\n".join(source_list)

    # 2. Update Prompt to request Citations & Links
    system_prompt = f"""You are a clinical assistant. 
    Answer the user question using ONLY the provided evidence.
    
    EVIDENCE:
    {context}
    
    SOURCE LIST:
    {source_text}
    
    INSTRUCTIONS:
    - Answer clearly and professionally.
    - When referencing specific data, cite the source index like [1] or [2].
    - If a source has a valid URL (not "N/A"), provide it as a clickable Markdown link in your answer.
      Example: "According to the [CDC Guidelines](https://cdc.gov)..."
    - If the evidence is insufficient, state that clearly.
    """
    
    input_messages = [HumanMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(input_messages)

    return {
        "final_answer": response.content,
        "messages": [response]
    }

# =====================================================
# 4. GRAPH BUILDER
# =====================================================

def build_clinical_graph():
    graph = StateGraph(OrchestratorState)

    graph.add_node("intent", detect_intent)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("drug", drug_intelligence_node)
    graph.add_node("literature", literature_node)
    graph.add_node("web", web_search_node)
    graph.add_node("compress", compress_context)
    graph.add_node("answer", answer_node)

    def route_from_intent(state: OrchestratorState):
        intent = state["intent"]
        if intent in ("dashboard_summary", "qa"): return "retrieval"
        if intent == "drug_intelligence": return "drug"
        if intent == "literature_search": return "literature"
        if intent == "web_search": return "web"
        return "retrieval"

    graph.add_conditional_edges(
        "intent",
        route_from_intent,
        {
            "retrieval": "retrieval",
            "drug": "drug",
            "literature": "literature",
            "web": "web",
        }
    )

    graph.add_edge("retrieval", "compress")
    graph.add_edge("drug", "compress")
    graph.add_edge("literature", "compress")
    graph.add_edge("web", "compress")
    graph.add_edge("compress", "answer")
    graph.add_edge("answer", END)

    graph.set_entry_point("intent")
    return graph.compile()

# Singleton instance
_clinical_bot = build_clinical_graph()

# =====================================================
# 5. PUBLIC API
# =====================================================

async def query_clinical_system(user_query: str, chat_history: List = None) -> Dict:
    if chat_history is None:
        chat_history = []
        
    initial_state = {
        "user_query": user_query,
        "messages": chat_history + [HumanMessage(content=user_query)],
        "entities": [],
        "retrieved_context": [],
        "tool_outputs": []
    }
    
    result = await _clinical_bot.ainvoke(initial_state)
    
    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "intent": result["intent"]
    }

# # =====================================================
# # Manual Test
# # =====================================================

# if __name__ == "__main__":
#     import asyncio

#     async def run():
#         # Setup initial state with a message
#         state = {
#             "user_query": "Recent studies on Imatinib resistance",
#             "messages": [HumanMessage(content="")]
#         }

#         print("Running Orchestrator...")
#         result = await orchestrator.ainvoke(state)

#         print("\n===== FINAL ANSWER =====\n")
#         print(result["final_answer"])
        
#         print("\n===== SOURCES FOR FRONTEND =====\n")
#         # This is what you will send to your UI
#         for s in result.get("sources", []):
#             print(f"[{s['agent']}] {s['title']} ({s['url']})")

#     asyncio.run(run())