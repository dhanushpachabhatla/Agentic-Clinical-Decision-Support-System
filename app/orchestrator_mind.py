"""
Clinical Orchestrator (LangGraph)
--------------------------------
Responsibility:
- Interpret user intent
- Route to correct agents
- Control retrieval mode (summary vs Q/A)
- Call tools when required
- Compress context safely
- Call LLM for final answer
- Maintain conversational memory
"""

# =====================================================
# 1. SETUP & IMPORTS (Fixed Order)
# =====================================================
import os
from dotenv import load_dotenv

# Load env vars BEFORE importing agents
load_dotenv()
GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2")

if not GEMINI_API_KEY2:
    raise ValueError("GEMINI_API_KEY2 not found in .env")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY2
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY2

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Agent Imports
from agents.retrieval_agent import RetrieverAgent
from agents.drug_intelligence_agent import extract_drug_names, fetch_drug_intelligence
from agents.literature_search_agent import literature_search
from agents.web_search_agent import web_search

# =====================================================
# 2. CONFIGURATION
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Updated to stable version
    temperature=0.3,
    max_tokens=None,
    max_retries=2,
)

# =====================================================
# 3. STATE DEFINITION
# =====================================================

class OrchestratorState(TypedDict):
    user_query: str
    intent: Optional[str]
    entities: Optional[List[Dict]]
    retrieved_context: List[Dict]
    tool_outputs: List[Dict]
    compressed_context: str
    final_answer: str
    
    # NEW: Structured list to hold source metadata for the frontend
    # Example: [{"type": "web", "title": "CDC Guide", "url": "..."}]
    sources: List[Dict] 
    
    messages: Annotated[List[BaseMessage], add_messages]


# =====================================================
# NODE 1: Intent Detection
# =====================================================

def detect_intent(state: OrchestratorState) -> OrchestratorState:
    prompt = f"""
You are a clinical AI ORCHESTRATOR.
Your job is ONLY to route the user query to the correct internal agent.

You have the following agents available:

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
     • "Is elevated troponin indicative of MI?"
     • "What does this lab value suggest?"
     • "Why was this medication stopped?"

3. drug_intelligence
   - Use when the user asks about a DRUG
   - Includes: mechanism, properties, contraindications, safety
   - Examples:
     • "What are the side effects of metformin?"
     • "Is aspirin safe in pregnancy?"

4. literature_search
   - Use when the user wants RESEARCH PAPERS, GUIDELINES, or STUDIES
   - Query-based (not patient-specific)
   - Examples:
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

Rules:
- Choose ONE and only ONE intent
- Do NOT explain your choice
- Do NOT add extra text
- Output must be exactly one of:
  dashboard_summary | qa | drug_intelligence | literature_search | web_search

User query:
{state['user_query']}
"""
    intent = llm.invoke(prompt).content.strip()
    
    # Initialize sources list empty for this turn
    state["intent"] = intent
    state["sources"] = [] 
    return state


# =====================================================
# NODE 2A: Retrieval (Internal Documents)
# =====================================================

async def retrieval_node(state: OrchestratorState) -> OrchestratorState:
    retriever = RetrieverAgent()

    if state["intent"] == "dashboard_summary":
        results = await retriever.retrieve_for_summary()
    else:
        results = await retriever.retrieve_for_qa(query=state["user_query"])

    state["retrieved_context"] = results
    
    # --- SOURCE EXTRACTION ---
    # We extract document names from the RAG chunks
    doc_sources = []
    seen_docs = set()
    
    for r in results:
        # Assuming your RAG chunks have 'source' or 'file_name' in metadata
        doc_name = r.get("metadata", {}).get("source", "Unknown Document")
        if doc_name not in seen_docs:
            doc_sources.append({
                "type": "internal_document",
                "title": doc_name,
                "url": None, # Internal docs might not have URLs
                "agent": "Vector Retrieval"
            })
            seen_docs.add(doc_name)
            
    state["sources"] = doc_sources
    return state


# =====================================================
# NODE 2B: Drug Intelligence
# =====================================================

def drug_intelligence_node(state: OrchestratorState) -> OrchestratorState:
    entities = state.get("entities", [])
    drugs = extract_drug_names(entities)

    if not drugs:
        state["tool_outputs"] = []
        return state

    outputs = fetch_drug_intelligence(drugs)
    state["tool_outputs"] = outputs
    
    # --- SOURCE EXTRACTION ---
    state["sources"] = [{
        "type": "database",
        "title": "OpenFDA / DrugBank",
        "url": "https://open.fda.gov",
        "agent": "Drug Intelligence"
    }]
    return state


# =====================================================
# NODE 2C: Literature Search
# =====================================================

def literature_node(state: OrchestratorState) -> OrchestratorState:
    # returns a dict with 'results': [...]
    output = literature_search(state["user_query"]) 
    state["tool_outputs"] = [output]
    
    # --- SOURCE EXTRACTION ---
    # Assuming literature_search returns a list of papers under "results"
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


# =====================================================
# NODE 2D: Web Search
# =====================================================

def web_search_node(state: OrchestratorState) -> OrchestratorState:
    # returns list of dicts: [{title, url, snippet, ...}]
    results = web_search(state["user_query"])
    state["tool_outputs"] = results
    
    # --- SOURCE EXTRACTION ---
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


# =====================================================
# NODE 3: Context Compression
# =====================================================

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


# =====================================================
# NODE 4: Answer Generation
# =====================================================

def answer_node(state: OrchestratorState) -> OrchestratorState:
    context = state.get("compressed_context", "")
    sources = state.get("sources", [])
    
    # Format sources for the LLM to see (optional, helps with citation)
    source_text = "\n".join([f"- {s['title']}" for s in sources])

    system_prompt = f"""You are a clinical assistant. 
    Answer using the evidence below. 
    
    Evidence:
    {context}
    
    Available Sources:
    {source_text}
    """
    
    input_messages = [HumanMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(input_messages)

    return {
        "final_answer": response.content,
        "messages": [response]
    }


# =====================================================
# GRAPH DEFINITION
# =====================================================

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
orchestrator = graph.compile()


# =====================================================
# Manual Test
# =====================================================

if __name__ == "__main__":
    import asyncio

    async def run():
        # Setup initial state with a message
        state = {
            "user_query": "Recent studies on Imatinib resistance",
            "messages": [HumanMessage(content="")]
        }

        print("Running Orchestrator...")
        result = await orchestrator.ainvoke(state)

        print("\n===== FINAL ANSWER =====\n")
        print(result["final_answer"])
        
        print("\n===== SOURCES FOR FRONTEND =====\n")
        # This is what you will send to your UI
        for s in result.get("sources", []):
            print(f"[{s['agent']}] {s['title']} ({s['url']})")

    asyncio.run(run())