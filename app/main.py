from dotenv import load_dotenv
load_dotenv()
import asyncio
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage  # Needed for memory
from pathlib import Path
from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator
# from app.orchestrator import ClinicalOrchestrator

async def start_chat_mode(orchestrator: ClinicalOrchestrator):
    """
    Runs the interactive chat loop using the Orchestrator.
    """
    print("\n" + "="*50)
    print(" CLINICAL AI READY - Ask me anything!")
    print("    (Type 'quit' to exit)")
    print("="*50 + "\n")

    history = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
            
            # Call the orchestrator's new async method
            response = await orchestrator.answer_user_query(user_input, history)

            print(f"\nAI: {response['answer']}")
            
            # Show sources if available
            if response.get('sources'):
                print("\nSources:")
                for i, s in enumerate(response['sources'], 1):
                    url = s.get('url')
                    link = f" ({url})" if url else ""
                    print(f" {i}. {s['title']}{link}")
            print("-" * 50)

            # Update Memory
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response['answer']))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    file_paths = [
        PROJECT_ROOT / "samples" / "file-1.png",
        PROJECT_ROOT / "samples" / "file-2.png",
        PROJECT_ROOT / "samples" / "file-3.png",
            # PROJECT_ROOT / "samples" / "file-4.png",
            # PROJECT_ROOT / "samples" / "file-5.png",  
            # PROJECT_ROOT / "samples" / "file-6.png",  
    ]

    file_paths = [str(p) for p in file_paths if p.exists()]
    if not file_paths:
        print("[ERROR] No valid input files")
        return

    state = ClinicalState(file_paths=file_paths)
    orchestrator = ClinicalOrchestrator()

    state = orchestrator.run_ingestion(state)
    state = orchestrator.run_clinical_nlp(state)
    state = orchestrator.run_embedding(state)
    state = orchestrator.run_vector_upsert(state)

    # 🔥 FINAL REASONING STEP
    state = orchestrator.run_reasoning(state)

    print("\n================ FINAL CLINICAL REASONING ================\n")
    if state.reasoning_result:
        print(state.reasoning_result)
    else:
        print("[ERROR] No reasoning output produced")

    if state.errors:
        print("\n[WARNINGS]")
        for e in state.errors:
            print("-", e)
    
    # -------------------------------------------------
    # 2. START INTERACTIVE CHAT
    # -------------------------------------------------
    # Since 'main' is synchronous, we use asyncio.run() to enter the async chat loop
    asyncio.run(start_chat_mode(orchestrator))



if __name__ == "__main__":
    main()
