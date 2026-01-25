import asyncio
from app.orchestrator_mind import query_clinical_system

async def chat_session():
    print("--- Clinical AI Ready (Type 'quit' to exit) ---")
    
    history = []
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        print("Thinking...")
        response = await query_clinical_system(user_input, history)
        
        print(f"\nAI: {response['answer']}")
        print(f"[Intent: {response['intent']}]")
        
        if response['sources']:
            print("Sources:")
            for s in response['sources']:
                print(f" - {s['title']}")

if __name__ == "__main__":
    asyncio.run(chat_session())