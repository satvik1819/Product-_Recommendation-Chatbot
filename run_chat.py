from src.conversation.responder import handle_user_query
from groq import Groq
import os

# Initialize Groq client ONCE (from environment variable)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

state = {}

print("Chat started (type 'exit' to quit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    response = handle_user_query(query, state, client)

    # update state safely
    state = response.get("state", state)

    print("\nAssistant:")
    print(response.get("summary", ""))

    if response.get("products"):
        print("\nProducts:")
        for p in response["products"][:5]:
            print(f"- {p.get('name')} | ₹{p.get('price')} | ⭐ {p.get('ratings')}")

    if response.get("explanation"):
        print("\nExplanation:")
        print(response["explanation"])

    if response.get("follow_up"):
        print("\nFollow-up:", response["follow_up"])

    print("\n" + "-" * 50 + "\n")