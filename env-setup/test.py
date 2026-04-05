from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import os

def build_chain():
    # Load environment variables from .env
    load_dotenv()

    # Gemini API key is read from GOOGLE_API_KEY env var
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment/.env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        google_api_key=api_key,  # optional if it's already in env, but explicit is clear
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a senior software architect helping developers make "
                    "good technical decisions. Be concise, practical, and specific. "
                    "Focus on architecture, tools, trade-offs, and best practices."
                ),
            ),
            (
                "human",
                (
                    "Developer question:\n"
                    "{question}\n\n"
                    "Answer for an experienced tech audience. "
                    "Use short paragraphs and bullets when helpful."
                ),
            ),
        ]
    )

    # LangChain Expression Language: prompt -> model
    chain = prompt | llm
    return chain


def main():
    chain = build_chain()

    print("Tech Stack Advisor (Gemini 2.5 Flash + LangChain)")
    print("Ask architecture / tooling questions (type 'exit' to quit).\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit", "q"}:
            print("Goodbye")
            break

        if not user_input.strip():
            continue

        try:
            response = chain.invoke({"question": user_input})
            print("\nAI:\n" + response.content + "\n")
        except Exception as e:
            print(f"Error while calling the model: {e}\n")


if __name__ == "__main__":
    main()
