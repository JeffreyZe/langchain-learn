import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GOOGLE_API_KEY before running the demo."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", api_key=api_key, temperature=0.7
    )

    # Example 1: Simple chain (Prompt | LLM | Parser)
    print("\n--- Simple Chain (Prompt | LLM | Parser) ---")

    topic = input("Enter a topic: ")

    prompt = ChatPromptTemplate.from_template(
        "Tell me a short less known interesting fact about {topic}."
    )

    simple_chain = prompt | llm | StrOutputParser()

    result = simple_chain.invoke({"topic": topic})
    print("\nGemini Response:")
    print(result)

    # Example 2: reusing the same chain with different inputs
    print("\n--- Reusing the same chain with different inputs ---")

    new_topic = input("Enter another topic: ")

    result2 = simple_chain.invoke({"topic": new_topic})
    print("\nGemini Response:")
    print(result2)

    # Example 3: Chain with mutiple variables
    print("\n--- Chain with multiple variables ---")
    role = input("Define AI role (e.g., expert teacher, mentor): ")
    concept = input("Enter a complex concept to explain: ")

    complex_prompt = ChatPromptTemplate.from_template(
        "You are an {role}. Explain the following concept in simple terms: {concept}"
    )

    multi_var_chain = complex_prompt | llm | StrOutputParser()

    result3 = multi_var_chain.invoke({"role": role, "concept": concept})
    print("\nGemini Response:")
    print(result3)


if __name__ == "__main__":
    main()
