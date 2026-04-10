import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()


def get_gemini_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def get_llm():
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY before running the demo."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        api_key=api_key
    )


def single_prompt_demo():
    llm = get_llm()
    print("\n--- PromptTemplate ---")

    # Ask user inputs
    topic = input("Enter a topic: ")
    tone = input("Choose tone (funny / formal / simple): ")
    length = input("Choose length (short / medium / long): ")

    # Template with 3 variables
    template_text = (
        "Write a {tone} explanation about the topic: {topic}. "
        "Make the explanation {length} and easy to understand."
    )

    # Create PromptTemplate
    prompt = PromptTemplate(
        input_variables=["tone", "topic", "length"],
        template=template_text
    )

    final_prompt = prompt.format(
        tone=tone,
        topic=topic,
        length=length
    )

    print("\nGenerated Prompt:")
    print(final_prompt)

    print("\nSending to Gemini...\n")
    response = llm.invoke(final_prompt)

    print("Gemini Response:")
    print(response.content)

def chat_prompt_demo():
    llm = get_llm()
    print("\n--- ChatPromptTemplate ---")

    system_role = input("Enter system instruction (e.g., You are a tutor): ")
    user_question = input("Enter user message: ")
    tone = input("Choose AI tone (friendly / strict / expert): ")

    # Chat template with structured message roles
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "{system_role}"),
        ("user", "{user_question}"),
        ("assistant", "Respond in a {tone} tone.")
    ])

    # Format chat messages
    formatted_chat = chat_template.format_messages(
        system_role=system_role,
        user_question=user_question,
        tone=tone
    )

    print("\nGenerated Chat Prompt:")
    for msg in formatted_chat:
        print(f"{msg.type.upper()}: {msg.content}")

    print("\nSending to Gemini...\n")
    response = llm.invoke(formatted_chat)

    print("Gemini Reply:")
    print(response.content[1].get("text", ""))
    
