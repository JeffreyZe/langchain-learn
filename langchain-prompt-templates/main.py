# Import the demo functions from prompt_demo.py
from prompt_demo import single_prompt_demo, chat_prompt_demo

def main():
    print("\nGemini Prompt Engineering (LangChain Integration)\n")
    
    # Show menu options to the user
    print("Choose an option:")
    print("1. PromptTemplate Workflow")
    print("2. ChatPromptTemplate Workflow")
    print("Type exit() to quit.\n")

    # Infinite loop to keep showing the menu until user exits
    while True:
        choice = input("Select 1 or 2: ").strip()  # Read user input

        # Exit condition
        if choice.lower() in ["exit", "exit()", "quit"]:
            print("\nExiting the Prompt Demo...\n")
            break

        # Call the PromptTemplate demo if user chooses 1
        if choice == "1":
            single_prompt_demo()

        # Call the ChatPromptTemplate demo if user chooses 2
        elif choice == "2":
            chat_prompt_demo()

        # Handle invalid input
        else:
            print("Invalid choice. Please type 1 or 2.\n")

# Standard Python idiom to ensure main() runs only when script is executed directly
if __name__ == "__main__":
    main()