# Guide

## Requirements.txt File

- *langchain*: We will need this for the **create_agent(for agent) and ChatPromptTemplate(for system prompt)** as well as for executing the agent
- *langchain-google-genai*: This is required to integrate the Gemini Models
- *google-generativeai*: This is something we must download for the langchain to function correctly. It will not be imported directly. This is Google's official Python SDK, required internally by LangChain. This will **authenticating your API key**, as well as **send requests to gemini**
- *python-dotenv*: This will store your API key away from the public eyes. 

## Creating Virtual Environment

Now we will create a virtual environment for the project. 
- We will use **py -m venv venv** to create the environment and then **.\venv\Scripts\activate (for windows)** to activate the environment.
- Once the environment is successfully activated, download all the dependencies in this virtual env will **pip install -r requiremnets.txt**

## Google API Key

Gemini requires your API key to create a connection with your agent since we will be using *gemini 1.5 flash model* as the LLM.
Also, developers do NOT hard-code keys into Python files therefore we must make it so that we can use it secretly in our main file.
- Create a **.env** file and add you API key like this *GOOGLE_API_KEY=YOUR_API_KEY_HERE*
- We will later load this key using the dotenv module.

Note: If you face **“Import 'dotenv' could not be resolved Pylance”** issue even though you have installed the dependecies, it might be because you are not using your venv interpreter. You can change that by typing *Ctrl + Shift + P > Python: Select Interpreter > Choose interpreter that has (venv) bracket*
## Defining System Prompt
Use the a variable and same the system prompt. 

## Tools to use

We will simply define three functions that the LLM can use as it sees fit. 
- extract_key_points → finds discussion/update lines
- extract_action_items → finds action items
- summarize_meeting → gives a short summary

The template for functions will be
'''
def function_name(argument):
    """description"""
    return {a dictionary element}
''' 

## Integrating the Model

We create the LLM reasoning engine that powers your agent. We will use the **gemini-1.5-flash** here. 

## Using create_agent
Now that we have all the core components it is time that combine them together and make it work. 
First we will give a template for giving prompts to LangChain using ChatPromptTemplate. This template tells LangChain:
- Always include your system prompt
- Then insert whatever the user provides as {input}

## Load the Transcript
We will have a transcript file. To let the LLM be able to read it, we will create a *transcript* variable that will have all the content stored as a string, which we will then pass on to the model as **input string**

## Invocation 
Finally, use the invoke function to trigger the agent by giving the transcript variable as an input. 

Here is what happens when you invoke the scripts. 
1. LLM reads your system prompt
2. LLM inspects the tools
3. LLM decides whether to call a tool
4. LangChain executes the tool
5. Output is fed back to the LLM
6. Loop continues until final answer is produced
