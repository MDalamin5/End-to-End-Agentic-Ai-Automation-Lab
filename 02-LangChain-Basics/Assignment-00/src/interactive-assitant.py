# Imports and Setup
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Annotated, List
from pydantic import BaseModel, Field
import sys

# Load environment variables from .env file
load_dotenv()

# API Keys and LangChain Setup
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if GROQ_API_KEY is available
if not groq_api_key:
    print("Error: GROQ_API_KEY not found in environment variables.")
    print("Please ensure you have a .env file with GROQ_API_KEY set or export it.")
    sys.exit(1) # Exit if key is missing

# Set LangChain environment variables (optional, provide defaults)
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = "Simple Chatbot"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize Model
# Using the model specified in the notebook
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
try:
    # Note: Ensure the specified model is available and supported by your Groq API key.
    # You might need to choose a different model like "llama3-8b-8192" if this one isn't accessible.
    model = ChatGroq(model_name=model_name, groq_api_key=groq_api_key)
except Exception as e:
    print(f"Error initializing ChatGroq model ({model_name}): {e}")
    print("Please check the model name and your Groq API key permissions.")
    sys.exit(1)

# Pydantic Output Schema
class ProductDetails(BaseModel):
    """This is product info Schema."""
    product_name: Annotated[str, Field(description="This is product name.")]
    product_details: Annotated[List[str], Field(description="List of product details like Ram, Rom, processor, Camera, battery, Screen size.")]
    product_price: Annotated[int, Field(description="Product price in USD (integer format price).")]

parser = PydanticOutputParser(pydantic_object=ProductDetails)

# System Message and Prompt Template
system_messages = """
    Think you are a tech expert. Your task is user will give a phone name and you will be provide a Structured output based on the {format_instruction} schema.
    1. product name which user will be give you.
    2. product details like Ram, Rom, processor, Camera, battery, Screen size and other which is necessary.
    3. product tentative price in USD format. must in USD integer format.\n"""


try:
    format_instructions = parser.get_format_instructions()
except Exception as e:
    print(f"Error getting format instructions from parser: {e}")
    sys.exit(1)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_messages),
        ("human", "{input}")
    ]
).partial(format_instruction=format_instructions)

# Chain Definition
chain = chat_prompt | model | parser

# Interactive Loop
print(f"Simple Product Assistant Initialized (using model: {model_name}).")
print("Type 'quit' or 'exit' to stop.")

while True:
    try:
        user_input = input("Enter product name: ")
    except EOFError:
        # Handle cases where input stream is closed (e.g., piping input)
        print("\nInput stream closed. Exiting.")
        break

    if user_input.strip().lower() in ["quit", "exit"]:
        print("Exiting assistant.")
        break

    if not user_input.strip():
        print("Please enter a product name.")
        continue

    try:
        print(f"\nFetching details for '{user_input}'...")
        result = chain.invoke({"input": user_input})

        print("\n--- Product Information ---")
        print(f"Name: {result.product_name}")
        print("Details:")
        if result.product_details:
            for detail in result.product_details:
                print(f"- {detail}")
        else:
            print("- No details provided.")
        print(f"Tentative Price (USD): {result.product_price}")
        print("-------------------------\n")

    except Exception as e:
        # Catch potential errors during chain invocation (API errors, parsing errors)
        print(f"\nAn error occurred: {e}")
        print("Could not retrieve or parse product information. Please try again or check the product name.")
        # Consider more specific error handling if needed (e.g., for Pydantic validation errors)
        print("-------------------------\n")

