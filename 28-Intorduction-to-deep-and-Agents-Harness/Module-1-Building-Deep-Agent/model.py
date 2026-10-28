from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")