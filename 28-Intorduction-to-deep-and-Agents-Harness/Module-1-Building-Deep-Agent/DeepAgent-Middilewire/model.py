from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek

import os
os.environ["DeepSeek_API_KEY"] = os.getenv("DeepSeek_API_KEY")

model = ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
        