from dotenv import load_dotenv
load_dotenv()


import os
os.environ["DeepSeek_API_KEY"] = os.getenv("DeepSeek_API_KEY")

class Setting:
    model = 