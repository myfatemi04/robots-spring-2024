from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

llm_client = OpenAI()
vlm_client = OpenAI()
