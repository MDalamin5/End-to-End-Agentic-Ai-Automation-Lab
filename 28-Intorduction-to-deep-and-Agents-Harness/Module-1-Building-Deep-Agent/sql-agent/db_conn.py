import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

from deepagents import create_deep_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool

from model import model

DB_PATH = Path(__file__).parent / "chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
