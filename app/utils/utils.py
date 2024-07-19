import dotenv
import os


def load_env():
    dotenv.load_dotenv()
    return os.environ
