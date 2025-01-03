from dotenv import load_dotenv
from src.app.semantic_engine import semantic_engine


def main():
    load_dotenv()
    semantic_engine()


if __name__ == "__main__":
    main()
