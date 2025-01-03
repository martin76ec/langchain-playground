from typing import Final
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def main():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    system_template: Final[str] = (
        "Translate the following from English into {language}"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    prompt = prompt_template.invoke(
        {"language": "spanish", "text": "vete al diablo"}
    )

    response = llm.invoke(prompt)
    print(response.content)
