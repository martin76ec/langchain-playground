from typing import Any, Final
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import numpy as np

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def files_load():
    file_path: Final = (
        "/home/martin/Documents/Preguntas frecuentes facturaci_n electr_nica.pdf"
    )
    loader: Final = PyPDFLoader(file_path)

    return loader.load()


def file_split(docs: Any):
    text_splitter: Final = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def model_get():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def main():
    load_dotenv()
    docs: Final = files_load()
    chunks: Final = file_split(docs)
    model: Final = model_get()

    for chunk in chunks:
        model.embed_query(chunk.page_content)

    chunk_embeddings = [
        (chunk, model.embed_query(chunk.page_content)) for chunk in chunks
    ]

    # Question to ask
    user_question = "What are the steps for electronic billing?"
    question_embedding = model.embed_query(user_question)

    # Find the most similar chunk to the question
    most_similar_chunk = max(
        chunk_embeddings,
        key=lambda x: cosine_similarity(question_embedding, x[1]),
    )

    most_similar_chunk = max(
        chunk_embeddings,
        key=lambda x: cosine_similarity(question_embedding, x[1]),
    )

    # Extract the chunk with the highest similarity
    best_chunk = most_similar_chunk[0]
    print(f"Most relevant chunk: {best_chunk.page_content}")

    # Optionally, pass the chunk and question to a language model
    system_template: Final[str] = (
        "Answer the question based on the following information: {context}"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    prompt = prompt_template.invoke(
        {"context": best_chunk.page_content, "text": user_question}
    )

    print("Generated Answer:")
    print(prompt)


main()
