from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


def store_save(embeddings: GoogleGenerativeAIEmbeddings):
    return InMemoryVectorStore(embeddings)
