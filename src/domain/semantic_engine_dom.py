from typing import Final
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.doc_loader import doc_load
from src.utils.embedds import embed_get
from src.utils.text_spltr import doc_split
from src.utils.vec_store import store_save


def doc_to_chunks(path: str):
    document: Final = doc_load(path)
    return doc_split(document)


def embed_chunks(embeddings: GoogleGenerativeAIEmbeddings, chunks: list[Document]):
    [embeddings.embed_query(chunk.page_content) for chunk in chunks]


def doc_to_store(path: str):
    chunks = doc_to_chunks(path)
    embeddings = embed_get()
    embed_chunks(embeddings, chunks)
    store = store_save(embeddings)
    store.add_documents(chunks)

    return store
