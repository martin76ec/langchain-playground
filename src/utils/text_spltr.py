from typing import Final
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def doc_split(doc: Document):
    text_splitter: Final = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    return text_splitter.split_documents([doc])
