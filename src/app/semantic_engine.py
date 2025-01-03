from typing import Final
from langchain.globals import set_debug
from src.domain.semantic_engine_dom import doc_to_store
import os


def semantic_engine():
    set_debug(True)
    dir = os.getcwd()
    filepath: Final[str] = f"{dir}/.files/test.pdf"
    store = doc_to_store(filepath)
    store.add_documents
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    output = retriever.batch(
        [
            "que es bancarizacion?",
            "cual es el costo deducible?",
        ],
    )

    print(output)
