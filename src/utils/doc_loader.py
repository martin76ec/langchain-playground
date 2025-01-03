from langchain_community.document_loaders import PyPDFLoader


def doc_load(path: str):
    return PyPDFLoader(path).load()
