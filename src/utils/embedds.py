from langchain_google_genai import GoogleGenerativeAIEmbeddings


def embed_get():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
