from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain



from langserve import add_routes

FILEPATH = "AI Engineer.pdf" 
MODEL = "gpt-3.5-turbo-0125"
EMBEDDING = "text-embedding-3-large"

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

loader = PyPDFLoader(FILEPATH)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, 
                chunk_overlap=5)

all_splits = text_splitter.split_documents(data)

persist_directory = 'data'

vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=OpenAIEmbeddings(model=EMBEDDING),
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model=MODEL)

chain = ConversationalRetrievalChain.from_llm(llm, retriever)

add_routes(
    app,
    chain,
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
