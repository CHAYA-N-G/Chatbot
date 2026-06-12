import os

from dotenv import load_dotenv

 

from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

 

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_cohere import CohereEmbeddings, ChatCohere

from langchain_community.vectorstores import FAISS

 

from pypdf import PdfReader

 

# Load API key

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

 

# ✅ Load PDF manually

def load_pdf(path):

    reader = PdfReader(path)

    docs = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            docs.append(Document(page_content=text, metadata={"page": i}))

    return docs

 

documents = load_pdf("sample.pdf")

 

# ✅ Split

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50

)

chunks = splitter.split_documents(documents)

 

# ✅ Embeddings

embeddings = CohereEmbeddings(

    cohere_api_key=api_key,

    model="embed-english-v3.0"

)

 

# ✅ Vector DB

db = FAISS.from_documents(chunks, embeddings)

 

retriever = db.as_retriever()

 

# ✅ LLM

llm = ChatCohere(

    cohere_api_key=api_key,

    model="command-a-plus-05-2026"

)

 

from langchain_core.prompts import ChatPromptTemplate

 

# ✅ Prompt template (VERY IMPORTANT)

prompt = ChatPromptTemplate.from_template("""

Answer the question based only on the context below:

 

Context:

{context}

 

Question:

{question}

""")

 

# ✅ Format retrieved documents

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)

 

# ✅ Correct RAG chain

rag_chain = (

    {

        "context": retriever | format_docs,

        "question": RunnablePassthrough()

    }

    | prompt   # ✅ ADD THIS LINE

    | llm

    | StrOutputParser()

)

 

# ✅ Chat loop

print("✅ Chatbot ready! Type 'exit' to quit.\n")

 

while True:

    question = input("You: ")

    if question.lower() == "exit":

        break

 

    answer = rag_chain.invoke(question)

    print("Bot:", answer, "\n")

