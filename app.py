# import dependencies
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from chromadb.utils import embedding_functions
import chromadb
from openai import OpenAI
import uuid

load_dotenv()

openai_key = ""
client = OpenAI(api_key=openai_key)

## create embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name="text-embedding-3-small")

# init chroma client
chroma_client = chromadb.PersistentClient(path="./data/vector_store")
collection_name = "pdf_documents"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef, metadata={"description": "PDF Documents Collection"})

# load docs
def load_docs_from_dir(dir_path): 
    documents = []
    pdf_dir = Path(dir_path)
    # fetch all pdf files
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for file in pdf_files:
        try: 
            loader = PyMuPDFLoader(str(file))
            documents = loader.load()
            # add metadata from source
            for i, doc in enumerate(documents):
                doc.id = str(uuid.uuid4()),
                doc.metadata["source"] = file.name
                doc.metadata['file_type'] = 'pdf'
            documents.extend(documents)
        except Exception as e:
            print(f" Error: {e}")
    return documents

# Text splitting get into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\nTotal documents after splitting: {len(chunks)}")
    return chunks

def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("...Generating embeddings...")
    return embedding

def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # extract relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("=== returning relevant chunks ===")
    return relevant_chunks

# function to generate a response from open ai
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of"
        "retrieved context to answer the question. If you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestions:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ]
    )
    answer = response.choices[0].message
    return answer
    

def main():
    dir_path = "./data/files"
    documents = load_docs_from_dir(dir_path)
    print(f"Loaded {len(documents)} documents")
    chunked_docs = split_documents(documents)
    # generate embedding for the document chunks
    for doc in chunked_docs:
        print(f"\nGenerating embeddings for {len(chunked_docs)} chunks...")
    # insert documents with embedding into chroma
    for doc in chunked_docs:
        print("inserting chunks into db")
        collection.upsert(
            ids=[str(doc.id)], documents=[doc.page_content], embeddings=get_openai_embedding(doc.page_content) 
        )
    question = ""
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(answer)


main()
