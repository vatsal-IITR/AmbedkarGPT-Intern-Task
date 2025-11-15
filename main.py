# AmbedkarGPT - simple local RAG demo 

import os
import sys

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# NOTE: depending on LangChain version, this import may need to change
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SPEECH_FILE = "speech.txt"
DB_DIR = "chroma_db"


def load_data():
    """Load local text file."""
    if not os.path.exists(SPEECH_FILE):
        raise FileNotFoundError("speech.txt missing. Place it next to this script.")

    loader = TextLoader(SPEECH_FILE, encoding="utf-8")
    return loader.load()


def split_into_chunks(docs):
    """Split long content into reasonably sized chunks."""
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    return splitter.split_documents(docs)


def setup_vector_db(chunks):
    """Create or load Chroma DB."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        # already built
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    db.persist()
    return db


def build_chain(db):
    """Create retrieval-based QA pipeline."""
    llm = Ollama(model="mistral")  # assumes model is already available in Ollama

    # Compact version of the earlier template
    template = (
        "Answer only based on the text below. "
        "If the info isn't there, say: 'I don't know based on the speech.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )


def interactive_chat(chain):
    """Basic CLI interface."""
    print("\nAmbedkarGPT ready. Ask questions about the speech.\n(Type 'exit' to quit.)\n")

    while True:
        q = input(">> ").strip()

        if q.lower() in {"exit", "quit"}:
            print("Closing. Bye.")
            break

        if not q:
            continue

        try:
            result = chain({"query": q})
            print("\n" + result["result"] + "\n")
        except Exception as e:
            print("Error:", e)
            # TODO: add logging later


def main():
    print("Loading text...")
    docs = load_data()

    print("Creating chunks...")
    chunks = split_into_chunks(docs)

    print(f"Total chunks: {len(chunks)}")

    print("Setting up vector DB...")
    db = setup_vector_db(chunks)

    print("Building QA pipeline...")
    chain = build_chain(db)

    interactive_chat(chain)


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("Python >= 3.8 required.")
        sys.exit(1)

    main()

