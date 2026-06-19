"""
RAG Engine — LangChain + FAISS + (Groq / OpenAI / Anthropic)
Handles document ingestion, embedding, retrieval, and generation.
"""

import os, json
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

METADATA_FILE = Path("doc_metadata.json")
FAISS_INDEX_DIR = Path("faiss_index")


class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.vectorstore: Optional[FAISS] = None
        self.metadata: dict = {}
        self.doc_chunk_ids: dict = {}

        self._load_persisted()
        self.llm = self._init_llm()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

    # ------------------------------------------------------------------ #
    # LLM Selection (priority: Groq > OpenAI > Anthropic)
    # ------------------------------------------------------------------ #
    def _init_llm(self):
        if os.getenv("GROQ_API_KEY"):
            logger.info("Using Groq LLM")
            return ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Using OpenAI LLM")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Using Anthropic LLM")
            return ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)
        raise EnvironmentError(
            "No LLM API key found. Set GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
        )

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _load_persisted(self):
        if FAISS_INDEX_DIR.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(FAISS_INDEX_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}")
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                data = json.load(f)
                self.metadata = data.get("metadata", {})
                self.doc_chunk_ids = data.get("doc_chunk_ids", {})

    def _persist(self):
        if self.vectorstore:
            self.vectorstore.save_local(str(FAISS_INDEX_DIR))
        with open(METADATA_FILE, "w") as f:
            json.dump({"metadata": self.metadata, "doc_chunk_ids": self.doc_chunk_ids}, f)

    # ------------------------------------------------------------------ #
    # Document ingestion
    # ------------------------------------------------------------------ #
    def _load_file(self, path: str) -> List:
        suffix = Path(path).suffix.lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader,
        }
        loader_cls = loaders.get(suffix)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {suffix}")
        return loader_cls(path).load()

    def add_document(self, path: str, filename: str, doc_id: str) -> int:
        docs = self._load_file(path)
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = filename

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self.metadata[doc_id] = {"filename": filename, "chunk_count": len(chunks)}
        self._persist()
        return len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.metadata:
            return False
        if self.vectorstore:
            self.metadata.pop(doc_id, None)
            self._persist()
        return True

    def list_documents(self):
        return [
            {"id": did, "filename": m["filename"], "chunk_count": m["chunk_count"]}
            for did, m in self.metadata.items()
        ]

    def get_document_count(self) -> int:
        return len(self.metadata)

    # ------------------------------------------------------------------ #
    # Query / RAG pipeline
    # ------------------------------------------------------------------ #
    def query(self, question: str, history: List[dict] = None) -> dict:
        if history is None:
            history = []

        if self.vectorstore is None or not self.metadata:
            return {
                "answer": "No documents uploaded yet. Please upload documents first via the Knowledge Base tab.",
                "sources": [],
            }

        retriever = self.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}
        )

        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in relevant_docs])

        system_prompt = (
            "You are a helpful AI assistant that answers questions based on the provided documents. "
            "Use the following context to answer the question accurately. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}"
        )

        lc_messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            role = msg.get("role") if isinstance(msg, dict) else msg[0]
            content = msg.get("content") if isinstance(msg, dict) else msg[1]
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        lc_messages.append(HumanMessage(content=question))

        response = self.llm.invoke(lc_messages)

        seen = set()
        sources = []
        for doc in relevant_docs:
            fname = doc.metadata.get("filename", "Unknown")
            if fname not in seen:
                seen.add(fname)
                sources.append({
                    "filename": fname,
                    "name": fname,
                    "snippet": doc.page_content[:200] + "...",
                    "doc_id": doc.metadata.get("doc_id", ""),
                })

        return {"answer": response.content, "sources": sources}
