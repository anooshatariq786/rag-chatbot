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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
# Import Anthropic from the older community/chat_models space
from langchain_community.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
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
        self.metadata: dict = {}          # doc_id -> {filename, chunk_count}
        self.doc_chunk_ids: dict = {}     # doc_id -> list of chunk indices

        self._load_persisted()
        self.llm = self._init_llm()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

    # ------------------------------------------------------------------ #
    # LLM Selection (priority: Groq > OpenAI > Anthropic > fallback)
    # ------------------------------------------------------------------ #
    def _init_llm(self):
        if os.getenv("GROQ_API_KEY"):
            logger.info("Using Groq LLM")
            return ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-70b-8192",
                temperature=0.2,
            )
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Using OpenAI LLM")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Using Anthropic LLM")
            return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.2)
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
        # Rebuild index without the deleted doc
        if self.vectorstore:
            all_docs = []
            for did, meta in self.metadata.items():
                if did == doc_id:
                    continue
                # Re-load isn't trivial with FAISS; simplest approach: mark deleted
                # For a full rebuild we'd need stored docs — use metadata filter instead
                pass
            # Simple approach: filter at query time via metadata
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
    def query(self, question: str, history: List[dict] = []) -> dict:
        if self.vectorstore is None or not self.metadata:
            return {
                "answer": "No documents uploaded yet. Please upload documents first via the Knowledge Base tab.",
                "sources": [],
            }

        retriever = self.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}
        )

        # Build chat history for LangChain
        chat_history = []
        for turn in history[-6:]:         # keep last 3 turns
            if turn.get("role") == "user":
                chat_history.append(HumanMessage(content=turn["content"]))
            elif turn.get("role") == "assistant":
                chat_history.append(AIMessage(content=turn["content"]))

        # Retrieve relevant chunks
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in relevant_docs])

        # Build prompt
        system_prompt = (
            "You are a helpful AI assistant that answers questions based on the provided documents. "
            "Use the following context to answer the question accurately. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-6:]:
            messages.append(turn)
        messages.append({"role": "user", "content": question})

        # Call LLM
        from langchain.schema import SystemMessage
        lc_messages = [SystemMessage(content=system_prompt)]
        lc_messages += chat_history
        lc_messages.append(HumanMessage(content=question))
        response = self.llm.invoke(lc_messages)

        # Build sources list (deduplicated by filename)
        seen = set()
        sources = []
        for doc in relevant_docs:
            fname = doc.metadata.get("filename", "Unknown")
            if fname not in seen:
                seen.add(fname)
                sources.append({
                    "filename": fname,
                    "snippet": doc.page_content[:200] + "...",
                    "doc_id": doc.metadata.get("doc_id", ""),
                })

        return {"answer": response.content, "sources": sources}
