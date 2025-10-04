# server/rag.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MilitaryRAG:
    def __init__(self):
        # Load vector database (already indexed documents)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory="./data/vectorstore",
            embedding_function=self.embeddings
        )
    
    def search(self, query: str, k: int = 3):
        """Search military docs for relevant info"""
        
        # Find relevant sections
        results = self.vectorstore.similarity_search(query, k=k)
        
        # Return text + sources
        context = []
        for doc in results:
            context.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
        
        return context

# Usage:
rag = MilitaryRAG()
docs = rag.search("How to format a SPOTREP for enemy contact?")