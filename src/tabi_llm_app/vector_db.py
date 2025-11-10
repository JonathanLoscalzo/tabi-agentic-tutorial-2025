"""
Módulo para gestionar la base de datos vectorial con ChromaDB
"""

from pathlib import Path
from typing import List, Dict
from uuid import uuid4
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import ollama


class VectorDatabase:
    """Gestiona la base de datos vectorial con ChromaDB y Ollama"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "tabi_docs",
        embedding_model: str = "embeddinggemma:latest",
    ):
        """
        Inicializa la base de datos vectorial

        Args:
            persist_directory: Directorio donde persistir la DB
            collection_name: Nombre de la colección
            embedding_model: Modelo de Ollama para embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

        # Crear directorio si no existe
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Inicializar cliente de ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(
                self.persist_directory,
            )
        )

        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

        print("🗄️  Base de datos vectorial inicializada")
        print(f"   Directorio: {self.persist_directory}")
        print(f"   Colección: {self.collection_name}")
        print(f"   Documentos existentes: {self.collection.count()}")

    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Añade documentos a la base de datos vectorial
        """
        uuids = [str(uuid4()) for _ in range(len(documents))]
        print("Añadiendo documentos a la base de datos vectorial...")
        print(f"   Documentos: {len(documents)}")
        print(f"   Batch size: {batch_size}")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]  # noqa: E203
            batch_end = min(i + batch_size, len(documents))

            print(f"   Procesando {i+1}-{batch_end}/{len(documents)}...")

            self.vector_store.add_documents(batch, ids=uuids[i : i + batch_size])  # noqa: E203

    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera un embedding para un texto usando Ollama

        Args:
            text: Texto a embedder

        Returns:
            Vector de embedding
        """
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def _add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Añade documentos a la base de datos vectorial

        Args:
            documents: Lista de documentos con 'content' y 'metadata'
            batch_size: Tamaño del batch para procesamiento
        """
        print("\n🔄 Generando embeddings y almacenando en ChromaDB...")
        print(f"   Modelo de embeddings: {self.embedding_model}")

        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]  # noqa: E203
            batch_end = min(i + batch_size, total_docs)

            msg = f"   Procesando {i+1}-{batch_end}/{total_docs}..."
            print(msg)

            # Preparar datos para ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []

            for j, doc in enumerate(batch):
                doc_id = f"doc_{i + j}"
                content = doc["content"]

                # Generar embedding
                embedding = self.generate_embedding(content)

                ids.append(doc_id)
                embeddings.append(embedding)
                metadatas.append(doc["metadata"])
                documents_text.append(content)

            # Añadir a ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text,
            )

        print(f"✅ {total_docs} documentos añadidos a ChromaDB")
        print(f"   Total en colección: {self.collection.count()}")

    def search(self, query: str, n_results: int = 5) -> list[Document]:
        """
        Busca documentos relevantes para una consulta
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": n_results})
        return retriever.invoke(query)

    # def _search(self, query: str, n_results: int = 5) -> Dict:
    #     """
    #     Busca documentos relevantes para una consulta

    #     Args:
    #         query: Consulta de búsqueda
    #         n_results: Número de resultados a devolver

    #     Returns:
    #         Diccionario con resultados de la búsqueda
    #     """
    #     # Generar embedding de la consulta
    #     query_embedding = self.generate_embedding(query)

    #     # Buscar en ChromaDB
    #     results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)

    #     return results

    def clear_collection(self):
        """Elimina todos los documentos de la colección"""
        self.vector_store.reset_collection(collection_name=self.collection_name)
        print(f"🗑️  Colección '{self.collection_name}' vacía")

    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas de la base de datos

        Returns:
            Diccionario con estadísticas
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(self.persist_directory),
            "embedding_model": self.embedding_model,
        }
