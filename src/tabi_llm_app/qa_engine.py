"""
Motor de preguntas y respuestas usando RAG (Retrieval Augmented Generation)
"""

from typing import Dict
from langchain_core.documents import Document
import ollama
from .vector_db import VectorDatabase


class QAEngine:
    """Motor de preguntas y respuestas con RAG"""

    def __init__(
        self,
        vector_db: VectorDatabase,
        llm_model: str = "deepseek-r1:1.5b",
        n_context_docs: int = 5,
        temperature: float = 0.2,
    ):
        """
        Inicializa el motor de Q&A

        Args:
            vector_db: Instancia de la base de datos vectorial
            llm_model: Modelo de Ollama para generar respuestas
            n_context_docs: Número de documentos de contexto a usar
            temperature: Temperatura para la generación (0-1)
        """
        self.vector_db = vector_db
        self.llm_model = llm_model
        self.n_context_docs = n_context_docs
        self.temperature = temperature

        print("🤖 Motor de Q&A inicializado")
        print(f"   Modelo LLM: {llm_model}")
        print(f"   Documentos de contexto: {n_context_docs}")

    def _format_context(self, search_results: list[Document]) -> str:
        """
        Formatea los resultados de búsqueda en un contexto legible

        Args:
            search_results: Resultados de la búsqueda vectorial

        Returns:
            Contexto formateado como string
        """
        context_parts = []
        for i, search_result in enumerate(search_results):
            content = search_result.page_content
            metadata = search_result.metadata
            # print(f"   Documento {i}: {content}")
            # print(f"   Metadatos: {metadata}")

            context_parts.append(f"[Documento {i} - {metadata.get('source', 'Desconocido')}]\n{content}\n")

        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """
        Crea el prompt para el LLM con contexto RAG

        Args:
            question: Pregunta del usuario
            context: Contexto recuperado de la base vectorial

        Returns:
            Prompt formateado
        """
        prompt = """Eres un asistente experto que responde preguntas basándose en documentos técnicos sobre análisis de datos y modelado dimensional.
            Usa ÚNICAMENTE la información proporcionada en el contexto a continuación para responder la pregunta.
            Si la respuesta no se encuentra en el contexto, di "No tengo información sobre eso" y no intentes adivinar.
            CONTEXTO: {context}
            \n\n
            PREGUNTA: {question}
            \n\n
            """  # noqa: E501

        return prompt.format(context=context, question=question)

    def generate_prompt(self, question: str, verbose: bool = True):
        # 1. Buscar documentos relevantes
        search_results = self.vector_db.search(query=question, n_results=self.n_context_docs)

        sources = [search_result.metadata.get("source", "N/A") for search_result in search_results]  # noqa: E501
        if verbose:
            print(f"   Documentos encontrados de: {set(sources)}\n")

        # 2. Formatear contexto
        context = self._format_context(search_results)

        # 3. Crear prompt
        prompt = self._create_prompt(question, context)

        if verbose:
            print("🤔 Generando respuesta con LLM...")

        return prompt, context, sources

    def ask(self, question: str, verbose: bool = True) -> Dict:
        """
        Hace una pregunta y obtiene una respuesta con RAG

        Args:
            question: Pregunta del usuario
            verbose: Si mostrar información detallada

        Returns:
            Diccionario con la respuesta y metadatos
        """
        if verbose:
            print(f"\n❓ Pregunta: {question}\n")
            print("🔍 Buscando documentos relevantes...")

        prompt, context, sources = self.generate_prompt(question, verbose)

        # 4. Generar respuesta con LLM
        response = ollama.generate(
            model=self.llm_model,
            prompt=prompt,
            options={"temperature": self.temperature},
        )

        answer = response["response"]

        if verbose:
            print(f"\n💬 Respuesta:\n{answer}\n")
            print("=" * 80)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": sources,
            "model": self.llm_model,
        }

    def ask_streaming(self, question: str):
        """
        Hace una pregunta y obtiene una respuesta en streaming

        Args:
            question: Pregunta del usuario

        Yields:
            Chunks de la respuesta
        """
        print(f"\n❓ Pregunta: {question}\n")
        print("🔍 Buscando documentos relevantes...")

        # Buscar documentos relevantes
        search_results = self.vector_db.search(query=question, n_results=self.n_context_docs)

        sources = [search_result.metadata.get("source", "N/A") for search_result in search_results]  # noqa: E501
        print(f"   Documentos encontrados de: {set(sources)}\n")

        # Formatear contexto y crear prompt
        context = self._format_context(search_results)
        prompt = self._create_prompt(question, context)

        print("🤔 Generando respuesta...\n")
        print("💬 Respuesta: ", end="", flush=True)

        # Generar respuesta en streaming
        stream = ollama.generate(
            model=self.llm_model,
            prompt=prompt,
            stream=True,
            options={"temperature": self.temperature, "num_predict": 500},
        )

        full_response = ""
        for chunk in stream:
            if chunk["response"]:
                print(chunk["response"], end="", flush=True)
                full_response += chunk["response"]
                yield chunk["response"]

        return {
            "question": question,
            "answer": full_response,
            "sources": sources,
        }


class QAEngineBuilder:
    """
    Fabrica de motores de preguntas y respuestas con RAG
    """

    def __init__(self):
        self.vector_db = None
        self.llm_model = None
        self.n_context_docs = 5
        self.temperature = 0.2

    # region With Methods Pattern

    def with_vector_db(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        return self

    def with_llm_model(self, llm_model: str):
        self.llm_model = llm_model
        return self

    def with_n_context_docs(self, n_context_docs: int):
        self.n_context_docs = n_context_docs
        return self

    def with_temperature(self, temperature: float):
        self.temperature = temperature
        return self

    # endregion

    def build(self):
        self.__validate()

        return QAEngine(
            vector_db=self.vector_db,
            llm_model=self.llm_model,
            n_context_docs=self.n_context_docs,
            temperature=self.temperature,
        )

    def __validate(self):
        """
        Valida la configuración del motor de preguntas y respuestas
        """
        if self.vector_db is None or not isinstance(self.vector_db, VectorDatabase):
            raise ValueError("La base de datos vectorial no está configurada")
        if self.llm_model is None or not isinstance(self.llm_model, str):
            raise ValueError("El modelo de LLM no está configurado")
        if self.n_context_docs is None or not isinstance(self.n_context_docs, int):
            raise ValueError("El número de documentos de contexto no está configurado")
        if self.temperature is None or not isinstance(self.temperature, float):
            raise ValueError("La temperatura no está configurada")
