"""
Sistema de agentes con LangGraph para Tabi QA
Incluye múltiples agentes especializados con memoria short-term y long-term
"""

import sqlite3
from typing import Any, TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .prompts import (
    CLASSIFIER_PROMPT,
    VECTOR_AGENT_PROMPT,
    VECTOR_REACT_AGENT_PROMPT,
    WEB_AGENT_PROMPT,
    WEB_REACT_AGENT_PROMPT,
    SUMMARY_AGENT_PROMPT,
)


# from langfuse import get_client
from langfuse.langchain import CallbackHandler

from loguru import logger

from tabi_llm_agent.tools import (
    search_vector_db,
    search_on_web,
    navigate_url,
    search_web,
    set_vector_db,
)

# ========== DEFINICIÓN DEL ESTADO ==========


class AgentState(TypedDict):
    """Estado compartido entre los agentes"""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    query: str
    is_relevant: bool
    vector_result: str  # Resultado del agente vectorial
    web_result: str  # Resultado del agente web
    final_answer: str  # Solo escrito por el summarizer
    iteration_count: int


# ========== CLASE PRINCIPAL DEL SISTEMA DE AGENTES ==========


class TabiAgentSystem:
    """Sistema de agentes para Tabi QA"""

    def __init__(
        self,
        vector_db,
        llm_model: str = "deepseek-r1:1.5b",
        memory_db_path: str = "./agent_memory.db",
        use_cra_agents: bool = False,
    ):
        """
        Inicializa el sistema de agentes

        Args:
            vector_db: Instancia de VectorDatabase
            llm_model: Modelo de Ollama a usar
            memory_db_path: Ruta a la base de datos de memoria
            use_cra_agents: Si True, usa agentes ReAct (CRA) para búsquedas más sofisticadas
        """
        self.vector_db = vector_db
        self.llm_model = llm_model
        self.memory_db_path = memory_db_path
        self.use_cra_agents = use_cra_agents

        # Configurar la base de datos vectorial para las herramientas
        set_vector_db(vector_db)

        # Inicializar el modelo LLM
        # self.llm = ChatOllama(model=llm_model, temperature=0.2)
        self.llm = ChatBedrockConverse(model="amazon.nova-lite-v1:0", temperature=0.2)
        # self.llm_classifier = ChatOllama(model=llm_model, temperature=0.0)
        self.llm_classifier = ChatBedrockConverse(model="amazon.nova-lite-v1:0", temperature=0.0)

        # Crear el grafo
        self.graph = self._build_graph()

        logger.info("🤖 Sistema de agentes inicializado")
        logger.info(f"   Modelo LLM: {llm_model}")
        logger.info(f"   Modo CRA: {'✓' if use_cra_agents else '✗'}")
        logger.info("   Flujo: Vector DB → Web Search → Summarizer")

    # ========== NODOS DEL GRAFO ==========

    def classifier_node(self, state: AgentState) -> AgentState:
        """
        Nodo clasificador: determina si la pregunta es relevante
        """
        query = state["query"]

        # Crear mensaje para el clasificador
        message = HumanMessage(content=CLASSIFIER_PROMPT.format(query=query))

        # Invocar al LLM
        response = self.llm_classifier.invoke([message])
        response_text = response.content.strip().upper()

        # Determinar si es relevante
        is_relevant = "RELEVANTE" in response_text and "NO_RELEVANTE" not in response_text

        return {
            **state,
            "is_relevant": is_relevant,
            "messages": [AIMessage(content=f"Clasificación: {'Relevante' if is_relevant else 'No relevante'}")],
        }

    def vector_search_llm_node(self, state: AgentState) -> AgentState:
        """
        Nodo de búsqueda vectorial: busca en la base de datos local
        IMPORTANTE: Este nodo SIEMPRE debe llamar a la tool search_vector_db
        """
        query = state["query"]

        # Crear el agente con herramientas
        llm_with_tools = self.llm.bind_tools([search_vector_db], tool_choice="search_vector_db")

        # Mensaje del sistema
        system_message = SystemMessage(content=VECTOR_AGENT_PROMPT.format(query=query))
        user_message = HumanMessage(content=query)

        # Invocar al agente
        response = llm_with_tools.invoke([system_message, user_message])

        # Si el agente quiere usar herramientas, ejecutarlas
        if response.tool_calls:
            tool_node = ToolNode([search_vector_db])
            tool_results = tool_node.invoke({"messages": [response]})

            # Obtener respuesta final con los resultados de las herramientas
            final_response = self.llm.invoke([system_message, user_message, response] + tool_results["messages"])

            return {
                **state,
                "messages": [AIMessage(content=final_response.content)],
                "vector_result": final_response.content,
            }

        return {
            **state,
            "messages": [AIMessage(content=response.content)],
            "vector_result": response.content,
        }

    def web_search_llm_node(self, state: AgentState) -> AgentState:
        """
        Nodo de búsqueda web: busca información en internet
        IMPORTANTE: Este nodo SIEMPRE debe llamar a las tools search_on_web y/o navigate_url
        """
        query = state["query"]

        # Crear el agente con herramientas
        llm_with_tools = self.llm.bind_tools([search_web(num_results=5), navigate_url], tool_choice="any")

        # Mensaje del sistema
        system_message = SystemMessage(content=WEB_AGENT_PROMPT.format(query=query))
        user_message = HumanMessage(content=query)

        # Invocar al agente (puede requerir múltiples iteraciones)
        messages = [system_message, user_message]
        max_iterations = 3
        iteration = 0

        while iteration < max_iterations:
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # Si no hay tool calls, terminamos
            if not response.tool_calls:
                break

            # Ejecutar herramientas
            tool_node = ToolNode([search_on_web, navigate_url])
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend(tool_results["messages"])

            iteration += 1

        # Obtener la última respuesta
        final_content = messages[-1].content if messages else "No se pudo obtener información."

        return {
            **state,
            "messages": [AIMessage(content=final_content)],
            "web_result": final_content,
        }

    def vector_search_cra_node(self, state: AgentState) -> AgentState:
        """
        Nodo de búsqueda vectorial con ReAct: usa create_react_agent para búsqueda más sofisticada
        Este agente usa el patrón Reasoning and Acting (ReAct) para una búsqueda más inteligente
        en la base de datos vectorial local. Puede llamar a la herramienta múltiples veces.
        IMPORTANTE: Este nodo SIEMPRE debe llamar a la tool search_vector_db
        """
        query = state["query"]

        # Crear un agente ReAct con la herramienta de búsqueda vectorial
        react_agent = create_agent(
            model=self.llm,
            tools=[search_vector_db],
            system_prompt=VECTOR_REACT_AGENT_PROMPT,
        )

        # Preparar los mensajes de entrada
        input_messages = [HumanMessage(content=query)]

        # Invocar el agente ReAct
        try:
            # El agente ReAct necesita un diccionario con "messages"
            result = react_agent.invoke({"messages": input_messages})

            # Extraer la última respuesta del agente
            if result and "messages" in result:
                last_message = result["messages"][-1]
                final_content = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                final_content = "No se pudo obtener una respuesta del agente ReAct vectorial."

        except Exception as e:
            final_content = f"Error en el agente ReAct vectorial: {str(e)}"

        logger.info(f"Respuesta del agente ReAct vectorial: {final_content}")
        logger.info(20 * "=")
        return {
            **state,
            "messages": [AIMessage(content=final_content)],
            "vector_result": final_content,
        }

    def web_search_cra_node(self, state: AgentState) -> AgentState:
        """
        Nodo de búsqueda web con ReAct: usa create_react_agent para búsqueda más sofisticada
        Este agente usa el patrón Reasoning and Acting (ReAct) para una búsqueda más inteligente
        IMPORTANTE: Este nodo SIEMPRE debe llamar a las tools search_on_web y/o navigate_url
        """
        query = state["query"]

        # Crear un agente ReAct con las herramientas de búsqueda web
        react_agent = create_agent(
            model=self.llm,
            tools=[search_web(num_results=5), navigate_url],
            system_prompt=WEB_REACT_AGENT_PROMPT,
        )

        # Preparar los mensajes de entrada
        input_messages = [HumanMessage(content=query)]

        # Invocar el agente ReAct
        try:
            # El agente ReAct necesita un diccionario con "messages"
            result = react_agent.invoke({"messages": input_messages})

            # Extraer la última respuesta del agente
            if result and "messages" in result:
                last_message = result["messages"][-1]
                final_content = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                final_content = "No se pudo obtener una respuesta del agente ReAct web."

        except Exception as e:
            final_content = f"Error en el agente ReAct web: {str(e)}"

        logger.info(f"Respuesta del agente ReAct web: {final_content}")
        logger.info(20 * "=")
        return {
            **state,
            "messages": [AIMessage(content=final_content)],
            "web_result": final_content,
        }

    def summary_node(self, state: AgentState) -> AgentState:
        """
        Nodo de resumen: sintetiza la información recopilada de los agentes vectorial y web.
        Este es el ÚNICO nodo que debe escribir en final_answer.
        """
        query = state["query"]
        vector_result = state.get("vector_result", "")
        web_result = state.get("web_result", "")

        # Construir contexto con los resultados de ambos agentes
        context_parts = []
        if vector_result:
            context_parts.append(f"=== Resultados de búsqueda en documentos locales ===\n{vector_result}")
        if web_result:
            context_parts.append(f"=== Resultados de búsqueda web ===\n{web_result}")

        context = "\n\n".join(context_parts)

        # Crear mensaje para el resumen
        prompt = SUMMARY_AGENT_PROMPT.format(
            context=context,
        )
        messages = [SystemMessage(content=prompt), HumanMessage(content=f"Pregunta original: {query}")]

        # Invocar al LLM
        response = self.llm.invoke(messages)

        return {
            **state,
            "messages": [AIMessage(content=response.content)],
            "final_answer": response.content,
        }

    # ========== FUNCIONES DE ROUTING ==========

    def should_continue_after_classifier(
        self, state: AgentState
    ) -> Literal["vector_search", "vector_search_cra", "end"]:
        """
        Decide si continuar procesando después del clasificador.
        Si es relevante, va directo al agente vectorial (simple o CRA según configuración).
        """
        if state.get("is_relevant", False):
            # Ir directo al agente vectorial según el modo configurado
            return "vector_search_cra" if self.use_cra_agents else "vector_search"
        return "end"

    # ========== CONSTRUCCIÓN DEL GRAFO ==========

    def _build_graph(self):
        """
        Construye el grafo de LangGraph con flujo fijo y secuencial:

        Flujo SIEMPRE:
        Usuario → Clasificador → ¿Relevante? → Vector DB → Web Search → Summary → Respuesta

        Los agentes vectorial y web ejecutan OBLIGATORIAMENTE sus herramientas (tools forzadas).
        Solo el nodo summary escribe en final_answer.
        """
        # Crear el grafo
        workflow = StateGraph(AgentState)

        # Añadir nodos básicos
        workflow.add_node("classifier", self.classifier_node)
        workflow.add_node("summary", self.summary_node)

        # Añadir nodos de búsqueda según configuración (simple o CRA)
        if self.use_cra_agents:
            # Usar agentes CRA (ReAct)
            workflow.add_node("vector_search_cra", self.vector_search_cra_node)
            workflow.add_node("web_search_cra", self.web_search_cra_node)

            # Definir el flujo: START → Classifier → Vector → Web → Summary → END
            workflow.add_edge(START, "classifier")
            workflow.add_conditional_edges(
                "classifier",
                self.should_continue_after_classifier,
                {
                    "vector_search_cra": "vector_search_cra",
                    "end": END,
                },
            )
            workflow.add_edge("vector_search_cra", "web_search_cra")
            workflow.add_edge("web_search_cra", "summary")
        else:
            # Usar agentes simples
            workflow.add_node("vector_search", self.vector_search_llm_node)
            workflow.add_node("web_search", self.web_search_llm_node)

            # Definir el flujo: START → Classifier → Vector → Web → Summary → END
            workflow.add_edge(START, "classifier")
            workflow.add_conditional_edges(
                "classifier",
                self.should_continue_after_classifier,
                {
                    "vector_search": "vector_search",
                    "end": END,
                },
            )
            workflow.add_edge("vector_search", "web_search")
            workflow.add_edge("web_search", "summary")

        # Después del resumen, terminar
        workflow.add_edge("summary", END)

        # Configurar memoria persistente
        conn = sqlite3.connect(self.memory_db_path, check_same_thread=False)
        memory = SqliteSaver(conn)
        store = SqliteStore(conn)

        # Compilar el grafo
        return workflow.compile(name="TabiResearcher", checkpointer=memory, store=store)

    # ========== MÉTODOS PÚBLICOS ==========

    def query(self, question: str, session_id: str = "default") -> str:
        """
        Procesa una consulta usando el sistema de agentes

        Args:
            question: Pregunta del usuario
            session_id: ID de la sesión

        Returns:
            Respuesta generada por el sistema
        """
        # Crear el estado inicial
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "session_id": session_id,
            "query": question,
            "is_relevant": False,
            "vector_result": "",
            "web_result": "",
            "final_answer": "",
            "iteration_count": 0,
        }

        # Ejecutar el grafo

        # Initialize Langfuse client
        # langfuse = get_client()

        # if langfuse.auth_check():
        #     logger.info("Langfuse client is authenticated and ready!")
        # else:
        #     logger.info("Authentication failed. Please check your credentials and host.")

        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()
        config: RunnableConfig = {"configurable": {"thread_id": session_id}, "callbacks": [langfuse_handler]}
        final_state = self.graph.invoke(initial_state, config)

        # Obtener la respuesta final
        answer = final_state.get("final_answer", "")

        # Si la pregunta no es relevante, responder apropiadamente
        if not final_state.get("is_relevant", False):
            answer = (
                "Lo siento, tu pregunta no está relacionada con los temas que puedo ayudarte: "
                "modelo dimensional, open source, datos, o machine learning. "
                "¿Tienes alguna pregunta sobre estos temas?"
            )

        return answer

    async def query_async(self, question: str, session_id: str = "default") -> str:
        """
        Procesa una consulta usando el sistema de agentes de forma asíncrona
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "session_id": session_id,
            "query": question,
            "is_relevant": False,
            "vector_result": "",
            "web_result": "",
            "final_answer": "",
            "iteration_count": 0,
        }

        # Ejecutar el grafo
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        return await self.graph.ainvoke(initial_state, config)
