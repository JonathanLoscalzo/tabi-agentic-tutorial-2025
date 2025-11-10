"""
Interfaz web con Streamlit para el sistema de Q&A
Segunda fase del proyecto
"""

import streamlit as st
from tabi_llm_app import QAEngine, TabiConfig, VectorDatabase
from tabi_llm_app.chat_qa_engine import ChatQAEngine
from tabi_llm_app.utils import retrieve_vector_database
from tabi_llm_app.qa_engine import QAEngineBuilder


# Configuración de la página
st.set_page_config(
    page_title="Tabi QA - Chat Interactivo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def initialize_system() -> tuple[VectorDatabase, QAEngine, ChatQAEngine]:
    """
    Inicializa la base de datos vectorial y el motor de Q&A
    (con cache)
    """
    with st.spinner("🔄 Cargando base de datos vectorial..."):
        config = TabiConfig.from_yaml("tabi-db.config.yml")
        config.display()

        # Verificar que existe la base de datos
        vector_db = retrieve_vector_database(
            chroma_dir=config.chroma_persist_directory,
            collection_name=config.chroma_collection_name,
            embedding_model=config.embedding_model,
        )
        if vector_db.collection.count() == 0:
            st.error("❌ Error: La base de datos está vacía")
            st.info("Asegúrate de haber ejecutado `python main.py` " "primero para crear la base de datos.")
            st.stop()

        st.info(f"✅ Base de datos cargada ({vector_db.collection.count()} documentos)\n")

        qa_engine = (
            QAEngineBuilder()  # noqa: F821
            .with_vector_db(vector_db)
            .with_llm_model(config.llm_model)
            .with_n_context_docs(config.n_context_docs)
            .with_temperature(config.temperature)
            .build()
        )

        chat_qa_engine = ChatQAEngine(qa_engine)

    return vector_db, qa_engine, chat_qa_engine


def main():
    """Función principal de la aplicación"""

    # Título principal
    st.title("🎯 Tabi QA - Chat Interactivo")
    st.markdown("Pregunta sobre **Modelado Dimensional** y " "**Análisis de Datos con Software Libre**")

    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")

        # Inicializar sistema
        try:
            vector_db, qa_engine, chat_qa_engine = initialize_system()
            stats = vector_db.get_stats()

            st.success("✅ Sistema iniciado")
            st.metric("Documentos en DB", stats["document_count"])
            st.info(f"**Modelo LLM:** {qa_engine.llm_model}")
            st.info(f"**Modelo Embeddings:** {stats['embedding_model']}")

        except Exception as e:
            st.error(f"❌ Error al inicializar: {e}")
            st.info("Asegúrate de haber ejecutado `python main.py` " "primero para crear la base de datos.")
            st.stop()

        st.divider()

        # Configuración
        st.header("⚙️ Configuración")

        n_context = st.slider(
            "Documentos de contexto",
            min_value=1,
            max_value=10,
            value=5,
            help="Número de documentos relevantes a usar",
        )

        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Creatividad del modelo (0=conservador, 1=creativo)",
        )

        model = st.selectbox(
            "Modelo LLM",
            options=["phi4:latest", "deepseek-r1:1.5b", "mistral:7b", "qwen3:0.6b" "gemma3:1b"],
            index=0,
            help="Modelo de LLM a usar",
        )

        # Actualizar configuración si cambió
        if model != qa_engine.llm_model:
            qa_engine.llm_model = model
            chat_qa_engine._chat.model = model
        if n_context != qa_engine.n_context_docs:
            qa_engine.n_context_docs = n_context
        if temperature != qa_engine.temperature:
            qa_engine.temperature = temperature
            chat_qa_engine._chat.temperature = temperature

        st.divider()

        # Ejemplos de preguntas
        st.header("💡 Preguntas de ejemplo")
        example_questions = [
            "¿Qué es el modelado dimensional?",
            "¿Qué es una tabla de hechos?",
            "¿Cuáles son las ventajas del software libre?",
            "Explica el concepto de grano",
        ]

        for question in example_questions:
            btn_key = f"example_{question}"
            if st.button(question, key=btn_key, use_container_width=True):
                st.session_state.example_question = question

    # Área principal de chat

    # Inicializar historial de chat para la UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Inicializar historial de mensajes para el LLM (formato LangChain)
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    # Inicializar flag de primera pregunta
    if "is_first_question" not in st.session_state:
        st.session_state.is_first_question = True

    # Inicializar fuentes consultadas
    if "sources" not in st.session_state:
        st.session_state.sources = []

    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Mostrar fuentes al final si existen
    if st.session_state.get("sources", None):
        with st.expander("📚 Fuentes consultadas (contexto inicial)"):
            unique_sources = set(st.session_state.sources)
            for source in unique_sources:
                st.text(f"• {source}")

    if st.session_state.get("context", None):
        with st.expander("📚 Contexto consultado"):
            st.markdown(st.session_state.context)

    # Input de chat
    prompt = st.chat_input("Haz tu pregunta aquí...")

    # Si hay una pregunta de ejemplo seleccionada, usarla
    if "example_question" in st.session_state:
        prompt = st.session_state.example_question
        del st.session_state.example_question

    # Procesar nueva pregunta
    if prompt:
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)

        # Añadir al historial de UI
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("🤔 Pensando..."):
                try:
                    # Si es la primera pregunta, hacer búsqueda RAG
                    if st.session_state.is_first_question:
                        # Buscar documentos relevantes y crear contexto
                        prompt_with_context, context, sources = chat_qa_engine.start_chat(prompt)

                        # Guardar fuentes para mostrarlas
                        st.session_state.sources += sources
                        st.session_state.context = context

                        # Crear mensaje del sistema con el contexto
                        st.session_state.llm_messages = [{"role": "system", "content": prompt_with_context}]

                        # Ya no es la primera pregunta
                        st.session_state.is_first_question = False

                    # Añadir pregunta del usuario al historial del LLM
                    st.session_state.llm_messages.append({"role": "user", "content": prompt})

                    # Generar respuesta usando el chat
                    response = chat_qa_engine.chat(st.session_state.llm_messages)
                    answer = response.content

                    # Mostrar respuesta
                    st.markdown(answer)

                    # Añadir respuesta al historial de UI
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Añadir respuesta al historial del LLM
                    st.session_state.llm_messages.append({"role": "assistant", "content": answer})

                    # Si fue la primera pregunta, hacer rerun para mostrar sources/context
                    if len(st.session_state.messages) == 2:  # User + Assistant
                        st.rerun()

                except Exception as e:
                    error_msg = f"❌ Error al generar respuesta: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Botón para limpiar conversación
    if st.session_state.messages:
        if st.button("🗑️ Limpiar conversación", type="secondary"):
            # Resetear todo el estado
            st.session_state.messages = []
            st.session_state.llm_messages = []
            st.session_state.is_first_question = True
            st.session_state.sources = []
            st.session_state.context = None
            st.rerun()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>🎯 Tabi QA - Sistema de preguntas y respuestas con RAG</p>
            <p><small>Powered by Ollama + ChromaDB + LangChain + Streamlit</small></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
