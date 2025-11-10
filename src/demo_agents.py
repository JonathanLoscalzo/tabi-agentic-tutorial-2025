#!/usr/bin/env python3
"""
Script de demostración del sistema de agentes Tabi
"""

import uuid
import time
from datetime import datetime
from pathlib import Path
from tabi_llm_app.utils import retrieve_vector_database
from tabi_llm_app.config import TabiConfig
from tabi_llm_agent import TabiAgentSystem
from dotenv import load_dotenv
from loguru import logger

logger.add("logs/demo_agents.log", rotation="100 MB", retention="30 days")

load_dotenv()


def save_qa_to_markdown(question: str, answer: str, session_id: str, answers_dir: str = "answers"):
    """
    Guarda una pregunta y respuesta en un archivo Markdown con formato de fecha/hora.

    Args:
        question: La pregunta realizada
        answer: La respuesta generada
        session_id: ID de la sesión
        answers_dir: Directorio donde guardar los archivos (default: "answers")
    """
    # Crear el directorio si no existe
    answers_path = Path(answers_dir)
    answers_path.mkdir(exist_ok=True)

    # Generar nombre de archivo con formato yyyymmdd_hhmmss
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.md"
    filepath = answers_path / filename

    # Crear contenido en formato Markdown
    content = f"""# Pregunta y Respuesta - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Sesión
- **ID**: `{session_id}`
- **Timestamp**: {datetime.now().isoformat()}

---

## Pregunta

{question}

---

## Respuesta

{answer}

---

*Generado automáticamente por el Sistema de Agentes Tabi QA*
"""

    # Guardar el archivo
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"💾 Respuesta guardada en: {filepath}")

    return filepath


def main():
    """Función principal de demostración"""

    logger.info("🤖 DEMO: Sistema de Agentes Tabi QA")

    # Cargar configuración
    logger.info("1️⃣  Cargando configuración...")
    config = TabiConfig.from_yaml("tabi-db.config.yml")
    config.display()

    # Inicializar base de datos vectorial
    logger.info("2️⃣  Inicializando base de datos vectorial...")
    vector_db = retrieve_vector_database(
        chroma_dir=config.chroma_persist_directory,
        collection_name=config.chroma_collection_name,
        embedding_model=config.embedding_model,
    )

    # Inicializar sistema de agentes
    logger.info("3️⃣  Inicializando sistema de agentes...")
    logger.info("Opciones de configuración:")
    logger.info("  - use_cra_agents: Usa agentes ReAct para búsquedas más sofisticadas (llaman tools múltiples veces)")
    logger.info("  - Flujo fijo: Vector DB → Web Search → Summarizer (siempre ejecuta ambas búsquedas)")
    agent_system = TabiAgentSystem(
        vector_db=vector_db,
        llm_model=config.llm_model,
        memory_db_path="./agent_memory.db",
        use_cra_agents=True,  # True = agentes ReAct (más profundo), False = agentes simples (más rápido)
    )

    logger.info("✅ Sistema inicializado correctamente")

    # Mostrar estadísticas
    logger.info("📊 Estadísticas del sistema:")
    vector_stats = vector_db.get_stats()

    logger.info(f"   📚 Documentos en base vectorial: {vector_stats['document_count']}")

    # Preguntas de ejemplo
    example_questions = [
        "¿Qué es el modelo dimensional de Kimball?",
        "¿Cuáles son las mejores herramientas open source para ETL?",
        "¿Qué es un esquema estrella en data warehousing?",
        "Dame información sobre machine learning en Python",
        "¿Cuál es la capital de Francia?",  # Esta no debería ser relevante
    ]

    logger.info("🔍 EJEMPLOS DE CONSULTAS")

    session_id = str(uuid.uuid4()) + "_demo_session"
    logger.info(f"Sesión ID: {session_id}")

    for i, question in enumerate(example_questions, 1):
        logger.info(f"Pregunta {i}: {question}")

        # Procesar pregunta
        answer = agent_system.query(question, session_id=session_id)

        logger.info(f"🤖 Respuesta:\n{answer}\n")

        # Guardar pregunta y respuesta en archivo Markdown
        save_qa_to_markdown(question, answer, session_id)

        # Pequeña pausa para mejor legibilidad
        time.sleep(1)

    logger.info("✅ Demo completada exitosamente!")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Demo del sistema de agentes Tabi QA")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Modo interactivo",
    )
    parser.add_argument(
        "--all-modes",
        "-a",
        action="store_true",
        help="Demo de todos los modos de operación (Simple, CRA)",
    )

    args = parser.parse_args()

    main()
