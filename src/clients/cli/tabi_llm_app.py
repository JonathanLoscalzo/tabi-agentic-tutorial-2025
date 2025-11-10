#!/usr/bin/env python
"""
CLI principal para Tabi QA usando argparse y questionary
"""

import argparse
import sys

import questionary
from questionary import Style

from tabi_llm_app import TabiConfig
from tabi_llm_app.utils import create_vector_database, retrieve_vector_database
from tabi_llm_app.config import load_config
from tabi_llm_app.qa_engine import QAEngineBuilder


# Estilo personalizado para questionary
custom_style = Style(
    [
        ("qmark", "fg:#673ab7 bold"),
        ("question", "bold"),
        ("answer", "fg:#f44336 bold"),
        ("pointer", "fg:#673ab7 bold"),
        ("highlighted", "fg:#673ab7 bold"),
        ("selected", "fg:#cc5454"),
        ("separator", "fg:#cc5454"),
        ("instruction", ""),
        ("text", ""),
    ]
)


def cmd_load_db(args):
    """
    Comando para cargar/crear la base de datos vectorial
    """
    # Cargar configuración
    config = load_config(args.config)
    config.display()

    vector_db = create_vector_database(
        data_dir=config.data_directory,
        chroma_dir=config.chroma_persist_directory,
        collection_name=config.chroma_collection_name,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        batch_size=config.batch_size,
        force_reload=args.force_reload,
    )

    print("\n✅ Base de datos cargada")
    print(f"   Documentos: {vector_db.collection.count()}")
    print(f"   Directorio: {config.chroma_persist_directory}")
    print(f"   Colección: {config.chroma_collection_name}")
    print(f"   Embedding model: {config.embedding_model}")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Chunk overlap: {config.chunk_overlap}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Force reload: {False}")


def cmd_ask(args):
    """
    Comando para hacer preguntas interactivas usando questionary
    """
    # Cargar configuración
    config = TabiConfig.from_yaml(args.config)
    config.display()

    print("\n🎯 Sistema de Q&A con RAG - Tabi")
    print("   ChromaDB + Ollama\n")

    # Verificar que existe la base de datos
    vector_db = retrieve_vector_database(
        chroma_dir=config.chroma_persist_directory,
        collection_name=config.chroma_collection_name,
        embedding_model=config.embedding_model,
    )

    if vector_db.collection.count() == 0:
        print("❌ Error: La base de datos está vacía")
        print("\nPrimero ejecuta: tabi-qa load")
        sys.exit(1)

    print(f"✅ Base de datos cargada ({vector_db.collection.count()} documentos)\n")

    # Inicializar motor de Q&A
    print("🤖 Inicializando motor de Q&A...")

    qa_engine = (
        QAEngineBuilder()
        .with_vector_db(vector_db)
        .with_llm_model(config.llm_model)
        .with_n_context_docs(config.n_context_docs)
        .with_temperature(config.temperature)
        .build()
    )

    print("\n" + "=" * 80)
    print("💬 MODO INTERACTIVO DE PREGUNTAS Y RESPUESTAS")
    print("=" * 80)
    print("\nOpciones:")
    print("  - Escribe tu pregunta y presiona Enter")
    print("  - Usa Ctrl+C o selecciona 'Salir' para terminar\n")

    # Preguntas de ejemplo
    example_questions = [
        "¿Qué es el modelado dimensional?",
        "¿Cuáles son las ventajas del software libre?",
        "¿Qué es una tabla de hechos?",
        "Explica el concepto de grano",
    ]

    while True:
        try:
            if questionary.confirm("¿Quieres usar una pregunta de ejemplo?", default=False, style=custom_style).ask():
                question = questionary.select(
                    "Elige una pregunta:",
                    choices=example_questions,
                    style=custom_style,
                ).ask()

            else:
                question = questionary.text("Tu pregunta:", style=custom_style).ask()

            # Si el usuario cancela (Ctrl+C o vacío)
            if not question:
                should_continue = questionary.confirm(
                    "¿Quieres hacer otra pregunta?", default=True, style=custom_style
                ).ask()

                if not should_continue:
                    print("\n👋 ¡Hasta luego!")
                    break
                continue

            # Si el usuario quiere salir
            if question.lower() in ["salir", "exit", "quit", "q"]:
                print("\n👋 ¡Hasta luego!")
                break

            # Procesar pregunta
            result = qa_engine.ask(question, verbose=False)
            print("=" * 80)
            print(f"   Documentos encontrados: {result['sources']}")
            print(f"   Modelo: {result['model']}")
            print(f"   Contexto: {result['context']}")
            print(f"   Pregunta: {result['question']}")
            print(f"   Respuesta: {result['answer']}")
            print(f"   Documentos encontrados: {result['sources']}")
            print(f"   Modelo: {result['model']}")
            print("=" * 80)
            print("\n💬 Respuesta:")
            print(result["answer"])
            print("=" * 80)

            # Preguntar si quiere continuar
            print()
            should_continue = questionary.confirm("¿Otra pregunta?", default=True, style=custom_style).ask()

            if not should_continue:
                print("\n👋 ¡Hasta luego!")
                break

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            should_continue = questionary.confirm("¿Quieres continuar?", default=True, style=custom_style).ask()

            if not should_continue:
                break


def main():
    """Función principal del CLI"""
    parser = argparse.ArgumentParser(
        description="Tabi QA - Sistema de preguntas y respuestas con RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Cargar la base de datos vectorial
  %(prog)s load
  
  # Cargar forzando recreación
  %(prog)s load --force-reload

  
  # Hacer preguntas interactivas
  %(prog)s ask
  
  # Usar un archivo de configuración personalizado
  %(prog)s load --config mi-config.yml
  %(prog)s ask --config mi-config.yml
        """,
    )

    # Argumento global para archivo de configuración
    parser.add_argument(
        "--config",
        type=str,
        default="tabi-db.config.yml",
        required=False,
        help="Ruta al archivo de configuración YAML (por defecto: tabi-db.config.yml)",
    )

    # Subcomandos
    subparsers = parser.add_subparsers(
        title="comandos", description="Comandos disponibles", dest="command", required=True
    )

    # Comando: load
    parser_load = subparsers.add_parser("load", help="Cargar/crear la base de datos vectorial desde PDFs")
    parser_load.add_argument("--force-reload", action="store_true", help="Forzar recreación de la base de datos")
    parser_load.set_defaults(func=cmd_load_db)

    # Comando: ask
    parser_ask = subparsers.add_parser("ask", help="Modo interactivo de preguntas y respuestas")
    parser_ask.set_defaults(func=cmd_ask)

    # Parsear argumentos
    args = parser.parse_args()

    # Ejecutar el comando correspondiente
    args.func(args)


if __name__ == "__main__":
    main()
