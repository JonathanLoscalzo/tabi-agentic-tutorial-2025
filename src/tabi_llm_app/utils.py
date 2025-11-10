from .document_loader import DirectoryLoaderArgs, DirectoryLoader
from .vector_db import VectorDatabase


def create_vector_database(
    data_dir: str = "./data",
    chroma_dir: str = "./chroma_db",
    collection_name: str = "tabi_docs",
    embedding_model: str = "embeddinggemma:latest",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 50,
    force_reload: bool = False,
) -> VectorDatabase:
    """
    Crea o carga la base de datos vectorial desde los PDFs

    Args:
        data_dir: Directorio con los PDFs
        chroma_dir: Directorio para ChromaDB
        collection_name: Nombre de la colección
        embedding_model: Modelo de embeddings de Ollama
        chunk_size: Tamaño de cada chunk
        chunk_overlap: Superposición entre chunks
        batch_size: Tamaño del batch para procesamiento
        force_reload: Si True, elimina la DB existente y la recrea

    Returns:
        Instancia de VectorDatabase
    """
    print("=" * 80)
    print("🚀 CREANDO BASE DE DATOS VECTORIAL")
    print("=" * 80)

    # 1. Inicializar el loader de documentos
    print("\n📚 Paso 1: Inicializando loader de documentos...")
    loader = DirectoryLoader(
        DirectoryLoaderArgs(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            directory_path=data_dir,
            add_start_index=True,
        )
    )

    # 2. Procesar todos los PDFs
    print("\n📖 Paso 2: Procesando PDFs...")
    documents = loader.load_documents()
    documents = loader.split_documents()

    # 3. Inicializar base de datos vectorial
    print("\n🗄️  Paso 3: Inicializando ChromaDB...")
    vector_db = VectorDatabase(
        persist_directory=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # 4. Limpiar DB si es necesario
    if force_reload and vector_db.collection.count() > 0:
        print("\n🗑️  Limpiando base de datos existente...")
        vector_db.clear_collection()

    # 5. Añadir documentos si es necesario
    if vector_db.collection.count() == 0:
        print("\n💾 Paso 4: Añadiendo documentos a ChromaDB...")
        vector_db.add_documents(documents, batch_size=batch_size)
    else:
        count = vector_db.collection.count()
        print(f"\n✅ Base de datos ya contiene {count} documentos")
        print("   (usa force_reload=True para recrear)")

    # 6. Mostrar estadísticas
    print("\n📊 Estadísticas de la base de datos:")
    stats = vector_db.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 80)
    print("✅ BASE DE DATOS VECTORIAL LISTA")
    print("=" * 80)

    return vector_db


def retrieve_vector_database(
    chroma_dir: str = "./chroma_db",
    collection_name: str = "tabi_docs",
    embedding_model: str = "embeddinggemma:latest",
) -> VectorDatabase:
    """
    Recupera la base de datos vectorial desde ChromaDB
    """
    return VectorDatabase(
        persist_directory=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
