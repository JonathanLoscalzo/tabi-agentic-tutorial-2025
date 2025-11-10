"""
Configuración del sistema usando Pydantic Settings
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TabiConfig(BaseSettings):
    """Configuración principal del sistema Tabi QA"""

    # Modelos de Ollama
    embedding_model: str = Field(default="embeddinggemma:latest", description="Modelo de embeddings de Ollama")
    llm_model: str = Field(default="deepseek-r1:1.5b", description="Modelo LLM de Ollama")

    # Configuración de chunks de documentos
    chunk_size: int = Field(default=1000, description="Tamaño de cada chunk de texto")
    chunk_overlap: int = Field(default=200, description="Superposición entre chunks")

    # Configuración de ChromaDB
    chroma_persist_directory: str = Field(default="./chroma_db", description="Directorio de persistencia de ChromaDB")
    chroma_collection_name: str = Field(default="tabi_docs", description="Nombre de la colección en ChromaDB")

    # Configuración del QA Engine
    n_context_docs: int = Field(default=5, description="Número de documentos de contexto")
    temperature: float = Field(default=0.2, description="Temperatura del modelo LLM (0-1)")

    # Directorio de datos
    data_directory: str = Field(default="./data", description="Directorio con los PDFs")

    # Procesamiento
    batch_size: int = Field(default=50, description="Tamaño de batch para embeddings")

    model_config = SettingsConfigDict(
        yaml_file="tabi-db.config.yml",
        yaml_file_encoding="utf-8",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def get_chroma_path(self) -> Path:
        """Obtiene el Path del directorio de ChromaDB"""
        return Path(self.chroma_persist_directory)

    def get_data_path(self) -> Path:
        """Obtiene el Path del directorio de datos"""
        return Path(self.data_directory)

    def display(self):
        """Muestra la configuración actual"""
        print("\n⚙️  CONFIGURACIÓN ACTUAL")
        print("=" * 80)
        print(f"  Modelo LLM: {self.llm_model}")
        print(f"  Modelo Embeddings: {self.embedding_model}")
        print(f"  Chunk Size: {self.chunk_size}")
        print(f"  Chunk Overlap: {self.chunk_overlap}")
        print(f"  ChromaDB Path: {self.chroma_persist_directory}")
        print(f"  Collection Name: {self.chroma_collection_name}")
        print(f"  Context Docs: {self.n_context_docs}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Data Directory: {self.data_directory}")
        print(f"  Batch Size: {self.batch_size}")
        print("=" * 80)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TabiConfig":
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configuration files. " "Install it with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested structure if config is under a "featurium" key
        tabi_data = data.get("tabi", data)

        return cls(**tabi_data)


def load_config(config_file: Optional[str] = None) -> TabiConfig:
    """
    Carga la configuración desde un archivo YAML

    Args:
        config_file: Ruta al archivo de configuración (opcional)

    Returns:
        Instancia de TabiConfig
    """
    if config_file:
        # Cargar desde archivo específico
        return TabiConfig.from_yaml(config_file)
    else:
        # Cargar desde archivo por defecto
        return TabiConfig()
