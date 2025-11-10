"""
Módulo para cargar y procesar documentos PDF
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document

# from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DirectoryLoader:
    """Carga y procesa documentos PDF de un directorio"""

    documents: List[Document] = None

    def __init__(self, args: "DirectoryLoaderArgs"):
        """
        Inicializa el loader de documentos
        """
        self.directory_path = args.directory_path
        self.loader = PyPDFDirectoryLoader(self.directory_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            add_start_index=args.add_start_index,
        )

    def load_documents(self) -> List[Dict]:
        """
        Carga y procesa todos los documentos PDF en el directorio
        """
        if self.documents is None:
            self.documents = self.loader.load()
            print(f"📄 Cargados {len(self.documents)} documentos")
        return self.documents

    def split_documents(self) -> List[Document]:
        """
        Divide los documentos en chunks
        """
        all_splits = self.text_splitter.split_documents(self.documents)
        print(f"📄 Divididos {len(all_splits)} chunks")
        return all_splits


@dataclass
class DirectoryLoaderArgs:
    directory_path: Path | str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    add_start_index: bool = True

    def __init__(
        self,
        directory_path: Path | str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
    ):
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

    def __validate(self):
        """
        Valida la configuración del loader de documentos
        """
        if self.directory_path is None or not isinstance(self.directory_path, Path):
            raise ValueError("El directorio de documentos no está configurado")
        if self.chunk_size is None or not isinstance(self.chunk_size, int):
            raise ValueError("El tamaño de chunk no está configurado")
        if self.chunk_overlap is None or not isinstance(self.chunk_overlap, int):
            raise ValueError("El tamaño de overlap no está configurado")
        if self.add_start_index is None or not isinstance(self.add_start_index, bool):
            raise ValueError("El índice de inicio no está configurado")

    def build(self) -> "DirectoryLoader":
        self.__validate()

        return DirectoryLoader(
            directory_path=self.directory_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=self.add_start_index,
        )
