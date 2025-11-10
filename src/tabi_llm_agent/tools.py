"""
Herramientas para los agentes
"""

import httpx
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import SearxSearchResults


# Instancia global de VectorDatabase (se inicializará en agent.py)
_vector_db = None


def set_vector_db(vector_db):
    """Establece la instancia de VectorDatabase para las herramientas"""
    global _vector_db
    _vector_db = vector_db


@tool
def search_vector_db(query: str, n_results: int = 5) -> str:
    """
    Busca información relevante en la base de datos vectorial local.
    Útil para encontrar información sobre modelo dimensional, bases de datos,
    y contenido de documentos previamente cargados.

    Args:
        query: La consulta de búsqueda
        n_results: Número de resultados a retornar (por defecto 5)

    Returns:
        String con los documentos encontrados y sus metadatos
    """
    print(f"Buscando en la base de datos vectorial: {query}")
    if _vector_db is None:
        return "Error: Base de datos vectorial no inicializada"

    try:
        # Usar el método search de VectorDatabase
        documents = _vector_db.search(query, n_results=n_results)

        if not documents:
            return "No se encontraron documentos relevantes en la base vectorial."

        # Formatear resultados
        resultados = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            content = doc.page_content

            resultado = f"\n--- Documento {i} ---"
            resultado += f"\nFuente: {metadata.get('source', 'N/A')}"
            resultado += f"\nPágina: {metadata.get('page', 'N/A')}"
            resultado += f"\nContenido:\n{content}\n"
            resultados.append(resultado)

        return "\n".join(resultados)

    except Exception as e:
        return f"Error al buscar en la base vectorial: {str(e)}"


def search_web(num_results: int = 5) -> SearxSearchResults:
    """
    Busca información en internet usando SearXNG.
    Útil para encontrar información actualizada sobre tecnologías,
    herramientas open source, machine learning, etc.

    Args:
        query: La consulta de búsqueda
        num_results: Número de resultados a retornar

    Returns:
        resultados de la búsqueda
    """
    searx = SearxSearchWrapper(searx_host="http://127.0.0.1:8080", k=num_results)
    return SearxSearchResults(name="WebSearchTool", wrapper=searx)


@tool
def search_on_web(query: str, num_results: int = 5) -> str:
    """
    Busca información en internet usando SearXNG.
    Útil para encontrar información actualizada sobre tecnologías,
    herramientas open source, machine learning, etc.

    Args:
        query: La consulta de búsqueda
        num_results: Número de resultados a retornar

    Returns:
        String con los resultados de la búsqueda
    """
    try:
        print(f"Buscando en web: {query}")
        # Configurar SearXNG (por defecto usa una instancia pública)
        # El usuario puede configurar su propia instancia
        searxng_url = "https://searx.be/search"

        params = {
            "q": query,
            "format": "json",
            "language": "es",
            "safesearch": 1,
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.get(searxng_url, params=params)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])[:num_results]

        if not results:
            return "No se encontraron resultados en la búsqueda web."

        # Formatear resultados
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"\n--- Resultado {i} ---"
            formatted_result += f"\nTítulo: {result.get('title', 'N/A')}"
            formatted_result += f"\nURL: {result.get('url', 'N/A')}"
            formatted_result += f"\nDescripción: {result.get('content', 'N/A')}\n"
            formatted_results.append(formatted_result)

        print(f"Resultados de la búsqueda web: {formatted_results}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error al buscar en web: {str(e)}"


@tool
def navigate_url(url: str) -> str:
    """
    Navega a una URL específica y extrae el contenido de texto principal.
    Útil para obtener información detallada de un artículo o documentación web.

    Args:
        url: La URL a visitar

    Returns:
        String con el contenido extraído de la página
    """
    try:
        print(f"Navegando a: {url}")
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        # Parsear HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Eliminar scripts y estilos
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Extraer texto
        text = soup.get_text()

        # Limpiar texto
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        # Limitar tamaño del texto (primeros 3000 caracteres)
        if len(text) > 3000:
            text = text[:3000] + "\n\n[... contenido truncado ...]"

        print(f"Contenido de {url}:\n\n{text}")

        return f"Contenido de {url}:\n\n{text}"

    except Exception as e:
        return f"Error al navegar a {url}: {str(e)}"


# Lista de herramientas disponibles
AVAILABLE_TOOLS = [
    search_vector_db,
    search_on_web,
    navigate_url,
]
