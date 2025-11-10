# 🎯 Agentic Tabi QA

Sistema de preguntas y respuestas con RAG (Retrieval Augmented Generation) usando **Ollama**, **ChromaDB** y **LangChain**.

## 📋 Descripción

Este proyecto procesa documentos PDF y crea una base de datos vectorial que permite hacer preguntas sobre el contenido de los documentos usando modelos de lenguaje local con Ollama. Los agentes usan BedRock.

### Características

- ✅ **Procesamiento de PDFs**: Carga y divide documentos en chunks manejables
- ✅ **Base de datos vectorial**: Usa ChromaDB para almacenamiento eficiente
- ✅ **Embeddings locales**: Genera embeddings con Ollama (nomic-embed-text)
- ✅ **RAG (Retrieval Augmented Generation)**: Responde preguntas con contexto relevante
- ✅ **100% Local**: Todo funciona en tu máquina, sin APIs externas
- ✅ **Modo interactivo**: Interfaz de línea de comandos para conversación

## 🏗️ Estructura del Proyecto

```
agentic-tabi-qa/
├── data/                           # PDFs a procesar
│   ....
├── src/                            # Código fuente
│   ├── __init__.py
│   ├── document_loader.py          # Carga y procesa PDFs
│   ├── vector_db.py                # Gestión de ChromaDB
│   └── qa_engine.py                # Motor de Q&A con RAG
├── chroma_db/                      # Base de datos vectorial (generado)
├── main.py                         # Script principal
├── pyproject.toml                  # Configuración del proyecto
├── requirements.txt                # Dependencias
└── README.md                       # Este archivo
```

## 🚀 Instalación

### 1. Requisitos Previos

#### Instalar Ollama

**macOS:**

```bash
brew install ollama
```

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Descarga desde [ollama.com](https://ollama.com)

#### Iniciar Ollama

```bash
ollama serve
```

#### Descargar modelos necesarios

```bash
# Modelo para embeddings
ollama pull embeddinggema

# Modelo para generación de respuestas (elige uno)
ollama pull llama3.2      # Recomendado (ligero y rápido)
# ollama pull llama3.1     # Alternativa más potente
# ollama pull mistral      # Otra alternativa
```

### 2. Instalar Dependencias de Python

- Instalar uv

```bash
uv pip install -r pyproject.toml
uv pip install -e .
```

## 📖 Uso

### CLI

```
tabi-qa load
tabi-qa ask
```

### 🎨 Interfaz Streamlit

Ejecutar

```
tabi-qa-st
```

### Agente

Ejecutar

```
tabi-qa-agent
```

## 📊 ¿Por qué ChromaDB?

**ChromaDB** fue seleccionada por las siguientes razones:

| Característica         | ChromaDB                         | Alternativas                       |
| ---------------------- | -------------------------------- | ---------------------------------- |
| **Facilidad de uso**   | ⭐⭐⭐⭐⭐ Simple, sin servidor  | Pinecone/Weaviate requieren config |
| **Local-first**        | ✅ 100% local                    | Pinecone es cloud-only             |
| **Integración Ollama** | ✅ Excelente                     | Variable                           |
| **Persistencia**       | ✅ Automática en disco           | Algunos requieren setup            |
| **Performance**        | ⭐⭐⭐⭐ Rápido para ~1000s docs | FAISS más rápido pero más complejo |
| **Python API**         | ⭐⭐⭐⭐⭐ Muy Pythonic          | Variable                           |

### Otras opciones consideradas:

- **FAISS**: Más rápido pero requiere más configuración
- **Pinecone**: Excelente pero cloud-only (no local)
- **Weaviate**: Potente pero requiere Docker/servidor
- **Qdrant**: Bueno pero más complejo de configurar

## 🧪 Ejemplos de Preguntas

Basado en los documentos incluidos, puedes preguntar:

### Modelado Dimensional

- "¿Qué es el modelado dimensional?"
- "¿Cuáles son las tablas de hechos y dimensiones?"
- "¿Qué es un data warehouse?"
- "Explica el concepto de grano en modelado dimensional"

### Análisis de Datos con Software Libre

- "¿Qué ventajas tiene usar software libre para análisis de datos?"
- "¿Qué herramientas de software libre menciona el documento?"
- "¿Cómo se compara Python con R para análisis de datos?"

## LangFuse

**LangFuse** es una plataforma open-source de observabilidad y análisis para aplicaciones LLM (Large Language Models). En este proyecto se utiliza para:

- 📊 **Tracing**: Rastrear y visualizar cada paso de la ejecución de los agentes (llamadas al LLM, uso de herramientas, etc.)
- 🔍 **Debugging**: Identificar problemas en el flujo de los agentes y optimizar prompts
- 📈 **Métricas**: Monitorear el rendimiento, costos, latencia y calidad de las respuestas
- 🧪 **Evaluación**: Comparar diferentes versiones de prompts y configuraciones

El sistema está integrado con LangFuse mediante el `CallbackHandler` de LangChain, lo que permite observabilidad completa sin modificar la lógica de los agentes.

## 🐛 Solución de Problemas

### Respuestas de baja calidad

- Ajusta `chunk_size` y `chunk_overlap` en `DocumentLoader`
- Incrementa `n_context_docs` en `QAEngine`
- Usa un modelo más potente (llama3.1 en lugar de llama3.2)
- Ajusta la `temperature` (valores más bajos = más conservador)

## 📝 Notas Técnicas

### Proceso de RAG

1. **Carga de documentos**: Los PDFs se extraen y dividen en chunks
2. **Generación de embeddings**: Cada chunk se convierte en un vector usando `nomic-embed-text`
3. **Almacenamiento**: Los vectores se guardan en ChromaDB con sus metadatos
4. **Consulta**: Cuando haces una pregunta:
   - Se genera un embedding de la pregunta
   - Se buscan los chunks más similares (cosine similarity)
   - Se construye un prompt con el contexto relevante
   - El LLM genera una respuesta basada en ese contexto

## 📄 Licencia

libre de usar y modificar este código.

## Referencias
- https://docs.langchain.com/oss/python/langgraph/graph-api
- https://docs.langchain.com/oss/python/langgraph/workflows-agents

