#Todo lo necesario para cargar las variables de ambiente
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
#se agrega MergedDataLoader (de langchain_avanzado.py) para poder cargar PDF y TXT al mismo tiempo
from langchain_community.document_loaders.merge import MergedDataLoader
#se reemplaza CharacterTextSplitter.from_huggingface_tokenizer por RecursiveCharacterTextSplitter,
#ya no depende de un tokenizer local de HuggingFace/Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
#se reemplaza OllamaEmbeddings/OllamaLLM por los equivalentes de Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

load_dotenv() #se cargan las variables de entorno

#se asignan a variables esas variables de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="langchain"

# Carga TODOS los archivos PDF que esten dentro de la carpeta 'documentos'
loader_pdfs = DirectoryLoader('documentos_pdf', glob='*.pdf', loader_cls=PyPDFLoader)
# Carga TODOS los archivos TXT que esten dentro de la carpeta 'documentos'
# (se agrega porque los documentos del proyecto final ya estan limpios en .txt)
loader_txts = DirectoryLoader('documentos_pdf', glob='*.txt', loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
# se combinan los dos loaders en uno solo, igual que en langchain_avanzado.py
documentos = MergedDataLoader(loaders=[loader_pdfs, loader_txts]).load()

# Antes se usaba el tokenizer de HuggingFace (BAAI/bge-m3) para medir los chunks en tokens.
# Ahora se usa RecursiveCharacterTextSplitter, que no depende de ningun modelo local
# y mide directamente por caracteres (mismo chunk_size/chunk_overlap que se usaba antes)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1250,          # cada chunk tendra aproximadamente 1250 caracteres
    chunk_overlap=150         # cada chunk comparte 150 caracteres con el chunk vecino,
                               # para no perder contexto en los cortes entre fragmentos
)

#se hace el chuking con las condiciones definidas en "splitter"
fragmentos = splitter.split_documents(documentos)

#Se crea la variable con el modelo de embedding de GEMINI (antes era de Ollama/bge-m3)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

#AHORA SE CREARA LA VECTOR STORE que almacena los chunks y embeddings
vector_store = FAISS.from_documents(documents=fragmentos, embedding=embeddings)


#Ahora se configura toda la LLM
prompt = ChatPromptTemplate(
    [("system", """Responde usando exclusivamente el contenido que se incluye a continuacion.
Si la respuesta no esta en el contenido, indica que no tienes esa informacion.

#CONTENIDO
{contexto}"""),
     ("human", "{query}")]
)

def buscar_fragmentos(texto_pregunta):
    vector_pregunta = embeddings.embed_documents([texto_pregunta])[0]
    fragmentos_encontrados = vector_store.similarity_search_by_vector(vector_pregunta, k=4)
    return "\n\n".join(fragmento.page_content for fragmento in fragmentos_encontrados)

retriever = RunnableLambda(buscar_fragmentos)

#para que las respuestas sean lo mas concretas posible (idea tomada de langchain_avanzado.py)
modelo = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.2
)

pregunta = "¿Que es el Cadmio, cuales son sus implicaciones a la salud y sus limites maximos permisibles?"

#instanciar un modelo sencillo para la reescritura (antes era OllamaLLM, ahora se reusa el mismo modelo de Gemini)
query_model = modelo

#template para reescribir las query del usuario

rewriter_prompt_template = """
Genera la consulta de búsqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario,
permitiendo una respuesta más precisa por medio de la búsqueda semántica.
Basta devolver la consulta revisada del Vector DB, entre comillas.

# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""

#se ejecuta la funcion que llama a la langhain pasando como contexto nuestro
rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)

#se crea la cadena pero ahora para reescribir con llm la query del usuario y que la respuesta final solo sea el texto respuesta
rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

#se arma la cadena completa del RAG con RunnablePassthrough (esto ya estaba planteado
#como comentario en la version de Ollama, ahora se deja activo)
rag_chain = (
    {
        "contexto": {"user_question": RunnablePassthrough()} | rewriter_chain | retriever, #modulo que toma la entrada y la coloca como siguiente entrada para el siguiente paso de la cadena
        "query": RunnablePassthrough()
    }
    | prompt | modelo | StrOutputParser()
)

respuesta = rag_chain.invoke(pregunta)
print(respuesta)


