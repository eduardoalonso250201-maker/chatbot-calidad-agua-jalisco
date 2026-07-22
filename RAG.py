from langchain.tools import BaseTool
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_FLASH
import ast
 
 
class HerramientaRAG(BaseTool):
    # mismo patron de nombre/description/return_direct que HerramientaAnalisisImagen y HerramientaExplicar
    name: str = "HerramientaRAG"
    description: str = """
                   Utiliza esta herramienta siempre que se pida informacion o explicacion de un
                  concepto, no un dato medido de una base de datos. Por ejemplo: preguntas sobre
                  rios vivos, humedales construidos, la problematica del agua en Jalisco (Lago de
                  Chapala, Rio Santiago), definiciones de los parametros de analisis de calidad del
                  agua, o informacion general sobre las comunidades involucradas (nombres,
                  poblacion, ubicacion, etc.).

                  # ENTRADA REQUERIDA
                  - "pregunta" (str): la pregunta del usuario, tal cual, en lenguaje natural.
                      """
    # se regresa la respuesta directo al usuario, sin que el agente la reprocese
    # (la cadena de abajo ya redacta la respuesta final por su cuenta)
    return_direct: bool = True
 
    def _run(self, accion):
        # se evalua el string de entrada como diccionario, igual que en las otras dos tools
        accion = ast.literal_eval(accion)
        pregunta = accion.get("pregunta", "")
 
        # ---------------------------------------------------------------
        # PASO 1: cargar y trocear los documentos (PDF + TXT) de la carpeta
        # 'documentos_pdf'. Se vuelve a hacer en cada llamada, sin cache,
        # igual que las otras tools reinstancian el LLM en cada _run().
        # ---------------------------------------------------------------
        loader_pdfs = DirectoryLoader('documentos_pdf', glob='*.pdf', loader_cls=PyPDFLoader)
        loader_txts = DirectoryLoader('documentos_pdf', glob='*.txt', loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documentos = MergedDataLoader(loaders=[loader_pdfs, loader_txts]).load()
 
        # RecursiveCharacterTextSplitter mide por caracteres, no depende de ningun
        # tokenizer local (a diferencia de la version vieja con Ollama/bge-m3)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1250,
            chunk_overlap=150
        )
        fragmentos = splitter.split_documents(documentos)
 
        # ---------------------------------------------------------------
        # PASO 2: embeddings + vector store (FAISS)
        # ---------------------------------------------------------------
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vector_store = FAISS.from_documents(documents=fragmentos, embedding=embeddings)
 
        # prompt anti-alucinacion: solo responde con lo que venga en {contexto}
        prompt = ChatPromptTemplate(
            [("system", """Si la respuesta no esta en el contenido, indica que no tienes esa informacion.
Redacta la respuesta en prosa natural, sin repetir encabezados de seccion, separadores
ni el formato original del documento fuente.
 
#CONTENIDO
{contexto}"""),
             ("human", "{query}")]
        )
 
        # ---------------------------------------------------------------
        # PASO 3: retriever manual. NOTA: no se usa vector_store.as_retriever()
        # porque ese metodo usa embeddings.embed_query() por dentro, y ese
        # metodo tiene un bug conocido en gemini-embedding-001 (error 500).
        # Por eso se vectoriza la pregunta con embed_documents() en vez de
        # embed_query(), y se busca directo con similarity_search_by_vector().
        # Al final se unen los fragmentos encontrados en un solo string,
        # porque el prompt de arriba espera texto plano en {contexto}.
        # ---------------------------------------------------------------
        def buscar_fragmentos(texto_pregunta):
            vector_pregunta = embeddings.embed_documents([texto_pregunta])[0]
            fragmentos_encontrados = vector_store.similarity_search_by_vector(vector_pregunta, k=5)
            return "\n\n".join(fragmento.page_content for fragmento in fragmentos_encontrados)
 
        retriever = RunnableLambda(buscar_fragmentos)
 
        # temperatura baja para respuestas mas concretas y menos creativas
        modelo = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH,
            temperature=0.2
        )
 
        # ---------------------------------------------------------------
        # PASO 4: reescritura de la pregunta antes de buscar en el vector
        # store, para que la busqueda semantica sea mas precisa.
        # ---------------------------------------------------------------
        rewriter_prompt_template = """
Genera la consulta de busqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario,
permitiendo una respuesta mas precisa por medio de la busqueda semantica.
Basta devolver la consulta revisada del Vector DB, entre comillas.
 
# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""
        rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)
        rewriter_chain = rewriter_prompt | modelo | StrOutputParser()
 
        # ---------------------------------------------------------------
        # PASO 5: cadena completa. RunnablePassthrough deja pasar la pregunta
        # tal cual hacia rewriter_chain -> retriever (para armar "contexto"),
        # y tambien tal cual hacia "query" (la pregunta original, sin reescribir,
        # es la que ve el modelo final junto con el contexto ya encontrado).
        # ---------------------------------------------------------------
        rag_chain = (
            {
                "contexto": {"user_question": RunnablePassthrough()} | rewriter_chain | retriever,
                "query": RunnablePassthrough()
            }
            | prompt | modelo | StrOutputParser()
        )
 
        respuesta = rag_chain.invoke(pregunta)
 
        return respuesta


