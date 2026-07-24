# Integration with Google's Gemini chat models for LLM inference
from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH
# API key for accessing the Gemini service
from my_keys import GEMINI_API_KEY
from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, Tool
from RAG import HerramientaRAG
from parametros_calidad import HerramientaParametrosCalidad


set_debug(False)

class AgenteOrquestador:
    def __init__(self):
        #Primer elemento de la clase, inicializa con Google Generative AI model con su API key y el nombre del modelo 
        self.llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH
        )

        # Ahora se usan las dos tools del proyecto de calidad del agua
        herramienta_rag = HerramientaRAG()
        herramienta_parametros_calidad = HerramientaParametrosCalidad()

        #Segundo elemento de clase, Lista de herramientas que el agente puede usar, con etiquetas y funciones definidas
        self.tools = [
            Tool(
                name=herramienta_rag.name,
                func=herramienta_rag.run,
                description=herramienta_rag.description,
                return_direct=herramienta_rag.return_direct
            ),
            Tool(
                name=herramienta_parametros_calidad.name,
                func=herramienta_parametros_calidad.run,
                description=herramienta_parametros_calidad.description,
                return_direct=herramienta_parametros_calidad.return_direct
            )
        ]

        # Tercer elemento de la calse, el promt. se le dice que herramientas tiene disponibles
        #que pasos seguir, cuanto tomar la decision de que herramienta usar
        #y se anade la parte del historial que se define en main
        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Historial de la conversacion hasta ahora:
            {chat_history}

            Question: {input}
            Thought:{agent_scratchpad}"""
        )

        # Se crea el agente ReAct con sus argmuentos de llm, herramientas y prompt
        self.agente = create_react_agent(self.llm, self.tools, prompt)