from langchain_classic.agents import AgentExecutor
from orquestador import AgenteOrquestador

def main():
    agente = AgenteOrquestador()
    ejecutor = AgentExecutor(
        agent=agente.agente,
        tools=agente.tools,
        verbose=True,
        handle_parsing_errors=True
    )

    pregunta = "¿Dame los resultados de todos los parametros del dia 29 de marzo del 2025 en JUA?"

    respuesta = ejecutor.invoke({"input": pregunta})
    print(respuesta["output"])

if __name__=="__main__":
    main()