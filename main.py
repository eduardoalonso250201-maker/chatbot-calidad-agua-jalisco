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

    pregunta = input("Ingresa tu pregunta (escribe 'fin' para terminar): ")
    while pregunta.strip().lower() != "fin":
        respuesta = ejecutor.invoke({"input": pregunta})
        print(respuesta["output"])
        pregunta = input("Ingresa tu pregunta (escribe 'fin' para terminar): ")

if __name__=="__main__":
    main()