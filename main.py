from langchain_classic.agents import AgentExecutor
from orquestador import AgenteOrquestador

def main():
    # se instancia el orquestador (arma el LLM, las tools y el agente ReAct)
    agente = AgenteOrquestador()

    # el AgentExecutor es el que realmente ejecuta el ciclo de razonamiento
    # (Thought -> Action -> Observation) usando el agente y las tools del orquestador
    ejecutor = AgentExecutor(
        agent=agente.agente,
        tools=agente.tools,
        verbose=True,               # muestra en la terminal cada paso del razonamiento
        handle_parsing_errors=True  # evita que el programa truene si el LLM responde en un formato invalido
    )

    # aqui se va acumulando toda la conversacion de esta sesion (memoria del chat).
    # empieza vacio porque al inicio no hay historial que mostrarle al agente
    historial = ""

    pregunta = input("Ingresa tu pregunta (escribe 'fin' para terminar): ")
    while pregunta.strip().lower() != "fin":
        # se manda la pregunta actual junto con todo el historial acumulado hasta ahora,
        # para que el agente pueda entender preguntas que dependen del contexto previo
        # (ej. "estos componentes", refiriendose a algo mencionado en una respuesta anterior)
        respuesta = ejecutor.invoke({"input": pregunta, "chat_history": historial})
        print(respuesta["output"])

        # se agrega la pregunta y la respuesta de este turno al historial,
        # para que la siguiente pregunta ya la incluya como contexto
        historial += f"Usuario: {pregunta}\nAsistente: {respuesta['output']}\n"

        pregunta = input("Ingresa tu pregunta (escribe 'fin' para terminar): ")

if __name__=="__main__":
    main()