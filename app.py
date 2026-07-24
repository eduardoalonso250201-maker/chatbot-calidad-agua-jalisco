import os
import gradio as gr
from langchain_classic.agents import AgentExecutor
from orquestador import AgenteOrquestador

# se crea el agente y el ejecutor UNA sola vez, no en cada pregunta
agente = AgenteOrquestador()
ejecutor = AgentExecutor(
    agent=agente.agente,
    tools=agente.tools,
    verbose=True,
    handle_parsing_errors=True
)

# se define fuera de la funcion para que persista entre cada pregunta;
# si estuviera dentro de responder_pregunta, se reiniciaria vacio en cada llamada
historial = ""

def responder_pregunta(pregunta):
    """Ejecuta el AgentExecutor con la pregunta del usuario y devuelve la respuesta final."""
    global historial
    respuesta = ejecutor.invoke({"input": pregunta, "chat_history": historial})
    historial += f"Usuario: {pregunta}\nAsistente: {respuesta['output']}\n"
    return respuesta["output"]

iface = gr.Interface(
    fn=responder_pregunta,
    inputs=gr.Textbox(label="Escribe tu pregunta:"),
    outputs=gr.Markdown(label="Respuesta"),
    title="Chatbot de Rios vivos",
    description="""Pregunta sobre: resultados de medición de parámetros de calidad de agua, cuerpos de agua en donde Rios Vivos actua, sistemas de tratamiento de agua realizados, catálogo de parametros de calidad de agua, microcistinas, contextualización de la problemática y màs."""
)

iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))