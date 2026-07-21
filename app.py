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

def responder_pregunta(pregunta):
    """Ejecuta el AgentExecutor con la pregunta del usuario y devuelve la respuesta final."""
    respuesta = ejecutor.invoke({"input": pregunta})
    return respuesta["output"]

iface = gr.Interface(
    fn=responder_pregunta,
    inputs=gr.Textbox(label="Escribe tu pregunta:"),
    outputs=gr.Markdown(label="Respuesta"),
    title="Chatbot de Rios vivos",
    description="Pregunta sobre conceptos de calidad del agua en Chapala y cuerpos de agua cercanos, consulta datos medidos, definiciones e informacion de contaminantes, sistemas de tratamiento y màs."
)

iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))