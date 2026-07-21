from RAG import HerramientaRAG

tool = HerramientaRAG()
respuesta = tool.run('{"pregunta": "Dime la definicion de Cadmio, sus fuentes de contmainacion principales, efectos para la salud y limites maximos permisibles de concentracion"}')
print(respuesta)