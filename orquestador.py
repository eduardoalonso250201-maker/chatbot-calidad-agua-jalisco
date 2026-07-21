from parametros_calidad import HerramientaParametrosCalidad

tool = HerramientaParametrosCalidad()
respuesta = tool.run('{"pregunta": "listame los resultados mas altos de DQO para juanacatlan, las pintas y el salto"}')
print(respuesta)