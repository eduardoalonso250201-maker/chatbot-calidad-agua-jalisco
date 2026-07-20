#Ahora se arma la parte que traduce la pregunta del usuario a un filtro que pandas pueda usar
#mismo patron que el rewriter de RAG.py: un modelo Pydantic define la "forma" de la respuesta
#y JsonOutputParser obliga al LLM a responder en ese formato
import pandas as pd
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv
import re

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

df_contaminantes = pd.read_csv("resultado.csv")

#se limpian espacios invisibles (caracteres \xa0) que trae la columna de codigos de localidad
df_contaminantes["Códigos de Localidad"] = df_contaminantes["Códigos de Localidad"].str.strip()

#se limpian los nombres de columnas: algunas traen varios espacios seguidos por error
#(ej. "DQO    (mg/L)" en vez de "DQO (mg/L)"), lo cual hacia que el LLM nunca coincidiera exacto
df_contaminantes.columns = [" ".join(col.split()) for col in df_contaminantes.columns]

#estas son las columnas de parametros que existen en la tabla (las sacamos del print de columns de arriba)
#se le pasan a la IA para que sepa exactamente como se llama cada una, con espacios y todo
columnas_parametros = [
    "Temperatura (°C)", "pH", "Oxigeno Disuelto (mg/L)", "Conductividad elecrica (mS/cm)",
    "Solidos Disueltos Totales SDT (mg/L)", "Nitratos (mg/L)", "Nitritos (mg/L)",
    "Fosfato (mg/L)", "Fósforo (mg/L)", "Dureza Total (mg/L)", "Cloro Total (mg/L)",
    "Cloro Libre (mg/L)", "Coeficiente de Absorscion Espectral", "Turbidez (UNT)",
    "DQO (mg/L)", "DBO5 (mg/L)"
]

#diccionario que traduce el codigo de localidad de la base de datos a su nombre completo
#esto se hace desde el codigo, no se modifica la base de datos
NOMBRES_LOCALIDADES = {
    "CASB": "CASA BLANCA",
    "OJOA": "OJO DE AGUA",
    "JUA": "JUANACATLÁN",
    "CAN": "LA CAÑADA",
    "PIN": "LAS PINTAS",
    "SAL": "EL SALTO"
}

#se define la "forma" que debe tener el filtro que regrese la IA
class FiltroConsulta(BaseModel):
    parametro: str = Field(default="", description="Nombre EXACTO de la columna de parametro consultado, tomado de esta lista: " + ", ".join(columnas_parametros) + ". Vacio si la pregunta no menciona ningun parametro.")
    sitio: str = Field(default="", description=(
        "Codigo EXACTO de localidad (nunca el nombre completo), tomado de esta lista: "
        + ", ".join(NOMBRES_LOCALIDADES.keys())
        + ". Si el usuario menciona el nombre completo del lugar en vez del codigo, tradúcelo "
        "al codigo correspondiente usando esta relacion: "
        + "; ".join(f"{codigo} = {nombre}" for codigo, nombre in NOMBRES_LOCALIDADES.items())
        + ". Vacio si no se menciona ningun sitio."
    ))
    fecha_inicio: str = Field(default="", description="Fecha inicial del rango preguntado, formato YYYY-MM-DD. Vacio si no aplica.")
    fecha_fin: str = Field(default="", description="Fecha final del rango preguntado, formato YYYY-MM-DD. Vacio si no aplica.")
    operacion: str = Field(default="listar", description="Una de estas cuatro palabras exactas: maximo, minimo, promedio, listar")

parser_filtro = JsonOutputParser(pydantic_object=FiltroConsulta)

modelo = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.2
)

template_filtro = PromptTemplate(
    template="""
Extrae de la siguiente pregunta del usuario los datos necesarios para filtrar una tabla
de calidad del agua. Si algun dato no se menciona explicitamente, dejalo vacio (o "listar"
para operacion, que es el valor por defecto).

PREGUNTA: {pregunta}

{formato_salida}
""",
    input_variables=["pregunta"],
    partial_variables={"formato_salida": parser_filtro.get_format_instructions()}
)

cadena_filtro = template_filtro | modelo | parser_filtro

#primera prueba con una de tus preguntas de ejemplo
pregunta_prueba = "Cual fue la fecha de valor mas alto de DQO en Juanacatlán"
filtro = cadena_filtro.invoke({"pregunta": pregunta_prueba})
print(filtro)

#funcion que saca la unidad de un nombre de columna, ej. "DQO (mg/L)" -> "mg/L"
#si la columna no tiene parentesis (como pH), regresa vacio
def extraer_unidad(nombre_columna):
    match = re.search(r"\(([^)]+)\)", nombre_columna)
    return match.group(1) if match else ""

#Ahora se aplica el filtro que extrajo la IA sobre la tabla real con pandas

#primero convertimos la columna de fecha a formato fecha de verdad
#(al leer el CSV, pandas la trae como texto, no como fecha)
df_contaminantes["Fecha de Muestreo"] = pd.to_datetime(df_contaminantes["Fecha de Muestreo"])

#se parte de una copia completa de la tabla y se le van aplicando los filtros que vengan llenos
df_filtrado = df_contaminantes.copy()

if filtro["sitio"]:
    df_filtrado = df_filtrado[df_filtrado["Códigos de Localidad"] == filtro["sitio"]]

if filtro["fecha_inicio"]:
    df_filtrado = df_filtrado[df_filtrado["Fecha de Muestreo"] >= filtro["fecha_inicio"]]

if filtro["fecha_fin"]:
    df_filtrado = df_filtrado[df_filtrado["Fecha de Muestreo"] <= filtro["fecha_fin"]]

#ahora, segun la operacion que pidio el usuario, se saca el resultado
parametro = filtro["parametro"]
operacion = filtro["operacion"]

if operacion == "maximo" and parametro:
    #idxmax() da el indice de la fila donde esta el valor mas alto, no solo el valor
    fila = df_filtrado.loc[df_filtrado[parametro].idxmax()]
    resultado = f"{fila[parametro]} {extraer_unidad(parametro)} (fecha: {fila['Fecha de Muestreo'].date()}, sitio: {fila['Códigos de Localidad']})".strip()
elif operacion == "minimo" and parametro:
    fila = df_filtrado.loc[df_filtrado[parametro].idxmin()]
    resultado = f"{fila[parametro]} {extraer_unidad(parametro)} (fecha: {fila['Fecha de Muestreo'].date()}, sitio: {fila['Códigos de Localidad']})".strip()
elif operacion == "promedio" and parametro:
    valor = df_filtrado[parametro].mean()
    resultado = f"{valor} {extraer_unidad(parametro)}".strip()
else:
    #operacion "listar" (o si no vino parametro): se regresa la tabla filtrada completa
    resultado = df_filtrado

print(resultado)