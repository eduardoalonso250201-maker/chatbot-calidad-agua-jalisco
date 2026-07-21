from langchain.tools import BaseTool
import pandas as pd
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_FLASH
from typing import List
import re
import ast
 
pd.set_option('display.max_columns', None)   # no ocultar columnas con "..."
pd.set_option('display.width', None)          # no cortar el ancho, usar el ancho real del contenido
 
 
class HerramientaParametrosCalidad(BaseTool):
    # mismo patron de name/description/return_direct que las otras tools del proyecto
    name: str = "HerramientaParametrosCalidad"
    description: str = """
                      Utiliza esta herramienta siempre que se pida un dato medido de la tabla de
                      calidad del agua: valores numericos de parametros (ej. DQO, nitratos, pH,
                      cloro, turbidez, etc.), filtrados por sitio y/o fecha, ya sea el maximo,
                      minimo, promedio o el listado completo de resultados.
                      No uses esta herramienta para preguntas conceptuales o explicativas
                      (para eso esta HerramientaRAG).
 
                      # ENTRADA REQUERIDA
                      - "pregunta" (str): la pregunta del usuario, tal cual, en lenguaje natural.
                      """
    # se regresa la respuesta directo al usuario, sin que el agente la reprocese.
    # esto es clave aqui: si el agente reformulara la respuesta con su propio LLM,
    # habria riesgo de que redondee o cambie el numero exacto que ya calculo pandas.
    return_direct: bool = True
 
    def _run(self, accion):
        # se evalua el string de entrada como diccionario, igual que en las otras tools
        accion = ast.literal_eval(accion)
        pregunta = accion.get("pregunta", "")
 
        # -----------------------------------------------------------
        # PASO 1: cargar el CSV y limpiarlo. Se hace en cada llamada (sin cache),
        # igual que las otras tools no cachean nada entre llamadas.
        # -----------------------------------------------------------
        df_contaminantes = pd.read_csv("resultado.csv")
 
        # la columna de codigo de sitio trae caracteres invisibles (\xa0) en varias filas,
        # eso hacia que comparaciones como == "JUA" fallaran en silencio; con .str.strip() se quita
        df_contaminantes["Códigos de Localidad"] = df_contaminantes["Códigos de Localidad"].str.strip()
 
        # algunas columnas del CSV traen espacios dobles por error (ej. "DQO    (mg/L)"),
        # se normalizan a un solo espacio para que coincidan exacto con columnas_parametros
        df_contaminantes.columns = [" ".join(col.split()) for col in df_contaminantes.columns]
 
        # nombres EXACTOS de columna tal como quedan despues de la limpieza de arriba;
        # se le pasan al LLM en el prompt para que sepa como se llama cada parametro
        columnas_parametros = [
            "Temperatura (°C)", "pH", "Oxigeno Disuelto (mg/L)", "Conductividad elecrica (mS/cm)",
            "Solidos Disueltos Totales SDT (mg/L)", "Nitratos (mg/L)", "Nitritos (mg/L)",
            "Fosfato (mg/L)", "Fósforo (mg/L)", "Dureza Total (mg/L)", "Cloro Total (mg/L)",
            "Cloro Libre (mg/L)", "Coeficiente de Absorscion Espectral", "Turbidez (UNT)",
            "DQO (mg/L)", "DBO5 (mg/L)"
        ]
 
        # traduccion codigo de sitio -> nombre completo, para que el usuario pueda preguntar
        # con el nombre real de la localidad y no solo con el codigo interno de la base
        NOMBRES_LOCALIDADES = {
            "CASB": "CASA BLANCA",
            "OJOA": "OJO DE AGUA",
            "JUA": "JUANACATLÁN",
            "CAN": "LA CAÑADA",
            "PIN": "LAS PINTAS",
            "SAL": "EL SALTO"
        }
 
        # -----------------------------------------------------------
        # PASO 2: pedirle al LLM que traduzca la pregunta libre a un filtro
        # estructurado. FiltroConsulta define la "forma" exacta que debe tener
        # esa respuesta, y JsonOutputParser obliga al LLM a devolver ese formato
        # (mismo patron Pydantic + JsonOutputParser usado en el script de prueba).
        # -----------------------------------------------------------
        class FiltroConsulta(BaseModel):
            # si la pregunta no menciona parametro, queda vacio; si menciona uno que
            # no esta en la tabla (arsenico, coliformes...) se marca NO_DISPONIBLE
            # para poder avisarle al usuario en vez de tronar o listar todo sin explicar
            parametro: str = Field(default="", description="Nombre EXACTO de la columna de parametro consultado, tomado de esta lista: " + ", ".join(columnas_parametros) + ". Vacio si la pregunta no menciona ningun parametro. Si la pregunta menciona un parametro que NO esta en esa lista (ejemplo: arsenico, coliformes), escribe exactamente NO_DISPONIBLE.")
            # lista (no un solo string) para poder soportar preguntas que piden
            # varios sitios a la vez, ej. "juanacatlan, las pintas y el salto"
            sitios: List[str] = Field(default_factory=list, description=(
                "Lista de codigos EXACTOS de localidad mencionados (puede ser uno, varios o ninguno). "
                "Codigos validos: " + ", ".join(NOMBRES_LOCALIDADES.keys())
                + ". Si el usuario menciona nombres completos en vez de codigos, tradúcelos usando esta relacion: "
                + "; ".join(f"{codigo} = {nombre}" for codigo, nombre in NOMBRES_LOCALIDADES.items())
                + ". Lista vacia si no se menciona ningun sitio."
            ))
            fecha_inicio: str = Field(default="", description="Fecha inicial del rango preguntado, formato YYYY-MM-DD. Vacio si no aplica.")
            fecha_fin: str = Field(default="", description="Fecha final del rango preguntado, formato YYYY-MM-DD. Vacio si no aplica.")
            # el default "listar" cubre tanto "dame todos los resultados de X"
            # como cualquier pregunta donde el LLM no identifique una operacion clara
            operacion: str = Field(default="listar", description="Una de estas cuatro palabras exactas: maximo, minimo, promedio, listar")
 
        parser_filtro = JsonOutputParser(pydantic_object=FiltroConsulta)
 
        modelo = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH,
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
        filtro = cadena_filtro.invoke({"pregunta": pregunta})
 
        # -----------------------------------------------------------
        # PASO 3: aplicar el filtro que extrajo el LLM sobre la tabla real con pandas.
        # Aqui ya no hay LLM de por medio: los numeros que salgan son los reales
        # de la tabla, filtrados con pandas puro (nada de que la IA "invente" datos).
        # -----------------------------------------------------------
 
        # saca la unidad de un nombre de columna, ej. "DQO (mg/L)" -> "mg/L"
        # (si la columna no tiene parentesis, como pH, regresa vacio)
        def extraer_unidad(nombre_columna):
            match = re.search(r"\(([^)]+)\)", nombre_columna)
            return match.group(1) if match else ""
 
        # el CSV trae la fecha como texto; se convierte a datetime real para
        # poder comparar (>=, <=) contra fecha_inicio/fecha_fin
        df_contaminantes["Fecha de Muestreo"] = pd.to_datetime(df_contaminantes["Fecha de Muestreo"])
        df_filtrado = df_contaminantes.copy()
 
        # se aplican los filtros uno por uno, solo si el LLM los lleno
        if filtro["sitios"]:
            df_filtrado = df_filtrado[df_filtrado["Códigos de Localidad"].isin(filtro["sitios"])]
        if filtro["fecha_inicio"]:
            df_filtrado = df_filtrado[df_filtrado["Fecha de Muestreo"] >= filtro["fecha_inicio"]]
        if filtro["fecha_fin"]:
            df_filtrado = df_filtrado[df_filtrado["Fecha de Muestreo"] <= filtro["fecha_fin"]]
 
        parametro = filtro["parametro"]
        operacion = filtro["operacion"]
 
        # ---- guards primero, para no tronar en idxmax/idxmin/mean mas abajo ----
        if df_filtrado.empty:
            # el filtro de sitio/fecha no dejo ninguna fila
            resultado = "No encontre resultados para esa combinacion de sitio y fecha. Intenta con otro sitio o rango de fechas."
        elif parametro and parametro not in df_contaminantes.columns:
            # el LLM regreso NO_DISPONIBLE o cualquier nombre que no es columna real
            resultado = f"El parametro '{parametro}' no esta en la tabla. Los parametros disponibles son: {', '.join(columnas_parametros)}"
 
        # ---- maximo/minimo: con sitios, uno por cada sitio (groupby + idxmax/idxmin);
        # sin sitios, un solo maximo/minimo global de toda la tabla ya filtrada ----
        elif operacion == "maximo" and parametro:
            if filtro["sitios"]:
                indices_max = df_filtrado.groupby("Códigos de Localidad")[parametro].idxmax()
                filas = df_filtrado.loc[indices_max]
                resultado = "\n".join(
                    f"{fila['Códigos de Localidad']}: {fila[parametro]} {extraer_unidad(parametro)} (fecha: {fila['Fecha de Muestreo'].date()})"
                    for _, fila in filas.iterrows()
                )
            else:
                fila = df_filtrado.loc[df_filtrado[parametro].idxmax()]
                resultado = f"{fila[parametro]} {extraer_unidad(parametro)} (fecha: {fila['Fecha de Muestreo'].date()}, sitio: {fila['Códigos de Localidad']})".strip()
        elif operacion == "minimo" and parametro:
            if filtro["sitios"]:
                indices_min = df_filtrado.groupby("Códigos de Localidad")[parametro].idxmin()
                filas = df_filtrado.loc[indices_min]
                resultado = "\n".join(
                    f"{fila['Códigos de Localidad']}: {fila[parametro]} {extraer_unidad(parametro)} (fecha: {fila['Fecha de Muestreo'].date()})"
                    for _, fila in filas.iterrows()
                )
            else:
                fila = df_filtrado.loc[df_filtrado[parametro].idxmin()]
                resultado = f"{fila[parametro]} {extraer_unidad(parametro)} (fecha: {fila['Fecha de Muestreo'].date()}, sitio: {fila['Códigos de Localidad']})".strip()
        elif operacion == "promedio" and parametro:
            valor = df_filtrado[parametro].mean()
            resultado = f"{valor} {extraer_unidad(parametro)}".strip()
 
        # ---- "listar" (o si no vino parametro): se regresa la tabla filtrada completa ----
        else:
            resultado = df_filtrado
 
        # -----------------------------------------------------------
        # PASO 4: siempre regresar un string. El script de prueba usaba print(),
        # pero una tool debe devolver el resultado con return para que el agente
        # (o el usuario, con return_direct=True) lo reciba. La tabla se transpone
        # (.T) para que quede legible: cada muestra como columna, cada parametro
        # como fila, en vez de 26 columnas amontonadas.
        # -----------------------------------------------------------
        if isinstance(resultado, pd.DataFrame):
            return resultado.set_index("ID").T.to_string()
        return resultado