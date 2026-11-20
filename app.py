import io
import os
import re
from fpdf import FPDF
from google import genai
from google.genai import types
import streamlit as st

st.set_page_config(
    page_title="Resumen de Sesión · Equipo", layout="centered"
)

st.title("Resumen de Sesión (60–75 min)")
st.caption(
    "Flujo: 1) Resumen IA de bloqueos · 2) Revisión y validación · 3) PDF +"
    " texto de Gmail para mañana"
)

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
  api_key = st.text_input(
      "Introduce tu clave gratuita de Google AI Studio (Gemini API Key)",
      type="password",
  )

SYSTEM_PROMPT_RESUMEN = """
Eres un asistente especializado en redactar documentos resumen para clientes a partir de
la transcripción de una sesión presencial de 60-75 minutos (llegada, trabajo corporal en camilla
y cierre/vestirse) en la que participan 3 personas:
1. Masajista: detecta tensiones, nudos o bloqueos físicos en el cuerpo.
2. Coach clariaudiente: explica el significado, lectura o sentido vital/emocional asociado al bloqueo.
3. Cliente/paciente: expresa lo que siente, si conecta o no con lo dicho, qué recuerda o vive.

NO entregues una transcripción. Genera un DOCUMENTO RESUMEN claro, cálido y práctico donde el eje
sean los BLOQUEOS trabajados en la sesión.

Reglas:
1. Basa todo exclusivamente en la transcripción. No inventes datos ni diagnósticos.
2. Si no queda claro quién dijo algo, usa redacción prudente ("durante el trabajo en...", "se exploró...").
3. Tono cercano, humano, claro y sin jerga fría; respeta palabras clave o metáforas que hayan resonado.
4. Extensión orientativa: 500 a 800 palabras.

Estructura obligatoria:
## 1. Contexto breve de la sesión (llegada, recorrido y cierre)
## 2. Bloqueos encontrados, significado y conexión contigo
(Para cada bloqueo: Bloqueo encontrado · Significado que se le dio · ¿Conectó contigo? y en qué punto quedó)
## 3. Cómo trabajarlo para resolverlo (si no se resolvió en sesión)
## 4. Cómo saber si se resolvió
## 5. Para comentar estos días
"""

SYSTEM_PROMPT_GMAIL = """
Eres un asistente que redacta mensajes cálidos, cercanos y respetuosos para enviar por Gmail al día siguiente
de una sesión presencial.
El mensaje debe:
- Preguntar cómo se ha sentido el cliente hoy tras la sesión de ayer.
- Invitarle a comentar si quiere compartir algo, alguna sensación física, sueño, duda o vivencia.
- Indicar que se adjunta el documento resumen en PDF para que pueda releerlo, recordar lo ocurrido,
  tener su registro y aplicarlo en su día a día (sin presión ni sensación de deberes).
- Mencionar de forma natural y breve (1 o 2 frases) el hilo central o los bloqueos trabajados según
  el resumen validado, sin sustituir al PDF.
- Ser breve, claro y humano (unas 100–160 palabras).
"""


def limpiar_texto_pdf(texto: str) -> str:
  texto = re.sub(r"^#{1,3}\s*", "", texto, flags=re.MULTILINE)
  texto = texto.replace("**", "").replace("*", "")
  return texto.encode("latin-1", errors="replace").decode("latin-1")


def crear_pdf_bytes(
    nombre_cliente: str, fecha_sesion: str, contenido_markdown: str
) -> bytes:
  pdf = FPDF()
  pdf.set_auto_page_break(auto=True, margin=18)
  pdf.add_page()

  pdf.set_font("Helvetica", "B", 16)
  pdf.cell(
      0,
      10,
      limpiar_texto_pdf(f"Resumen de tu sesion - {fecha_sesion}"),
      new_x="LMARGIN",
      new_y="NEXT",
  )

  pdf.set_font("Helvetica", "", 11)
  pdf.cell(
      0,
      8,
      limpiar_texto_pdf(f"Cliente: {nombre_cliente}"),
      new_x="LMARGIN",
      new_y="NEXT",
  )
  pdf.ln(4)

  for linea in contenido_markdown.splitlines():
    linea_limpia = limpiar_texto_pdf(linea).strip()
    if not linea_limpia:
      pdf.ln(3)
      continue
    if linea.lstrip().startswith("## "):
      pdf.ln(3)
      pdf.set_font("Helvetica", "B", 12)
      pdf.multi_cell(0, 7, linea_limpia)
      pdf.ln(1)
    elif linea.lstrip().startswith("- "):
      pdf.set_font("Helvetica", "", 11)
      pdf.multi_cell(0, 6.5, f"  - {linea_limpia[2:]}")
    else:
      pdf.set_font("Helvetica", "", 11)
      pdf.multi_cell(0, 6.5, linea_limpia)

  salida = pdf.output()
  return bytes(salida) if isinstance(salida, (bytearray, bytes)) else salida


col1, col2 = st.columns(2)
with col1:
  nombre_cliente = st.text_input("Nombre o código del cliente", value="Cliente")
with col2:
  fecha_sesion = st.text_input("Fecha de la sesión", value="2026-11-20")

correo_cliente = st.text_input(
    "Correo del cliente (para el envío de mañana)", value=""
)

transcripcion = st.text_area(
    "Pega aquí la transcripción bruta de la sesión (60–75 min)",
    height=180,
    placeholder="Solo sirve de entrada para generar el resumen...",
)

if "resumen" not in st.session_state:
  st.session_state.resumen = ""
if "texto_gmail" not in st.session_state:
  st.session_state.texto_gmail = ""
if "pdf_bytes" not in st.session_state:
  st.session_state.pdf_bytes = None

if st.button("1. Generar documento resumen con IA", type="primary"):
  if not api_key:
    st.error("Falta la clave de API.")
  elif not transcripcion.strip():
    st.warning("Pega primero la transcripción de la sesión.")
  else:
    with st.spinner("Generando resumen de bloqueos, significado y resolución..."):
      client = genai.Client(api_key=api_key)
      prompt = (
          f"Cliente: {nombre_cliente}\n"
          f"Fecha de sesión: {fecha_sesion}\n\n"
          f"TRANSCRIPCIÓN DE ENTRADA:\n{transcripcion}"
      )
      resp = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=prompt,
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_PROMPT_RESUMEN,
              temperature=0.3,
          ),
      )
      st.session_state.resumen = resp.text
      st.session_state.texto_gmail = ""
      st.session_state.pdf_bytes = None

st.divider()
st.subheader("2. Revisión de coherencia y corrección de las terapeutas")

resumen_corregido = st.text_area(
    "Texto final del resumen (aplica aquí tu revisión de coherencia y las correcciones de las terapeutas):",
    value=st.session_state.resumen,
    height=320,
)
st.session_state.resumen = resumen_corregido

coherencia_ok = st.checkbox("He revisado la coherencia general del documento")
terapeutas_ok = st.checkbox(
    "Las terapeutas han corregido y validado el contenido"
)

st.divider()
st.subheader("3. Generar PDF adjunto y texto de Gmail para mañana")

if st.button("Generar PDF y texto de Gmail desde el resumen corregido"):
  if not resumen_corregido.strip():
    st.warning("No hay contenido en el resumen para generar el PDF o el correo.")
  elif not (coherencia_ok and terapeutas_ok):
    st.warning(
        "Marca primero las dos casillas de revisión de coherencia y validación"
        " de las terapeutas."
    )
  elif not api_key:
    st.error("Falta la clave de API para redactar el mensaje de Gmail.")
  else:
    with st.spinner("Creando PDF y redactando el mensaje de Gmail..."):
      st.session_state.pdf_bytes = crear_pdf_bytes(
          nombre_cliente=nombre_cliente,
          fecha_sesion=fecha_sesion,
          contenido_markdown=resumen_corregido,
      )

      client = genai.Client(api_key=api_key)
      prompt_gmail = (
          f"Nombre del cliente: {nombre_cliente}\n"
          f"Fecha de la sesión: {fecha_sesion}\n"
          f"Resumen validado:\n{resumen_corregido}"
      )
      resp_gmail = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=prompt_gmail,
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_PROMPT_GMAIL,
              temperature=0.4,
          ),
      )
      st.session_state.texto_gmail = resp_gmail.text

if st.session_state.pdf_bytes:
  nombre_archivo_pdf = f"Resumen_sesion_{nombre_cliente}_{fecha_sesion}.pdf".replace(
      " ", "_"
  )
  st.download_button(
      label="Descargar PDF para adjuntar en Gmail",
      data=st.session_state.pdf_bytes,
      file_name=nombre_archivo_pdf,
      mime="application/pdf",
  )

if st.session_state.texto_gmail:
  st.markdown("**Texto sugerido para Gmail (envío al día siguiente):**")
  bloque_correo = (
      f"Para: {correo_cliente if correo_cliente else '[correo del cliente]'}\n"
      f"Asunto: ¿Cómo te has sentido hoy? · Resumen de tu sesión ({fecha_sesion})\n\n"
      f"{st.session_state.texto_gmail}"
  )
  st.code(bloque_correo, language="text")
