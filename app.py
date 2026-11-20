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

MODELOS_RESPALDO_POR_DEFECTO = [
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
    "gemini-3.8-flash-lite",
]


def obtener_claves_gemini() -> list[str]:
  claves: list[str] = []

  if "GEMINI_API_KEYS" in st.secrets:
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    if isinstance(raw_keys, (list, tuple)):
      for k in raw_keys:
        limpia = str(k).strip()
        if limpia and limpia not in claves:
          claves.append(limpia)
    elif isinstance(raw_keys, str):
      for linea in raw_keys.splitlines():
        limpia = linea.strip().strip(",").strip('"').strip("'")
        if limpia and limpia not in claves:
          claves.append(limpia)

  for clave_sola in [
      st.secrets.get("GEMINI_API_KEY", ""),
      os.environ.get("GEMINI_API_KEY", ""),
  ]:
    limpia = str(clave_sola).strip()
    if limpia and limpia not in claves:
      claves.append(limpia)

  return claves


@st.cache_data(ttl=600, show_spinner=False)
def listar_modelos_disponibles(primera_clave: str) -> list[str]:
  """Consulta qué modelos acepta la API para esa clave."""
  if not primera_clave:
    return []
  try:
    client = genai.Client(api_key=primera_clave)
    nombres: list[str] = []
    for m in client.models.list():
      nombre = getattr(m, "name", "") or ""
      # Quita el prefijo "models/" si viene así
      nombre_limpio = nombre.replace("models/", "").strip()
      if nombre_limpio and "gemini" in nombre_limpio.lower():
        if nombre_limpio not in nombres:
          nombres.append(nombre_limpio)
    return nombres
  except Exception:
    return []


def extraer_aviso_tiempo(mensaje_error: str) -> str:
  match_seg = re.search(
      r"(\d+(?:\.\d+)?)\s*s(?:ec|econds?|eg|egundos?)",
      mensaje_error,
      re.IGNORECASE,
  )
  if match_seg:
    return f" (tiempo estimado de espera: ~{match_seg.group(1)} s)"
  return (
      " (Google no indica la hora exacta de restablecimiento; revisa el uso en"
      " Gemini)"
  )


def generar_con_respaldo(
    claves: list[str],
    prompt: str,
    modelos_candidatos: list[str],
    system_instruction: str | None = None,
    temperature: float = 0.3,
) -> tuple[str, int, str]:
  """Devuelve (texto_generado, indice_cuenta_usada, modelo_usado)."""
  if not claves:
    raise RuntimeError(
        "No hay claves configuradas. Añade GEMINI_API_KEYS en Secrets."
    )

  config = types.GenerateContentConfig(
      system_instruction=system_instruction,
      temperature=temperature,
  )

  ultimo_error = None
  total_claves = len(claves)

  for idx, clave in enumerate(claves, start=1):
    client = genai.Client(api_key=clave)
    for modelo_actual in modelos_candidatos:
      try:
        resp = client.models.generate_content(
            model=modelo_actual,
            contents=prompt,
            config=config,
        )
        texto = (getattr(resp, "text", None) or "").strip()
        if not texto:
          raise RuntimeError("La respuesta ha llegado vacía.")

        return texto, idx, modelo_actual

      except Exception as e:
        ultimo_error = e
        texto_err = str(e)
        # Si el modelo está saturado (503) o no existe/no está disponible (404), prueba el siguiente modelo
        if (
            "503" in texto_err
            or "UNAVAILABLE" in texto_err
            or "404" in texto_err
            or "NOT_FOUND" in texto_err
        ):
          continue
        # Si hay límite de cuota o problema de la cuenta/clave, pasa a la siguiente cuenta
        break

    aviso_tiempo = extraer_aviso_tiempo(str(ultimo_error))
    if idx < total_claves:
      st.warning(
          f"La cuenta/clave #{idx} de {total_claves} no está disponible ahora"
          f" (límite/tiempo agotado o alta demanda){aviso_tiempo}. Buscando en"
          f" la cuenta #{idx + 1}..."
      )
    else:
      st.warning(
          f"La cuenta/clave #{idx} de {total_claves} tampoco está disponible"
          f" ahora{aviso_tiempo}."
      )

  raise RuntimeError(
      f"Ninguna de las {total_claves} cuentas ni modelos de respaldo ha podido"
      f" completar la petición. Último error: {ultimo_error}"
  )


claves_disponibles = obtener_claves_gemini()
if not claves_disponibles:
  clave_manual = st.text_input(
      "Introduce tu clave gratuita de Google AI Studio (Gemini API Key)",
      type="password",
  )
  if clave_manual.strip():
    claves_disponibles = [clave_manual.strip()]

modelos_api = (
    listar_modelos_disponibles(claves_disponibles[0])
    if claves_disponibles
    else []
)

opciones_selector = ["Automático (probar modelos disponibles)"]
for m in MODELOS_RESPALDO_POR_DEFECTO + modelos_api:
  if m not in opciones_selector:
    opciones_selector.append(m)

modelo_elegido = st.selectbox(
    "Modelo de Gemini a priorizar",
    options=opciones_selector,
    index=0,
)

if modelo_elegido == "Automático (probar modelos disponibles)":
  modelos_a_usar = []
  for m in MODELOS_RESPALDO_POR_DEFECTO + modelos_api:
    if m not in modelos_a_usar:
      modelos_a_usar.append(m)
else:
  modelos_a_usar = [modelo_elegido] + [
      m
      for m in (MODELOS_RESPALDO_POR_DEFECTO + modelos_api)
      if m != modelo_elegido
  ]

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
if "info_generacion" not in st.session_state:
  st.session_state.info_generacion = ""

if st.button("1. Generar documento resumen con IA", type="primary"):
  if not claves_disponibles:
    st.error("Falta la clave de API.")
  elif not transcripcion.strip():
    st.warning("Pega primero la transcripción de la sesión.")
  else:
    with st.spinner("Generando resumen de bloqueos, significado y resolución..."):
      try:
        prompt = (
            f"Cliente: {nombre_cliente}\n"
            f"Fecha de sesión: {fecha_sesion}\n\n"
            f"TRANSCRIPCIÓN DE ENTRADA:\n{transcripcion}"
        )
        texto_resumen, num_cuenta, modelo_usado = generar_con_respaldo(
            claves=claves_disponibles,
            prompt=prompt,
            modelos_candidatos=modelos_a_usar,
            system_instruction=SYSTEM_PROMPT_RESUMEN,
            temperature=0.3,
        )
        st.session_state.resumen = texto_resumen
        st.session_state.info_generacion = (
            f"Generado con la cuenta #{num_cuenta} de"
            f" {len(claves_disponibles)} usando el modelo `{modelo_usado}`."
        )
        st.session_state.texto_gmail = ""
        st.session_state.pdf_bytes = None
      except Exception as err:
        st.error(f"No se ha podido generar el resumen: {err}")

if st.session_state.info_generacion:
  st.info(st.session_state.info_generacion)

st.divider()
st.subheader("2. Revisión de coherencia y corrección de las terapeutas")

resumen_corregido = st.text_area(
    "Texto final del resumen (aplica aquí tu revisión de coherencia y las"
    " correcciones de las terapeutas):",
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
  elif not claves_disponibles:
    st.error("Falta la clave de API para redactar el mensaje de Gmail.")
  else:
    with st.spinner("Creando PDF y redactando el mensaje de Gmail..."):
      st.session_state.pdf_bytes = crear_pdf_bytes(
          nombre_cliente=nombre_cliente,
          fecha_sesion=fecha_sesion,
          contenido_markdown=resumen_corregido,
      )

      try:
        prompt_gmail = (
            f"Nombre del cliente: {nombre_cliente}\n"
            f"Fecha de la sesión: {fecha_sesion}\n"
            f"Resumen validado:\n{resumen_corregido}"
        )
        texto_mail, num_cuenta_mail, modelo_mail = generar_con_respaldo(
            claves=claves_disponibles,
            prompt=prompt_gmail,
            modelos_candidatos=modelos_a_usar,
            system_instruction=SYSTEM_PROMPT_GMAIL,
            temperature=0.4,
        )
        st.session_state.texto_gmail = texto_mail
        st.caption(
            f"Correo redactado con la cuenta #{num_cuenta_mail} y modelo"
            f" `{modelo_mail}`."
        )
      except Exception as err:
        st.error(
            f"Se creó el PDF, pero no se pudo redactar el correo: {err}"
        )

if st.session_state.pdf_bytes:
  nombre_archivo_pdf = (
      f"Resumen_sesion_{nombre_cliente}_{fecha_sesion}.pdf".replace(" ", "_")
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
      f"Asunto: ¿Cómo te has sentido hoy? · Resumen de tu sesión"
      f" ({fecha_sesion})\n\n"
      f"{st.session_state.texto_gmail}"
  )
  st.code(bloque_correo, language="text")
