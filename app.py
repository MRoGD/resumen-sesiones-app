from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import os
import re
from fpdf import FPDF
from google import genai
from google.genai import types
import streamlit as st

st.set_page_config(
    page_title="Resumen de Sesión · Equipo",
    layout="centered",
)

st.title("Resumen de Sesión (60–75 min)")
st.caption(
    "Flujo: 1) Resumen IA de bloqueos · 2) Revisión y validación · 3) PDF +"
    " texto de Gmail para mañana"
)

MODELOS_PRIORITARIOS = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
    "gemini-3.1-pro-preview",
]

EXCLUIR_MODELOS = (
    "embedding",
    "imagen",
    "veo",
    "audio",
    "live",
    "tts",
    "aqa",
    "gemini-1.0",
    "gemini-1.5",
    "gemini-2.0",
    "gemini-2.5",
)

TIMEOUT_POR_INTENTO_SEG = 25
MAX_MODELOS_POR_CUENTA = 5

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


def _agregar_unico(destino: list[str], candidato: str) -> None:
  limpio = str(candidato).replace("models/", "").strip().strip(",").strip('"').strip("'")
  if limpio and limpio not in destino:
    destino.append(limpio)


def obtener_claves_gemini() -> list[str]:
  """Recupera y deduplica las claves API desde st.secrets o variables de entorno."""
  claves: list[str] = []

  if "GEMINI_API_KEYS" in st.secrets:
    raw_keys = st.secrets["GEMINI_API_KEYS"]
    if isinstance(raw_keys, (list, tuple)):
      for k in raw_keys:
        _agregar_unico(claves, k)
    elif isinstance(raw_keys, str):
      for linea in raw_keys.splitlines():
        _agregar_unico(claves, linea)

  for respaldo in (
      st.secrets.get("GEMINI_API_KEY", ""),
      os.environ.get("GEMINI_API_KEY", ""),
  ):
    _agregar_unico(claves, respaldo)

  return claves


def es_modelo_texto_valido(nombre: str) -> bool:
  baja = nombre.lower()
  if "gemini" not in baja:
    return False
  return not any(palabra in baja for palabra in EXCLUIR_MODELOS)


@st.cache_data(ttl=600, show_spinner=False)
def listar_modelos_disponibles(primera_clave: str) -> list[str]:
  """Consulta qué modelos de texto acepta la API para la primera clave válida."""
  if not primera_clave:
    return []
  try:
    client = genai.Client(api_key=primera_clave)
    nombres: list[str] = []
    for m in client.models.list():
      nombre = getattr(m, "name", "") or ""
      limpio = nombre.replace("models/", "").strip()
      if limpio and es_modelo_texto_valido(limpio) and limpio not in nombres:
        nombres.append(limpio)
      if len(nombres) >= MAX_MODELOS_POR_CUENTA:
        break
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


def _llamar_modelo_con_timeout(
    client: genai.Client,
    modelo: str,
    prompt: str,
    config: types.GenerateContentConfig,
    timeout_seg: int = TIMEOUT_POR_INTENTO_SEG,
) -> str:
  def _ejecutar() -> str:
    resp = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config=config,
    )
    return (getattr(resp, "text", None) or "").strip()

  with ThreadPoolExecutor(max_workers=1) as executor:
    futuro = executor.submit(_ejecutar)
    try:
      return futuro.result(timeout=timeout_seg)
    except FuturesTimeoutError as exc:
      raise TimeoutError(
          f"El modelo `{modelo}` tardó más de {timeout_seg} segundos en"
          " responder."
      ) from exc


def generar_con_respaldo(
    claves: list[str],
    prompt: str,
    modelos_candidatos: list[str],
    system_instruction: str | None = None,
    temperature: float = 0.3,
    estado_placeholder=None,
) -> tuple[str, int, str]:
  """Itera por claves y modelos de respaldo devolviendo (texto, num_cuenta, modelo)."""
  if not claves:
    raise RuntimeError(
        "No hay claves configuradas. Añade GEMINI_API_KEYS en Secrets."
    )

  config = types.GenerateContentConfig(
      system_instruction=system_instruction,
      temperature=temperature,
  )

  candidatos: list[str] = []
  for m in modelos_candidatos:
    _agregar_unico(candidatos, m)
    if len(candidatos) >= MAX_MODELOS_POR_CUENTA:
      break

  ultimo_error: Exception | None = None
  total_claves = len(claves)

  for idx, clave in enumerate(claves, start=1):
    client = genai.Client(api_key=clave)
    for modelo_actual in candidatos:
      if estado_placeholder is not None:
        estado_placeholder.caption(
            f"Probando cuenta #{idx}/{total_claves} con `{modelo_actual}`"
            f" (máx. {TIMEOUT_POR_INTENTO_SEG}s)..."
        )
      try:
        texto = _llamar_modelo_con_timeout(
            client=client,
            modelo=modelo_actual,
            prompt=prompt,
            config=config,
            timeout_seg=TIMEOUT_POR_INTENTO_SEG,
        )
        if not texto:
          raise RuntimeError("La respuesta ha llegado vacía.")

        if estado_placeholder is not None:
          estado_placeholder.empty()
        return texto, idx, modelo_actual

      except Exception as e:
        ultimo_error = e
        texto_err = str(e)
        es_fallo_modelo = (
            isinstance(e, TimeoutError)
            or "503" in texto_err
            or "UNAVAILABLE" in texto_err
            or "404" in texto_err
            or "NOT_FOUND" in texto_err
        )
        if es_fallo_modelo:
          continue
        break

    aviso_tiempo = extraer_aviso_tiempo(str(ultimo_error or ""))
    if idx < total_claves:
      st.warning(
          f"La cuenta/clave #{idx} de {total_claves} no está disponible ahora"
          f" (límite/tiempo agotado o demora){aviso_tiempo}. Buscando en la"
          f" cuenta #{idx + 1}..."
      )
    else:
      st.warning(
          f"La cuenta/clave #{idx} de {total_claves} tampoco está disponible"
          f" ahora{aviso_tiempo}."
      )

  if estado_placeholder is not None:
    estado_placeholder.empty()

  raise RuntimeError(
      f"Ninguna de las {total_claves} cuentas ni modelos de respaldo ha podido"
      f" completar la petición. Último error: {ultimo_error}"
  )


def limpiar_texto_pdf(texto: str) -> str:
  """Limpia sintaxis Markdown básica y adapta el texto a Latin-1 para FPDF."""
  texto = re.sub(r"^#{1,4}\s*", "", texto, flags=re.MULTILINE)
  texto = re.sub(r"\*{1,2}([^*\n]+?)\*{1,2}", r"\1", texto)
  texto = texto.replace("**", "").replace("*", "")
  texto = re.sub(r"(\S{45})", r"\1 ", texto)
  return texto.encode("latin-1", errors="replace").decode("latin-1")


def crear_pdf_bytes(
    nombre_cliente: str,
    fecha_sesion: str,
    contenido_markdown: str,
) -> bytes:
  """Genera un PDF en memoria a partir del resumen validado."""
  pdf = FPDF()
  pdf.set_auto_page_break(auto=True, margin=18)
  pdf.add_page()

  ancho_util = max(50.0, pdf.w - pdf.l_margin - pdf.r_margin)

  def escribir_bloque(texto: str, alto: float = 6.5) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(ancho_util, alto, texto)

  pdf.set_font("Helvetica", "B", 16)
  escribir_bloque(
      limpiar_texto_pdf(f"Resumen de tu sesion - {fecha_sesion}"),
      alto=9.0,
  )

  pdf.set_font("Helvetica", "", 11)
  escribir_bloque(
      limpiar_texto_pdf(f"Cliente: {nombre_cliente}"),
      alto=7.5,
  )
  pdf.ln(4)

  for linea in contenido_markdown.splitlines():
    stripped = linea.lstrip()
    linea_limpia = limpiar_texto_pdf(stripped).strip()
    if not linea_limpia:
      pdf.ln(3)
      continue
    if stripped.startswith("## "):
      pdf.ln(3)
      pdf.set_font("Helvetica", "B", 12)
      escribir_bloque(linea_limpia, alto=7.0)
      pdf.ln(1)
    elif stripped.startswith(("- ", "* ")):
      pdf.set_font("Helvetica", "", 11)
      escribir_bloque(f"  - {linea_limpia[2:].lstrip()}", alto=6.5)
    else:
      pdf.set_font("Helvetica", "", 11)
      escribir_bloque(linea_limpia, alto=6.5)

  salida = pdf.output()
  return bytes(salida) if isinstance(salida, (bytearray, bytes)) else salida


def sanitizar_nombre_archivo(nombre: str) -> str:
  limpio = re.sub(r"[^\w\-.]+", "_", nombre.strip())
  return limpio.strip("_") or "Cliente"


# --- Carga de credenciales y selección de modelos ---
claves_disponibles = obtener_claves_gemini()
if not claves_disponibles:
  clave_manual = st.text_input(
      "Introduce tu clave gratuita de Google AI Studio (Gemini API Key)",
      type="password",
  )
  _agregar_unico(claves_disponibles, clave_manual)

modelos_api = (
    listar_modelos_disponibles(claves_disponibles[0])
    if claves_disponibles
    else []
)

opciones_selector = ["Automático (probar modelos disponibles)"]
for m in MODELOS_PRIORITARIOS + modelos_api:
  _agregar_unico(opciones_selector, m)

modelo_elegido = st.selectbox(
    "Modelo de Gemini a priorizar",
    options=opciones_selector,
    index=0,
)

modelos_a_usar: list[str] = []
if modelo_elegido != "Automático (probar modelos disponibles)":
  _agregar_unico(modelos_a_usar, modelo_elegido)
for m in MODELOS_PRIORITARIOS + modelos_api:
  _agregar_unico(modelos_a_usar, m)

# --- Estado de sesión ---
for clave_estado, valor_defecto in (
    ("resumen", ""),
    ("texto_gmail", ""),
    ("pdf_bytes", None),
    ("info_generacion", ""),
    ("info_gmail", ""),
):
  if clave_estado not in st.session_state:
    st.session_state[clave_estado] = valor_defecto

# --- Paso 1: Entrada y generación del resumen ---
col1, col2 = st.columns(2)
with col1:
  nombre_cliente = st.text_input("Nombre o código del cliente", value="Cliente")
with col2:
  fecha_sesion = st.text_input("Fecha de la sesión", value="2026-11-20")

correo_cliente = st.text_input(
    "Correo del cliente (para el envío de mañana)",
    value="",
)

transcripcion = st.text_area(
    "Pega aquí la transcripción bruta de la sesión (60–75 min)",
    height=180,
    placeholder="Solo sirve de entrada para generar el resumen...",
)

estado_paso1 = st.empty()

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
            estado_placeholder=estado_paso1,
        )
        st.session_state.resumen = texto_resumen
        st.session_state.info_generacion = (
            f"Generado con la cuenta #{num_cuenta} de"
            f" {len(claves_disponibles)} usando el modelo `{modelo_usado}`."
        )
        st.session_state.texto_gmail = ""
        st.session_state.info_gmail = ""
        st.session_state.pdf_bytes = None
      except Exception as err:
        st.error(f"No se ha podido generar el resumen: {err}")

if st.session_state.info_generacion:
  st.info(st.session_state.info_generacion)

# --- Paso 2: Revisión y validación ---
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

# --- Paso 3: PDF y texto de Gmail ---
st.divider()
st.subheader("3. Generar PDF adjunto y texto de Gmail para mañana")

estado_paso3 = st.empty()

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
            estado_placeholder=estado_paso3,
        )
        st.session_state.texto_gmail = texto_mail
        st.session_state.info_gmail = (
            f"Correo redactado con la cuenta #{num_cuenta_mail} y modelo"
            f" `{modelo_mail}`."
        )
      except Exception as err:
        st.session_state.info_gmail = ""
        st.error(
            f"Se creó el PDF, pero no se pudo redactar el correo: {err}"
        )

if st.session_state.pdf_bytes:
  cliente_seguro = sanitizar_nombre_archivo(nombre_cliente)
  fecha_segura = sanitizar_nombre_archivo(fecha_sesion)
  nombre_archivo_pdf = f"Resumen_sesion_{cliente_seguro}_{fecha_segura}.pdf"
  st.download_button(
      label="Descargar PDF para adjuntar en Gmail",
      data=st.session_state.pdf_bytes,
      file_name=nombre_archivo_pdf,
      mime="application/pdf",
  )

if st.session_state.texto_gmail:
  if st.session_state.info_gmail:
    st.caption(st.session_state.info_gmail)
  st.markdown("**Texto sugerido para Gmail (envío al día siguiente):**")
  bloque_correo = (
      f"Para: {correo_cliente if correo_cliente else '[correo del cliente]'}\n"
      f"Asunto: ¿Cómo te has sentido hoy? · Resumen de tu sesión"
      f" ({fecha_sesion})\n\n"
      f"{st.session_state.texto_gmail}"
  )
  st.code(bloque_correo, language="text")
