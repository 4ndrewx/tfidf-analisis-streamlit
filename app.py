import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Búsqueda y Análisis TF-IDF", page_icon="🔍", layout="wide"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    .hero-banner {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%);
    }
    .result-box {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        color: #F8FAFC;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- CABECERA PRINCIPAL ---
st.markdown(
    """
    <div class="hero-banner">
        <h1 style="margin:0; font-size: 2.2rem;">🔍 Búsqueda y Análisis TF-IDF</h1>
        <p style="margin-top: 8px; font-size: 1rem; opacity: 0.9;">Procesamiento de Lenguaje Natural en Español utilizando Matriz TF-IDF y Similitud de Coseno</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Corpus predeterminado con ejemplos nuevos (Tecnología, IA y Sistemas)
corpus_por_defecto = (
    "Las redes neuronales artificiales simulan el aprendizaje del cerebro"
    " humano.\nEl procesamiento de imágenes facilita la visión artificial en"
    " tiempo real.\nLos protocolos MQTT optimizan la transmisión de datos en"
    " dispositivos IoT.\nEl desarrollo en Streamlit permite crear interfaces"
    " web interactivas.\nEl reconocimiento óptico de caracteres extrae texto"
    " legible desde fotografías.\nLos modelos de lenguaje natural analizan"
    " sentimientos y contextos léxicos."
)

# Consultas sugeridas para los botones de ejemplo
ejemplos_consultas = [
    "¿Cómo funcionan las redes neuronales?",
    "¿Para qué sirven los protocolos MQTT?",
    "¿Qué herramientas permiten crear interfaces web?",
    "¿Cómo procesa imágenes la visión artificial?",
]

# --- DISEÑO DE DOS COLUMNAS ---
col1, col2 = st.columns([1.3, 1])

with col1:
  st.markdown("### 📚 Base de Documentos (Corpus)")
  corpus_input = st.text_area(
      "Edita o añade documentos (uno por línea):",
      value=corpus_por_defecto,
      height=180,
  )

  st.markdown("### 🔎 Consulta de Búsqueda")
  # Estado de sesión para manejar los botones de ejemplo
  if "query_val" not in st.session_state:
    st.session_state.query_val = "¿Cómo funcionan las redes neuronales?"

  query_input = st.text_input(
      "Escribe tu pregunta o término de búsqueda:",
      value=st.session_state.query_val,
  )

  analizar_btn = st.button("📊 Analizar y Buscar")

with col2:
  st.markdown("### 💡 Consultas de Ejemplo")
  st.write("Haz clic en cualquier ejemplo para autocompletar la búsqueda:")

  for ej in ejemplos_consultas:
    if st.button(f"📌 {ej}", key=ej):
      st.session_state.query_val = ej
      st.rerun()

# --- PROCESAMIENTO TF-IDF Y SIMILITUD DE COSENO ---
st.markdown("---")
st.markdown("### 📊 Resultado del Análisis")

if analizar_btn or query_input:
  # Separar el corpus en documentos individuales por línea
  documentos = [
      doc.strip() for doc in corpus_input.split("\n") if doc.strip()
  ]

  if not documentos:
    st.warning("El corpus de documentos está vacío.")
  else:
    # Inicializar Vectorizador TF-IDF
    vectorizer = TfidfVectorizer(stop_words=None)  # En español básico

    try:
      # Ajustar y transformar documentos y consulta
      tfidf_matrix = vectorizer.fit_transform(documentos)
      query_vec = vectorizer.transform([query_input])

      # Calcular similitud de coseno
      cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

      # Obtener el mejor resultado
      best_idx = np.argmax(cosine_similarities)
      best_score = cosine_similarities[best_idx]

      if best_score > 0.0:
        mejor_documento = documentos[best_idx]

        st.markdown(
            f"""
            <div class="result-box">
                <p style="color: #10B981; font-weight: bold; margin-bottom: 5px;">✅ Respuesta Encontrada (Coincidencia Óptima)</p>
                <h3 style="margin: 0; color: white;">"{mejor_documento}"</h3>
                <p style="margin-top: 10px; font-size: 0.85rem; color: #94A3B8;">Puntuación de Similitud Coseno: <b>{best_score:.4f}</b></p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Mostrar matriz de similitudes para transparencia analítica
        with st.expander("Ver detalles de similitud con todos los documentos"):
          df_resultados = pd.DataFrame(
              {"Documento": documentos, "Similitud": cosine_similarities}
          )
          df_resultados = df_resultados.sort_values(
              by="Similitud", ascending=False
          )
          st.dataframe(df_resultados, use_container_width=True)

      else:
        st.warning(
            "No se encontró una coincidencia relevante en el corpus para la"
            " consulta ingresada."
        )

    except Exception as e:
      st.error(
          f"Ocurrió un error al procesar el modelo TF-IDF (asegúrate de que"
          f" haya términos válidos): {e}"
      )
