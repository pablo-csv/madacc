# -----------
# DEMO MADACC
# -----------

# -----------  añadido -----------
import nest_asyncio
nest_asyncio.apply()
# --------------------------------

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium import Popup, IFrame

import os
import time
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import Point
import matplotlib.pyplot as plt           # para los gráficos

from google import genai          # ← credenciales ya configuradas
from google.genai import types


# credenciales GCP
# 1. Tomar el contenido del secret (el JSON como string)
service_account_info = st.secrets["gcp"]["service_account"]

# 2. Guardarlo en un archivo temporal (p.e., "gcp_credentials.json")
with open("gcp_credentials.json", "w") as f:
    f.write(service_account_info)

# 3. Ajustar la variable de entorno para que las librerías de Google lo detecten
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"


# ---------- utilidades ------------------------------------------------------

@st.cache_data
def load_data():
    return (
        pd.read_csv("df_accs.csv"),
        gpd.read_file("zonas_cluster_siniestralidad.geojson"),
    )

@st.cache_resource
def get_genai_client():
    import nest_asyncio, asyncio
    nest_asyncio.apply()
    # si no hubiera loop, créalo
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return genai.Client(
        vertexai=True,
        project="sonic-dialect-452009-d4",
        location="us-central1"
    )

def calcular_metricas_siniestralidad(x_utm, y_utm, df_accs, zonas):
    punto       = Point(x_utm, y_utm)
    n_clusteres = zonas.geometry.contains(punto).sum()

    dx = df_accs["coordenada_x_utm"] - x_utm
    dy = df_accs["coordenada_y_utm"] - y_utm
    dist = np.sqrt(dx**2 + dy**2)

    mask_100, mask_10 = dist <= 100, dist <= 10
    num_accs_100, num_accs_10 = mask_100.sum(), mask_10.sum()
    imp_100, imp_10 = df_accs.loc[mask_100, "num_implicados"].sum(), df_accs.loc[mask_10, "num_implicados"].sum()
    tipos_100 = (
        df_accs.loc[mask_100]
        .groupby("tipo_accidente")
        .size()
        .to_dict()
    )
    return num_accs_100, num_accs_10, n_clusteres, imp_100, imp_10, tipos_100


def generate_warning(prompt, llm_model="gemini-2.0-flash-001", temperature=0.8):
    client = get_genai_client()
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt)
        ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature = temperature,
        top_p = 0.95,
        max_output_tokens = 8192,
        response_modalities = ["TEXT"],
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
    )

    # ① llamada no-stream
    response = client.models.generate_content(
        model    = llm_model,
        contents = contents,
        config   = generate_content_config,
    )

    # ② extraer el texto completo del primer candidato
    full_text = "".join(
        part.text for part in response.candidates[0].content.parts
    )

    return full_text


# proyección UTM zona 30 (Madrid)
proj_utm = pyproj.Proj(proj="utm", zone=30, ellps="WGS84", south=False)

# ---------- interfaz --------------------------------------------------------

df_accs, zonas = load_data()

st.title("Simulación de rutas seguras en Madrid 🚗")
st.markdown("Esta es una demostración de cómo calcularíamos la peligrosidad y generaríamos avisos durante una ruta real. Haz **dos clics** en el mapa: primero el **origen** y luego el **destino**. Después verás la ruta y los **avisos** en los puntos de siniestralidad.")

# estado de la selección
for k in ("origin", "destination"): st.session_state.setdefault(k, None)

# mapa base
base_map = folium.Map(location=[40.416835, -3.703417], zoom_start=14, control_scale=True)
if st.session_state["origin"]:
    folium.Marker(st.session_state["origin"], tooltip="Origen", icon=folium.Icon(color="green")).add_to(base_map)
if st.session_state["destination"]:
    folium.Marker(st.session_state["destination"], tooltip="Destino", icon=folium.Icon(color="red")).add_to(base_map)

# ---------------------------------------------------------------------------
# 1) MOSTRAR MAPA BASE Y CAPTURAR CLICS ──────────── cuando falta origen/destino
# ---------------------------------------------------------------------------
if not (st.session_state["origin"] and st.session_state["destination"]):

    out = st_folium(base_map, height=600, width=800)

    if out and out["last_clicked"]:
        lat, lon = out["last_clicked"]["lat"], out["last_clicked"]["lng"]
        if st.session_state["origin"] is None:
            st.session_state["origin"] = (lat, lon)
            st.rerun()
        elif st.session_state["destination"] is None:
            st.session_state["destination"] = (lat, lon)
            st.rerun()

# ---------------------------------------------------------------------------
# 2) CALCULAR RUTA Y MOSTRARLA ───────────────────── cuando ya hay origen y destino
# ---------------------------------------------------------------------------
else:
    with st.spinner("Calculando ruta y analizando siniestralidad..."):
        origin, dest = st.session_state["origin"], st.session_state["destination"]

        # bbox alrededor de la ruta
        lat_min, lat_max = min(origin[0], dest[0]) - .01, max(origin[0], dest[0]) + .01
        lon_min, lon_max = min(origin[1], dest[1]) - .01, max(origin[1], dest[1]) + .01

        G = ox.graph_from_bbox([lon_min, lat_min, lon_max, lat_max], network_type="drive")
        n_o = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
        n_d = ox.distance.nearest_nodes(G, X=dest[1], Y=dest[0])
        ruta = nx.shortest_path(G, n_o, n_d, weight="length")
        puntos_ruta = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in ruta]

        # nuevo mapa con ruta
        route_map = folium.Map(location=origin, zoom_start=14, control_scale=True)
        folium.PolyLine(puntos_ruta, color="blue", weight=5).add_to(route_map)
        folium.Marker(origin, tooltip="Origen", icon=folium.Icon(color="green")).add_to(route_map)
        folium.Marker(dest, tooltip="Destino", icon=folium.Icon(color="red")).add_to(route_map)

        # evaluar cada punto
        info = []
        for lat, lon in puntos_ruta:
            x, y = proj_utm(lon, lat)
            info.append(dict(zip(
                ("num_accs_100","num_accs_10","n_clusteres","imp_100","imp_10","tipos_100"),
                calcular_metricas_siniestralidad(x, y, df_accs, zonas)
            ), lat=lat, lon=lon))

        # filtrar y ordenar por riesgo
        peligrosos = [p for p in info if p["n_clusteres"] > 0]
        peligrosos.sort(key=lambda p: (-p["n_clusteres"], -p["num_accs_10"], -p["num_accs_100"], -p["imp_10"], -p["imp_100"]))
        top = peligrosos[:3]   # puede cambiarse en función de la demo

        # avisos Gemini
        for p in top:
            prompt = f"""
            Eres un asistente de navegación urbano para conductores de coche.

            El conductor se encuentra ahora mismo en una ubicación:
            1. Donde hay de media {p['num_accs_100']/4:.2f} accidentes anuales en 100 m a la redonda (media general 0.875). Esto indica la siniestralidad en la zona.
            2. Donde hay de media {p['num_accs_10']/4:.2f} accidentes anuales en 10 m a la redonda (media general 0.009). Esto indica la siniestralidad en el lugar concreto.
            3. Que está incluida en {p['n_clusteres']} clústeres de siniestralidad (media general 0.093).
            4. Donde hay de media {p['imp_100']/4:.2f} personas implicadas al año en accidentes en 100 m (media general 2.044).
            5. Donde hay de media {p['imp_10']/4:.2f} personas implicadas al año en accidentes en 10 m (media general 0.021).
            6. Donde los tipos de accidente son los siguientes: {p['tipos_100']}

            Genera un aviso corto, conciso, preciso y convincente para que el conductor adecúe su conducción.
            Adáptalo en función de los datos proporcionados y del significado de cada cifra.
            No des múltiples opciones. Genera el mensaje definitivo que se va a mostrar al conductor. No uses saltos de línea.
            """
            aviso = generate_warning(prompt=prompt, llm_model='gemini-2.0-flash-001', temperature=0.8)
            #aviso = 'sin gemini'

            folium.Marker(
                location=(p["lat"], p["lon"]),
                popup=Popup(aviso),
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")
            ).add_to(route_map)

    st.success("Ruta generada.")
    time.sleep(0.1)
    st_folium(route_map, height=600, width=800, returned_objects=[])   # el returned_objects=[] es ESENCIAL para que el mapa no se siga actualizando

    # ───────────────────────────────────────────────
    # BOTONES: «Nueva ruta» y «Generar estadísticas»
    # ───────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns([1, 1], gap="small")
    
    with col_btn1:
        if st.button("🔄 Nueva ruta"):
            st.session_state["origin"] = st.session_state["destination"] = None
            st.rerun()
    
    with col_btn2:
        generate_stats = st.button("📊 Generar estadísticas")
    
    # ───────────────────────────────────────────────
    # ESTADÍSTICAS (fuera de las columnas ⇒ ancho completo)
    # ───────────────────────────────────────────────
    if generate_stats:
        # índice de puntos en la ruta
        d_acum = list(range(len(puntos_ruta)))
    
        # pesos
        w_cl, w_100, w_10 = 0, 1, 2   # los pesos pueden cambiarse
        
        # extraer variables por punto
        pel_clus  = np.array([p["n_clusteres"]   for p in info])
        pel_100   = np.array([10_000 * (p["num_accs_100"]/(3.1415*(100**2)))  for p in info])
        pel_10    = np.array([10_000 * (p["num_accs_10"]/(3.1415*(10**2)))   for p in info])
        
        # score por punto
        score_punto = w_cl * pel_clus + w_100 * pel_100 + w_10 * pel_10
        
        # score global como media
        score = score_punto.mean()
    
        # ── panel de resultados ─────────────────────
        st.subheader("📈 Estadísticas de siniestralidad")
        st.markdown(f"**Índice global de peligrosidad de la ruta:** `{score:0.1f}` (más alto ⇒ más peligrosa. Media general: 270.0)")
    
        # línea de peligrosidad
        df_line = pd.DataFrame({
            "Punto de ruta": d_acum,
            #"Peligrosidad combinada": score_punto,
            "Siniestralidad (100 m alrededor)":   pel_100,
            "Siniestralidad (10 m alrededor)":    pel_10,
        }).set_index("Punto de ruta")
        
        st.line_chart(df_line)
    
        # gráfico circular con tipos de accidente
        tipos_tot = {}
        for p in peligrosos:  # solo puntos marcados con aviso
            for k, v in p["tipos_100"].items():
                tipos_tot[k] = tipos_tot.get(k, 0) + v
        
        if tipos_tot:
            total_accidentes = sum(tipos_tot.values())
        
            tipos_agrupados = {}
            otros = 0
        
            for tipo, count in tipos_tot.items():
                porcentaje = 100 * count / total_accidentes
                if porcentaje > 5:
                    tipos_agrupados[tipo] = count
                else:
                    otros += count
        
            if otros > 0:
                tipos_agrupados["Otros"] = otros
        
            # gráfico
            fig, ax = plt.subplots()
            ax.pie(
                tipos_agrupados.values(),
                labels=tipos_agrupados.keys(),
                autopct="%1.0f%%",
                startangle=90
            )
            ax.set_title("Distribución de tipos de accidente")
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No hay datos suficientes de tipos de accidente para esta ruta.")


