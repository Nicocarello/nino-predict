import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium

print("1. Cargando datos de simulacion 2026/27 con cuantiles...")
df_sim = pd.read_csv('data/processed/df_2026_super_nino.csv')

# Mapear Chascomus
df_sim['cde'] = df_sim.apply(lambda r: f"{int(r['id_provincia']):02d}{int(r['id_departamento']):03d}", axis=1)
df_sim.loc[df_sim['cde'] == '06217', 'cde'] = '06218'

# Resumen agregado por departamento
depto_summary = df_sim.groupby(['cde', 'id_provincia', 'provincia', 'id_departamento', 'departamento'], as_index=False).agg(
    sup_sembrada_total_ha=('sup_sembrada_ha', 'sum'),
    sup_cosechada_total_ha=('pred_sup_cosechada_ha', 'sum'),
    sup_perdida_total_ha=('pred_sup_perdida_ha', 'sum'),
    sup_perdida_p10_total_ha=('pred_sup_perdida_p10_ha', 'sum'),
    sup_perdida_p90_total_ha=('pred_sup_perdida_p90_ha', 'sum'),
    produccion_total_tn=('pred_produccion_tn', 'sum')
)
depto_summary['tasa_perdida_total'] = np.where(
    depto_summary['sup_sembrada_total_ha'] > 0,
    (depto_summary['sup_perdida_total_ha'] / depto_summary['sup_sembrada_total_ha']) * 100.0,
    0.0
)

# Metricas por cultivo
for crop in ['Maíz', 'Soja', 'Trigo', 'Girasol']:
    crop_col = crop.lower().replace('í', 'i')
    cdf = df_sim[df_sim['cultivo_grupo'] == crop][['cde', 'sup_sembrada_ha', 'pred_sup_cosechada_ha', 'pred_sup_perdida_ha', 'pred_tasa_perdida', 'pred_rendimiento_kg_ha']].copy()
    cdf = cdf.rename(columns={
        'sup_sembrada_ha': f'sembrada_{crop_col}_ha',
        'pred_sup_cosechada_ha': f'cosechada_{crop_col}_ha',
        'pred_sup_perdida_ha': f'perdida_{crop_col}_ha',
        'pred_tasa_perdida': f'tasa_perdida_{crop_col}',
        'pred_rendimiento_kg_ha': f'rinde_{crop_col}_kg_ha'
    })
    cdf[f'tasa_perdida_{crop_col}'] = cdf[f'tasa_perdida_{crop_col}'] * 100.0
    depto_summary = depto_summary.merge(cdf, on='cde', how='left')

for crop in ['maiz', 'soja', 'trigo', 'girasol']:
    depto_summary[f'sembrada_{crop}_ha'] = depto_summary[f'sembrada_{crop}_ha'].fillna(0)
    depto_summary[f'perdida_{crop}_ha'] = depto_summary[f'perdida_{crop}_ha'].fillna(0)
    depto_summary[f'tasa_perdida_{crop}'] = depto_summary[f'tasa_perdida_{crop}'].fillna(0)
    depto_summary[f'rinde_{crop}_kg_ha'] = depto_summary[f'rinde_{crop}_kg_ha'].fillna(0)

def get_tasa_alert(tasa):
    if tasa >= 35: return '🔴 CRÍTICO (>35% de las hectáreas)'
    elif tasa >= 20: return '🟠 ALTO (20% - 35% de las hectáreas)'
    elif tasa >= 10: return '🟡 MODERADO (10% - 20% de las hectáreas)'
    elif tasa > 0: return '🟢 BAJO (<10% de las hectáreas)'
    return '⚪ Sin actividad agrícola'

def get_ha_impact(ha):
    if ha >= 30000: return '🌊 MUY SEVERO (>30.000 Ha)'
    elif ha >= 15000: return '🌊 ALTO (15.000 - 30.000 Ha)'
    elif ha >= 5000: return '🌊 MODERADO (5.000 - 15.000 Ha)'
    elif ha > 0: return 'LEVE (<5.000 Ha)'
    return 'Sin anegamiento'

depto_summary['alerta_riesgo'] = depto_summary['tasa_perdida_total'].apply(get_tasa_alert)
depto_summary['impacto_ha'] = depto_summary['sup_perdida_total_ha'].apply(get_ha_impact)

depto_summary['f_sembrada'] = depto_summary['sup_sembrada_total_ha'].apply(lambda x: f"{x:,.0f} ha" if x > 0 else "0 ha")
depto_summary['f_perdida'] = depto_summary.apply(
    lambda r: f"{r['sup_perdida_total_ha']:,.0f} ha [{r['sup_perdida_p10_total_ha']:,.0f} a {r['sup_perdida_p90_total_ha']:,.0f} ha al 80% CI]" if r['sup_perdida_total_ha'] > 0 else "0 ha", axis=1
)
depto_summary['f_tasa'] = depto_summary['tasa_perdida_total'].apply(lambda x: f"{x:.1f}% de las ha sembradas")
depto_summary['f_prod'] = depto_summary['produccion_total_tn'].apply(lambda x: f"{x/1000:,.1f} kTn" if x > 0 else "-")

depto_summary['f_maiz'] = depto_summary.apply(
    lambda r: f"{r['sembrada_maiz_ha']:,.0f} ha | Pérd: {r['tasa_perdida_maiz']:.1f}% ({r['perdida_maiz_ha']:,.0f} ha) | Rinde: {r['rinde_maiz_kg_ha']:,.0f} kg/ha" if r['sembrada_maiz_ha'] > 0 else "No sembrado", axis=1
)
depto_summary['f_soja'] = depto_summary.apply(
    lambda r: f"{r['sembrada_soja_ha']:,.0f} ha | Pérd: {r['tasa_perdida_soja']:.1f}% ({r['perdida_soja_ha']:,.0f} ha) | Rinde: {r['rinde_soja_kg_ha']:,.0f} kg/ha" if r['sembrada_soja_ha'] > 0 else "No sembrado", axis=1
)
depto_summary['f_trigo'] = depto_summary.apply(
    lambda r: f"{r['sembrada_trigo_ha']:,.0f} ha | Pérd: {r['tasa_perdida_trigo']:.1f}% ({r['perdida_trigo_ha']:,.0f} ha) | Rinde: {r['rinde_trigo_kg_ha']:,.0f} kg/ha" if r['sembrada_trigo_ha'] > 0 else "No sembrado", axis=1
)
depto_summary['f_girasol'] = depto_summary.apply(
    lambda r: f"{r['sembrada_girasol_ha']:,.0f} ha | Pérd: {r['tasa_perdida_girasol']:.1f}% ({r['perdida_girasol_ha']:,.0f} ha) | Rinde: {r['rinde_girasol_kg_ha']:,.0f} kg/ha" if r['sembrada_girasol_ha'] > 0 else "No sembrado", axis=1
)

# 2. Geometrias INDEC WFS
print("2. Procesando GeoJSON INDEC...")
gdf = gpd.read_file('data/external/indec_departamentos.geojson')
if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs('EPSG:4326')

gdf = gdf[~gdf['nam'].astype(str).str.contains('Antártida|Islas del Atlántico Sur', case=False, na=False)].copy()
gdf = gdf[['cde', 'nam', 'jur', 'geometry']].copy()
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.005, preserve_topology=True)

gdf_merged = gdf.merge(depto_summary, on='cde', how='left')
gdf_merged['has_data'] = ~gdf_merged['sup_sembrada_total_ha'].isna() & (gdf_merged['sup_sembrada_total_ha'] > 0)
gdf_merged['tasa_perdida_total'] = gdf_merged['tasa_perdida_total'].fillna(0.0)
gdf_merged['sup_perdida_total_ha'] = gdf_merged['sup_perdida_total_ha'].fillna(0.0)

for c in ['maiz', 'soja', 'trigo', 'girasol']:
    gdf_merged[f'tasa_perdida_{c}'] = gdf_merged[f'tasa_perdida_{c}'].fillna(0.0)
    gdf_merged[f'sembrada_{c}_ha'] = gdf_merged[f'sembrada_{c}_ha'].fillna(0.0)

gdf_merged['alerta_riesgo'] = gdf_merged['alerta_riesgo'].fillna('⚪ Sin actividad agrícola')
gdf_merged['f_sembrada'] = gdf_merged['f_sembrada'].fillna('Sin registro')
gdf_merged['f_perdida'] = gdf_merged['f_perdida'].fillna('0 ha')
gdf_merged['f_tasa'] = gdf_merged['f_tasa'].fillna('0.0%')
gdf_merged['f_prod'] = gdf_merged['f_prod'].fillna('-')
gdf_merged['f_maiz'] = gdf_merged['f_maiz'].fillna('No sembrado')
gdf_merged['f_soja'] = gdf_merged['f_soja'].fillna('No sembrado')
gdf_merged['f_trigo'] = gdf_merged['f_trigo'].fillna('No sembrado')
gdf_merged['f_girasol'] = gdf_merged['f_girasol'].fillna('No sembrado')

cols_to_keep = [
    'cde', 'nam', 'jur', 'geometry', 'has_data',
    'tasa_perdida_total', 'sup_perdida_total_ha',
    'tasa_perdida_maiz', 'sembrada_maiz_ha',
    'tasa_perdida_soja', 'sembrada_soja_ha',
    'tasa_perdida_trigo', 'sembrada_trigo_ha',
    'tasa_perdida_girasol', 'sembrada_girasol_ha',
    'alerta_riesgo', 'f_sembrada', 'f_perdida', 'f_tasa', 'f_prod',
    'f_maiz', 'f_soja', 'f_trigo', 'f_girasol'
]
gdf_final = gdf_merged[cols_to_keep].copy()

# 3. Mapa Folium con ÚNICAMENTE MODO OSCURO (sin selector de capas base)
print("3. Construyendo mapa Leaflet exclusivamente en Modo Oscuro...")
m = folium.Map(
    location=[-34.5, -63.5],
    zoom_start=6,
    tiles=None,
    control_scale=True,
    prefer_canvas=True
)

# ÚNICA capa base: CartoDB Dark Matter fija (control=False elimina los radio buttons del menú)
folium.TileLayer(
    'CartoDB dark_matter',
    name='Modo Oscuro',
    control=False
).add_to(m)

def create_tooltip():
    return folium.GeoJsonTooltip(
        fields=['nam', 'jur', 'alerta_riesgo', 'f_tasa', 'f_perdida', 'f_sembrada'],
        aliases=['📍 Departamento:', 'Provincia:', 'Riesgo:', '⚠️ % Ha Perdidas:', '🌊 Sup. Perdida (p50 [p10-p90]):', '🌱 Sup. Sembrada:'],
        localize=False,
        sticky=True,
        style="""
            background-color: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 11px;
            padding: 10px 14px;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            line-height: 1.5;
        """
    )

def create_popup():
    return folium.GeoJsonPopup(
        fields=['nam', 'jur', 'alerta_riesgo', 'f_tasa', 'f_perdida', 'f_sembrada', 'f_prod', 'f_maiz', 'f_soja', 'f_trigo', 'f_girasol'],
        aliases=[
            '📍 Departamento:', 
            'Provincia:', 
            'Riesgo Departamental (%):', 
            '⚠️ % HECTÁREAS PERDIDAS:', 
            '🌊 Anegamiento Total [p10-p90]:', 
            '🌱 Superficie Sembrada:', 
            '🌾 Producción Estimada:', 
            '🌽 Maíz:', 
            '🌱 Soja:', 
            '🌾 Trigo:',
            '🌻 Girasol:'
        ],
        localize=False,
        labels=True,
        style="""
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 12px;
            min-width: 360px;
            line-height: 1.6;
        """
    )

# --- ESTILOS COROPLÉTICOS OPTIMIZADOS PARA FONDO OSCURO ---

def style_global_loss(f):
    props = f['properties']
    if not props.get('has_data', False):
        return {'fillColor': '#1e293b', 'color': '#334155', 'weight': 0.4, 'fillOpacity': 0.25}
    t = props.get('tasa_perdida_total', 0.0)
    if t >= 35: c = '#991b1b' # Rojo oscuro intenso
    elif t >= 20: c = '#ea580c' # Naranja brillante
    elif t >= 10: c = '#f59e0b' # Ambar
    elif t > 0: c = '#10b981' # Esmeralda brillante
    else: c = '#475569'
    return {'fillColor': c, 'color': '#0f172a', 'weight': 0.7, 'fillOpacity': 0.82}

def style_hectares(f):
    props = f['properties']
    if not props.get('has_data', False):
        return {'fillColor': '#1e293b', 'color': '#334155', 'weight': 0.4, 'fillOpacity': 0.25}
    ha = props.get('sup_perdida_total_ha', 0.0)
    if ha >= 30000: c = '#881337' # Borravino brillante
    elif ha >= 15000: c = '#be123c' # Carmín
    elif ha >= 8000: c = '#e11d48' # Rosa fuerte
    elif ha >= 3000: c = '#fb7185' # Coral
    elif ha > 0: c = '#fed7aa' # Durazno suave
    else: c = '#1e293b'
    return {'fillColor': c, 'color': '#0f172a', 'weight': 0.7, 'fillOpacity': 0.82}

def style_maiz(f):
    props = f['properties']
    if not props.get('has_data', False) or props.get('sembrada_maiz_ha', 0) == 0:
        return {'fillColor': '#1e293b', 'color': '#334155', 'weight': 0.3, 'fillOpacity': 0.15}
    t = props.get('tasa_perdida_maiz', 0.0)
    if t >= 40: c = '#9333ea' # Violeta brillante (destaca en oscuro)
    elif t >= 25: c = '#dc2626' # Rojo
    elif t >= 15: c = '#f97316' # Naranja
    elif t >= 8: c = '#facc15' # Amarillo
    else: c = '#22c55e' # Verde
    return {'fillColor': c, 'color': '#0f172a', 'weight': 0.7, 'fillOpacity': 0.85}

def style_soja(f):
    props = f['properties']
    if not props.get('has_data', False) or props.get('sembrada_soja_ha', 0) == 0:
        return {'fillColor': '#1e293b', 'color': '#334155', 'weight': 0.3, 'fillOpacity': 0.15}
    t = props.get('tasa_perdida_soja', 0.0)
    if t >= 8.0: c = '#dc2626'
    elif t >= 6.0: c = '#ea580c'
    elif t >= 4.0: c = '#f59e0b'
    elif t >= 2.5: c = '#84cc16'
    else: c = '#10b981'
    return {'fillColor': c, 'color': '#0f172a', 'weight': 0.7, 'fillOpacity': 0.85}

def style_trigo(f):
    props = f['properties']
    if not props.get('has_data', False) or props.get('sembrada_trigo_ha', 0) == 0:
        return {'fillColor': '#1e293b', 'color': '#334155', 'weight': 0.3, 'fillOpacity': 0.15}
    t = props.get('tasa_perdida_trigo', 0.0)
    if t >= 10.0: c = '#b45309'
    elif t >= 7.0: c = '#ea580c'
    elif t >= 4.5: c = '#f59e0b'
    elif t >= 2.5: c = '#84cc16'
    else: c = '#10b981'
    return {'fillColor': c, 'color': '#0f172a', 'weight': 0.7, 'fillOpacity': 0.85}

def style_girasol(f):
    props = f['properties']
    if not props.get('has_data', False) or props.get('sembrada_girasol_ha', 0) == 0:
        return {'fillColor': '#1e293b', 'color': '#334155', 'weight': 0.3, 'fillOpacity': 0.15}
    t = props.get('tasa_perdida_girasol', 0.0)
    if t >= 5.0: c = '#dc2626'
    elif t >= 3.0: c = '#ea580c'
    elif t >= 1.5: c = '#facc15'
    else: c = '#22c55e'
    return {'fillColor': c, 'color': '#0f172a', 'weight': 0.7, 'fillOpacity': 0.85}

# --- CAPAS TEMÁTICAS ---

layer1 = folium.GeoJson(
    gdf_final,
    name='⚠️ Tasa Pérdida Global (%) [Activa]',
    style_function=style_global_loss,
    highlight_function=lambda x: {'weight': 2.5, 'color': '#38bdf8', 'fillOpacity': 0.95},
    tooltip=create_tooltip(),
    popup=create_popup()
)
layer1.add_to(m)

layer2 = folium.GeoJson(
    gdf_final,
    name='🌊 Superficie Perdida Total (Ha)',
    style_function=style_hectares,
    highlight_function=lambda x: {'weight': 2.5, 'color': '#38bdf8', 'fillOpacity': 0.95},
    tooltip=create_tooltip(),
    popup=create_popup(),
    show=False
)
layer2.add_to(m)

layer3 = folium.GeoJson(
    gdf_final,
    name='🌽 Pérdida Específica Maíz (%)',
    style_function=style_maiz,
    highlight_function=lambda x: {'weight': 2.5, 'color': '#38bdf8', 'fillOpacity': 0.95},
    tooltip=create_tooltip(),
    popup=create_popup(),
    show=False
)
layer3.add_to(m)

layer4 = folium.GeoJson(
    gdf_final,
    name='🌱 Pérdida Específica Soja (%)',
    style_function=style_soja,
    highlight_function=lambda x: {'weight': 2.5, 'color': '#38bdf8', 'fillOpacity': 0.95},
    tooltip=create_tooltip(),
    popup=create_popup(),
    show=False
)
layer4.add_to(m)

layer5 = folium.GeoJson(
    gdf_final,
    name='🌾 Pérdida Específica Trigo (%)',
    style_function=style_trigo,
    highlight_function=lambda x: {'weight': 2.5, 'color': '#38bdf8', 'fillOpacity': 0.95},
    tooltip=create_tooltip(),
    popup=create_popup(),
    show=False
)
layer5.add_to(m)

layer6 = folium.GeoJson(
    gdf_final,
    name='🌻 Pérdida Específica Girasol (%)',
    style_function=style_girasol,
    highlight_function=lambda x: {'weight': 2.5, 'color': '#38bdf8', 'fillOpacity': 0.95},
    tooltip=create_tooltip(),
    popup=create_popup(),
    show=False
)
layer6.add_to(m)

# Control de Capas (como base_layers tiene control=False, solo mostrará las 6 capas temáticas)
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Panel y Leyenda
panel_html = """
<div style="
    position: fixed; 
    top: 20px; left: 60px; max-width: 380px; 
    background-color: rgba(15, 23, 42, 0.94);
    backdrop-filter: blur(12px);
    z-index: 9999; 
    border-radius: 12px;
    padding: 16px 18px;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 11px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.15);
">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
        <span style="font-weight: 700; font-size: 13px; color: #38bdf8;">SIMULACIÓN SÚPER EL NIÑO 2026/27</span>
        <span style="background: #dc2626; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 9px;">+2.7°C ONI</span>
    </div>
    <div style="font-size: 10.5px; color: #94a3b8; margin-bottom: 10px; line-height: 1.4;">
        <strong>¿Qué es la Tasa de Pérdida?</strong> Es el <u>% de hectáreas sembradas que quedan bajo agua</u> y no llegan a cosecharse (NO mide el rinde).
    </div>

    <div style="font-weight: 600; margin-bottom: 6px; color: #cbd5e1;">Código de Colores:</div>
    <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #9333ea; border-radius: 3px; box-shadow: 0 0 8px rgba(147, 51, 234, 0.6);"></span>
            <span><strong>Violeta:</strong> &gt; 40% Pérdida Extrema (Común en Maíz)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #991b1b; border-radius: 3px;"></span>
            <span><strong>Rojo Oscuro:</strong> 25% - 40% (Alerta Crítica / Emergencia)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #ea580c; border-radius: 3px;"></span>
            <span><strong>Naranja:</strong> 20% - 25% (Riesgo Alto)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #f59e0b; border-radius: 3px;"></span>
            <span><strong>Amarillo:</strong> 10% - 20% (Riesgo Moderado)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #10b981; border-radius: 3px;"></span>
            <span><strong>Verde:</strong> &lt; 10% (Bajo Impacto Relativo)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #1e293b; border: 1px solid #334155; border-radius: 3px;"></span>
            <span style="color: #64748b;"><strong>Gris Oscuro:</strong> Sin ese cultivo sembrado</span>
        </div>
    </div>
    
    <div style="font-size: 10px; color: #38bdf8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px; line-height: 1.4;">
        💡 <strong>Intervalos al 80% CI:</strong> Los popups incluyen la banda de incertidumbre [p10 a p90] para hectáreas bajo anegamiento.
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(panel_html))

# Estilo CSS 100% Dark Mode para Popups y LayerControl
custom_dark_css = """
<style>
/* Menú de Capas en Modo Oscuro */
.leaflet-control-layers {
    background: rgba(15, 23, 42, 0.94) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6) !important;
    padding: 10px 14px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    font-size: 11px !important;
}
.leaflet-control-layers-base {
    display: none !important; /* Asegurar remoción completa */
}
.leaflet-control-layers-separator {
    display: none !important;
}
.leaflet-control-layers label {
    color: #f8fafc !important;
    margin-bottom: 4px !important;
}
.leaflet-control-layers-overlays label span {
    color: #e2e8f0 !important;
}

/* Popups en Modo Oscuro Elegante */
.leaflet-popup-content-wrapper {
    background: #0f172a !important;
    border-radius: 12px !important;
    box-shadow: 0 25px 30px -5px rgba(0, 0, 0, 0.7) !important;
    padding: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    color: #f8fafc !important;
}
.leaflet-popup-content {
    margin: 12px 14px !important;
    line-height: 1.5 !important;
}
.leaflet-popup-content table {
    width: 100% !important;
    border-collapse: collapse !important;
}
.leaflet-popup-content tr {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}
.leaflet-popup-content tr:last-child {
    border-bottom: none !important;
}
.leaflet-popup-content th, .leaflet-popup-content td {
    padding: 4px 6px !important;
    font-size: 11px !important;
    text-align: left !important;
}
.leaflet-popup-content th {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    width: 42% !important;
}
.leaflet-popup-content td {
    color: #f8fafc !important;
    font-weight: 500 !important;
}
/* Destacar tasa de perdida en rojo neon */
.leaflet-popup-content tr:nth-child(4) td {
    color: #f87171 !important;
    font-weight: 800 !important;
    font-size: 13px !important;
}
.leaflet-popup-tip {
    background: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
}
</style>
"""
m.get_root().html.add_child(folium.Element(custom_dark_css))

out_path_nb = 'notebooks/mapa_inundaciones_2026.html'
m.save(out_path_nb)
sz_mb = os.path.getsize(out_path_nb) / (1024 * 1024)
print(f"Mapa guardado en {out_path_nb} ({sz_mb:.2f} MB)")

artifact_path = r'C:\Users\nico_\.gemini\antigravity\brain\2d03620f-8152-4c6c-974b-cbd13d67b3ee\mapa_inundaciones_super_nino.html'
m.save(artifact_path)
print(f"Mapa guardado en artefacto {artifact_path}")
