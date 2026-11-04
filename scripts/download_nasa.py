"""
scripts/download_nasa.py
========================
Script modular para la descarga, agregación fenológica y cacheo local
de series agroclimáticas diarias de la API de NASA POWER para los departamentos agrícolas de Argentina.

Parámetros descargados:
- T2M, T2M_MAX, T2M_MIN, T2MDEW: Temperatura media, máxima, mínima y punto de rocío
- PRECTOTCORR: Precipitación diaria corregida
- GWETTOP, GWETROOT: Humedad del suelo superficial y a nivel de raíces (0-1)
- RH2M: Humedad relativa
- ALLSKY_SFC_SW_DWN: Radiación solar incidente
- WS2M: Velocidad del viento a 2 metros
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

NASA_PARAMS = [
    'T2M', 'T2M_MAX', 'T2M_MIN', 'T2MDEW', 'PRECTOTCORR',
    'GWETTOP', 'GWETROOT', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M'
]
NASA_DAILY_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'

def descargar_clima_depto_nasa(id_prov: int, id_depto: int, lat: float, lon: float, out_dir: Path, start_year: int = 1981, end_year: int = 2025, max_retries: int = 3):
    """
    Descarga y cachea la serie diaria histórica de NASA POWER para un centroide departamental.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"prov_{int(id_prov):02d}_depto_{int(id_depto):03d}.parquet"
    
    if out_file.exists():
        return out_file

    params = {
        'parameters': ','.join(NASA_PARAMS),
        'community': 'AG',
        'longitude': lon,
        'latitude': lat,
        'start': f"{start_year}0101",
        'end': f"{end_year}1231",
        'format': 'JSON'
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(NASA_DAILY_URL, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()['properties']['parameter']
                df = pd.DataFrame(data)
                df.index = pd.to_datetime(df.index, format='%Y%m%d')
                df.index.name = 'fecha'
                df['id_provincia'] = int(id_prov)
                df['id_departamento'] = int(id_depto)
                df.to_parquet(out_file)
                time.sleep(0.5) # Respetar rate limiting de NASA
                return out_file
            elif resp.status_code == 429:
                wait_time = (attempt + 1) * 15
                print(f"Rate limited (429). Esperando {wait_time}s...")
                time.sleep(wait_time)
            else:
                time.sleep(3)
        except Exception as e:
            time.sleep(3)
            if attempt == max_retries - 1:
                print(f"Error descargando Depto {id_prov}-{id_depto}: {e}")
                return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Descarga de clima diario NASA POWER para departamentos agrícolas de Argentina.")
    parser.add_argument('--start-year', type=int, default=1981, help="Año de inicio (defecto: 1981)")
    parser.add_argument('--end-year', type=int, default=2025, help="Año de fin (defecto: 2025)")
    parser.add_argument('--cache-dir', type=str, default='data/external/nasa_power_daily', help="Directorio de cache parquet")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    print(f"Iniciando descarga NASA POWER ({args.start_year} - {args.end_year}) en {cache_path}...")
    
    # Cargar coordenadas departamentales
    est_file = Path('data/processed/df_2026_super_nino.csv')
    if not est_file.exists():
        print(f"Archivo de departamentos {est_file} no encontrado.")
        sys.exit(1)

    df_deptos = pd.read_csv(est_file)[['id_provincia', 'id_departamento']].drop_duplicates().sort_values(['id_provincia', 'id_departamento'])
    print(f"Total departamentos a procesar: {len(df_deptos)}")
    
if __name__ == '__main__':
    main()

