import os
from pathlib import Path

import pandas as pd
import numpy as np
from functools import reduce

import matplotlib.pyplot as plt
import seaborn as sns


def ensure_outputs_dir(base_dir: Path) -> Path:
    out = base_dir / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_stations(base_dir: Path) -> pd.DataFrame:
    stations_path = base_dir / "estaciones.csv"
    if stations_path.exists():
        stations_df = pd.read_csv(stations_path)
        return stations_df
    else:
        return pd.DataFrame()


def load_variable_csv(path: Path, variable_name: str) -> pd.DataFrame:
    """
    Convierte un CSV por variable a formato largo:
    date | station | variable
    Detecta la columna de fecha si no se llama 'date'.
    """
    df = pd.read_csv(path)

    # Detectar columna de fecha
    candidate_names = {"date", "fecha", "fechahora", "datetime", "time"}
    date_col = None
    for col in df.columns:
        if str(col).strip().lower() in candidate_names:
            date_col = col
            break

    # Si no se detecta, usar la primera columna como fecha
    if date_col is None:
        date_col = df.columns[0]

    # Convertir a datetime y renombrar a 'date'
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.rename(columns={date_col: "date"}, inplace=True)

    df_long = df.melt(
        id_vars="date",
        var_name="station",
        value_name=variable_name,
    )

    # Asegurar unicidad por (date, station) agregando si hay duplicados
    df_long = (
        df_long.groupby(["date", "station"], as_index=False)[variable_name]
        .mean()
    )

    return df_long


def visualize_missingness(df: pd.DataFrame, save_path: Path) -> None:
    try:
        import missingno as msno
    except ImportError:
        # Fallback simple heatmap if missingno no está disponible
        plt.figure(figsize=(15, 5))
        sns.heatmap(df.isna(), cbar=False)
        plt.title("Missing Data Heatmap (fallback)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    plt.figure(figsize=(15, 5))
    msno.matrix(df)
    plt.title("Missing Data Across Variables and Stations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def impute_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """Imputación por estación usando SAITS si está disponible; si no, KNNImputer."""
    num_cols = df.select_dtypes(include=np.number).columns

    # Intentar importar SAITS
    use_saits = False
    try:
        from saits.imputation import SAITS  # type: ignore
        use_saits = True
    except Exception:
        use_saits = False

    # Fallback KNNImputer
    from sklearn.impute import KNNImputer

    dfs_imputed = []
    for station in df["station"].dropna().unique():
        temp = df[df["station"] == station].copy()

        if len(temp) == 0:
            continue

        # Separar columnas sin datos para evitar discrepancias de forma
        cols_with_data = [c for c in num_cols if temp[c].notna().any()]
        cols_all_na = [c for c in num_cols if c not in cols_with_data]

        if use_saits:
            try:
                # SAITS tipicamente espera (n_muestras, n_tiempo, n_features); aquí usamos 2D.
                # Algunos wrappers aceptan 2D y asumen n_steps como filas.
                model = SAITS(n_steps=temp[cols_with_data].shape[0])
                imputed = model.fit_transform(temp[cols_with_data].values)

                # Convertir a DataFrame con mismas columnas si devuelve ndarray
                if isinstance(imputed, np.ndarray):
                    temp[cols_with_data] = pd.DataFrame(imputed, index=temp.index, columns=cols_with_data)
                else:
                    temp[cols_with_data] = imputed
            except Exception:
                # Si falla SAITS, usar KNNImputer
                imputer = KNNImputer(n_neighbors=5, weights="distance")
                imputed = imputer.fit_transform(temp[cols_with_data])
                temp[cols_with_data] = pd.DataFrame(imputed, index=temp.index, columns=cols_with_data)
        else:
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            imputed = imputer.fit_transform(temp[cols_with_data])
            temp[cols_with_data] = pd.DataFrame(imputed, index=temp.index, columns=cols_with_data)

        # Mantener columnas totalmente vacías como NaN
        for c in cols_all_na:
            temp[c] = temp[c]

        dfs_imputed.append(temp)

    result = pd.concat(dfs_imputed).sort_values(["station", "date"]) if dfs_imputed else df
    return result


def main():
    # Base directory = repo root where CSVs live
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = ensure_outputs_dir(base_dir)

    # PASO 2 — Cargar estaciones (datos estáticos)
    stations_df = load_stations(base_dir)
    if not stations_df.empty:
        print(f"Stations loaded: {stations_df.shape}")
    else:
        print("Warning: estaciones.csv not found or empty.")

    # PASO 3/4 — Cargar todas las variables
    pm25 = load_variable_csv(base_dir / "PM2.5.csv", "PM2.5")
    pm10 = load_variable_csv(base_dir / "PM10.csv", "PM10")
    no2 = load_variable_csv(base_dir / "NO2.csv", "NO2")
    o3 = load_variable_csv(base_dir / "O3.csv", "O3")
    so2 = load_variable_csv(base_dir / "SO2.csv", "SO2")
    co = load_variable_csv(base_dir / "CO.csv", "CO")

    tmp = load_variable_csv(base_dir / "TMP.csv", "temperature")
    hum = load_variable_csv(base_dir / "HUM.csv", "humidity")
    vel = load_variable_csv(base_dir / "VEL.csv", "wind_velocity")
    dirc = load_variable_csv(base_dir / "DIR.csv", "wind_direction")
    llu = load_variable_csv(base_dir / "LLU.csv", "precipitation")
    pre = load_variable_csv(base_dir / "PRE.csv", "pressure")
    rs = load_variable_csv(base_dir / "RS.csv", "solar_radiation")

    dfs = [
        pm25, pm10, no2, o3, so2, co,
        tmp, hum, vel, dirc, llu, pre, rs,
    ]

    # PASO 5 — Unir todas las variables (estrategia con índices para evitar explosión de memoria)
    dfs_indexed = [d.set_index(["date", "station"]) for d in dfs]
    df = reduce(lambda left, right: left.join(right, how="outer"), dfs_indexed).reset_index()

    # PASO 6 — Limpieza básica
    df.sort_values(["station", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Merged dataframe:")
    print(df.info())
    print(df.head())

    # PASO 7 — Convertir columnas a numéricas
    num_cols = df.columns.difference(["date", "station"])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # PASO 8 — Visualizar valores faltantes (guardar a archivo)
    missing_plot_path = outputs_dir / "missing_matrix.png"
    visualize_missingness(df, missing_plot_path)
    print(f"Missingness plot saved to: {missing_plot_path}")

    # PASO 9 — Imputación con SAITS (por estación)
    df = impute_per_station(df)

    # PARTE B — Preparación para modelado
    # PASO 10 — Resampling mensual (municipio)
    monthly = (
        df.set_index("date")
          .groupby("station")
          .resample("M")
          .agg({
              "PM2.5": "mean",
              "PM10": "mean",
              "NO2": "mean",
              "O3": "mean",
              "SO2": "mean",
              "CO": "mean",
              "temperature": "mean",
              "humidity": "mean",
              "wind_velocity": "mean",
              "precipitation": "sum",  # lluvia = suma
              "pressure": "mean",
              "solar_radiation": "mean",
          })
          .reset_index()
    )

    # PASO 11 — Feature engineering básico
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    # PASO 12 — Lags de PM2.5
    for lag in [1, 3, 6, 12]:
        monthly[f"pm25_lag_{lag}"] = (
            monthly.groupby("station")["PM2.5"].shift(lag)
        )

    # PASO 13 — Rolling statistics (ventana 3)
    monthly["pm25_roll_3"] = (
        monthly.groupby("station")["PM2.5"]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )

    # PASO 14 — Eliminar NaNs generados
    monthly.dropna(inplace=True)

    # PASO 15 — Guardar dataset final listo para modelos
    final_path = outputs_dir / "monthly_dataset.csv"
    monthly.to_csv(final_path, index=False)

    print("Final monthly dataset:")
    print(monthly.head())
    print(monthly.info())
    print(f"Monthly dataset saved to: {final_path}")


if __name__ == "__main__":
    main()
