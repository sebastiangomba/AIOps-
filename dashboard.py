from __future__ import annotations

"""
Dashboard interactivo construido con Dash para explorar las métricas de la
base de datos `clean_cloud_usage.csv`.

Ejecuta con:
    python dashboard.py
Luego abre http://127.0.0.1:8050 en tu navegador.
"""
from pathlib import Path

import dash
from dash import Dash, Input, Output, State, dcc, html
import joblib
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Compatibilidad con paquetes que aún llaman a pkgutil.find_loader (removido en Python 3.12+).
import pkgutil
from importlib.util import find_spec

if not hasattr(pkgutil, "find_loader"):
    def _find_loader(name: str):
        spec = find_spec(name)
        return spec.loader if spec else None

    pkgutil.find_loader = _find_loader  # type: ignore[attr-defined]


BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "clean_cloud_usage.csv"
MODEL_PATH = BASE_PATH / "random_forest_model.pkl"
GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select", "lasso2d", "zoomIn2d", "zoomOut2d"],
    "responsive": True,
}
GRAPH_HEIGHT = 360
FEATURE_COLUMNS = ["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_IO"]
METRIC_LABELS = {
    "CPU_Usage": "CPU Usage (%)",
    "Memory_Usage": "Memoria (%)",
    "Disk_IO": "Disco IO (MB/s)",
    "Network_IO": "Red IO (MB/s)",
}
INPUT_IDS = {
    "CPU_Usage": "input-cpu",
    "Memory_Usage": "input-memory",
    "Disk_IO": "input-disk",
    "Network_IO": "input-network",
}


def load_data() -> pd.DataFrame:
    """Carga el dataset y aplica el preprocesamiento mínimo."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    numeric_cols = ["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_IO"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["Anomaly_Label"] = df["Anomaly_Label"].astype(int)
    df = df.dropna(subset=numeric_cols)
    return df


def style_fig(fig):
    """Uniforma los estilos de los gráficos para mantener marcos consistentes."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        margin=dict(l=20, r=20, t=70, b=40),
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(148,163,184,0.4)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False)
    return fig


df = load_data()
workload_options = [{"label": w, "value": w} for w in sorted(df["Workload_Type"].unique())]
all_workloads = [opt["value"] for opt in workload_options]
palette = px.colors.qualitative.Vivid + px.colors.qualitative.Set3 + px.colors.qualitative.Plotly
WORKLOAD_COLOR_MAP = {
    workload: palette[idx % len(palette)] for idx, workload in enumerate(all_workloads)
}
metric_stats = {
    col: {"min": df[col].min(), "max": df[col].max(), "mean": df[col].mean()} for col in FEATURE_COLUMNS
}
prediction_defaults = {col: round(stats["mean"], 2) for col, stats in metric_stats.items()}
workload_metric_defaults = (
    df.groupby("Workload_Type")[FEATURE_COLUMNS].mean().round(2).to_dict("index")
)
scaler = StandardScaler().fit(df[FEATURE_COLUMNS])
model_error: str | None = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:  # pragma: no cover - logging visual feedback
    model = None
    model_error = f"No se pudo cargar el modelo Random Forest: {exc}"

app = Dash(__name__)
app.title = "Dashboard de Métricas Cloud"


def kpi_card(card_id: str, title: str) -> html.Div:
    """Pequeño helper para no repetir estructura de las tarjetas KPI."""
    return html.Div(
        [
            html.Span(title, className="kpi-label"),
            html.Span("—", id=card_id, className="kpi-value"),
        ],
        className="kpi-card glass-panel",
    )


def metric_input(field_key: str) -> html.Div:
    """Crea un control numérico con rango de referencia para el formulario de predicción."""
    stats = metric_stats[field_key]
    return html.Div(
        [
            html.Label(METRIC_LABELS[field_key], htmlFor=INPUT_IDS[field_key]),
            dcc.Input(
                id=INPUT_IDS[field_key],
                type="number",
                min=float(stats["min"]),
                max=float(stats["max"]),
                step=0.1,
                value=prediction_defaults[field_key],
                debounce=True,
                className="metric-input",
            ),
            html.Small(f"Min: {stats['min']:.2f} | Max: {stats['max']:.2f}", className="field-hint"),
        ],
        className="form-field",
    )


app.layout = html.Div(
    [
        html.H1("Dashboard de Métricas Cloud"),
        html.P("Explora las métricas clave, filtra por workload y detecta rápidamente comportamientos anómalos."),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Tipos de carga"),
                        dcc.Dropdown(
                            id="workload-filter",
                            options=workload_options,
                            value=all_workloads,
                            multi=True,
                            placeholder="Selecciona workloads",
                            persistence=True,
                            className="styled-dropdown",
                        ),
                        html.Div(
                            [
                                html.Button("Todos", id="select-all-workloads", n_clicks=0, className="ghost-button"),
                                html.Button("Limpiar", id="clear-workloads", n_clicks=0, className="ghost-button"),
                            ],
                            className="button-row",
                        ),
                    ],
                    className="control glass-panel",
                ),
                html.Div(
                    [
                        html.Label("Modo de visualización"),
                        dcc.Checklist(
                            id="anomaly-toggle",
                            options=[{"label": "Ver solo anomalías", "value": "anomaly"}],
                            value=[],
                            inputClassName="switch-input",
                            labelClassName="switch-label",
                        ),
                        html.Div(id="active-filters", className="filter-chip-container"),
                    ],
                    className="control glass-panel",
                ),
            ],
            className="controls-container",
        ),
        html.Div(
            [
                kpi_card("kpi-cpu", "CPU promedio (%)"),
                kpi_card("kpi-memory", "Memoria promedio (%)"),
                kpi_card("kpi-disk", "Disco IO promedio (MB/s)"),
                kpi_card("kpi-network", "Red IO promedio (MB/s)"),
            ],
            className="kpi-grid",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Evolución temporal de métricas"),
                        dcc.Graph(id="time-series-fig", config=GRAPH_CONFIG, style={"height": f"{GRAPH_HEIGHT}px"}),
                    ],
                    className="graph-card glass-panel",
                ),
                html.Div(
                    [
                        html.H3("Uso promedio por workload"),
                        dcc.Graph(id="workload-share-fig", config=GRAPH_CONFIG, style={"height": f"{GRAPH_HEIGHT}px"}),
                    ],
                    className="graph-card glass-panel",
                ),
                html.Div(
                    [
                        html.H3("Eventos diarios normales vs anómalos"),
                        dcc.Graph(id="anomaly-trend-fig", config=GRAPH_CONFIG, style={"height": f"{GRAPH_HEIGHT}px"}),
                    ],
                    className="graph-card glass-panel wide",
                ),
            ],
            className="charts-grid",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Predicción de anomalías"),
                        html.P("Ingresa métricas actuales y valida rápidamente si el modelo espera un comportamiento anómalo."),
                    ],
                    className="prediction-header",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Preset por tipo de workload"),
                                dcc.Dropdown(
                                    id="prediction-workload",
                                    options=workload_options,
                                    value=all_workloads[0] if all_workloads else None,
                                    placeholder="Selecciona un workload para precargar valores",
                                    clearable=True,
                                    className="styled-dropdown preset-dropdown",
                                ),
                                html.Small(
                                    "Selecciona un workload para rellenar los campos con valores promedio o ingresa tus propios datos.",
                                    className="field-hint",
                                ),
                            ],
                            className="form-field full-width",
                        ),
                        *[metric_input(key) for key in FEATURE_COLUMNS],
                    ],
                    className="prediction-grid",
                ),
                html.Div(
                    [
                        html.Button("Predecir anomalía", id="predict-button", n_clicks=0, className="primary-button"),
                        html.Span(
                            model_error or "Modelo Random Forest cargado correctamente.",
                            className=f"model-status {'error' if model_error else 'ok'}",
                        ),
                    ],
                    className="prediction-actions",
                ),
                dcc.Loading(
                    id="prediction-loader",
                    type="circle",
                    children=html.Div(
                        "Ingresa valores y presiona \"Predecir anomalía\".",
                        id="prediction-output",
                        className="prediction-message neutral",
                    ),
                ),
            ],
            className="prediction-section glass-panel",
        ),
    ],
    className="page",
)


def filter_dataframe(selected_workloads: list[str] | None, anomaly_only: bool) -> pd.DataFrame:
    """Aplica filtros de workloads y anomalías al dataframe."""
    data = df.copy()
    if selected_workloads:
        data = data[data["Workload_Type"].isin(selected_workloads)]
    if anomaly_only:
        data = data[data["Anomaly_Label"] == 1]
    return data


@app.callback(
    Output("kpi-cpu", "children"),
    Output("kpi-memory", "children"),
    Output("kpi-disk", "children"),
    Output("kpi-network", "children"),
    Output("time-series-fig", "figure"),
    Output("workload-share-fig", "figure"),
    Output("anomaly-trend-fig", "figure"),
    Input("workload-filter", "value"),
    Input("anomaly-toggle", "value"),
)
def update_dashboard(selected_workloads: list[str], anomaly_toggle: list[str]):
    anomaly_only = "anomaly" in (anomaly_toggle or [])
    filtered = filter_dataframe(selected_workloads or [], anomaly_only)

    if filtered.empty:
        empty_fig = style_fig(px.scatter(title="Sin datos para los filtros aplicados"))
        empty_fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return ("—",) * 4 + (empty_fig,) * 3

    cpu_avg = f"{filtered['CPU_Usage'].mean():.2f}"
    mem_avg = f"{filtered['Memory_Usage'].mean():.2f}"
    disk_avg = f"{filtered['Disk_IO'].mean():.2f}"
    network_avg = f"{filtered['Network_IO'].mean():.2f}"

    long_metrics = filtered.melt(
        id_vars=["Timestamp"],
        value_vars=["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_IO"],
        var_name="Métrica",
        value_name="Valor",
    )
    time_series_fig = style_fig(
        px.line(
            long_metrics,
            x="Timestamp",
            y="Valor",
            color="Métrica",
            title="Evolución temporal de las métricas",
        )
    )
    time_series_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    workload_share = (
        filtered.groupby("Workload_Type")
        .agg(
            cpu_avg=("CPU_Usage", "mean"),
            mem_avg=("Memory_Usage", "mean"),
            count=("Workload_Type", "size"),
        )
        .reset_index()
    )
    workload_long = workload_share.melt(
        id_vars=["Workload_Type"],
        value_vars=["cpu_avg", "mem_avg"],
        var_name="Métrica",
        value_name="Promedio",
    )
    workload_share_fig = style_fig(
        px.bar(
            workload_long,
            x="Métrica",
            y="Promedio",
            color="Workload_Type",
            barmode="group",
            title="Uso promedio por tipo de carga",
            labels={"Promedio": "Promedio (%)", "Métrica": "Métrica", "Workload_Type": "Tipo de carga"},
            color_discrete_map=WORKLOAD_COLOR_MAP,
        )
    )

    anomaly_trend = (
        filtered.assign(date=filtered["Timestamp"].dt.date)
        .groupby(["date", "Anomaly_Label"])
        .size()
        .reset_index(name="Eventos")
    )
    anomaly_trend_fig = style_fig(
        px.bar(
            anomaly_trend,
            x="date",
            y="Eventos",
            color="Anomaly_Label",
            title="Eventos diarios (0=normal,1=anomalía)",
            labels={"date": "Fecha", "Anomaly_Label": "Clase"},
        )
    )

    return (
        cpu_avg,
        mem_avg,
        disk_avg,
        network_avg,
        time_series_fig,
        workload_share_fig,
        anomaly_trend_fig,
    )


@app.callback(
    Output("workload-filter", "value"),
    Input("select-all-workloads", "n_clicks"),
    Input("clear-workloads", "n_clicks"),
    State("workload-filter", "value"),
    prevent_initial_call=True,
)
def toggle_workloads(select_all_clicks, clear_clicks, current_values):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_values
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "select-all-workloads":
        return all_workloads
    if trigger_id == "clear-workloads":
        return []
    return current_values


@app.callback(
    Output("active-filters", "children"),
    Input("workload-filter", "value"),
)
def render_active_filters(selected_workloads):
    if not selected_workloads:
        return html.Span("Sin workloads seleccionados", className="filter-chip neutral-chip")
    if len(selected_workloads) == len(all_workloads):
        return html.Span("Mostrando todos los workloads", className="filter-chip neutral-chip")
    return [
        html.Span(
            workload,
            className="filter-chip",
            style={
                "background": WORKLOAD_COLOR_MAP.get(workload, "#38bdf8"),
                "color": "#020617",
                "borderColor": WORKLOAD_COLOR_MAP.get(workload, "#38bdf8"),
            },
        )
        for workload in selected_workloads
    ]


@app.callback(
    Output("prediction-output", "children"),
    Output("prediction-output", "className"),
    Input("predict-button", "n_clicks"),
    State("prediction-workload", "value"),
    State("input-cpu", "value"),
    State("input-memory", "value"),
    State("input-disk", "value"),
    State("input-network", "value"),
    prevent_initial_call=True,
)
def predict_anomaly(n_clicks, workload_value, cpu, memory, disk, network):
    base_class = "prediction-message"
    if model is None:
        return (
            "⚠️ No se pudo cargar el modelo Random Forest. Verifica el archivo random_forest_model.pkl.",
            f"{base_class} error",
        )
    if scaler is None:
        return ("No hay scaler para estandarizar los datos.", f"{base_class} error")
    values = [cpu, memory, disk, network]
    if any(value is None for value in values):
        return (
            "Completa todos los campos numéricos antes de predecir (puedes usar el preset de workload para rellenarlos).",
            f"{base_class} warning",
        )

    features = pd.DataFrame(
        [
            {
                "CPU_Usage": float(cpu),
                "Memory_Usage": float(memory),
                "Disk_IO": float(disk),
                "Network_IO": float(network),
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(scaled_features)[0]
        probability = float(proba_values[prediction])

    label = "Se detecta una anomalía" if prediction == 1 else "Condición normal"
    emoji = "⚠️" if prediction == 1 else "✅"
    detail = f"{emoji} {label}."
    if probability is not None:
        detail += f" Confianza: {probability * 100:.2f}%."

    status_class = "anomaly" if prediction == 1 else "normal"
    return (detail, f"{base_class} {status_class}")


@app.callback(
    Output("input-cpu", "value"),
    Output("input-memory", "value"),
    Output("input-disk", "value"),
    Output("input-network", "value"),
    Input("prediction-workload", "value"),
    prevent_initial_call=True,
)
def apply_workload_preset(workload_value):
    if not workload_value:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    preset = workload_metric_defaults.get(workload_value)
    if not preset:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    return (
        float(preset["CPU_Usage"]),
        float(preset["Memory_Usage"]),
        float(preset["Disk_IO"]),
        float(preset["Network_IO"]),
    )


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: radial-gradient(circle at 20% 20%, #1e1b4b, #0b1120 60%);
                color: #f8fafc;
            }
            .page {
                padding: 32px;
                display: flex;
                flex-direction: column;
                gap: 24px;
                max-width: 1400px;
                margin: 0 auto;
            }
            .controls-container { display: flex; gap: 24px; flex-wrap: wrap; }
            .control { flex: 1; min-width: 280px; display: flex; flex-direction: column; gap: 12px; }
            .glass-panel {
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 20px 35px rgba(2, 6, 23, 0.6);
                backdrop-filter: blur(12px);
            }
            .button-row { display: flex; gap: 12px; }
            .ghost-button {
                flex: 1;
                background: rgba(94, 234, 212, 0.08);
                color: #67e8f9;
                border: 1px solid rgba(94, 234, 212, 0.5);
                border-radius: 999px;
                padding: 8px 0;
                font-weight: 600;
                cursor: pointer;
            }
            .ghost-button:hover { background: rgba(94, 234, 212, 0.18); }
            .filter-chip-container { display:flex; flex-wrap:wrap; gap:8px; min-height:40px; }
            .filter-chip {
                display:inline-flex;
                align-items:center;
                padding:6px 14px;
                border-radius:999px;
                border:1px solid rgba(148, 163, 184, 0.35);
                font-size:0.85rem;
                font-weight:600;
                letter-spacing:0.02em;
                background:rgba(148, 163, 184, 0.1);
                color:#e2e8f0;
            }
            .filter-chip.neutral-chip {
                border-style:dashed;
            }
            .switch-label { font-size: 0.95rem; color: #e2e8f0; }
            .switch-input { margin-right: 8px; transform: scale(1.2); accent-color: #38bdf8; }
            .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:16px; }
            .kpi-card { display:flex; flex-direction:column; gap:12px; }
            .kpi-label { font-size:0.8rem; color:#93c5fd; text-transform:uppercase; letter-spacing:0.08em; }
            .kpi-value { font-size:2rem; font-weight:700; color:#38bdf8; }
            .charts-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); gap:24px; }
            .graph-card { display:flex; flex-direction:column; gap:12px; min-height:420px; }
            .graph-card.wide { grid-column: span 2; }
            .graph-card h3 { margin:0; font-size:1rem; color:#cbd5f5; letter-spacing:0.05em; text-transform:uppercase; }
            .dash-graph { background:#0f172a !important; border-radius:16px; padding:8px; flex:1; }
            .prediction-section { display:flex; flex-direction:column; gap:20px; margin-top:12px; }
            .prediction-header h2 { margin:0; letter-spacing:0.04em; text-transform:uppercase; color:#e0f2fe; }
            .prediction-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:16px; }
            .form-field { display:flex; flex-direction:column; gap:6px; }
            .form-field label { font-size:0.85rem; color:#cbd5f5; letter-spacing:0.04em; }
            .form-field.full-width { grid-column: 1 / -1; }
            .styled-dropdown .Select-control,
            .styled-dropdown .Select__control {
                background:rgba(15,23,42,0.85);
                border-radius:14px;
                border:1px solid rgba(148,163,184,0.45);
                box-shadow:none;
            }
            .styled-dropdown .Select-value,
            .styled-dropdown .Select__single-value {
                color:#e2e8f0;
                font-weight:600;
                letter-spacing:0.03em;
            }
            .styled-dropdown .Select-menu-outer,
            .styled-dropdown .Select__menu {
                background:rgba(2,6,23,0.95);
                border-radius:14px;
                border:1px solid rgba(59,130,246,0.4);
                overflow:hidden;
            }
            .styled-dropdown .Select-option,
            .styled-dropdown .Select__option {
                padding:10px 14px;
                color:#cbd5f5;
            }
            .styled-dropdown .Select-option.is-focused,
            .styled-dropdown .Select__option--is-focused {
                background:rgba(59,130,246,0.25);
                color:#f8fafc;
            }
            .styled-dropdown .Select-multi-value-wrapper .Select-value,
            .styled-dropdown .Select__multi-value {
                background:rgba(59,130,246,0.2);
                border-radius:999px;
                border:1px solid rgba(59,130,246,0.5);
                color:#e0f2fe;
            }
            .metric-input {
                border:1px solid rgba(148,163,184,0.4);
                border-radius:12px;
                padding:10px 12px;
                background:rgba(15,23,42,0.6);
                color:#f8fafc;
                font-size:1rem;
            }
            .metric-input:focus { outline:none; border-color:#38bdf8; box-shadow:0 0 0 1px rgba(56,189,248,0.6); }
            .field-hint { color:#94a3b8; font-size:0.75rem; letter-spacing:0.03em; }
            .prediction-actions { display:flex; flex-wrap:wrap; gap:16px; align-items:center; justify-content:space-between; }
            .primary-button {
                padding:12px 28px;
                border:none;
                border-radius:999px;
                background:linear-gradient(120deg, #38bdf8, #8b5cf6);
                color:#020617;
                font-weight:700;
                letter-spacing:0.08em;
                text-transform:uppercase;
                cursor:pointer;
                box-shadow:0 15px 35px rgba(56,189,248,0.35);
            }
            .primary-button:hover { filter:brightness(1.08); }
            .model-status { font-size:0.85rem; letter-spacing:0.04em; }
            .model-status.ok { color:#34d399; }
            .model-status.error { color:#f87171; }
            .prediction-message {
                border-radius:16px;
                padding:18px 22px;
                font-size:1rem;
                font-weight:600;
                letter-spacing:0.03em;
                border:1px solid transparent;
            }
            .prediction-message.neutral { border-color:rgba(148,163,184,0.35); color:#e2e8f0; }
            .prediction-message.normal { border-color:rgba(52,211,153,0.4); background:rgba(16,185,129,0.1); color:#34d399; }
            .prediction-message.anomaly { border-color:rgba(248,113,113,0.45); background:rgba(248,113,113,0.08); color:#f87171; }
            .prediction-message.warning, .prediction-message.error { border-color:rgba(248,189,88,0.6); background:rgba(248,189,88,0.08); color:#fcd34d; }
            .prediction-message.error { border-color:rgba(248,113,113,0.6); color:#fca5a5; }
            @media(max-width:1024px){ .graph-card.wide { grid-column:auto; } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
