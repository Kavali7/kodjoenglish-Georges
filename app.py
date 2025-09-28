import logging
import os
import json
from datetime import date
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local helpers for GHL integration
from ghl_client import (
    GHLClient,
    GHLSettings,
    GHLConfigurationError,
    build_dashboard_dataframe,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ------------------------ Data loading ------------------------------------

@st.cache_data(ttl=900)
def load_ghl_dataframe(start_date: Optional[date] = None, end_date: Optional[date] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load fresh data from the GoHighLevel API and build the dashboard frame.

    Reads settings from Streamlit secrets under `ghl`.
    """
    settings = GHLSettings.from_streamlit(st.secrets)
    client = GHLClient(settings)
    diag: Dict[str, Any] = {}
    # Diagnostics: surface base URL, headers, and location used
    try:
        diag["base_url"] = settings.base_url
        diag["location_id"] = settings.location_id or ""
        diag["headers"] = {
            "LocationId": client.session.headers.get("LocationId"),
            "Version": client.session.headers.get("Version"),
        }
    except Exception:
        pass
    df = build_dashboard_dataframe(client, settings, start_date=start_date, end_date=end_date, diag=diag)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    diag["source"] = "GHL API"
    return df, diag


def load_google_sheet_data() -> pd.DataFrame:
    """Optional fallback: load first worksheet of a Google Sheet if configured.

    Requires `st.secrets["gcp_service_account"]` and `st.secrets["google_sheet_id"]`.
    Returns an empty DataFrame if not configured or on failure.
    """
    try:
        if "gcp_service_account" not in st.secrets or "google_sheet_id" not in st.secrets:
            return pd.DataFrame()
        import gspread

        sa = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        sh = sa.open_by_key(st.secrets["google_sheet_id"])
        ws = sh.get_worksheet(0)
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records)
        # Normalize columns if present
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Setter" not in df.columns and not df.empty:
            df["Setter"] = "Team"
        numeric_cols = [c for c in df.columns if c not in ("Date", "Setter")]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception as exc:  # pragma: no cover - best-effort fallback
        logger.exception("Google Sheets fallback failed", exc_info=exc)
        return pd.DataFrame()


def load_data(fast_days: Optional[int] = None) -> pd.DataFrame:
    """Try GHL first, then fallback to Google Sheets if configured."""
    # Attempt GHL
    try:
        start = end = None
        if fast_days and fast_days > 0:
            end = date.today()
            start = end  # default to same day; build() adjusts to a range if needed
            # build_dashboard_dataframe uses provided start/end; we’ll subtract below for clarity
            from datetime import timedelta
            start = end - timedelta(days=fast_days)

        df_ghl, diag = load_ghl_dataframe(start, end)
        if not df_ghl.empty:
            st.session_state["data_source"] = "GHL API"
            st.session_state["diagnostics"] = diag
            return df_ghl
    except GHLConfigurationError as cfg_err:
        logger.info("GHL configuration incomplete: %s", cfg_err)
        st.info("Configuration GHL absente ou incomplète. Passage sur Google Sheets si configuré.")
    except Exception as exc:
        logger.exception("Failed to fetch from GHL", exc_info=exc)
        st.warning("Impossible de récupérer les données via l'API GHL. Utilisation d'une source de secours si disponible.")

    # Fallback: Google Sheet
    df_sheet = load_google_sheet_data()
    if not df_sheet.empty:
        st.session_state["data_source"] = "Google Sheets"
        st.session_state["diagnostics"] = {
            "source": "Google Sheets",
            "counts": {"sheet_rows": int(len(df_sheet))},
            "setters_rows": df_sheet["Setter"].value_counts(dropna=False).to_dict() if "Setter" in df_sheet else {},
            "setters_total_leads": df_sheet.groupby("Setter")["Total_Leads"].sum().astype(int).to_dict() if "Setter" in df_sheet and "Total_Leads" in df_sheet else {},
        }
        return df_sheet

    # Fallback: Local JSON files in project root (contacts.json, ...)
    try:
        class LocalJsonClient:
            def __init__(self, settings: GHLSettings, base_dir: str = ".") -> None:
                self.settings = settings
                self.base_dir = base_dir

            def _load(self, filename: str, keys: list[str]) -> list[dict]:
                try:
                    path = os.path.join(self.base_dir, filename)
                    if not os.path.exists(path):
                        return []
                    # Read with encoding auto-fallback
                    with open(path, "rb") as f:
                        raw = f.read()
                    text = None
                    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
                        try:
                            text = raw.decode(enc)
                            break
                        except Exception:
                            continue
                    if text is None:
                        text = raw.decode("latin-1", errors="ignore")
                    data = json.loads(text)
                    if isinstance(data, dict):
                        for k in keys:
                            if k in data and isinstance(data[k], list):
                                return data[k]
                    if isinstance(data, list):
                        return data
                except Exception as e:
                    logger.warning("Local JSON load failed for %s: %s", filename, e)
                return []

            def get_contacts(self, start, end):
                return self._load("contacts.json", ["contacts", "items"])

            def get_opportunities(self, start, end):
                return self._load("opportunities.json", ["opportunities", "items"])

            def get_forms(self, start, end):
                return self._load("forms.json", ["forms", "submissions", "items"])

            def get_messages(self, start, end):
                return self._load("messages.json", ["messages", "items"])

            def get_payments(self, start, end):
                return self._load("payments.json", ["transactions", "items"])

            def get_calls(self, start, end):
                return self._load("calls.json", ["calls", "items"])

        # Build settings: prefer Streamlit secrets, else use safe defaults for local JSON
        try:
            settings = GHLSettings.from_streamlit(st.secrets)
        except Exception:
            tz = "UTC"
            try:
                if "ghl" in st.secrets and "timezone" in st.secrets["ghl"]:
                    tz = str(st.secrets["ghl"]["timezone"])
            except Exception:
                pass
            settings = GHLSettings(
                api_key="dummy",
                timezone=tz,
                user_mapping={},
                source_mapping={
                    "facebook": "From_FB",
                    "instagram": "From_Insta",
                    "tiktok": "From_TikTok",
                    "linkedin": "From_LinkedIn",
                    "forms": "From_Forms",
                    "direct": "From_Direct",
                },
                history_days=7,
            )
        local_client = LocalJsonClient(settings, base_dir=os.getcwd())
        diag_local: Dict[str, Any] = {}
        df_local = build_dashboard_dataframe(local_client, settings, diag=diag_local)
        if not df_local.empty:
            if "Date" in df_local.columns:
                df_local["Date"] = pd.to_datetime(df_local["Date"], errors="coerce")
            st.session_state["data_source"] = "Local JSON"
            diag_local["source"] = "Local JSON"
            st.session_state["diagnostics"] = diag_local
            return df_local
    except Exception as exc:
        logger.info("Local JSON fallback not available or failed: %s", exc)

    st.session_state["data_source"] = "Aucune"
    return pd.DataFrame()


# ------------------------ UI helpers --------------------------------------

def display_kpis(df_filtered: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)

    total_rev = float(df_filtered.get("Revenue_Generated", pd.Series(dtype=float)).sum()) if not df_filtered.empty else 0.0
    total_leads = int(df_filtered.get("Total_Leads", pd.Series(dtype=float)).sum()) if not df_filtered.empty else 0
    total_closes = int(df_filtered.get("Total_Closes", pd.Series(dtype=float)).sum()) if not df_filtered.empty else 0
    if total_leads > 0:
        avg_close_rate = (total_closes / total_leads) * 100
    else:
        avg_close_rate = 0.0

    with col1:
        st.metric("Total Revenue", f"{total_rev:,.0f}")
    with col2:
        st.metric("Total Leads", f"{total_leads}")
    with col3:
        st.metric("Total Closes", f"{total_closes}")
    with col4:
        st.metric("Avg Close Rate", f"{avg_close_rate:.1f}%")


def revenue_trend_plot(df_filtered: pd.DataFrame) -> None:
    if df_filtered.empty or "Date" not in df_filtered or "Revenue_Generated" not in df_filtered:
        st.info("Revenue trend: données insuffisantes.")
        return
    df_t = df_filtered.sort_values("Date")
    fig = px.line(
        df_t,
        x="Date",
        y="Revenue_Generated",
        markers=True,
        title="Revenue Trend",
        labels={"Revenue_Generated": "Revenue", "Date": "Date"},
        color_discrete_sequence=["#C53030"],
    )
    st.plotly_chart(fig, use_container_width=True)


def leads_by_source_plot(df_filtered: pd.DataFrame) -> None:
    source_cols = [c for c in df_filtered.columns if c.startswith("From_")]
    if not source_cols:
        st.info("Aucune donnée de source.")
        return
    df_sum = df_filtered[source_cols].sum().reset_index()
    df_sum.columns = ["Source", "Leads"]
    df_sum["Source"] = df_sum["Source"].str.replace("From_", "", regex=False)
    fig = px.bar(
        df_sum,
        x="Source",
        y="Leads",
        text="Leads",
        title="Leads par source",
        color="Source",
        color_discrete_sequence=["#DC2626", "#EA580C", "#D97706", "#CA8A04", "#65A30D"],
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def funnel_plot(df_filtered: pd.DataFrame) -> None:
    stages = [
        "Total_Leads",
        "Follow_ups",
        "Initial_msg_Sent",
        "Offers_Made",
        "Links_Sent",
        "Total_Closes",
    ]
    labels = ["Leads", "Follow Ups", "Messages Sent", "Offers", "Links Sent", "Closed"]
    values = []
    for s in stages:
        values.append(int(df_filtered[s].sum()) if s in df_filtered.columns else 0)

    if sum(values) == 0:
        st.info("Pipeline: données insuffisantes.")
        return

    fig = go.Figure(
        go.Funnel(y=labels, x=values, textinfo="value+percent initial", marker=dict(color=[
            "#DC2626", "#EA580C", "#F59E0B", "#84CC16", "#10B981", "#06B6D4"
        ]))
    )
    st.plotly_chart(fig, use_container_width=True)


def diagnostics_panel(df: pd.DataFrame) -> None:
    diag = st.session_state.get("diagnostics") or {}
    with st.expander("Diagnostics", expanded=False):
        st.write({k: v for k, v in diag.items() if k in ("source", "period", "pipelines", "base_url", "location_id")})
        headers = diag.get("headers")
        if headers:
            st.caption("En-têtes API")
            st.json(headers)
        counts = diag.get("counts") or {}
        if counts:
            st.subheader("Objets récupérés")
            st.table(pd.DataFrame([counts]).T.rename(columns={0: "count"}))
        # Setters
        rows_by_setter = diag.get("setters_rows") or {}
        leads_by_setter = diag.get("setters_total_leads") or {}
        if rows_by_setter or leads_by_setter:
            st.subheader("Setters détectés")
            all_setters = sorted(set(rows_by_setter) | set(leads_by_setter))
            data = []
            for s in all_setters:
                data.append({
                    "Setter": s,
                    "Rows": rows_by_setter.get(s, 0),
                    "Total_Leads": leads_by_setter.get(s, 0),
                })
            st.table(pd.DataFrame(data))
        errors = diag.get("errors") or []
        if errors:
            st.subheader("Erreurs API récentes")
            st.json(errors)


def show_dashboard() -> None:
    st.sidebar.header("Filtres")

    fast_mode = st.sidebar.checkbox("Mode rapide (30 jours)", value=True, help="Limite la période des appels API pour éviter les temps d'attente.")

    if st.sidebar.button("Actualiser les données"):
        load_ghl_dataframe.clear()
        st.toast("Cache vidé. Rafraîchissez si besoin.")

    df = load_data(fast_days=(30 if fast_mode else None))

    if df.empty:
        st.warning("Aucune donnée à afficher. Configurez GHL ou Google Sheets dans `.streamlit/secrets.toml`.")
        return

    # Date filter
    min_date: Optional[date] = pd.to_datetime(df["Date"]).dt.date.min() if "Date" in df else None
    max_date: Optional[date] = pd.to_datetime(df["Date"]).dt.date.max() if "Date" in df else None
    start = st.sidebar.date_input("Début", value=min_date or date.today())
    end = st.sidebar.date_input("Fin", value=max_date or date.today())

    df_filtered = df.copy()
    if "Date" in df_filtered:
        df_filtered = df_filtered[(df_filtered["Date"].dt.date >= start) & (df_filtered["Date"].dt.date <= end)]

    # Setter filter
    if "Setter" in df_filtered:
        setters = sorted([s for s in df_filtered["Setter"].dropna().unique()])
        selected = st.sidebar.multiselect("Setters", options=setters, default=setters)
        if selected:
            df_filtered = df_filtered[df_filtered["Setter"].isin(selected)]

    st.caption(f"Source des données: {st.session_state.get('data_source', 'Inconnue')}")

    display_kpis(df_filtered)

    col1, col2 = st.columns(2)
    with col1:
        revenue_trend_plot(df_filtered)
    with col2:
        leads_by_source_plot(df_filtered)

    st.markdown("### Pipeline")
    funnel_plot(df_filtered)
    diagnostics_panel(df)


def _welcome_name() -> str:
    try:
        info_path = ".streamlit/Info"
        with open(info_path, "r", encoding="utf-8") as f:
            name = f.read().strip()
        return name or "Invité"
    except Exception:
        return "Invité"


def main() -> None:
    st.set_page_config(page_title="KodjoEnglish Sales Dashboard", layout="wide")
    st.sidebar.success(f"Bienvenue {_welcome_name()}")
    st.title("KodjoEnglish Sales Metrics")
    show_dashboard()


if __name__ == "__main__":
    main()
