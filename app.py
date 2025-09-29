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

# Optional default worksheet gid for Google Sheets (user-provided request)
DEFAULT_GOOGLE_SHEET_GID = 180120810


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


@st.cache_data(ttl=600)
def load_google_sheet_data() -> pd.DataFrame:
    """Optional fallback: load first worksheet of a Google Sheet if configured.

    Requires `st.secrets["gcp_service_account"]` and `st.secrets["google_sheet_id"]`.
    Returns an empty DataFrame if not configured or on failure.
    """
    try:
        if "gcp_service_account" not in st.secrets:
            return pd.DataFrame()
        import gspread

        sa = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        raw_sheet_id = (
            st.secrets.get("google_sheet_id")
            or dict(st.secrets.get("gcp_service_account", {})).get("google_sheet_id")
        )
        raw_gid = (
            st.secrets.get("google_sheet_gid")
            or dict(st.secrets.get("gcp_service_account", {})).get("google_sheet_gid")
        )
        if not raw_sheet_id:
            return pd.DataFrame()

        sheet_id = str(raw_sheet_id)
        ws_gid: Optional[int] = None
        try:
            import re
            if raw_gid:
                m = re.search(r"(\d+)", str(raw_gid))
                if m:
                    ws_gid = int(m.group(1))
            if not ws_gid and isinstance(sheet_id, str) and "gid=" in sheet_id:
                m = re.search(r"gid=(\d+)", sheet_id)
                if m:
                    ws_gid = int(m.group(1))
        except Exception:
            ws_gid = None

        # If a URL was provided, open by URL; else by key
        if sheet_id.startswith("http://") or sheet_id.startswith("https://"):
            sh = sa.open_by_url(sheet_id)
        else:
            # If user mistakenly passed only a gid here, cannot open
            if sheet_id.strip().lower().startswith("gid="):
                return pd.DataFrame()
            sh = sa.open_by_key(sheet_id)

        ws = None
        if ws_gid is None:
            try:
                import re, os as _os
                env_gid = _os.getenv("GOOGLE_SHEET_GID")
                if env_gid:
                    m = re.search(r"(\d+)", env_gid)
                    if m:
                        ws_gid = int(m.group(1))
                if ws_gid is None and DEFAULT_GOOGLE_SHEET_GID:
                    ws_gid = int(DEFAULT_GOOGLE_SHEET_GID)
            except Exception:
                ws_gid = None

        if ws_gid is not None:
            try:
                ws = sh.get_worksheet_by_id(int(ws_gid))  # type: ignore[attr-defined]
            except Exception:
                ws = None
        if ws is None:
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


def load_local_tabular_data() -> pd.DataFrame:
    """Load local Excel/CSV file if present in project root.

    Looks for `data.xlsx` (first worksheet) or `data.csv`.
    Returns a normalized DataFrame compatible with the dashboard or empty if not found.
    """
    try:
        cwd = os.getcwd()
        xlsx_candidates = ["base.xlsx", "data.xlsx"]
        csv_candidates = ["base.csv", "data.csv"]

        df: pd.DataFrame
        xlsx_path = next((os.path.join(cwd, n) for n in xlsx_candidates if os.path.exists(os.path.join(cwd, n))), None)
        csv_path = next((os.path.join(cwd, n) for n in csv_candidates if os.path.exists(os.path.join(cwd, n))), None)

        if xlsx_path:
            try:
                df = pd.read_excel(xlsx_path)
            except Exception:
                df = pd.read_excel(xlsx_path, engine="openpyxl")
        elif csv_path:
            # try comma then semicolon separators
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.read_csv(csv_path, sep=";")
        else:
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Normalize columns
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            # If no Date column, cannot chart; return empty to avoid confusion
            return pd.DataFrame()
        if "Setter" not in df.columns:
            df["Setter"] = "Team"

        # Force numeric for the remaining metric columns
        for col in [c for c in df.columns if c not in ("Date", "Setter")]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception as exc:
        logger.exception("Local Excel/CSV fallback failed", exc_info=exc)
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_local_json_dataframe() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load local JSON files as a lightweight offline source.

    Looks for files like contacts.json, opportunities.json, etc. in the project root.
    Returns a dataframe and diagnostics.
    """
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

        # Build settings from secrets if possible, else minimal defaults
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
        if not df_local.empty and "Date" in df_local.columns:
            df_local["Date"] = pd.to_datetime(df_local["Date"], errors="coerce")
        diag_local["source"] = "Local JSON"
        return df_local, diag_local
    except Exception as exc:
        logger.info("Local JSON load failed: %s", exc)
        return pd.DataFrame(), {"source": "Local JSON", "errors": [str(exc)]}


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
        # Considérer GHL comme valide uniquement si au moins un endpoint a renvoyé > 0
        counts = (diag or {}).get("counts", {}) if isinstance(diag, dict) else {}
        has_any = any(int(v) > 0 for v in counts.values()) if counts else False
        if not df_ghl.empty and has_any:
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

    # Fallback: Local Excel/CSV in project root
    df_tab = load_local_tabular_data()
    if not df_tab.empty:
        st.session_state["data_source"] = "Local Excel/CSV"
        st.session_state["diagnostics"] = {
            "source": "Local Excel/CSV",
            "counts": {"rows": int(len(df_tab))},
            "setters_rows": df_tab["Setter"].value_counts(dropna=False).to_dict() if "Setter" in df_tab else {},
            "setters_total_leads": df_tab.groupby("Setter")["Total_Leads"].sum().astype(int).to_dict() if "Setter" in df_tab and "Total_Leads" in df_tab else {},
        }
        return df_tab

    # Fallback: Local JSON files in project root (contacts.json, ...)
    try:
        df_local, diag_local = load_local_json_dataframe()
        if not df_local.empty:
            st.session_state["data_source"] = "Local JSON"
            diag_local["source"] = "Local JSON"
            st.session_state["diagnostics"] = diag_local
            return df_local
    except Exception as exc:
        logger.info("Local JSON fallback not available or failed: %s", exc)

    st.session_state["data_source"] = "Aucune"
    return pd.DataFrame()


# ------------------------ UI helpers --------------------------------------

def _period_bounds(start: date, end: date) -> tuple[date, date, int]:
    days = (end - start).days + 1
    return start, end, max(1, days)


def _previous_period(start: date, end: date) -> tuple[date, date]:
    _, _, days = _period_bounds(start, end)
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start.date(), prev_end.date()


def _sum_col(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not df.empty else 0.0


def _delta(curr: float, prev: float, pct: bool = False) -> tuple[str, bool]:
    if prev == 0:
        return ("+0%" if pct else "+0"), True
    diff = curr - prev
    if pct:
        val = (diff / prev) * 100
        return (f"{val:+.1f}%", val >= 0)
    else:
        return (f"{diff:+,.0f}", diff >= 0)


def _sparkline(df: pd.DataFrame, y: str):
    if df.empty or y not in df:
        return None
    s = df.groupby("Date")[y].sum().reset_index().sort_values("Date")
    if s[y].sum() == 0:
        return None
    fig = px.area(s, x="Date", y=y, template=plotly_template())
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=70, showlegend=False)
    fig.update_traces(line_color="#ef4444", fillcolor="rgba(239,68,68,0.15)")
    return fig


def display_kpis(df_filtered: pd.DataFrame, *, start: Optional[date] = None, end: Optional[date] = None, compare_prev: bool = False) -> None:
    col1, col2, col3, col4 = st.columns(4)

    # Current sums
    total_rev = _sum_col(df_filtered, "Revenue_Generated")
    total_leads = _sum_col(df_filtered, "Total_Leads")
    total_offers = _sum_col(df_filtered, "Offers_Made")
    total_links = _sum_col(df_filtered, "Links_Sent")
    total_closes = _sum_col(df_filtered, "Total_Closes")

    avg_close_rate = (total_closes / total_leads * 100) if total_leads > 0 else 0.0
    offer_rate = _sum_col(df_filtered, "Offer_Rate") / max(1, len(df_filtered)) if "Offer_Rate" in df_filtered else 0.0
    link_rate = _sum_col(df_filtered, "Link_Rate") / max(1, len(df_filtered)) if "Link_Rate" in df_filtered else 0.0
    close_rate = _sum_col(df_filtered, "Close_Rate") / max(1, len(df_filtered)) if "Close_Rate" in df_filtered else avg_close_rate
    aov = (total_rev / total_closes) if total_closes > 0 else 0.0

    # Previous period if requested
    d_rev = d_leads = d_offers = d_links = d_closes = ("", True)
    d_cvr = d_aov = ("", True)
    if compare_prev and start and end and "Date" in df_filtered:
        prev_start, prev_end = _previous_period(start, end)
        df_prev = df_filtered[(df_filtered["Date"].dt.date >= prev_start) & (df_filtered["Date"].dt.date <= prev_end)]
        prev_rev = _sum_col(df_prev, "Revenue_Generated")
        prev_leads = _sum_col(df_prev, "Total_Leads")
        prev_offers = _sum_col(df_prev, "Offers_Made")
        prev_links = _sum_col(df_prev, "Links_Sent")
        prev_closes = _sum_col(df_prev, "Total_Closes")
        prev_cvr = (prev_closes / prev_leads * 100) if prev_leads > 0 else 0.0
        prev_aov = (prev_rev / prev_closes) if prev_closes > 0 else 0.0
        d_rev = _delta(total_rev, prev_rev, pct=True)
        d_leads = _delta(total_leads, prev_leads, pct=True)
        d_offers = _delta(total_offers, prev_offers, pct=True)
        d_links = _delta(total_links, prev_links, pct=True)
        d_closes = _delta(total_closes, prev_closes, pct=True)
        d_cvr = _delta(avg_close_rate, prev_cvr, pct=True)
        d_aov = _delta(aov, prev_aov, pct=True)

    with col1:
        st.metric("Revenue", f"{total_rev:,.0f}", delta=d_rev[0])
        fig = _sparkline(df_filtered, "Revenue_Generated")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Leads", f"{int(total_leads)}", delta=d_leads[0])
        fig = _sparkline(df_filtered, "Total_Leads")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.metric("Offers", f"{int(total_offers)}", delta=d_offers[0])
        fig = _sparkline(df_filtered, "Offers_Made")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.metric("Links Sent", f"{int(total_links)}", delta=d_links[0])
        fig = _sparkline(df_filtered, "Links_Sent")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Closes", f"{int(total_closes)}", delta=d_closes[0])
        fig = _sparkline(df_filtered, "Total_Closes")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col6:
        st.metric("Close Rate", f"{close_rate:.1f}%", delta=d_cvr[0])
    with col7:
        st.metric("Offer Rate", f"{offer_rate:.1f}%")
    with col8:
        st.metric("AOV", f"{aov:,.0f}", delta=d_aov[0])


def source_share_donut(df_filtered: pd.DataFrame) -> None:
    source_cols = [c for c in df_filtered.columns if c.startswith("From_")]
    if not source_cols:
        st.info("Aucune donnée de source.")
        return
    df_sum = df_filtered[source_cols].sum().reset_index()
    df_sum.columns = ["Source", "Leads"]
    df_sum["Source"] = df_sum["Source"].str.replace("From_", "", regex=False)
    if df_sum["Leads"].sum() == 0:
        st.info("Répartition des sources: données insuffisantes.")
        return
    fig = px.pie(
        df_sum,
        names="Source",
        values="Leads",
        hole=0.5,
        title="Répartition des sources",
        template=plotly_template(),
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def revenue_trend_plot(df_filtered: pd.DataFrame, *, start: Optional[date] = None, end: Optional[date] = None, compare_prev: bool = False) -> None:
    if df_filtered.empty or "Date" not in df_filtered or "Revenue_Generated" not in df_filtered:
        st.info("Revenue trend: données insuffisantes.")
        return
    df_t = df_filtered.groupby("Date")["Revenue_Generated"].sum().reset_index().sort_values("Date")
    fig = px.line(
        df_t,
        x="Date",
        y="Revenue_Generated",
        markers=True,
        title="Revenue Trend",
        labels={"Revenue_Generated": "Revenue", "Date": "Date"},
        color_discrete_sequence=["#C53030"],
        template=plotly_template(),
    )
    # Optional previous period overlay
    if compare_prev and start and end and "Date" in df_filtered:
        prev_start, prev_end = _previous_period(start, end)
        df_prev = df_filtered[(df_filtered["Date"].dt.date >= prev_start) & (df_filtered["Date"].dt.date <= prev_end)]
        if not df_prev.empty:
            series_prev = df_prev.groupby("Date")["Revenue_Generated"].sum().reset_index().sort_values("Date")
            # Align previous series to current x-axis by shifting dates forward by period length
            _, _, days = _period_bounds(start, end)
            series_prev["Date"] = pd.to_datetime(series_prev["Date"]) + pd.Timedelta(days=days)
            fig.add_trace(
                go.Scatter(
                    x=series_prev["Date"],
                    y=series_prev["Revenue_Generated"],
                    name="Période précédente (décalée)",
                    mode="lines",
                    line=dict(color="#9CA3AF", dash="dash"),
                    showlegend=True,
                )
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
        template=plotly_template(),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def setter_performance_plot(df_filtered: pd.DataFrame) -> None:
    if df_filtered.empty or "Setter" not in df_filtered:
        st.info("Aucune donnée par setter.")
        return
    # Summaries per setter
    rev = df_filtered.groupby("Setter")["Revenue_Generated"].sum() if "Revenue_Generated" in df_filtered else pd.Series(dtype=float)
    closes = df_filtered.groupby("Setter")["Total_Closes"].sum() if "Total_Closes" in df_filtered else pd.Series(dtype=float)
    if rev.empty and closes.empty:
        st.info("Aucune métrique de performance disponible.")
        return
    df = pd.DataFrame({
        "Revenue": rev.fillna(0),
        "Closes": closes.fillna(0)
    }).reset_index().sort_values("Revenue", ascending=False)
    df_melt = df.melt(id_vars=["Setter"], value_vars=["Revenue", "Closes"], var_name="Metric", value_name="Value")
    fig = px.bar(
        df_melt,
        x="Setter",
        y="Value",
        color="Metric",
        barmode="group",
        title="Performance par setter",
        color_discrete_sequence=["#16A34A", "#2563EB"],
        template=plotly_template(),
    )
    st.plotly_chart(fig, use_container_width=True)


def daily_activity_plot(df_filtered: pd.DataFrame) -> None:
    if df_filtered.empty or "Date" not in df_filtered:
        st.info("Activité quotidienne: données insuffisantes.")
        return
    cols = [c for c in ["Total_Leads", "Offers_Made", "Links_Sent"] if c in df_filtered.columns]
    if not cols:
        st.info("Colonnes d'activité absentes.")
        return
    daily = df_filtered.groupby("Date")[cols].sum().reset_index().sort_values("Date")
    df_m = daily.melt(id_vars=["Date"], value_vars=cols, var_name="Metric", value_name="Value")
    fig = px.bar(
        df_m,
        x="Date",
        y="Value",
        color="Metric",
        title="Activité quotidienne",
        barmode="stack",
        color_discrete_sequence=["#0EA5E9", "#F59E0B", "#8B5CF6"],
        template=plotly_template(),
    )
    st.plotly_chart(fig, use_container_width=True)


def weekly_hour_heatmap(diag: Dict[str, Any]) -> None:
    data = (diag or {}).get("activity_heatmap") if isinstance(diag, dict) else None
    if not data:
        st.info("Heatmap indisponible pour cette source.")
        return
    try:
        import numpy as np
        import plotly.express as px
        mat = np.array(data.get("matrix", []))
        if mat.size == 0:
            st.info("Aucune activité à afficher.")
            return
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        hours = list(range(24))
        fig = px.imshow(
            mat,
            labels=dict(x="Heure", y="Jour", color="Activité"),
            x=hours,
            y=days,
            aspect="auto",
            color_continuous_scale="Reds",
            title="Chaleur d'activité (Jour × Heure)",
            template=plotly_template(),
        )
        fig.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Heatmap non disponible (erreur de rendu).")


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
    values: list[int] = []
    for s in stages:
        values.append(int(df_filtered[s].sum()) if s in df_filtered.columns else 0)

    if sum(values) == 0:
        st.info("Pipeline: données insuffisantes.")
        return

    # Compute stage-to-stage conversion rates on totals
    leads, _, _, offers, links, closes = values
    offer_rate = (offers / leads * 100) if leads else 0.0
    link_rate = (links / offers * 100) if offers else 0.0
    close_rate = (closes / links * 100) if links else 0.0
    texts = [
        f"{values[0]:,}",
        f"{values[1]:,}",
        f"{values[2]:,}",
        f"{values[3]:,}  ({offer_rate:.1f}%)",
        f"{values[4]:,}  ({link_rate:.1f}%)",
        f"{values[5]:,}  ({close_rate:.1f}%)",
    ]

    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        text=texts,
        textposition="inside",
        textinfo="text",
        marker=dict(color=["#DC2626", "#EA580C", "#F59E0B", "#84CC16", "#10B981", "#06B6D4"]),
    ))
    fig.update_layout(template=plotly_template(), margin=dict(l=20, r=20, t=30, b=20))
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
        # Tester dynamiquement les endpoints
        if st.button("Tester les endpoints API", type="secondary"):
            res = test_ghl_endpoints()
            st.subheader("Résultats des endpoints")
            st.json(res)
            # Suggestion d'hôte
            try:
                by_host = {}
                for item in res.get("results", []):
                    host = item.get("host")
                    ok = 1 if item.get("ok") else 0
                    by_host[host] = by_host.get(host, 0) + ok
                if by_host:
                    best = max(by_host, key=by_host.get)
                    st.info(f"Hôte recommandé: {best} (ok={by_host[best]})")
            except Exception:
                pass


def _guided_tips(diag: Dict[str, Any]) -> list[str]:
    tips: list[str] = []
    counts = (diag or {}).get("counts", {}) or {}
    # Récupération des secrets à tolérance de panne
    ghl: Dict[str, Any] = {}
    try:
        if hasattr(st, "secrets"):
            # L'accès peut lever si le TOML est invalide
            ghl = dict(st.secrets.get("ghl", {}))  # type: ignore[arg-type]
    except Exception:
        tips.append("Erreur de lecture de secrets.toml: vérifiez la syntaxe (sections [ghl], clés = valeurs).")
        ghl = {}
    if counts.get("opportunities", 0) == 0:
        if not ghl.get("pipeline_ids"):
            tips.append("Aucune opportunité: renseignez 'pipeline_ids' dans secrets.")
        else:
            tips.append("Aucune opportunité sur la période test: vérifiez stages/pipelines et permissions API.")
    if counts.get("forms", 0) == 0:
        if not ghl.get("form_ids"):
            tips.append("Aucun formulaire: ajoutez des 'form_ids' si vous souhaitez suivre les soumissions.")
        else:
            tips.append("Formulaires vides: vérifiez les dates et l'ID des formulaires.")
    if counts.get("payments", 0) == 0:
        tips.append("Paiements vides: vérifiez l'accès 'payments' et les transactions. Option: calculer le revenu via opportunités 'won'.")
    if counts.get("messages", 0) == 0:
        tips.append("Messages vides: vérifiez les droits 'conversations' et l'activité récente.")
    if counts.get("calls", 0) == 0:
        tips.append("Appels vides: vérifiez l'intégration téléphonie ou les droits 'calls'.")
    return tips


def test_ghl_endpoints() -> Dict[str, Any]:
    """Probe common GHL endpoints across candidate hosts and return statuses.

    Returns a dict with results list: [{host, path, status, ok, items}].
    """
    out: Dict[str, Any] = {"results": []}
    try:
        settings = GHLSettings.from_streamlit(st.secrets)
        client = GHLClient(settings)
        hosts = []
        for h in [settings.base_url, "https://rest.gohighlevel.com", "https://services.leadconnectorhq.com"]:
            h = (h or "").rstrip("/")
            if h and h not in hosts:
                hosts.append(h)

        from datetime import timedelta
        end = date.today()
        start = end
        start = end - timedelta(days=2)
        base_params = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
            "limit": 1,
            "offset": 0,
        }
        # Build candidate paths
        paths = [
            ("contacts", "/v1/contacts/"),
            ("contacts", f"/v2/locations/{settings.location_id}/contacts/search") if settings.location_id else None,
            ("opportunities", "/v1/opportunities/"),
            ("opportunities", "/v1/opportunities/search"),
            ("opportunities", f"/v2/locations/{settings.location_id}/opportunities/search") if settings.location_id else None,
            ("messages", "/v1/conversations/messages/"),
            ("messages", "/v1/conversations/messages/search"),
            ("payments", "/v1/payments/transactions/"),
            ("payments", f"/v2/locations/{settings.location_id}/payments/transactions") if settings.location_id else None,
            ("calls", "/v1/calls/"),
            ("calls", f"/v2/locations/{settings.location_id}/calls") if settings.location_id else None,
        ]
        paths = [p for p in paths if p]

        session = client.session
        for host in hosts:
            for label, path in paths:
                try:
                    params = dict(base_params)
                    if label == "opportunities" and settings.pipeline_ids:
                        params["pipelineId"] = list(settings.pipeline_ids)
                    r = session.get(f"{host}{path}", params=params, timeout=10)
                    status = r.status_code
                    ok = 200 <= status < 300
                    items = 0
                    try:
                        payload = r.json()
                        # rough count
                        for key in ("contacts", "opportunities", "messages", "transactions", "calls", "forms"):
                            if isinstance(payload.get(key), list):
                                items = len(payload.get(key))
                                break
                    except Exception:
                        pass
                    out["results"].append({"host": host, "path": path, "endpoint": label, "status": status, "ok": ok, "items": items})
                except Exception as e:
                    out["results"].append({"host": host, "path": path, "endpoint": label, "status": str(e), "ok": False, "items": 0})
    except Exception as e:
        out["error"] = str(e)
    return out


def show_dashboard() -> None:
    st.sidebar.header("Filtres")

    fast_mode = st.sidebar.checkbox("Mode rapide (30 jours)", value=True, help="Limite la période des appels API pour éviter les temps d'attente.")

    # Permettre de forcer l'utilisation du fichier local Excel/CSV
    use_local_tabular = st.sidebar.checkbox("Forcer source locale (Excel/CSV)", value=False,
                                            help="Ignore GHL/Google Sheets si coché et utilise base.xlsx/csv s'ils existent")
    source_choice = st.sidebar.radio(
        "Source des données",
        options=["Auto", "GHL API", "Google Sheets", "Local JSON"],
        index=0,
        help="Choisir la source: Auto (GHL puis fallback), GHL direct, Google Sheets ou JSON local."
    )

    # Auto-check GHL connectivity (période courte) une seule fois par session
    if not st.session_state.get("_startup_checked"):
        try:
            from datetime import timedelta
            end_short = date.today()
            start_short = end_short - timedelta(days=2)
            _df_test, diag_test = load_ghl_dataframe(start_short, end_short)
            st.session_state["_startup_diag"] = diag_test
        except Exception as e:
            st.session_state["_startup_diag"] = {"errors": [str(e)]}
        finally:
            st.session_state["_startup_checked"] = True

    # Afficher des conseils guidés si des endpoints sont vides
    if st.session_state.get("_startup_diag"):
        tips = _guided_tips(st.session_state["_startup_diag"])
        if tips:
            st.info("\n".join(f"• {t}" for t in tips))

    if st.sidebar.button("Actualiser les données"):
        try:
            load_ghl_dataframe.clear()
        except Exception:
            pass
        try:
            load_google_sheet_data.clear()
        except Exception:
            pass
        try:
            load_local_json_dataframe.clear()
        except Exception:
            pass
        st.toast("Caches vidés. Rechargement…")

    # Thème
    dark_selected = st.sidebar.checkbox("Mode sombre", value=is_dark_mode())
    st.session_state["dark_mode"] = dark_selected
    inject_theme_css()

    # Charger selon le choix
    df: pd.DataFrame
    if source_choice == "Auto":
        if use_local_tabular:
            df_local = load_local_tabular_data()
            if df_local.empty:
                st.warning("Aucun fichier local trouvé (base.xlsx/csv ou data.xlsx/csv) ou format invalide.")
                return
            st.session_state["data_source"] = "Local Excel/CSV (forcé)"
            st.session_state["diagnostics"] = {
                "source": "Local Excel/CSV",
                "counts": {"rows": int(len(df_local))},
            }
            df = df_local
        else:
            df = load_data(fast_days=(30 if fast_mode else None))
    elif source_choice == "GHL API":
        from datetime import timedelta
        start = end = None
        if fast_mode:
            end = date.today()
            start = end - timedelta(days=30)
        try:
            df_ghl, diag_ghl = load_ghl_dataframe(start, end)
            st.session_state["data_source"] = "GHL API"
            st.session_state["diagnostics"] = diag_ghl
            df = df_ghl
        except Exception as e:
            st.error(f"Erreur GHL: {e}")
            df = pd.DataFrame()
    elif source_choice == "Google Sheets":
        df_sheet = load_google_sheet_data()
        if not df_sheet.empty:
            st.session_state["data_source"] = "Google Sheets"
            st.session_state["diagnostics"] = {
                "source": "Google Sheets",
                "counts": {"sheet_rows": int(len(df_sheet))},
            }
        df = df_sheet
    else:  # Local JSON
        df_local, diag_local = load_local_json_dataframe()
        st.session_state["data_source"] = "Local JSON"
        st.session_state["diagnostics"] = diag_local
        df = df_local

    if df.empty:
        st.warning("Aucune donnée à afficher. Configurez GHL ou Google Sheets dans `.streamlit/secrets.toml`.")
        return

    # Date filter
    min_date: Optional[date] = pd.to_datetime(df["Date"]).dt.date.min() if "Date" in df else None
    max_date: Optional[date] = pd.to_datetime(df["Date"]).dt.date.max() if "Date" in df else None
    start = st.sidebar.date_input("Début", value=min_date or date.today())
    end = st.sidebar.date_input("Fin", value=max_date or date.today())
    compare_prev = st.sidebar.checkbox("Comparer à la période précédente", value=False)

    df_filtered = df.copy()
    if "Date" in df_filtered:
        df_filtered = df_filtered[(df_filtered["Date"].dt.date >= start) & (df_filtered["Date"].dt.date <= end)]

    # Setter filter
    if "Setter" in df_filtered:
        setters = sorted([s for s in df_filtered["Setter"].dropna().unique()])
        # Pré-sélectionner "Ingénieur Coffi" si présent, sinon tout
        try:
            preferred = _welcome_name()
        except Exception:
            preferred = None
        default_setters = [preferred] if preferred and preferred in setters else setters
        selected = st.sidebar.multiselect("Setters", options=setters, default=default_setters)
        if selected:
            df_filtered = df_filtered[df_filtered["Setter"].isin(selected)]

    st.caption(f"Source des données: {st.session_state.get('data_source', 'Inconnue')}")

    # Tableau de bord avec onglets
    display_kpis(df_filtered, start=start, end=end, compare_prev=compare_prev)

    tab_overview, tab_sources, tab_setters, tab_pipeline, tab_diag = st.tabs([
        "Vue d'ensemble", "Sources", "Setters", "Pipeline", "Diagnostics"
    ])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            revenue_trend_plot(df_filtered, start=start, end=end, compare_prev=compare_prev)
        with c2:
            daily_activity_plot(df_filtered)
        st.markdown("#### Heatmap d'activité")
        weekly_hour_heatmap(st.session_state.get("diagnostics") or {})

    with tab_sources:
        c3, c4 = st.columns(2)
        with c3:
            leads_by_source_plot(df_filtered)
        with c4:
            source_share_donut(df_filtered)

    with tab_setters:
        setter_performance_plot(df_filtered)

    with tab_pipeline:
        funnel_plot(df_filtered)

    with tab_diag:
        diagnostics_panel(df)


def _welcome_name() -> str:
    """Return first non-empty, non-comment line from .streamlit/Info.

    Prevents dumping long notes into the sidebar. Falls back to a generic
    label if file is missing or empty.
    """
    try:
        info_path = ".streamlit/Info"
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                return s
        return "Invité"
    except Exception:
        return "Invité"


def is_dark_mode() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def plotly_template() -> str:
    return "plotly_dark" if is_dark_mode() else "plotly_white"


def inject_theme_css() -> None:
    # Base tokens (light + dark)
    base_css = """
    <style>
      :root {
        --ke-radius: 12px;
        --ke-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 6px 12px rgba(0,0,0,0.06);
        --ke-card-bg: #ffffff;
        --ke-card-fg: #111827;
        --ke-muted: #6B7280;
      }
      .ke-card { background: var(--ke-card-bg); color: var(--ke-card-fg); border-radius: var(--ke-radius); box-shadow: var(--ke-shadow); padding: 10px 12px; }
      /* Compact metrics */
      div[data-testid="stMetric"] { padding: 8px 10px; border-radius: var(--ke-radius); box-shadow: var(--ke-shadow); }
      div[data-testid="stMetric"] label { color: var(--ke-muted); font-weight: 600; }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if not is_dark_mode():
        return
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #0F172A; color: #E5E7EB; }
        [data-testid="stSidebar"] { background-color: #111827; color: #E5E7EB; }
        .stMetric label, .stMetric [data-testid="stMetricValue"] { color: #F3F4F6 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_header() -> None:
    # Theming-aware banner
    if is_dark_mode():
        gradient_start = "#1F2937"
        gradient_end = "#0B1220"
        border_color = "#334155"
        title_color = "#F8FAFC"
    else:
        gradient_start = "#FFE4E6"
        gradient_end = "#FFFFFF"
        border_color = "#FEE2E2"
        title_color = "#1F2937"

    st.markdown(
        f"""
        <style>
        .ke-banner {{ padding: 12px 18px; background: linear-gradient(90deg, {gradient_start} 0%, {gradient_end} 70%); border: 1px solid {border_color}; border-radius: 12px; margin-bottom: 12px; }}
        .ke-banner-row {{ display: flex; align-items: center; gap: 16px; }}
        .ke-logo {{ height: 48px; width: auto; border-radius: 6px; }}
        .ke-title {{ margin: 0; font-weight: 700; color: {title_color}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        import base64 as _b64
        with open("kodjoenglish_logo.jpeg", "rb") as f:
            b64 = _b64.b64encode(f.read()).decode("ascii")
        st.markdown(
            f"""
            <div class=\"ke-banner\">
              <div class=\"ke-banner-row\">
                <img src=\"data:image/jpeg;base64,{b64}\" class=\"ke-logo\" />
                <h1 class=\"ke-title\">KodjoEnglish Sales Metrics</h1>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        cols = st.columns([1, 6, 1])
        with cols[0]:
            try:
                st.image("kodjoenglish_logo.jpeg", use_container_width=True)
            except Exception:
                pass
        with cols[1]:
            st.title("KodjoEnglish Sales Metrics")


def main() -> None:
    st.set_page_config(page_title="KodjoEnglish Sales Dashboard", layout="wide", page_icon="kodjoenglish_logo.jpeg")
    try:
        st.sidebar.image("kodjoenglish_logo.jpeg", use_container_width=True)
        st.sidebar.markdown("---")
    except Exception:
        pass
    st.sidebar.success(f"Bienvenue {_welcome_name()}")
    _render_header()
    show_dashboard()


if __name__ == "__main__":
    main()
