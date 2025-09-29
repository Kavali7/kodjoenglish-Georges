"""
Utilities to fetch and transform data from the GoHighLevel (GHL) REST API
into a tabular structure consumed by the KodjoEnglish dashboard.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, time as dt_time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://rest.gohighlevel.com"
API_VERSION = "2021-07-28"
PAGE_SIZE = 100
TIMEOUT_SECONDS = 15


class GHLConfigurationError(RuntimeError):
    """Raised when a required configuration value is missing."""


@dataclass
class GHLSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timezone: str = "UTC"
    # Optional: some endpoints require LocationId header
    location_id: Optional[str] = None
    pipeline_ids: Sequence[str] = field(default_factory=list)
    stage_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)
    form_ids: Sequence[str] = field(default_factory=list)
    product_ids: Mapping[str, Sequence[str]] = field(default_factory=dict)
    user_mapping: Mapping[str, str] = field(default_factory=dict)
    source_mapping: Mapping[str, str] = field(default_factory=dict)
    custom_source_field_id: Optional[str] = None
    link_indicators: Sequence[str] = field(default_factory=lambda: ("checkout", "pay", "http"))
    direct_inbound_sources: Sequence[str] = field(default_factory=lambda: ("direct", "inbound"))
    ready_to_pay_stage_ids: Sequence[str] = field(default_factory=list)
    source_counter_ids: Mapping[str, str] = field(default_factory=dict)
    history_days: int = 90
    page_size: int = PAGE_SIZE
    max_pages: int = 50
    request_pause: float = 0.3
    max_retries: int = 2
    retry_backoff: float = 1.0

    @staticmethod
    def from_streamlit(secrets: Mapping[str, Any]) -> "GHLSettings":
        if "ghl" not in secrets:
            raise GHLConfigurationError("Section 'ghl' missing from Streamlit secrets")
        config = secrets["ghl"]
        api_key = config.get("api_key")
        if not api_key:
            raise GHLConfigurationError("'ghl.api_key' is required in Streamlit secrets")

        # Try to obtain location id from secrets or infer from JWT if possible
        explicit_location = config.get("location_id") or config.get("locationId")
        inferred_location = None
        if not explicit_location:
            try:
                # JWT format: header.payload.sig (base64url). Extract payload and read location_id
                import json, base64

                parts = api_key.split(".")
                if len(parts) >= 2:
                    payload_b64 = parts[1] + "==="  # pad for base64url if needed
                    payload = base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8", errors="ignore")
                    data = json.loads(payload)
                    inferred_location = (
                        data.get("location_id")
                        or data.get("locationId")
                        or data.get("location")
                    )
            except Exception:
                inferred_location = None

        return GHLSettings(
            api_key=api_key,
            base_url=config.get("base_url", DEFAULT_BASE_URL),
            timezone=config.get("timezone", "UTC"),
            location_id=explicit_location or inferred_location,
            pipeline_ids=config.get("pipeline_ids", []) or [],
            stage_ids={k: _ensure_sequence(v) for k, v in (config.get("stage_ids") or {}).items()},
            form_ids=_ensure_sequence(config.get("form_ids", [])),
            product_ids={k: _ensure_sequence(v) for k, v in (config.get("product_ids") or {}).items()},
            user_mapping=config.get("user_mapping", {}) or {},
            source_mapping=config.get("source_mapping", {}) or {},
            source_counter_ids=_ensure_str_mapping(config.get("source_counter_ids")),

            custom_source_field_id=config.get("custom_source_field_id"),
            link_indicators=_ensure_sequence(config.get("link_indicators", ("checkout", "pay", "http"))),
            direct_inbound_sources=_ensure_sequence(config.get("direct_inbound_sources", ("direct", "inbound"))),
            ready_to_pay_stage_ids=_ensure_sequence(config.get("ready_to_pay_stage_ids", [])),
            page_size=int(config.get("page_size", PAGE_SIZE)),
            max_pages=int(config.get("max_pages", 50)),
            request_pause=float(config.get("request_pause", 0.3)),
            max_retries=int(config.get("max_retries", 2)),
            retry_backoff=float(config.get("retry_backoff", 1.0)),
        )


def _ensure_sequence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _ensure_str_mapping(mapping: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not mapping:
        return {}
    return {str(k): str(v) for k, v in mapping.items()}


class GHLClient:
    """Lightweight wrapper around the GHL REST API with pagination helpers."""

    def __init__(self, settings: GHLSettings, session: Optional[requests.Session] = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {settings.api_key}",
                "Version": API_VERSION,
            }
        )
        # Some GHL endpoints require the LocationId header
        if getattr(settings, "location_id", None):
            self.session.headers["LocationId"] = str(settings.location_id)

    def fetch_paginated(
        self,
        path: str,
        *,
        params: Optional[MutableMapping[str, Any]] = None,
        data_keys: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        params = params.copy() if params else {}
        params.setdefault("limit", getattr(self.settings, "page_size", PAGE_SIZE))
        params.setdefault("offset", 0)

        max_pages = max(1, getattr(self.settings, "max_pages", 50))
        pause = max(0.0, getattr(self.settings, "request_pause", 0.3))
        retries = max(0, getattr(self.settings, "max_retries", 2))
        backoff = max(0.0, getattr(self.settings, "retry_backoff", 1.0))

        # Candidate hosts: configured base + known alternates (deduplicated)
        base = self.settings.base_url.rstrip("/")
        candidates = [base]
        for alt in ("https://rest.gohighlevel.com", "https://services.leadconnectorhq.com"):
            alt = alt.rstrip("/")
            if alt not in candidates:
                candidates.append(alt)

        last_error: Optional[requests.HTTPError] = None

        for host in candidates:
            page_count = 0
            current_params = params.copy()
            results: List[Dict[str, Any]] = []
            while True:
                host_404 = False
                attempt = 0
                while True:
                    response = self.session.get(
                        f"{host}{path}", params=current_params, timeout=TIMEOUT_SECONDS
                    )
                    try:
                        response.raise_for_status()
                        break
                    except requests.HTTPError as exc:
                        status = getattr(exc.response, "status_code", None)
                        logger.error("GHL request failed: %s", exc)
                        # Try retries for transient errors
                        if status in (429, 500, 502, 503, 504) and attempt < retries:
                            sleep_for = backoff * (2 ** attempt)
                            time.sleep(sleep_for)
                            attempt += 1
                            continue
                        # If 404, break this host and try next candidate
                        if status == 404:
                            last_error = exc
                            results = []
                            host_404 = True
                            break
                        # Other errors: give up for this host
                        last_error = exc
                        results = []
                        break

                if host_404:
                    # Switch to next host
                    break

                payload = response.json()
                items = _extract_items(payload, data_keys)
                if not items:
                    break

                results.extend(items)

                page_count += 1
                if page_count >= max_pages:
                    logger.warning(
                        "Reached max_pages=%s for %s; stopping pagination.", max_pages, path
                    )
                    break

                if len(items) < current_params["limit"]:
                    break
                current_params["offset"] += current_params["limit"]
                if pause:
                    time.sleep(pause)

            if results:
                return results

        # If all hosts failed or returned empty
        return []

    def get_contacts(self, start: date, end: date) -> List[Dict[str, Any]]:
        params = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
        }
        # Try v1 then v2 scoped to location
        paths = [
            "/v1/contacts/",
        ]
        if self.settings.location_id:
            paths.append(f"/v2/locations/{self.settings.location_id}/contacts/search")
        for p in paths:
            try:
                items = self.fetch_paginated(p, params=params, data_keys=("contacts", "items"))
                if items:
                    return items
            except requests.HTTPError:
                continue
        return []

    def get_opportunities(self, start: date, end: date) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
        }
        if self.settings.pipeline_ids:
            # Use repeated query params for multiple pipelines
            params["pipelineId"] = list(self.settings.pipeline_ids)
        # Try common variants depending on account routing
        paths = [
            "/v1/opportunities/",
            "/v1/opportunities/search",
        ]
        if self.settings.location_id:
            paths.insert(1, f"/v2/locations/{self.settings.location_id}/opportunities/search")
        for p in paths:
            try:
                items = self.fetch_paginated(p, params=params, data_keys=("opportunities", "items"))
                if items:
                    return items
            except requests.HTTPError:
                continue
        return []

    def get_forms(self, start: date, end: date) -> List[Dict[str, Any]]:
        # Some accounts require explicit formId(s) for submissions; avoid noisy 400 if none configured
        if not self.settings.form_ids:
            return []
        params: Dict[str, Any] = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
            "formId": list(self.settings.form_ids),
        }
        # Try both marketing and forms endpoints
        for p in ("/v1/forms/submissions", "/v1/marketing/forms/submissions"):
            try:
                items = self.fetch_paginated(
                    p,
                    params=params,
                    data_keys=("forms", "submissions", "items"),
                )
                if items:
                    return items
            except requests.HTTPError:
                continue
        return []

    def get_messages(self, start: date, end: date) -> List[Dict[str, Any]]:
        params = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
        }
        for p in ("/v1/conversations/messages/", "/v1/conversations/messages/search"):
            try:
                items = self.fetch_paginated(p, params=params, data_keys=("messages", "items"))
                if items:
                    return items
            except requests.HTTPError:
                continue
        return []

    def get_payments(self, start: date, end: date) -> List[Dict[str, Any]]:
        params = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
        }
        for p in ("/v1/payments/transactions/", (
            f"/v2/locations/{self.settings.location_id}/payments/transactions" if self.settings.location_id else None
        )):
            if not p:
                continue
            try:
                items = self.fetch_paginated(p, params=params, data_keys=("transactions", "items"))
                if items:
                    return items
            except requests.HTTPError:
                continue
        return []

    def get_calls(self, start: date, end: date) -> List[Dict[str, Any]]:
        params = {
            "startDate": start.isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
        }
        for p in ("/v1/calls/", (
            f"/v2/locations/{self.settings.location_id}/calls" if self.settings.location_id else None
        )):
            if not p:
                continue
            try:
                items = self.fetch_paginated(p, params=params, data_keys=("calls", "items"))
                if items:
                    return items
            except requests.HTTPError:
                continue
        return []


# --- Transformation layer -------------------------------------------------

METRIC_COLUMNS = [
    "Setter",
    "Date",
    "From_Forms",
    "From_Direct",
    "From_FB",
    "From_Insta",
    "From_TikTok",
    "From_LinkedIn",
    "Total_Leads",
    "Follow_ups",
    "Initial_msg_Sent",
    "Offers_Made",
    "Links_Sent",
    "Total_Closes",
    "Calls_Connected",
    "Revenue_Generated",
    "Forms_Submitted",
    "Total_Filled",
    "Forms_Calls_Connected",
    "First_Offer",
    "Upsell_Offer",
    "Downsell_Offer",
    "Direct_Inbound",
    "Response_Rate",
    "Offer_Rate",
    "Link_Rate",
    "Close_Rate",
    "Calls_Rate",
    "Leads_Contacted",
    "Ready_to_Pay",
]


def build_dashboard_dataframe(
    client: GHLClient,
    settings: GHLSettings,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    start, end = _resolve_dates(start_date, end_date, settings.history_days)
    tz = _get_timezone(settings.timezone)

    def _safe_get(fetch_fn, label: str) -> pd.DataFrame:
        try:
            df = _to_dataframe(fetch_fn(start, end))
            if diag is not None:
                counts = diag.setdefault("counts", {})
                counts[label] = int(len(df))
            return df
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", label, e)
            if diag is not None:
                errors = diag.setdefault("errors", [])
                errors.append({"endpoint": label, "error": str(e)})
            return pd.DataFrame()

    contacts = _safe_get(client.get_contacts, "contacts")
    opportunities = _safe_get(client.get_opportunities, "opportunities")
    forms = _safe_get(client.get_forms, "forms")
    messages = _safe_get(client.get_messages, "messages")
    payments = _safe_get(client.get_payments, "payments")
    calls = _safe_get(client.get_calls, "calls")

    logger.info(
        "Fetched %s contacts, %s opportunities, %s forms, %s messages, %s payments, %s calls",
        len(contacts),
        len(opportunities),
        len(forms),
        len(messages),
        len(payments),
        len(calls),
    )

    # Build activity heatmap diagnostics (day-of-week x hour) if possible
    if diag is not None:
        try:
            def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            def _to_hour_series(df: pd.DataFrame, candidates: Sequence[str]):
                col = _first_present(df, candidates)
                if not col:
                    return pd.Series(dtype="datetime64[ns, UTC]")
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                ts = ts.dropna()
                if ts.empty:
                    return ts
                local = ts.dt.tz_convert(tz)
                return local.dt.floor("H")

            all_ts = []
            if not contacts.empty:
                all_ts.append(_to_hour_series(contacts, ("dateAdded", "createdAt")))
            if not opportunities.empty:
                all_ts.append(_to_hour_series(opportunities, ("updatedAt", "dateUpdated", "createdAt")))
            if not forms.empty:
                all_ts.append(_to_hour_series(forms, ("dateAdded", "createdAt", "submittedOn")))
            if not messages.empty:
                all_ts.append(_to_hour_series(messages, ("dateCreated", "createdOn", "timestamp")))
            if not payments.empty:
                all_ts.append(_to_hour_series(payments, ("dateCreated", "createdAt", "paidOn")))
            if not calls.empty:
                all_ts.append(_to_hour_series(calls, ("startTime", "dateCreated", "createdAt")))

            if all_ts:
                import numpy as np
                ts_all = pd.concat(all_ts).dropna()
                if not ts_all.empty:
                    dow = ts_all.dt.dayofweek.astype(int)
                    hr = ts_all.dt.hour.astype(int)
                    df_counts = pd.DataFrame({"dow": dow, "hour": hr})
                    mat = np.zeros((7, 24), dtype=int)
                    for (d, h), cnt in df_counts.value_counts().items():
                        mat[int(d), int(h)] = int(cnt)
                    diag["activity_heatmap"] = {
                        "matrix": mat.tolist(),
                        "day_index": list(range(7)),
                        "hour_index": list(range(24)),
                    }
        except Exception as _e:
            # best-effort; do not fail build if diagnostics aggregation errors
            pass

    # Prepare base index (Date x Setter)
    setters = _collect_setters([contacts, opportunities, forms, messages, payments, calls], settings=settings)
    if not setters:
        setters = ["Unknown"]
    index = pd.MultiIndex.from_product(
        (_date_range(start, end, tz), sorted(setters)),
        names=["Date", "Setter"],
    )
    df = pd.DataFrame(0, index=index, columns=METRIC_COLUMNS[2:])
    df = df.reset_index()

    _populate_sources(df, contacts, settings, tz)
    _populate_messages(df, messages, settings, tz)
    _populate_opportunities(df, opportunities, settings, tz)
    _populate_forms(df, forms, settings, tz)
    _populate_payments(df, payments, settings, tz)
    _populate_calls(df, calls, settings, tz)
    _compute_totals_and_rates(df)

    # Ensure columns order and types
    df = df[METRIC_COLUMNS]
    df.sort_values(["Date", "Setter"], inplace=True)

    if diag is not None:
        try:
            diag["period"] = {"start": start.isoformat(), "end": end.isoformat()}
            diag["pipelines"] = list(settings.pipeline_ids)
            # Setters diagnostics
            rows_by_setter = df["Setter"].value_counts(dropna=False).to_dict()
            leads_by_setter = (
                df.groupby("Setter")["Total_Leads"].sum().astype(int).to_dict()
                if "Total_Leads" in df.columns
                else {}
            )
            diag["setters_rows"] = rows_by_setter
            diag["setters_total_leads"] = leads_by_setter
        except Exception as _:
            pass
    return df


# --- Helpers ---------------------------------------------------------------

def _resolve_dates(start: Optional[date], end: Optional[date], history_days: int) -> tuple[date, date]:
    today = datetime.now(timezone.utc).date()
    if end is None:
        end = today
    if start is None:
        start = end - timedelta(days=30)
    if start > end:
        raise ValueError("start_date must be before end_date")
    return start, end


def _get_timezone(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        # fallback to UTC
        logger.warning("Unknown timezone '%s', falling back to UTC", name)
        return timezone.utc


def _to_dataframe(items: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    return pd.json_normalize(items)


def _collect_setters(dfs: Sequence[pd.DataFrame], settings: GHLSettings) -> List[str]:
    names: set[str] = set()
    user_map = settings.user_mapping
    for df in dfs:
        if df.empty:
            continue
        for col in ("assignedUserId", "assignedTo", "userId", "ownerId", "createdBy"):
            if col in df.columns:
                names.update(user_map.get(val, str(val)) for val in df[col].dropna().unique())
    # Always include a generic bucket to receive rows that have no clear owner
    names.add("Unknown")
    return sorted(n for n in names if n and n != "nan")


def _date_range(start: date, end: date, tz) -> Iterable[date]:
    cursor = datetime.combine(start, dt_time.min, tzinfo=timezone.utc).astimezone(tz).date()
    final = datetime.combine(end, dt_time.min, tzinfo=timezone.utc).astimezone(tz).date()
    while cursor <= final:
        yield cursor
        cursor += timedelta(days=1)


def _map_setter(raw_value: Any, settings: GHLSettings) -> str:
    if raw_value is None:
        return "Unknown"
    return settings.user_mapping.get(str(raw_value), settings.user_mapping.get(raw_value, str(raw_value)))


def _normalize_datetime(series: pd.Series, tz) -> pd.Series:
    if series.empty:
        return series
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(tz).dt.date


def _populate_sources(df: pd.DataFrame, contacts: pd.DataFrame, settings: GHLSettings, tz) -> None:
    if contacts.empty:
        return
    contacts = contacts.copy()
    # Choose appropriate datetime column without triggering Series truthiness
    date_col = "dateAdded" if "dateAdded" in contacts.columns else ("createdAt" if "createdAt" in contacts.columns else None)
    if not date_col:
        return
    contacts["Date"] = _normalize_datetime(contacts[date_col], tz)
    setter_col = None
    for c in ("assignedUserId", "assignedTo", "ownerId", "createdBy"):
        if c in contacts.columns:
            setter_col = c
            break
    if setter_col:
        contacts["Setter"] = contacts[setter_col].apply(lambda v: _map_setter(v, settings))
    else:
        contacts["Setter"] = "Unknown"

    custom_fields_col = next((col for col in ("customFields", "customField", "custom_fields") if col in contacts.columns), None)

    if settings.source_counter_ids and custom_fields_col:
        if _populate_sources_from_counters(df, contacts, settings, custom_fields_col):
            return

    if settings.custom_source_field_id and custom_fields_col:
        contacts["lead_source"] = contacts[custom_fields_col].apply(
            lambda fields: _extract_custom_field(fields, settings.custom_source_field_id)
        )
    else:
        base_source = contacts.get("source")
        if base_source is not None:
            contacts["lead_source"] = base_source
        else:
            contacts["lead_source"] = "unknown"

    contacts["lead_source"] = contacts["lead_source"].fillna("unknown").astype(str).str.lower()
    source_mapping = {k.lower(): v for k, v in settings.source_mapping.items()}

    for label, column in (
        ("forms", "From_Forms"),
        ("direct", "From_Direct"),
        ("facebook", "From_FB"),
        ("instagram", "From_Insta"),
        ("tiktok", "From_TikTok"),
        ("linkedin", "From_LinkedIn"),
    ):
        mapped_values = {label}
        mapped_values.update({k for k, v in source_mapping.items() if v == column})
        mask = contacts["lead_source"].isin(mapped_values)
        _aggregate_assign(df, contacts[mask], column)

    _aggregate_assign(df, contacts, "Total_Leads")

    direct_values = {s.lower() for s in settings.direct_inbound_sources}
    mask_direct = contacts["lead_source"].isin(direct_values)
    _aggregate_assign(df, contacts[mask_direct], "Direct_Inbound")


def _populate_sources_from_counters(
    df: pd.DataFrame,
    contacts: pd.DataFrame,
    settings: GHLSettings,
    custom_fields_col: str,
) -> bool:
    counter_ids = settings.source_counter_ids
    if not counter_ids:
        return False

    base = contacts.dropna(subset=["Date"]).copy()
    if base.empty:
        return False

    used = False
    for label, field_id in counter_ids.items():
        values = base[custom_fields_col].apply(lambda fields: _extract_custom_field(fields, field_id))
        numeric = pd.to_numeric(values, errors="coerce").fillna(0)
        if numeric.sum() <= 0:
            continue
        used = True
        aggregated = (
            base.assign(_counter=numeric)
            .groupby(["Date", "Setter"], dropna=False)["_counter"]
            .sum()
            .reset_index()
        )
        column = label if label in df.columns else label
        for _, row in aggregated.iterrows():
            _add_value(df, row["Date"], row["Setter"], column, row["_counter"])
            if column == "From_Direct":
                _add_value(df, row["Date"], row["Setter"], "Direct_Inbound", row["_counter"])
    return used



def _populate_messages(df: pd.DataFrame, messages: pd.DataFrame, settings: GHLSettings, tz) -> None:
    if messages.empty:
        return
    msgs = messages.copy()
    date_col = next((col for col in ("dateCreated", "createdOn", "timestamp") if col in msgs), None)
    if not date_col:
        return

    msgs["Date"] = _normalize_datetime(msgs[date_col], tz)
    if "userId" in msgs:
        msgs["Setter"] = msgs["userId"].apply(lambda v: _map_setter(v, settings))
    else:
        msgs["Setter"] = "Unknown"

    if "direction" in msgs:
        msgs["direction"] = msgs["direction"].str.lower()
    else:
        msgs["direction"] = "outbound"

    outbound = msgs[msgs["direction"].isin(["outbound", "outgoing", "sent"])].copy()
    inbound = msgs[msgs["direction"].isin(["inbound", "incoming", "received"])].copy()

    contact_col = "contactId" if "contactId" in msgs else None

    if contact_col:
        outbound = outbound.dropna(subset=[contact_col])
        inbound = inbound.dropna(subset=[contact_col])

        sort_cols = [contact_col, date_col]
        outbound_sorted = outbound.sort_values(sort_cols)
        first_outbound = outbound_sorted.drop_duplicates(subset=[contact_col], keep="first")
        _aggregate_assign(df, first_outbound, "Initial_msg_Sent")

        key_col = next((col for col in ("id", "_id", "messageId") if col in outbound.columns), None)
        follow_ups = outbound.iloc[0:0]
        if key_col and key_col in first_outbound:
            first_ids = set(first_outbound[key_col].dropna())
            follow_ups = outbound[~outbound[key_col].isin(first_ids)]
        if not follow_ups.empty:
            _aggregate_assign(df, follow_ups, "Follow_ups")
        else:
            if not outbound.empty:
                counts_all = outbound.groupby(["Date", "Setter"]).size()
                counts_first = first_outbound.groupby(["Date", "Setter"]).size()
                diff = (counts_all - counts_first).reset_index(name="value")
                diff = diff[diff["value"] > 0]
                for _, row in diff.iterrows():
                    _add_value(df, row["Date"], row["Setter"], "Follow_ups", row["value"])

        _aggregate_assign(df, outbound, "Leads_Contacted", unique_contacts=True)

        outbound_counts = outbound.groupby(["Date", "Setter"])[contact_col].nunique().reset_index(name="outbound_contacts")
        inbound_counts = inbound.groupby(["Date", "Setter"])[contact_col].nunique().reset_index(name="inbound_contacts")
        merged = outbound_counts.merge(inbound_counts, on=["Date", "Setter"], how="left").fillna(0)
        for _, row in merged.iterrows():
            value = (row["inbound_contacts"] / row["outbound_contacts"] * 100) if row["outbound_contacts"] else 0
            _set_value(df, row["Date"], row["Setter"], "Response_Rate", round(value, 1))
    else:
        _aggregate_assign(df, outbound, "Initial_msg_Sent")

    if not outbound.empty:
        text_col = next((col for col in ("body", "text", "message", "content") if col in outbound.columns), None)
        if text_col:
            indicators = tuple(ind.lower() for ind in settings.link_indicators)
            if indicators:
                mask = outbound[text_col].astype(str).str.lower().apply(lambda text: any(ind in text for ind in indicators))
                links = outbound[mask]
                _aggregate_assign(df, links, "Links_Sent")

def _populate_opportunities(df: pd.DataFrame, opportunities: pd.DataFrame, settings: GHLSettings, tz) -> None:

    if opportunities.empty:
        return
    opps = opportunities.copy()
    date_col = None
    for candidate in ("updatedAt", "dateUpdated", "createdAt"):
        if candidate in opps:
            date_col = candidate
            break
    if not date_col:
        return

    opps["Date"] = _normalize_datetime(opps[date_col], tz)
    opps["Setter"] = opps.get("assignedUserId").apply(lambda v: _map_setter(v, settings)) if "assignedUserId" in opps else "Unknown"

    # closes: status won
    mask_won = opps.get("status").str.lower().eq("won") if "status" in opps else False
    if mask_won is not False:
        _aggregate_assign(df, opps[mask_won], "Total_Closes")

    offers_stage_ids = set(settings.stage_ids.get("offers_made", []))
    if offers_stage_ids and "pipelineStageId" in opps:
        offers = opps[opps["pipelineStageId"].isin(offers_stage_ids)]
        _aggregate_assign(df, offers, "Offers_Made")

    ready_stage_ids = set(settings.ready_to_pay_stage_ids)
    if ready_stage_ids and "pipelineStageId" in opps:
        ready = opps[opps["pipelineStageId"].isin(ready_stage_ids)]
        _aggregate_assign(df, ready, "Ready_to_Pay")


def _populate_forms(df: pd.DataFrame, forms: pd.DataFrame, settings: GHLSettings, tz) -> None:
    if forms.empty:
        return
    forms = forms.copy()
    date_col = None
    for candidate in ("dateAdded", "createdAt", "submittedOn"):
        if candidate in forms:
            date_col = candidate
            break
    if not date_col:
        return

    forms["Date"] = _normalize_datetime(forms[date_col], tz)
    forms["Setter"] = forms.get("assignedUserId").apply(lambda v: _map_setter(v, settings)) if "assignedUserId" in forms else "Unknown"

    _aggregate_assign(df, forms, "Forms_Submitted")

    if "status" in forms:
        filled = forms[forms["status"].str.lower().isin(["completed", "filled", "submitted"])]
        _aggregate_assign(df, filled, "Total_Filled")


def _populate_payments(df: pd.DataFrame, payments: pd.DataFrame, settings: GHLSettings, tz) -> None:
    if payments.empty:
        return
    payments = payments.copy()
    date_col = None
    for candidate in ("dateCreated", "createdAt", "paidOn"):
        if candidate in payments:
            date_col = candidate
            break
    if not date_col:
        return

    payments["Date"] = _normalize_datetime(payments[date_col], tz)
    payments["Setter"] = payments.get("userId").apply(lambda v: _map_setter(v, settings)) if "userId" in payments else "Unknown"

    if "status" in payments:
        payments = payments[payments["status"].str.lower().isin(["paid", "succeeded", "complete", "completed"])]

    amount_col = "amount"
    if amount_col not in payments:
        # sometimes stored under total
        if "total" in payments:
            amount_col = "total"
        else:
            payments["amount"] = 0
            amount_col = "amount"

    payments[amount_col] = pd.to_numeric(payments[amount_col], errors="coerce").fillna(0)
    revenue = payments.groupby(["Date", "Setter"])[amount_col].sum().reset_index()
    for _, row in revenue.iterrows():
        _add_value(df, row["Date"], row["Setter"], "Revenue_Generated", row[amount_col])

    for label, column in (
        ("first", "First_Offer"),
        ("upsell", "Upsell_Offer"),
        ("downsell", "Downsell_Offer"),
    ):
        product_ids = set(settings.product_ids.get(label, []))
        if not product_ids or "productId" not in payments:
            continue
        subset = payments[payments["productId"].isin(product_ids)]
        sums = subset.groupby(["Date", "Setter"])[amount_col].sum().reset_index()
        for _, row in sums.iterrows():
            _add_value(df, row["Date"], row["Setter"], column, row[amount_col])


def _populate_calls(df: pd.DataFrame, calls: pd.DataFrame, settings: GHLSettings, tz) -> None:
    if calls.empty:
        return
    calls = calls.copy()
    date_col = None
    for candidate in ("startTime", "dateCreated", "createdAt"):
        if candidate in calls:
            date_col = candidate
            break
    if not date_col:
        return

    calls["Date"] = _normalize_datetime(calls[date_col], tz)
    calls["Setter"] = calls.get("userId").apply(lambda v: _map_setter(v, settings)) if "userId" in calls else "Unknown"

    if "status" in calls:
        connected = calls[calls["status"].str.lower().isin(["connected", "answered", "completed"])]
    else:
        connected = calls
    _aggregate_assign(df, connected, "Calls_Connected")

    if "formId" in calls:
        _aggregate_assign(df, connected[calls["formId"].notna()], "Forms_Calls_Connected")


def _aggregate_assign(df: pd.DataFrame, data: pd.DataFrame, column: str, *, unique_contacts: bool = False) -> None:
    if data.empty:
        return
    working = data.copy()
    if "Date" not in working.columns:
        return
    if "Setter" not in working.columns:
        working["Setter"] = "Unknown"

    if unique_contacts and "contactId" in working:
        grouped = working.dropna(subset=["contactId"]).drop_duplicates(["Date", "Setter", "contactId"])
        counts = grouped.groupby(["Date", "Setter"]).size().reset_index(name="value")
    else:
        counts = working.groupby(["Date", "Setter"]).size().reset_index(name="value")

    for _, row in counts.iterrows():
        _add_value(df, row["Date"], row["Setter"], column, row["value"])


def _add_value(df: pd.DataFrame, date_value: Any, setter: Any, column: str, amount: float) -> None:
    mask = (df["Date"] == date_value) & (df["Setter"] == setter)
    if not mask.any():
        return
    df.loc[mask, column] = df.loc[mask, column] + amount


def _set_value(df: pd.DataFrame, date_value: Any, setter: Any, column: str, value: float) -> None:
    mask = (df["Date"] == date_value) & (df["Setter"] == setter)
    if not mask.any():
        return
    df.loc[mask, column] = value

def _compute_totals_and_rates(df: pd.DataFrame) -> None:
    df["Total_Leads"] = (
        df[["From_Forms", "From_Direct", "From_FB", "From_Insta", "From_TikTok", "From_LinkedIn"]]
        .sum(axis=1)
    )

    df["Offer_Rate"] = _safe_rate(df["Offers_Made"], df["Total_Leads"])
    df["Link_Rate"] = _safe_rate(df["Links_Sent"], df["Offers_Made"])
    df["Close_Rate"] = _safe_rate(df["Total_Closes"], df["Offers_Made"])
    df["Calls_Rate"] = _safe_rate(df["Calls_Connected"], df["Total_Leads"])

    # Response rate can be pre-filled during message processing; ensure numeric
    df["Response_Rate"] = pd.to_numeric(df["Response_Rate"], errors="coerce").fillna(0)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: pd.NA})
    rate = (numerator / denom) * 100
    return rate.fillna(0).round(1)


def _extract_custom_field(custom_fields: Any, field_id: str) -> Optional[str]:
    if not custom_fields:
        return None
    if isinstance(custom_fields, list):
        for item in custom_fields:
            if isinstance(item, Mapping):
                if item.get("id") == field_id:
                    return item.get("value")
    elif isinstance(custom_fields, Mapping):
        return custom_fields.get(field_id)
    return None


def _extract_items(payload: Mapping[str, Any], data_keys: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if data_keys:
        for key in data_keys:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    # Fallback: first list in payload
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []


__all__ = [
    "GHLClient",
    "GHLSettings",
    "GHLConfigurationError",
    "build_dashboard_dataframe",
]









