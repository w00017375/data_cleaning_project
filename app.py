"""
Data Analysis Studio - A Streamlit application for data exploration,
cleaning, visualization, and reporting.

Tabs:
    1. Upload & Overview  - load files (CSV/Excel/JSON/Google Sheets) and view summary
    2. Cleaning & Prep    - filter, transform, log every step into a reusable recipe
    3. Visualization      - build charts from the cleaned dataframe
    4. Report & Export    - export cleaned data, recipe and a summary report
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

# Plotting / Google Sheets are optional (handled gracefully if missing)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe inside Streamlit
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

try:
    import gspread
    GSPREAD_OK = True
except Exception:
    GSPREAD_OK = False

try:
    import statsmodels.api as _sm  # noqa: F401  (needed by plotly OLS trendline)
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend; Streamlit serves the figure
    import matplotlib.pyplot as _plt  # noqa: F401  (imported lazily where used)
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

try:
    from openai import OpenAI as _OpenAIClient
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# Load environment variables from a .env file in the project root (if any).
# This lets users keep their GROQ_API_KEY (or any other secret) in a local
# .env file instead of pasting it into the UI every time.
try:
    from dotenv import load_dotenv
    load_dotenv()  # silently does nothing if no .env file is found
    DOTENV_OK = True
except Exception:
    DOTENV_OK = False


# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Analysis Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.caption("Build version: 2026-06-07-final-check")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #f3f4f6;
            color: #dc2626 !important;
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #dc2626 !important;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #dc2626 !important;
            border-bottom: 3px solid #dc2626 !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #dc2626 !important;
        }
        .metric-card {
            background: linear-gradient(135deg,#1e3a8a 0%,#2563eb 100%);
            padding: 18px; border-radius: 10px; color:#fff;
        }
        .step-box {
            background:#0f172a; color:#f1f5f9;
            border-left:4px solid #ef4444;
            padding:10px 14px; margin:6px 0; border-radius:6px;
            font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
            font-size:.85rem; line-height:1.55;
            box-shadow: 0 1px 2px rgba(0,0,0,.08);
        }
        .step-box b { color:#fca5a5; }
        .step-box .ts { color:#94a3b8; font-size:.78rem; margin-left:8px; }
        .step-box .params { color:#cbd5e1; word-break: break-word; }
        .step-box .rows { color:#bef264; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults: dict[str, Any] = {
        "df_original": None,   # the raw dataframe, never mutated
        "df": None,            # the working dataframe
        "recipe": [],          # list of logged cleaning steps
        "file_name": None,
        "load_error": None,
        "last_action": None,   # {"msg": str, "ts": iso-str} — flashed after rerun
        "pending_delete": None,  # list[str] of columns awaiting delete confirmation
        "pending_drop_sparse": None,  # pending sparse-column deletion confirmation
        "charts": [],            # list of created chart configs (Visualization tab)
        "chart_seq": 0,          # monotonic id generator for charts
        "ai_enabled": False,     # toggle for AI suggestions (Groq)
        "ai_suggestions": [],    # list of suggestion dicts from the model
        "ai_pending": None,      # suggestion awaiting create-confirmation
        "ai_error": None,        # last error to display (dict: title, detail)
        "ai_source": None,       # "ai" | "fallback" | "fallback_manual" | None
        "validation_rules": [],  # list of validation rule dicts
        "validation_results": None,  # last validation run output (dict)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session() -> None:
    for k in ("df_original", "df", "recipe", "file_name", "load_error",
              "last_action", "pending_delete", "pending_drop_sparse",
              "charts", "chart_seq",
              "validation_rules", "validation_results"):
        if k in ("recipe", "charts", "validation_rules"):
            st.session_state[k] = []
        elif k == "chart_seq":
            st.session_state[k] = 0
        else:
            st.session_state[k] = None


def flash(msg: str, tab: str = "tab2") -> None:
    """Store a message to display once on the next rerun, scoped to a tab."""
    st.session_state.last_action = {
        "msg": msg,
        "ts": datetime.now().strftime("%H:%M:%S"),
        "tab": tab,
    }


def show_flash(tab: str) -> None:
    """Display and clear the pending flash message if it belongs to `tab`."""
    la = st.session_state.last_action
    if la and la.get("tab", "tab2") == tab:
        st.success(f"✅ {la['msg']} _(at {la['ts']})_")
        st.session_state.last_action = None


def _infer_affected_columns(params: dict) -> list:
    """Best-effort extraction of columns touched by an operation, based on
    common parameter keys used across the cleaning steps."""
    cols: list = []
    # Singular column keys
    for k in ("column", "name", "group_by"):
        v = params.get(k)
        if isinstance(v, str):
            cols.append(v)
    # Plural / list-style keys
    for k in ("columns", "new_columns"):
        v = params.get(k)
        if isinstance(v, list):
            cols.extend(c for c in v if isinstance(c, str))
    # Mappings (rename, replace_values, redetect_dtypes, etc.)
    for k in ("mapping", "changes"):
        v = params.get(k)
        if isinstance(v, dict):
            cols.extend(v.keys())
    # Filters specification used by the multi-level filter step
    if isinstance(params.get("filters"), list):
        for f in params["filters"]:
            c = f.get("column") or f.get("col")
            if isinstance(c, str):
                cols.append(c)
    # De-duplicate while preserving order
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def log_step(action: str, params: dict, rows_before: int, rows_after: int,
             affected_columns: list | None = None) -> None:
    """
    Append a single transformation step to the recipe.

    The record follows the canonical structure:
        operation          (string)  — what was done
        parameters         (dict)    — how it was configured
        affected_columns   (list)    — which columns were touched
        timestamp          (string)  — UTC ISO8601, second precision

    Row counts (before/after/changed) are also recorded for the UI but
    aren't part of the canonical schema (they're derived state, not input).
    """
    cols = affected_columns if affected_columns is not None else _infer_affected_columns(params)
    st.session_state.recipe.append(
        {
            "operation": action,
            "parameters": params,
            "affected_columns": cols,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            # Auxiliary fields:
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "rows_changed": int(rows_before - rows_after),
        }
    )


init_state()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_uploaded_file(uploaded) -> pd.DataFrame:
    """Read CSV / Excel / JSON from a Streamlit UploadedFile. Raises on failure."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    if name.endswith(".json"):
        # try records-orient first, then fall back to generic
        raw = uploaded.read()
        try:
            return pd.read_json(io.BytesIO(raw))
        except ValueError:
            data = json.loads(raw.decode("utf-8"))
            return pd.json_normalize(data)
    raise ValueError(
        f"Unsupported file type: '{uploaded.name}'. "
        "Allowed: .csv, .xlsx, .xls, .json"
    )


def load_google_sheet(url: str) -> pd.DataFrame:
    """
    Load a public Google Sheet via its 'export?format=csv' endpoint.
    For private sheets a service-account JSON would be required (gspread).
    """
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not find a spreadsheet ID in that URL.")
    sheet_id = m.group(1)

    gid_match = re.search(r"[#&?]gid=([0-9]+)", url)
    gid = gid_match.group(1) if gid_match else "0"

    export_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
        f"?format=csv&gid={gid}"
    )
    try:
        return pd.read_csv(export_url)
    except Exception as e:
        raise ValueError(
            "Could not read the sheet. Make sure link-sharing is enabled "
            f"('Anyone with the link – Viewer'). Details: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Overview helpers
# ---------------------------------------------------------------------------
def df_fingerprint(df: pd.DataFrame) -> tuple:
    """
    Cheap, stable cache key for a dataframe. Used as a hashable proxy so
    @st.cache_data doesn't have to walk the entire frame on every call.

    Encoding: (n_rows, n_cols, columns_tuple, dtypes_tuple, recipe_len).
    Whenever any cleaning step has been applied the recipe length changes,
    which invalidates dependent caches.
    """
    try:
        recipe_len = len(st.session_state.get("recipe", []))
    except Exception:
        recipe_len = 0
    return (
        int(len(df)),
        int(df.shape[1]),
        tuple(str(c) for c in df.columns),
        tuple(str(t) for t in df.dtypes),
        int(recipe_len),
    )


def apply_recipe(df: pd.DataFrame, recipe: list[dict]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Replay a transformation recipe against a fresh dataframe.

    Returns (transformed_df, applied_operations, errors). Operations whose
    columns/parameters don't fit the current dataframe are skipped (with a
    note in `errors`) so a partial replay still completes instead of bailing
    on the first incompatibility.

    Supports the same set of operations the cleaning UI emits.
    """
    df = df.copy()
    applied: list[str] = []
    errors: list[str] = []

    def _get(step, *keys, default=None):
        for k in keys:
            if k in step:
                return step[k]
        return default

    for i, step in enumerate(recipe, 1):
        op = _get(step, "operation", "action", default="?")
        params = _get(step, "parameters", "params", default={}) or {}
        try:
            if op == "filter":
                # Build a combined mask, supporting AND/OR/NOT connectors
                fs = params.get("filters", []) or []
                conns = params.get("connectors", []) or []
                masks: list = []
                for f in fs:
                    col = f.get("column"); fop = f.get("op"); fval = f.get("value")
                    if col not in df.columns:
                        raise KeyError(col)
                    s = df[col]
                    if pd.api.types.is_numeric_dtype(s):
                        if fop == "between":
                            m = s.between(*fval)
                        else:
                            m = {">": s > fval, "<": s < fval,
                                 ">=": s >= fval, "<=": s <= fval,
                                 "==": s == fval, "!=": s != fval}[fop]
                    else:
                        cs = s.astype(str)
                        if fop == "is in":   m = cs.isin(fval)
                        elif fop == "not in": m = ~cs.isin(fval)
                        elif fop == "contains": m = cs.str.contains(fval or "", case=False, na=False)
                        else: m = pd.Series(True, index=df.index)
                    masks.append(m.fillna(False))
                if masks:
                    combined = masks[0]
                    for k, m in enumerate(masks[1:]):
                        c = conns[k] if k < len(conns) else "AND"
                        if c == "AND":   combined = combined & m
                        elif c == "OR":  combined = combined | m
                        else:            combined = combined & (~m)
                    df = df.loc[combined].reset_index(drop=True)

            elif op == "missing_values":
                strat = params.get("strategy", "")
                cols = params.get("columns", []) or []
                if strat == "Drop rows with any null":
                    df = df.dropna().reset_index(drop=True)
                elif strat == "Drop rows null in selected columns":
                    df = df.dropna(subset=[c for c in cols if c in df.columns]).reset_index(drop=True)
                elif strat in ("Fill with mean (numeric)", "Fill with median (numeric)"):
                    fn = "mean" if "mean" in strat else "median"
                    for c in cols:
                        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = df[c].fillna(getattr(df[c], fn)())
                elif strat == "Fill with mode (numeric)":
                    for c in cols:
                        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                            m = df[c].dropna().mode()
                            if not m.empty:
                                df[c] = df[c].fillna(m.iloc[0])
                elif strat == "Fill with most frequent value (categorical)":
                    for c in cols:
                        if (c in df.columns
                                and not pd.api.types.is_numeric_dtype(df[c])
                                and not pd.api.types.is_datetime64_any_dtype(df[c])):
                            m = df[c].dropna().mode()
                            if not m.empty:
                                df[c] = df[c].fillna(m.iloc[0])
                elif "by group" in strat:
                    fn = "mean" if "mean" in strat else "median"
                    grp = params.get("group_by")
                    for c in cols:
                        if c in df.columns and grp in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = df.groupby(grp)[c].transform(lambda s: s.fillna(getattr(s, fn)()))
                elif strat == "Fill with custom value":
                    val = params.get("value", "")
                    for c in cols:
                        if c in df.columns:
                            try: df[c] = df[c].fillna(df[c].dtype.type(val))
                            except Exception: df[c] = df[c].fillna(val)
                elif strat in ("Forward-fill (ffill)", "Backward-fill (bfill)"):
                    method = params.get("method") or (
                        "ffill" if "ffill" in strat else "bfill")
                    sort_specs = params.get("sort_by") or []
                    sort_specs = [s for s in sort_specs
                                  if s.get("column") in df.columns]
                    if sort_specs:
                        keys = [s["column"] for s in sort_specs]
                        asc = [bool(s.get("ascending", True)) for s in sort_specs]
                        original_idx = df.index.copy()
                        df = df.sort_values(by=keys, ascending=asc,
                                            kind="mergesort")
                        for c in cols:
                            if c in df.columns:
                                df[c] = getattr(df[c], method)()
                        df = df.loc[original_idx]
                    else:
                        for c in cols:
                            if c in df.columns:
                                df[c] = getattr(df[c], method)()

            elif op == "replace_values":
                col = params.get("column")
                mapping = params.get("mapping", {}) or {}
                if col in df.columns and mapping:
                    new = df[col].astype(str).replace(mapping)
                    new = new.mask(df[col].isna())
                    try: df[col] = new.astype(df[col].dtype)
                    except Exception: df[col] = new
                    df[col] = infer_better_dtype(df[col])

            elif op == "remove_duplicates":
                scope = params.get("scope", "full_row")
                cols = params.get("columns") or []
                keep_str = params.get("keep", "first")
                keep_param = {"first": "first", "last": "last",
                              "none": False}.get(keep_str, "first")
                if scope == "full_row" or not cols:
                    df = df.drop_duplicates(keep=keep_param).reset_index(drop=True)
                else:
                    cols = [c for c in cols if c in df.columns]
                    if cols:
                        df = df.drop_duplicates(
                            subset=cols, keep=keep_param
                        ).reset_index(drop=True)

            elif op == "clean_numeric_strings":
                cols = params.get("columns", []) or []
                for c in cols:
                    if c in df.columns:
                        try:
                            stripped = (
                                df[c].astype(str)
                                .str.replace(r"[^\d.\-]", "", regex=True)
                                .replace({"": np.nan})
                            )
                            df[c] = pd.to_numeric(stripped, errors="coerce")
                        except Exception:
                            pass

            elif op == "trim_whitespace":
                cols = params.get("columns", []) or []
                mode = params.get("mode", "both")
                def _trim_one(value, mode=mode):
                    try:
                        if value is None or pd.isna(value):
                            return value
                        s = str(value)
                        if mode == "lstrip":
                            return s.lstrip()
                        if mode == "rstrip":
                            return s.rstrip()
                        return s.lstrip().rstrip()
                    except Exception:
                        return value
                for c in cols:
                    if c in df.columns:
                        try:
                            df[c] = df[c].map(_trim_one)
                        except Exception:
                            pass

            elif op == "outlier_trim":
                col = params.get("column")
                lo = float(params.get("low_value"))
                hi = float(params.get("high_value"))
                if col in df.columns:
                    df = df[df[col].between(lo, hi)].reset_index(drop=True)

            elif op == "outlier_winsorize":
                col = params.get("column")
                lo = float(params.get("low_value"))
                hi = float(params.get("high_value"))
                if col in df.columns:
                    df[col] = df[col].clip(lower=lo, upper=hi)

            elif op == "scale":
                col = params.get("column")
                method = params.get("method")
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if method == "min_max":
                        rng = s.max() - s.min()
                        if rng: df[col] = (s - s.min()) / rng
                    elif method == "z_score":
                        sd = s.std()
                        if sd: df[col] = (s - s.mean()) / sd

            elif op == "numeric_format":
                col = params.get("column"); mode = params.get("mode")
                decimals = int(params.get("decimals", 2))
                if col in df.columns:
                    if mode == "round":
                        df[col] = df[col].round(decimals)
                    elif mode == "to_percent":
                        df[col] = (df[col] * 100).round(decimals).astype(str) + "%"

            elif op == "date_format":
                col = params.get("column"); fmt = params.get("format", "%Y-%m-%d")
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime(fmt)

            elif op == "convert_dtype":
                col = params.get("column"); target = params.get("target_choice")
                if col in df.columns:
                    s = df[col]
                    if target == "String / text":     df[col] = s.astype("string")
                    elif target == "Integer":
                        n = pd.to_numeric(s, errors="coerce")
                        df[col] = n.round().astype("Int64") if n.isna().any() else n.round().astype("int64")
                    elif target == "Float":           df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
                    elif target == "Datetime":        df[col] = pd.to_datetime(s, errors="coerce")
                    elif target == "Category":       df[col] = s.astype("category")

            elif op == "redetect_dtypes":
                changes = params.get("changes", {}) or {}
                for c in changes:
                    if c in df.columns:
                        df[c] = infer_better_dtype(df[c])

            elif op == "case_change":
                col = params.get("column"); case = params.get("case")
                if col in df.columns:
                    s = df[col].astype(str)
                    if case == "lower":   df[col] = s.str.lower()
                    elif case == "UPPER": df[col] = s.str.upper()
                    else:                 df[col] = s.str.title()

            elif op == "split_column":
                col = params.get("column"); sep = params.get("separator", " ")
                if col in df.columns:
                    max_splits = int(params.get("max_splits", -1))
                    parts = df[col].astype(str).str.split(sep, n=max_splits, expand=True)
                    taken = list(df.columns); new_names = []
                    for k in range(parts.shape[1]):
                        nm = unique_column_name(f"{col}_part{k+1}", taken)
                        new_names.append(nm); taken.append(nm)
                    parts.columns = new_names
                    for c in parts.columns:
                        parts[c] = infer_better_dtype(parts[c])
                    df = pd.concat([df, parts], axis=1)

            elif op == "rename_columns":
                mapping = params.get("mapping", {}) or {}
                mapping = {k: v for k, v in mapping.items() if k in df.columns}
                df = df.rename(columns=mapping)

            elif op == "delete_columns":
                cols = params.get("columns", []) or []
                df = df.drop(columns=[c for c in cols if c in df.columns])

            elif op == "drop_sparse_columns":
                # Re-evaluate against the current dataframe state so the same
                # rule produces sensible results even if column null-rates
                # have changed since the recipe was first recorded.
                # New recipes store a fraction, e.g. 50% -> 0.5. Older
                # recipes may only have threshold_pct, so keep that fallback.
                threshold_fraction = float(
                    params.get(
                        "threshold_fraction",
                        float(params.get("threshold_pct", 50.0)) / 100.0,
                    )
                )
                exceptions = set(params.get("exceptions") or [])
                to_drop = [
                    c for c in df.columns
                    if df[c].isna().mean() > threshold_fraction
                    and c not in exceptions
                ]
                df = df.drop(columns=to_drop)

            elif op == "group_rare_categories":
                col = params.get("column"); label = params.get("other_label", "Other")
                # We don't have the original threshold context, but the
                # log records which values were merged via the affected count.
                # Simplest reliable replay: re-compute from current data using
                # the original cutoff if stored, else skip.
                cutoff = params.get("threshold")
                if col in df.columns and cutoff:
                    # cutoff is a display string like "< 10 rows" — parse number
                    import re as _re
                    m = _re.search(r"<\s*([\d.]+)\s*(rows|%)", str(cutoff))
                    if m:
                        n = float(m.group(1))
                        if "%" in m.group(2):
                            n = (n / 100.0) * len(df)
                        vc = df[col].value_counts(dropna=False)
                        rare = set(vc[vc < n].index.tolist())
                        df[col] = df[col].where(~df[col].isin(rare), label)

            elif op == "one_hot_encode":
                col = params.get("column")
                if col in df.columns:
                    dummies = pd.get_dummies(
                        df[col], prefix=col,
                        prefix_sep=params.get("prefix_sep", "_"),
                        drop_first=params.get("drop_first", False), dtype=int,
                    )
                    taken = list(df.columns); rename_map = {}
                    for c in dummies.columns:
                        nm = unique_column_name(c, taken)
                        if nm != c: rename_map[c] = nm
                        taken.append(nm)
                    if rename_map: dummies = dummies.rename(columns=rename_map)
                    df = pd.concat([df, dummies], axis=1)
                    if params.get("drop_original", True):
                        df = df.drop(columns=[col])

            elif op == "bin_numeric":
                col = params.get("column")
                new_name = params.get("new_name")
                n = int(params.get("bins", 5))
                method = params.get("method", "cut")
                labels_key = params.get("labels", "range")
                if col in df.columns and new_name:
                    try:
                        source = pd.to_numeric(df[col], errors="coerce")
                        if method == "qcut":
                            binned = pd.qcut(source, q=n, duplicates="drop")
                        else:
                            binned = pd.cut(source, bins=n,
                                            include_lowest=True,
                                            duplicates="drop")
                        if labels_key == "ordinal":
                            cats = binned.cat.categories
                            mapping = {iv: f"Bin {i+1}" for i, iv in enumerate(cats)}
                            df[new_name] = binned.map(mapping).astype("string")
                        else:
                            def _fmt(iv):
                                if pd.isna(iv):
                                    return None
                                return f"[{iv.left:.0f}, {iv.right:.0f}]"
                            df[new_name] = binned.map(_fmt).astype("string")
                    except Exception:
                        pass

            elif op == "new_column":
                name = params.get("name")
                if params.get("mode") == "formula":
                    parts = params.get("parts")
                    if parts and name:
                        # New structured-parts evaluator
                        n = len(df)
                        placeholders: list = []
                        expr_tokens: list[str] = []
                        for p in parts:
                            k = p["kind"]
                            if k == "op":
                                expr_tokens.append(p["value"]); continue
                            if k == "num":
                                placeholders.append(float(p["value"]))
                            elif k == "col":
                                placeholders.append(df[p["name"]])
                            elif k == "fn":
                                pname = p["name"]
                                if "args" in p and "col" not in p:
                                    cols = p["args"]
                                    if pname in ("mean","std","median","min","max"):
                                        val = df[cols].agg(pname, axis=1)
                                    else:
                                        val = {"log": np.log,"log10": np.log10,
                                               "sqrt": np.sqrt,"abs": np.abs}[pname](df[cols[0]])
                                else:
                                    col_ = p["col"]; gb = p.get("group_by")
                                    s = df[col_]
                                    if p.get("agg_mode") == "transform" or pname in ("log","log10","sqrt","abs"):
                                        val = {"log": np.log,"log10": np.log10,
                                               "sqrt": np.sqrt,"abs": np.abs}[pname](s)
                                    elif gb:
                                        val = df.groupby(gb)[col_].transform(pname)
                                    else:
                                        val = pd.Series([s.agg(pname)] * n, index=df.index)
                                placeholders.append(val)
                            expr_tokens.append(f"__v[{len(placeholders) - 1}]")
                        out = ""
                        for t in expr_tokens:
                            if not out: out = t
                            elif out.endswith("(") or t == ")": out += t
                            else: out += " " + t
                        result = eval(out, {"__builtins__": {}}, {"__v": placeholders})
                        if not isinstance(result, pd.Series):
                            result = pd.Series([result] * n, index=df.index)
                        df[name] = result
                    else:
                        # Legacy path: expression string with __frame__ refs
                        expr = params.get("expression")
                        if name and expr and "__frame__" in expr:
                            def _row_agg(fn_name):
                                def f(*args):
                                    return pd.concat(args, axis=1).agg(fn_name, axis=1)
                                return f
                            ns = {"__frame__": df, "log": np.log, "log10": np.log10,
                                  "sqrt": np.sqrt, "abs": np.abs,
                                  "mean": _row_agg("mean"), "std": _row_agg("std"),
                                  "median": _row_agg("median"), "min": _row_agg("min"),
                                  "max": _row_agg("max")}
                            df[name] = eval(expr, {"__builtins__": {}}, ns)
                elif params.get("mode") in ("mean", "sum", "std", "max", "min", "median"):
                    cols = params.get("columns", []) or []
                    cols = [c for c in cols if c in df.columns]
                    if cols and name:
                        df[name] = df[cols].agg(params["mode"], axis=1)
                elif params.get("mode") == "ratio":
                    a, b = params.get("a"), params.get("b")
                    if a in df.columns and b in df.columns and name:
                        df[name] = df[a] / df[b].replace(0, np.nan)
                elif params.get("mode") == "custom":
                    # Legacy: a free-text formula with backtick-quoted columns
                    text = params.get("formula", "")
                    if text:
                        try:
                            df[name] = df.eval(text)
                        except Exception:
                            pass

            else:
                errors.append(f"Step #{i} `{op}`: not supported by replay.")
                continue

            applied.append(op)
        except Exception as e:
            errors.append(f"Step #{i} `{op}`: {e}")

    return df, applied, errors


def undo_last_recipe_action() -> tuple[bool, str, list[str]]:
    """Undo the latest transformation by replaying the remaining recipe.

    This removes the last recipe record and rebuilds the working dataframe
    from df_original, so the original dataframe stays untouched and the
    transformation recipe stays consistent with the visible dataframe.
    """
    recipe = list(st.session_state.get("recipe") or [])
    original = st.session_state.get("df_original")

    if original is None:
        return False, "No original dataframe is available to rebuild from.", []
    if not recipe:
        return False, "No transformation action to undo.", []

    removed_step = recipe[-1]
    remaining_recipe = recipe[:-1]
    rebuilt_df, _applied, errors = apply_recipe(original.copy(), remaining_recipe)

    st.session_state.df = rebuilt_df
    st.session_state.recipe = remaining_recipe
    st.session_state.pending_delete = None
    st.session_state.pending_drop_sparse = None
    st.session_state.validation_results = None

    operation = removed_step.get("operation") or removed_step.get("action") or "last action"
    return (
        True,
        f"Undid `{operation}` and removed it from the transformation recipe.",
        errors,
    )


def overview_blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every summary table used on the overview tab."""
    info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": df.notna().sum().values,
            "nulls": df.isna().sum().values,
            "null_pct": (df.isna().mean() * 100).round(2).values,
            "unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )

    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(include=["object", "category", "bool"])

    return {
        "info": info,
        "numeric": num.describe().T.round(4) if not num.empty else pd.DataFrame(),
        "categorical": cat.describe().T if not cat.empty else pd.DataFrame(),
    }


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Recipe import validation
# ---------------------------------------------------------------------------
SUPPORTED_RECIPE_OPERATIONS: set[str] = {
    "filter",
    "missing_values",
    "replace_values",
    "remove_duplicates",
    "clean_numeric_strings",
    "trim_whitespace",
    "outlier_trim",
    "outlier_winsorize",
    "scale",
    "numeric_format",
    "date_format",
    "convert_dtype",
    "redetect_dtypes",
    "case_change",
    "split_column",
    "rename_columns",
    "delete_columns",
    "drop_sparse_columns",
    "group_rare_categories",
    "one_hot_encode",
    "bin_numeric",
    "new_column",
}


def validate_recipe_payload(payload: Any) -> tuple[list[dict], list[str], list[str]]:
    """Validate imported recipe JSON before preview/replay.

    Returns (steps, errors, warnings). Errors block replay because the script
    cannot read/apply the recipe safely. Warnings do not block replay, but tell
    the user which readability metadata is missing.
    """
    errors: list[str] = []
    warnings: list[str] = []
    steps: list[dict] = []

    if not isinstance(payload, dict):
        return [], [
            "Top-level JSON must be an object containing a `steps` or `recipe` list."
        ], []

    if "steps" in payload:
        raw_steps = payload.get("steps")
    elif "recipe" in payload:
        raw_steps = payload.get("recipe")
    else:
        return [], [
            "Missing top-level key: expected `steps` or `recipe`."
        ], []

    if not isinstance(raw_steps, list):
        return [], [
            "`steps` / `recipe` must be a list of transformation step objects."
        ], []
    if not raw_steps:
        return [], ["Recipe contains no steps to replay."], []

    for i, step in enumerate(raw_steps, 1):
        if not isinstance(step, dict):
            errors.append(f"Step #{i}: each recipe step must be a JSON object.")
            continue

        operation = step.get("operation") or step.get("action")
        if not isinstance(operation, str) or not operation.strip():
            errors.append(
                f"Step #{i}: missing readable `operation` string "
                "(legacy key `action` is also accepted)."
            )
        elif operation not in SUPPORTED_RECIPE_OPERATIONS:
            errors.append(
                f"Step #{i}: unsupported operation `{operation}`. "
                "This script cannot replay that action."
            )

        has_params_key = "parameters" in step or "params" in step
        params = step.get("parameters") if "parameters" in step else step.get("params")
        if not has_params_key:
            errors.append(
                f"Step #{i}: missing `parameters` object "
                "(legacy key `params` is also accepted)."
            )
        elif not isinstance(params, dict):
            errors.append(f"Step #{i}: `parameters` / `params` must be a JSON object.")

        affected = step.get("affected_columns", [])
        if "affected_columns" not in step:
            warnings.append(
                f"Step #{i}: missing `affected_columns`; preview will show `—`."
            )
        elif not isinstance(affected, list) or not all(isinstance(c, str) for c in affected):
            errors.append(
                f"Step #{i}: `affected_columns` must be a list of column-name strings."
            )

        timestamp = step.get("timestamp")
        if "timestamp" not in step:
            warnings.append(
                f"Step #{i}: missing `timestamp`; preview will show `—`."
            )
        elif not isinstance(timestamp, str):
            errors.append(f"Step #{i}: `timestamp` must be a string.")

    if not errors:
        steps = raw_steps
    return steps, errors, warnings



# ---------------------------------------------------------------------------
# Cached versions of the expensive computations
# ---------------------------------------------------------------------------
# All of these use the fingerprint pattern: pass a cheap tuple as the cache
# key, and the dataframe as a leading underscore arg so Streamlit doesn't try
# to hash it. The fingerprint changes whenever rows, columns, dtypes, or the
# recipe length change — which covers every real mutation we care about.

@st.cache_data(show_spinner=False, max_entries=8)
def cached_overview_blocks(_df: pd.DataFrame, fp: tuple) -> dict[str, pd.DataFrame]:
    return overview_blocks(_df)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_csv_bytes(_df: pd.DataFrame, fp: tuple) -> bytes:
    return _df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=16)
def cached_excel_bytes(_df: pd.DataFrame, fp: tuple) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        _df.to_excel(w, index=False, sheet_name="data")
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=16)
def cached_json_bytes(_df: pd.DataFrame, fp: tuple) -> bytes:
    return _df.to_json(orient="records", indent=2).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=4)
def cached_eligible_date_cols(_df: pd.DataFrame, fp: tuple) -> list:
    """Columns whose every non-null value parses as a date (excluding numerics)."""
    out = []
    for c in _df.columns:
        s = _df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c); continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        if pd.to_datetime(non_null, errors="coerce").notna().all():
            out.append(c)
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def cached_dtype_proposals(_df: pd.DataFrame, fp: tuple) -> dict:
    """Auto-detect proposals for the bulk-promote feature."""
    proposals: dict = {}
    for c in _df.columns:
        s = _df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        inferred = infer_better_dtype(s)
        if str(inferred.dtype) != str(s.dtype):
            proposals[c] = (str(s.dtype), str(inferred.dtype))
    return proposals


@st.cache_data(show_spinner=False, max_entries=32)
def cached_value_counts(_df: pd.DataFrame, col: str, fp: tuple) -> pd.Series:
    """Cached value-counts; used by the rare-category grouping UI."""
    return _df[col].value_counts(dropna=False)


def sparse_columns_preview_table(
    df: pd.DataFrame,
    threshold_fraction: float,
    exceptions: list[str] | set[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return columns whose null share is greater than the threshold.

    ``threshold_fraction`` must be a decimal value between 0 and 1.
    For example, a 50% slider value is passed as 0.5. The returned
    dataframe is only a preview for the UI and does not mutate the
    working dataframe.
    """
    if df is None or df.empty and df.shape[1] == 0:
        return pd.DataFrame(columns=["column", "null_count", "null_pct"])

    exceptions_set = set(exceptions or [])
    total_rows = max(len(df), 1)

    rows: list[dict[str, Any]] = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_fraction = null_count / total_rows
        null_pct = round(null_fraction * 100, 2)
        if null_fraction > float(threshold_fraction) and col not in exceptions_set:
            rows.append(
                {
                    "column": col,
                    "null_count": null_count,
                    "null_pct": null_pct,
                }
            )

    return (
        pd.DataFrame(rows, columns=["column", "null_count", "null_pct"])
        .sort_values(["null_pct", "null_count", "column"],
                     ascending=[False, False, True])
        .reset_index(drop=True)
    )


def render_drop_sparse_columns_tool(df: pd.DataFrame) -> None:
    """Render a confirmed sparse-column deletion tool.

    Workflow:
      1. User picks a null-percentage threshold and exception columns.
      2. App shows exactly which columns match the deletion rule.
      3. User clicks "Review deletion".
      4. App asks for final confirmation.
      5. Only after confirmation are columns dropped, logged, and replayable.
    """
    st.markdown("---")
    st.markdown("**🗂️ Drop columns by missing-value percentage**")
    st.caption(
        "Remove columns whose null percentage is greater than the threshold. "
        "The table below shows the exact columns that match the condition "
        "before anything is deleted."
    )

    threshold_fraction = st.slider(
        "Drop columns where missing share is greater than",
        min_value=0.0,
        max_value=0.99,
        value=0.0,
        step=0.01,
        format="%.2f",
        key="drop_sparse_threshold",
        help=(
            "A column is dropped only if its missing-value share is greater "
            "than this value. Example: 0.50 means > 50%, not >= 50%."
        ),
    )
    threshold_absolute = threshold_fraction
    threshold_fraction = float(threshold_fraction)
    threshold_pct_display = threshold_absolute * 100.0
      

    exceptions = st.multiselect(
        "Exception columns (never drop these)",
        list(df.columns),
        key="drop_sparse_exceptions",
        help="Selected columns will be kept even if they exceed the threshold.",
    )

    preview = sparse_columns_preview_table(df, threshold_fraction, exceptions)
    sparse_exception_count = sum(
        1
        for col in df.columns
        if (df[col].isna().mean() > threshold_fraction) and col in set(exceptions)
    )

    st.markdown(
        f"<div class='step-box'>"
        f"Condition: <b>missing share &gt; {threshold_absolute:.2f}</b> "
        f"({threshold_pct_display:g}%)<br>"
        f"<span class='rows'>{len(preview)} column(s) match deletion rule</span>"
        f" · {df.shape[1] - len(preview)} would remain"
        + (
            f"<br><span style='opacity:.8'>{sparse_exception_count} sparse "
            f"column(s) kept by exception</span>"
            if sparse_exception_count
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if preview.empty:
        st.info("No columns currently match this deletion condition.")
    else:
        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
            height=min(320, 40 + 35 * len(preview)),
        )

    with st.expander("Show null percentage for all columns", expanded=False):
        all_nulls = (
            pd.DataFrame(
                {
                    "column": list(df.columns),
                    "null_count": [int(df[c].isna().sum()) for c in df.columns],
                    "null_pct": [
                        round((int(df[c].isna().sum()) / max(len(df), 1)) * 100, 2)
                        for c in df.columns
                    ],
                }
            )
            .sort_values(["null_pct", "null_count", "column"],
                         ascending=[False, False, True])
            .reset_index(drop=True)
        )
        st.dataframe(
            all_nulls,
            use_container_width=True,
            hide_index=True,
            height=min(320, 40 + 35 * len(all_nulls)),
        )

    c_review, c_cancel = st.columns([1, 1])
    if c_review.button(
        "Review deletion",
        type="primary",
        key="drop_sparse_review",
        disabled=preview.empty,
    ):
        st.session_state.pending_drop_sparse = {
            "threshold_pct": float(threshold_pct_display),
            "threshold_fraction": float(threshold_fraction),
            "exceptions": list(exceptions),
            "columns": preview["column"].tolist(),
            "cols_before": int(df.shape[1]),
        }
        st.rerun()

    if c_cancel.button(
        "Clear pending deletion",
        key="drop_sparse_clear_pending",
        disabled=st.session_state.get("pending_drop_sparse") is None,
    ):
        st.session_state.pending_drop_sparse = None
        st.rerun()

    pending = st.session_state.get("pending_drop_sparse")
    if not pending:
        return

    pending_threshold_pct = float(pending.get("threshold_pct", threshold_pct_display))
    pending_threshold_fraction = float(
        pending.get("threshold_fraction", threshold_fraction)
    )
    pending_exceptions = list(pending.get("exceptions") or [])
    confirm_preview = sparse_columns_preview_table(
        df,
        pending_threshold_fraction,
        pending_exceptions,
    )

    st.warning(
        f"Confirm deletion of {len(confirm_preview)} column(s) where "
        f"missing share > {pending_threshold_fraction:.2f} "
        f"({pending_threshold_pct:g}%)."
    )

    if confirm_preview.empty:
        st.info(
            "The dataframe changed after review. No columns now match the "
            "pending deletion condition."
        )
    else:
        st.dataframe(
            confirm_preview,
            use_container_width=True,
            hide_index=True,
            height=min(260, 40 + 35 * len(confirm_preview)),
        )

    c_confirm, c_abort = st.columns([1, 1])
    if c_confirm.button(
        "Confirm drop sparse columns",
        type="primary",
        key="drop_sparse_confirm",
        disabled=confirm_preview.empty,
    ):
        before_rows = len(df)
        before_cols = df.shape[1]
        columns_to_drop = confirm_preview["column"].tolist()

        updated_df = df.drop(columns=columns_to_drop)
        st.session_state.df = updated_df
        st.session_state.pending_drop_sparse = None
        st.session_state.validation_results = None

        log_step(
            "drop_sparse_columns",
            {
                "threshold_pct": pending_threshold_pct,
                "threshold_fraction": pending_threshold_fraction,
                "exceptions": pending_exceptions,
                "columns_dropped": columns_to_drop,
                "cols_before": before_cols,
                "cols_after": updated_df.shape[1],
            },
            before_rows,
            len(updated_df),
            affected_columns=columns_to_drop,
        )
        flash(
            f"Dropped {len(columns_to_drop)} sparse column(s) with "
            f"> {pending_threshold_fraction:.2f} missing share "
            f"({pending_threshold_pct:g}% nulls)."
        )
        st.rerun()

    if c_abort.button("Cancel deletion", key="drop_sparse_cancel"):
        st.session_state.pending_drop_sparse = None
        st.rerun()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def apply_viz_filters(df: pd.DataFrame, filters: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Apply a list of filter dicts to df, returning (filtered_df, labels)."""
    out = df.copy()
    labels: list[str] = []
    for f in filters:
        col, op, val = f.get("col"), f.get("op"), f.get("val")
        if col not in out.columns:
            continue
        s = out[col]
        try:
            if f.get("kind") == "numeric":
                if op == "between":
                    out = out[out[col].between(val[0], val[1])]
                    labels.append(f"{col} in [{val[0]:.3g}, {val[1]:.3g}]")
                else:
                    cond = {
                        ">": out[col] > val, "<": out[col] < val,
                        "==": out[col] == val, "!=": out[col] != val,
                        ">=": out[col] >= val, "<=": out[col] <= val,
                    }[op]
                    out = out[cond]
                    labels.append(f"{col} {op} {val}")
            elif f.get("kind") == "date":
                start, end = pd.to_datetime(val[0]), pd.to_datetime(val[1])
                out = out[(out[col] >= start) & (out[col] <= end)]
                labels.append(f"{col} in {start.date()}…{end.date()}")
            else:  # categorical
                cs = out[col].astype(str)
                if op == "is in":
                    out = out[cs.isin(val)]
                    labels.append(f"{col} in {val}")
                elif op == "not in":
                    out = out[~cs.isin(val)]
                    labels.append(f"{col} not in {val}")
                else:
                    cond = {
                        "contains": cs.str.contains(val or "", case=False, na=False),
                        "equals": cs == val,
                        "starts with": cs.str.startswith(val or ""),
                        "ends with": cs.str.endswith(val or ""),
                        "not contains": ~cs.str.contains(val or "", case=False, na=False),
                        "not equals": cs != val,
                    }[op]
                    out = out[cond]
                    labels.append(f"{col} {op} '{val}'")
        except Exception:
            continue
    return out, labels


def summarize_for_ai(df: pd.DataFrame, n_sample: int = 5) -> dict:
    """
    Build a compact, model-friendly description of the dataframe.
    Stays under a few KB so it's cheap to send to a chat model.
    """
    cols_info = []
    for c in df.columns:
        s = df[c]
        info = {
            "name": str(c),
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            try:
                info["min"] = float(s.min())
                info["max"] = float(s.max())
                info["mean"] = float(s.mean())
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(s):
            try:
                info["min"] = str(s.min())
                info["max"] = str(s.max())
            except Exception:
                pass
        else:
            top = s.dropna().astype(str).value_counts().head(5).index.tolist()
            info["sample_values"] = top
        cols_info.append(info)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": cols_info,
        "sample_rows": df.head(n_sample).astype(str).to_dict(orient="records"),
    }


def request_ai_suggestions(summary: dict, api_key: str,
                           model: str = "llama-3.1-8b-instant") -> list[dict]:
    """
    Ask Groq's LLM API to propose chart suggestions tailored to the dataframe.
    Groq uses an OpenAI-compatible endpoint, so we reuse the OpenAI Python SDK
    with a custom base_url. Free-tier rate limits are generous.

    Returns a list of suggestion dicts. Raises on errors so the caller can show them.
    """
    if not OPENAI_OK:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    if not api_key:
        raise RuntimeError("No Groq API key provided.")

    client = _OpenAIClient(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system_msg = (
        "You are a data-visualization advisor. Given a compact summary of a "
        "pandas dataframe, propose 3 to 5 chart suggestions that would reveal "
        "the most useful insights. Prioritise: "
        "(1) time-series patterns when a datetime column exists, "
        "(2) correlations between numeric columns, "
        "(3) distributions of important variables, "
        "(4) group comparisons (numeric by category). "
        "Return ONLY valid JSON of the form: "
        "{\"suggestions\": [{"
        "\"chart_type\": \"<Scatter|Bubble|Line|Bar|Histogram|Box|Violin|Area|Pie|Correlation heatmap>\","
        "\"x\": \"<column name or null>\","
        "\"y\": \"<column name or null>\","
        "\"color\": \"<column name or null>\","
        "\"size\": \"<column name or null>\","
        "\"agg\": \"<mean|sum|median|max|min|count or null>\","
        "\"purpose\": \"<one sentence: what insight this chart reveals>\""
        "}, ...]}. "
        "Use only column names that exist in the provided summary. "
        "Set fields to null when not applicable to the chart type."
    )
    user_msg = ("Dataset summary:\n```json\n"
                + json.dumps(summary, default=str)[:8000] + "\n```")

    # Not every backend supports response_format={"type":"json_object"}.
    # Try with it first; if the server rejects it, retry without.
    call_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )
    try:
        resp = client.chat.completions.create(
            **call_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Retry without response_format for backends that don't support it
        resp = client.chat.completions.create(**call_kwargs)

    text = resp.choices[0].message.content or "{}"

    # Some models wrap JSON in ```json ... ``` fences. Strip if present.
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()

    # Find the first { ... } block in case the model added prose around it
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s:e+1]

    parsed = json.loads(text)
    suggestions = parsed.get("suggestions", [])
    if not isinstance(suggestions, list):
        raise ValueError("Model did not return a 'suggestions' list.")
    return suggestions


CHART_ICONS: dict[str, str] = {
    "Scatter":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<circle cx="10" cy="30" r="3" fill="#3b82f6"/>'
        '<circle cx="22" cy="20" r="3" fill="#3b82f6"/>'
        '<circle cx="34" cy="14" r="3" fill="#3b82f6"/>'
        '<circle cx="46" cy="8" r="3" fill="#3b82f6"/>'
        '<circle cx="18" cy="25" r="3" fill="#3b82f6"/>'
        '<circle cx="40" cy="22" r="3" fill="#3b82f6"/>'
        '</svg>',
    "Bubble":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<circle cx="14" cy="25" r="5" fill="#3b82f6" opacity=".7"/>'
        '<circle cx="32" cy="14" r="8" fill="#3b82f6" opacity=".7"/>'
        '<circle cx="48" cy="28" r="4" fill="#3b82f6" opacity=".7"/>'
        '</svg>',
    "Line":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<polyline points="4,32 14,22 24,26 34,12 44,18 56,6" '
        'fill="none" stroke="#3b82f6" stroke-width="2"/>'
        '</svg>',
    "Area":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<polygon points="4,32 14,22 24,26 34,12 44,18 56,6 56,36 4,36" '
        'fill="#3b82f6" opacity=".4"/>'
        '<polyline points="4,32 14,22 24,26 34,12 44,18 56,6" '
        'fill="none" stroke="#3b82f6" stroke-width="2"/>'
        '</svg>',
    "Bar":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<rect x="6"  y="20" width="8" height="16" fill="#3b82f6"/>'
        '<rect x="18" y="10" width="8" height="26" fill="#3b82f6"/>'
        '<rect x="30" y="14" width="8" height="22" fill="#3b82f6"/>'
        '<rect x="42" y="6"  width="8" height="30" fill="#3b82f6"/>'
        '</svg>',
    "Histogram":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<rect x="4"  y="28" width="6" height="8"  fill="#3b82f6"/>'
        '<rect x="11" y="22" width="6" height="14" fill="#3b82f6"/>'
        '<rect x="18" y="14" width="6" height="22" fill="#3b82f6"/>'
        '<rect x="25" y="10" width="6" height="26" fill="#3b82f6"/>'
        '<rect x="32" y="16" width="6" height="20" fill="#3b82f6"/>'
        '<rect x="39" y="22" width="6" height="14" fill="#3b82f6"/>'
        '<rect x="46" y="28" width="6" height="8"  fill="#3b82f6"/>'
        '</svg>',
    "Box":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<line x1="20" y1="6"  x2="20" y2="34" stroke="#3b82f6"/>'
        '<rect x="14" y="14" width="12" height="14" fill="none" stroke="#3b82f6" stroke-width="1.5"/>'
        '<line x1="14" y1="22" x2="26" y2="22" stroke="#3b82f6" stroke-width="1.5"/>'
        '<line x1="42" y1="6"  x2="42" y2="34" stroke="#3b82f6"/>'
        '<rect x="36" y="10" width="12" height="18" fill="none" stroke="#3b82f6" stroke-width="1.5"/>'
        '<line x1="36" y1="20" x2="48" y2="20" stroke="#3b82f6" stroke-width="1.5"/>'
        '</svg>',
    "Violin":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<path d="M22,4 Q12,16 14,20 Q12,28 22,36 Q32,28 30,20 Q32,16 22,4 Z" '
        'fill="#3b82f6" opacity=".5"/>'
        '<path d="M44,4 Q34,16 36,20 Q34,28 44,36 Q54,28 52,20 Q54,16 44,4 Z" '
        'fill="#3b82f6" opacity=".5"/>'
        '</svg>',
    "Pie":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<circle cx="30" cy="20" r="14" fill="#3b82f6"/>'
        '<path d="M30,20 L30,6 A14,14 0 0,1 43.6,22.4 Z" fill="#93c5fd"/>'
        '<path d="M30,20 L43.6,22.4 A14,14 0 0,1 35,33 Z" fill="#1e40af"/>'
        '</svg>',
    "Correlation heatmap":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<rect x="6"  y="6"  width="10" height="10" fill="#1e40af"/>'
        '<rect x="18" y="6"  width="10" height="10" fill="#60a5fa"/>'
        '<rect x="30" y="6"  width="10" height="10" fill="#bfdbfe"/>'
        '<rect x="6"  y="18" width="10" height="10" fill="#60a5fa"/>'
        '<rect x="18" y="18" width="10" height="10" fill="#1e40af"/>'
        '<rect x="30" y="18" width="10" height="10" fill="#93c5fd"/>'
        '<rect x="6"  y="30" width="10" height="6"  fill="#bfdbfe"/>'
        '<rect x="18" y="30" width="10" height="6"  fill="#93c5fd"/>'
        '<rect x="30" y="30" width="10" height="6"  fill="#1e40af"/>'
        '</svg>',
    "_default":
        '<svg viewBox="0 0 60 40" width="80" height="50">'
        '<rect x="6" y="22" width="48" height="14" fill="none" stroke="#64748b" stroke-width="1.5" rx="2"/>'
        '<polyline points="10,16 20,10 30,14 40,6 50,12" fill="none" stroke="#3b82f6" stroke-width="2"/>'
        '</svg>',
}


def classify_ai_error(err: Exception | str) -> tuple[str, str]:
    """
    Inspect an exception/message string from the AI call and return a tuple
    (title, friendly_explanation) the UI can show without exposing raw HTTP
    error spam. Recognises the common failure modes so users see actionable
    advice instead of an opaque traceback.
    """
    msg = str(err)
    low = msg.lower()

    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return (
            "Quota exhausted",
            "The AI provider account has run out of credit / free quota. "
            "Top up your Groq account, or wait for the daily free-tier "
            "limit to reset. Using local rule-based suggestions instead."
        )
    if "rate" in low and "limit" in low or "429" in low or "rate_limit" in low:
        return (
            "Rate limit hit",
            "You're sending requests too quickly. Wait a moment and try "
            "again. Using local rule-based suggestions in the meantime."
        )
    if "invalid_api_key" in low or "401" in low or "incorrect api key" in low:
        return (
            "API key rejected",
            "The Groq API key is invalid or expired. Generate a new one at "
            "console.groq.com/keys and update your `.env` file."
        )
    if "timed out" in low or "timeout" in low or "connection" in low or \
       "name or service not known" in low or "network" in low:
        return (
            "Network problem",
            "Couldn't reach the AI service. Check your internet connection. "
            "Using local rule-based suggestions instead."
        )
    if "json" in low or "model did not return" in low:
        return (
            "Bad response from model",
            "The model returned something we couldn't parse. Try a different "
            "model, or the local rule-based suggestions below."
        )
    # Generic fallback
    return ("AI service error", msg[:300])


def run_validations(df: pd.DataFrame, rules: list[dict]) -> dict:
    """
    Evaluate a list of validation rules against the dataframe.

    Each rule is a dict with shape:
        {kind: "range" | "allowed" | "not_null",
         column: str,
         ... rule-specific params ...}

    Returns a dict with:
        - "violations":  DataFrame (rule_index, rule, column, row_index,
                          row_position, bad_value, message)
        - "summary":     DataFrame (rule_index, rule, column, total_checked,
                          n_violations, status, severity)
        - "overall_severity": "Info" | "Critical"

    Severity is "Critical" if any rule produced at least one violation,
    otherwise "Info". This matches the convention requested in the spec.
    """
    violations_rows = []
    summary_rows = []

    for idx, rule in enumerate(rules):
        kind = rule.get("kind")
        col = rule.get("column")
        rule_text = ""
        n_checked = 0
        n_bad = 0
        violators_idx = pd.Index([])

        if col not in df.columns:
            summary_rows.append({
                "rule_index": idx + 1, "rule": kind, "column": col or "—",
                "total_checked": 0, "n_violations": 0,
                "status": "skipped", "severity": "Info",
                "detail": f"column `{col}` not present"})
            continue

        s = df[col]

        if kind == "range":
            # Numeric or datetime min/max
            lo = rule.get("min"); hi = rule.get("max")
            rule_text = (f"range [{lo if lo is not None else '−∞'}, "
                         f"{hi if hi is not None else '+∞'}]")
            cmp_series = s
            # Coerce dates if needed
            if pd.api.types.is_datetime64_any_dtype(s):
                try:
                    lo = pd.to_datetime(lo) if lo is not None else None
                    hi = pd.to_datetime(hi) if hi is not None else None
                except Exception:
                    lo = hi = None
            elif not pd.api.types.is_numeric_dtype(s):
                # try numeric coerce
                cmp_series = pd.to_numeric(s, errors="coerce")
            non_null = cmp_series.dropna()
            n_checked = len(non_null)
            mask_bad = pd.Series(False, index=cmp_series.index)
            if lo is not None:
                mask_bad |= (cmp_series < lo).fillna(False)
            if hi is not None:
                mask_bad |= (cmp_series > hi).fillna(False)
            n_bad = int(mask_bad.sum())
            violators_idx = cmp_series.index[mask_bad]

        elif kind == "allowed":
            allowed = list(rule.get("allowed", []) or [])
            rule_text = f"value in {allowed[:8]}{'…' if len(allowed) > 8 else ''}"
            case_insensitive = bool(rule.get("case_insensitive"))
            cs = s.astype(str)
            allowed_for_check = (set(a.lower() for a in allowed)
                                 if case_insensitive else set(allowed))
            check = cs.str.lower() if case_insensitive else cs
            non_null = s.dropna()
            n_checked = len(non_null)
            # Bad rows: non-null and NOT in the allowed set
            mask_bad = (~check.isin(allowed_for_check)) & s.notna()
            n_bad = int(mask_bad.sum())
            violators_idx = s.index[mask_bad]

        elif kind == "not_null":
            rule_text = "value must not be null"
            n_checked = len(s)
            mask_bad = s.isna()
            n_bad = int(mask_bad.sum())
            violators_idx = s.index[mask_bad]

        else:
            summary_rows.append({
                "rule_index": idx + 1, "rule": kind, "column": col,
                "total_checked": 0, "n_violations": 0,
                "status": "unknown", "severity": "Info", "detail": ""})
            continue

        # Build violation rows (cap to a sensible number per rule to keep
        # the table manageable on huge datasets)
        MAX_PER_RULE = 5000
        for pos, ridx in enumerate(violators_idx[:MAX_PER_RULE]):
            try:
                bad_val = df.loc[ridx, col]
            except Exception:
                bad_val = None
            violations_rows.append({
                "rule_index": idx + 1,
                "rule": f"{kind}: {rule_text}",
                "column": col,
                "row_index": ridx,
                "row_position": int(df.index.get_loc(ridx))
                                if isinstance(ridx, (int, str)) else pos,
                "bad_value": bad_val,
                "message": f"`{col}` = {bad_val!r} violates {rule_text}",
            })

        summary_rows.append({
            "rule_index": idx + 1,
            "rule": f"{kind}: {rule_text}",
            "column": col,
            "total_checked": n_checked,
            "n_violations": n_bad,
            "status": "fail" if n_bad else "pass",
            "severity": "Critical" if n_bad else "Info",
        })

    violations_df = pd.DataFrame(violations_rows)
    summary_df = pd.DataFrame(summary_rows)
    overall = "Critical" if any(r.get("n_violations", 0) > 0 for r in summary_rows) else "Info"
    return {
        "violations": violations_df,
        "summary": summary_df,
        "overall_severity": overall,
    }


def fallback_suggestions(df: pd.DataFrame) -> list[dict]:
    """
    Produce a small set of chart suggestions WITHOUT calling any external API.

    These are based on simple rules over the dataframe's structure:
      * one suggestion per datetime column (time-series line chart)
      * a correlation heatmap if there are 3+ numeric columns
      * a histogram of the most-varying numeric column
      * a grouped box plot if a low-cardinality categorical and a numeric exist
      * a scatter of the two top numeric columns

    Returns the same dict shape as request_ai_suggestions, so the UI doesn't
    need to know whether suggestions came from AI or this fallback.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category", "string"]).columns
                if df[c].nunique(dropna=True) <= 30]

    out: list[dict] = []

    # 1) Time-series for each datetime column (cap at 1 per dataset)
    if date_cols and num_cols:
        d = date_cols[0]
        y = num_cols[0]
        out.append({
            "chart_type": "Line", "x": d, "y": y,
            "color": cat_cols[0] if cat_cols else None,
            "size": None, "agg": None,
            "purpose": f"Track `{y}` over time using `{d}`"
                       + (f", split by `{cat_cols[0]}`." if cat_cols else "."),
        })

    # 2) Correlation heatmap if multiple numerics
    if len(num_cols) >= 3:
        out.append({
            "chart_type": "Correlation heatmap",
            "x": None, "y": None, "color": None, "size": None, "agg": None,
            "purpose": f"See which of the {len(num_cols)} numeric columns "
                       "move together.",
        })

    # 3) Distribution of the most-varying numeric column
    if num_cols:
        # Pick the numeric column with the highest std (most variation)
        try:
            stds = df[num_cols].std()
            top = stds.sort_values(ascending=False).index[0]
        except Exception:
            top = num_cols[0]
        out.append({
            "chart_type": "Histogram", "x": top,
            "y": None, "color": None, "size": None, "agg": None,
            "purpose": f"Distribution of `{top}` — useful to spot skew "
                       "and outliers.",
        })

    # 4) Numeric-by-category box plot
    if cat_cols and num_cols:
        out.append({
            "chart_type": "Box", "x": cat_cols[0], "y": num_cols[0],
            "color": None, "size": None, "agg": None,
            "purpose": f"Compare `{num_cols[0]}` distributions across "
                       f"`{cat_cols[0]}` groups.",
        })

    # 5) Scatter of the two most-varying numeric columns
    if len(num_cols) >= 2:
        try:
            stds = df[num_cols].std().sort_values(ascending=False)
            x_, y_ = stds.index[0], stds.index[1]
        except Exception:
            x_, y_ = num_cols[0], num_cols[1]
        out.append({
            "chart_type": "Scatter", "x": x_, "y": y_,
            "color": cat_cols[0] if cat_cols else None,
            "size": None, "agg": None,
            "purpose": f"Relationship between `{x_}` and `{y_}`"
                       + (f", coloured by `{cat_cols[0]}`." if cat_cols else "."),
        })

    return out


def suggestion_to_config(sug: dict, df: pd.DataFrame) -> dict | None:
    """
    Convert a model suggestion into a config dict matching what build_figure
    expects. Returns None if the suggestion references unknown columns OR
    if the suggestion would obviously fail at render time (e.g. it tries to
    aggregate a text column with mean/sum, or use a text column as a Scatter
    Y axis). Filtering those here is cheaper than letting the chart engine
    throw a confusing error in the UI.
    """
    NONE = "—"
    available_cols = list(df.columns)
    num_cols = set(df.select_dtypes(include="number").columns)
    date_cols = set(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    def _resolve(name):
        if name in (None, "", "null"):
            return None
        return name if name in available_cols else None

    chart = sug.get("chart_type")
    if chart not in ("Scatter", "Bubble", "Line", "Bar", "Histogram", "Box",
                     "Violin", "Area", "Pie", "Correlation heatmap"):
        return None
    x = _resolve(sug.get("x"))
    y = _resolve(sug.get("y"))
    color = _resolve(sug.get("color"))
    size = _resolve(sug.get("size"))
    agg = sug.get("agg")

    # ---- Dtype sanity checks: refuse impossible combinations ----
    # Y axis must be numeric for charts that statistically summarise it.
    needs_numeric_y = chart in ("Scatter", "Bubble", "Box", "Violin", "Line", "Area")
    if needs_numeric_y and y and y not in num_cols:
        return None
    # X must be numeric (or at least sortable) for Scatter/Bubble.
    if chart in ("Scatter", "Bubble") and x and x not in num_cols:
        return None
    # Bubble size must be numeric.
    if chart == "Bubble" and size and size not in num_cols:
        return None
    # Bar with mean/sum/std/median/max/min requires a numeric Y.
    if chart == "Bar" and y and y not in (None, NONE):
        if agg in ("mean", "sum", "std", "median", "max", "min") and y not in num_cols:
            return None
    # Pie's "value" (passed as y) should be numeric when provided.
    if chart == "Pie" and y and y not in num_cols:
        return None
    # Histogram x should be numeric.
    if chart == "Histogram" and x and x not in num_cols:
        return None
    # Line/Area x should be datetime, numeric, or at least sortable —
    # text X with many uniques produces a useless plot. We allow text X here
    # only if the column has reasonably few unique values (<= 100).
    if chart in ("Line", "Area") and x:
        if x not in num_cols and x not in date_cols:
            if df[x].nunique(dropna=True) > 100:
                return None

    cfg: dict = {
        "chart_type": chart,
        "engine": "Plotly",
        "x": x, "y": y,
        "color": color or NONE,
        "facet": NONE,
        "palette": "Plotly",
        "template": "plotly_white",
        "title": (sug.get("purpose") or "")[:80],
        "show_legend": True,
        "opacity": 0.8,
        "height": 500,
        "log_x": False, "log_y": False,
        "filters": [],
        "trendline": False,
    }
    if chart == "Bubble":
        cfg["size"] = size
    if chart in ("Line", "Area"):
        cfg["y"] = [y] if y else []
    if chart == "Bar" and y and agg:
        cfg["agg"] = agg
    if chart == "Histogram":
        cfg["bins"] = 30

    needs_x = chart != "Correlation heatmap"
    needs_y = chart in ("Scatter", "Bubble", "Box", "Violin")
    if needs_x and not x:
        return None
    if needs_y and not y:
        return None
    if chart == "Bubble" and not size:
        return None
    return cfg


def build_matplotlib_figure(df: pd.DataFrame, cfg: dict):
    """Build a static matplotlib figure from a chart-config dict.

    Returns (fig, error). Supports the same chart types as build_figure but
    uses matplotlib for rendering — useful when the user wants a static image
    that can be saved as PNG/PDF or embedded in a print-friendly report.
    """
    if not MATPLOTLIB_OK:
        return None, "matplotlib is not installed."
    import matplotlib.pyplot as plt

    ct = cfg["chart_type"]
    x, y, size = cfg.get("x"), cfg.get("y"), cfg.get("size")
    NONE = "—"
    color_arg = None if cfg.get("color") in (None, NONE) else cfg["color"]
    title = cfg.get("title") or None
    height_px = cfg.get("height", 500)

    # Translate Plotly templates to matplotlib styles where possible.
    template = cfg.get("template", "plotly_white")
    style_map = {
        "plotly": "default",
        "plotly_white": "seaborn-v0_8-whitegrid",
        "plotly_dark": "dark_background",
        "ggplot2": "ggplot",
        "seaborn": "seaborn-v0_8",
        "simple_white": "seaborn-v0_8-white",
    }
    try:
        plt.style.use(style_map.get(template, "default"))
    except Exception:
        plt.style.use("default")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Same binning logic as the Plotly path — only for chart types where
    # grouping a numeric X into bins makes sense.
    if ct in ("Bar", "Box", "Violin", "Pie"):
        df = apply_x_binning(df, cfg)

    # Top N is visualization-only: these copies are used for plotting only,
    # while the filtered dataframe fragment remains unchanged.
    if ct == "Scatter":
        df = _apply_top_n_rows(df, cfg, [y])
    elif ct == "Bubble":
        df = _apply_top_n_rows(df, cfg, [size, y])
    elif ct in ("Line", "Area"):
        ys_for_top = y if isinstance(y, list) else [y]
        df = _apply_top_n_rows(df, cfg, [c for c in ys_for_top if c])
    elif ct in ("Histogram",):
        df = _apply_top_n_rows(df, cfg, [x])
    elif ct in ("Box", "Violin"):
        df = _apply_top_n_rows(df, cfg, [y])

    try:
        fig, ax = plt.subplots(figsize=(8, height_px / 100))

        # Build a colour-by-group iterator if color column is set
        def _groups():
            if color_arg and color_arg in df.columns:
                for name, grp in df.groupby(color_arg, dropna=False):
                    yield str(name), grp
            else:
                yield None, df

        if ct == "Scatter":
            for name, grp in _groups():
                ax.scatter(grp[x], grp[y], alpha=cfg.get("opacity", 0.8),
                           s=(cfg.get("marker_size") or 7) ** 2, label=name)
            ax.set_xlabel(x); ax.set_ylabel(y)

        elif ct == "Bubble":
            for name, grp in _groups():
                # Scale bubble size to a usable range
                s_vals = grp[size].astype(float)
                s_norm = ((s_vals - s_vals.min()) /
                          (s_vals.max() - s_vals.min() + 1e-9)) * 400 + 30
                ax.scatter(grp[x], grp[y], s=s_norm,
                           alpha=cfg.get("opacity", 0.5), label=name,
                           edgecolors="white", linewidth=0.5)
            ax.set_xlabel(x); ax.set_ylabel(y)

        elif ct in ("Line", "Area"):
            ys = y if isinstance(y, list) else [y]
            ys = [c for c in ys if c]
            if not ys:
                return None, "Pick at least one Y column."
            # Sort by X for the same reason as the Plotly branch — without this
            # an unsorted date axis produces a tangled line.
            plot_df = df.dropna(subset=[x]).sort_values(by=x, kind="mergesort")
            for col in ys:
                if ct == "Line":
                    ax.plot(plot_df[x], plot_df[col], label=col,
                            alpha=cfg.get("opacity", 0.9))
                else:
                    ax.fill_between(plot_df[x], plot_df[col],
                                    alpha=cfg.get("opacity", 0.4), label=col)
                    ax.plot(plot_df[x], plot_df[col], alpha=0.9)
            ax.set_xlabel(x)
            if len(ys) == 1:
                ax.set_ylabel(ys[0])

        elif ct == "Bar":
            agg = cfg.get("agg")
            if y in (None, NONE) or agg == "count":
                vc = df[x].value_counts().reset_index()
                vc.columns = [x, "count"]
                vc = _apply_top_n_grouped(vc, cfg, "count")
                ax.bar(vc[x].astype(str), vc["count"],
                       alpha=cfg.get("opacity", 0.8))
                ax.set_ylabel("count")
            else:
                grp = df.groupby(x)[y].agg(agg).reset_index()
                grp = _apply_top_n_grouped(grp, cfg, y)
                ax.bar(grp[x].astype(str), grp[y],
                       alpha=cfg.get("opacity", 0.8))
                ax.set_ylabel(y)
            ax.set_xlabel(x)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        elif ct == "Histogram":
            ax.hist(df[x].dropna(), bins=cfg.get("bins", 30),
                    alpha=cfg.get("opacity", 0.8))
            ax.set_xlabel(x); ax.set_ylabel("count")

        elif ct == "Box":
            if x in (None, NONE):
                ax.boxplot(df[y].dropna(), labels=[y])
            else:
                groups = [g[y].dropna().values for _, g in df.groupby(x)]
                labels = [str(k) for k, _ in df.groupby(x)]
                ax.boxplot(groups, labels=labels)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_ylabel(y)

        elif ct == "Violin":
            if x in (None, NONE):
                ax.violinplot(df[y].dropna())
                ax.set_xticks([1]); ax.set_xticklabels([y])
            else:
                groups = [g[y].dropna().values for _, g in df.groupby(x)]
                labels = [str(k) for k, _ in df.groupby(x)]
                ax.violinplot(groups, showmedians=True)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(y)

        elif ct == "Pie":
            if y in (None, NONE):
                vc = df[x].value_counts().reset_index()
                vc.columns = [x, "count"]
                vc = _apply_top_n_grouped(vc, cfg, "count")
                ax.pie(vc["count"], labels=vc[x].astype(str), autopct="%1.1f%%",
                       wedgeprops={"alpha": cfg.get("opacity", 0.85)})
            else:
                grp = df.groupby(x, dropna=False)[y].sum().reset_index()
                grp = _apply_top_n_grouped(grp, cfg, y)
                ax.pie(grp[y], labels=grp[x].astype(str), autopct="%1.1f%%",
                       wedgeprops={"alpha": cfg.get("opacity", 0.85)})
            ax.axis("equal")

        else:  # Correlation heatmap
            if len(num_cols) < 2:
                return None, "Need at least two numeric columns."
            corr = df[num_cols].corr().round(2)
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(num_cols)))
            ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=45, ha="right")
            ax.set_yticklabels(num_cols)
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                            ha="center", va="center",
                            color="white" if abs(corr.iloc[i,j]) > 0.5 else "black",
                            fontsize=8)
            fig.colorbar(im, ax=ax)

        if title:
            ax.set_title(title)
        if cfg.get("show_legend", True) and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
        if cfg.get("log_x"):
            ax.set_xscale("log")
        if cfg.get("log_y"):
            ax.set_yscale("log")

        fig.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, str(e)




# ---------------------------------------------------------------------------
# Minimalist wireframe previews (no data — just chart shape)
# ---------------------------------------------------------------------------
# These are low-contrast SVGs shown in the configure form so the user can see
# what the chart shape will look like before clicking Create. Using #d0d6dd
# stroke and a very light fill keeps the contrast intentionally low — this is
# a *template*, not a real chart.
WIREFRAMES: dict[str, str] = {
    "Scatter":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg" '
        'preserveAspectRatio="xMidYMid meet">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        + "".join(f'<circle cx="{60+i*16}" cy="{170-((i*9)%140)}" r="4" '
                  f'fill="#c7d0da" opacity="0.7"/>'
                  for i in range(20))
        + '<text x="200" y="210" font-size="10" fill="#94a3b8" '
        'text-anchor="middle" font-family="sans-serif">X axis</text>'
        '<text x="20" y="100" font-size="10" fill="#94a3b8" '
        'text-anchor="middle" font-family="sans-serif" '
        'transform="rotate(-90 20 100)">Y axis</text>'
        '</svg>',
    "Bubble":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        '<circle cx="80" cy="160" r="10" fill="#c7d0da" opacity=".6"/>'
        '<circle cx="140" cy="120" r="22" fill="#c7d0da" opacity=".6"/>'
        '<circle cx="220" cy="80" r="32" fill="#c7d0da" opacity=".6"/>'
        '<circle cx="300" cy="140" r="16" fill="#c7d0da" opacity=".6"/>'
        '<circle cx="350" cy="60" r="14" fill="#c7d0da" opacity=".6"/>'
        '</svg>',
    "Line":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        '<polyline points="50,150 100,110 150,130 200,80 250,100 300,50 360,70" '
        'fill="none" stroke="#c7d0da" stroke-width="2"/>'
        '</svg>',
    "Area":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        '<polygon points="50,150 100,110 150,130 200,80 250,100 300,50 360,70 '
        '360,180 50,180" fill="#c7d0da" opacity=".5"/>'
        '<polyline points="50,150 100,110 150,130 200,80 250,100 300,50 360,70" '
        'fill="none" stroke="#a8b2bd" stroke-width="2"/>'
        '</svg>',
    "Bar":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        + "".join(f'<rect x="{60+i*45}" y="{180-(40+(i*22)%110)}" '
                  f'width="30" height="{40+(i*22)%110}" fill="#c7d0da"/>'
                  for i in range(7))
        + '</svg>',
    "Histogram":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        + "".join(f'<rect x="{50+i*22}" y="{180-h}" width="20" height="{h}" '
                  f'fill="#c7d0da"/>'
                  for i, h in enumerate(
                      [25, 40, 65, 95, 130, 145, 130, 100, 75, 55, 35, 25, 15]))
        + '</svg>',
    "Box":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        # Three boxes
        + "".join((
            f'<line x1="{100+i*100}" y1="{30+i*10}" '
            f'x2="{100+i*100}" y2="{160-i*10}" stroke="#a8b2bd"/>'
            f'<rect x="{75+i*100}" y="{60+i*10}" width="50" height="{60+i*5}" '
            f'fill="#e6eaef" stroke="#a8b2bd"/>'
            f'<line x1="{75+i*100}" y1="{95+i*10}" '
            f'x2="{125+i*100}" y2="{95+i*10}" stroke="#a8b2bd"/>'
        ) for i in range(3))
        + '</svg>',
    "Violin":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<line x1="40" y1="180" x2="380" y2="180" stroke="#d0d6dd"/>'
        '<line x1="40" y1="20" x2="40" y2="180" stroke="#d0d6dd"/>'
        + "".join(f'<path d="M{100+i*100},40 Q{70+i*100},90 {80+i*100},100 '
                  f'Q{60+i*100},140 {100+i*100},170 Q{140+i*100},140 '
                  f'{120+i*100},100 Q{130+i*100},90 {100+i*100},40 Z" '
                  f'fill="#e6eaef" stroke="#a8b2bd"/>' for i in range(3))
        + '</svg>',
    "Pie":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        '<circle cx="200" cy="110" r="80" fill="#e6eaef" stroke="#a8b2bd"/>'
        '<path d="M200,110 L200,30 A80,80 0 0,1 277,135 Z" fill="#d0d6dd"/>'
        '<path d="M200,110 L277,135 A80,80 0 0,1 220,187 Z" fill="#c7d0da"/>'
        '<path d="M200,110 L220,187 A80,80 0 0,1 130,150 Z" fill="#bfc6ce"/>'
        '</svg>',
    "Correlation heatmap":
        '<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="400" height="220" fill="#fafbfc"/>'
        + "".join(
            f'<rect x="{120+col*40}" y="{30+row*35}" width="40" height="35" '
            f'fill="rgba(167,182,196,{0.2+((row+col)%3)*0.25:.2f})" '
            f'stroke="#fff"/>'
            for row in range(5) for col in range(5))
        + '</svg>',
}


def wireframe_for(chart_type: str) -> str:
    """Return a minimalist SVG showing what the chosen chart shape looks like.
    The output is plain HTML/SVG with low-contrast strokes; it carries no
    real data — purely a layout cue for the user before they click Create."""
    return WIREFRAMES.get(chart_type, WIREFRAMES["Bar"])


# Friendly palette labels — show which framework each palette comes from.
# Internal palette KEYS stay the same so build_figure doesn't need changes;
# only the labels presented in the UI are user-friendly.
PALETTE_OPTIONS: list[tuple[str, str]] = [
    # (UI label, internal key)
    ("Plotly · default (qualitative)", "Plotly"),
    ("Plotly · D3 (qualitative)",      "D3"),
    ("Plotly · Pastel (qualitative)",  "Pastel"),
    ("Plotly · Bold (qualitative)",    "Bold"),
    ("Plotly · Dark24 (qualitative)",  "Dark24"),
    ("ColorBrewer · Set2 (qualitative)", "Set2"),
    ("Matplotlib · Viridis (sequential)", "Viridis"),
]
PALETTE_LABEL_TO_KEY = {lbl: key for lbl, key in PALETTE_OPTIONS}
PALETTE_KEY_TO_LABEL = {key: lbl for lbl, key in PALETTE_OPTIONS}
PALETTE_LABELS = [lbl for lbl, _ in PALETTE_OPTIONS]


def apply_x_binning(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    If the chart config has a binning spec on the X column, return a copy of
    the dataframe with the X column replaced by its binned (categorical)
    version. Otherwise return df unchanged.

    Binning specs supported (cfg["binning"] dict):
        {"enabled": True,
         "method": "cut" | "qcut",
         "n_bins": int,            # used by both
         "width": float,           # OPTIONAL — for cut, custom uniform width
         "edges": list[float],     # OPTIONAL — for cut, explicit edges
         "labels": "range" | "ordinal"}

    Anything missing falls back to safe defaults (10 bins, range labels).
    """
    bcfg = cfg.get("binning") or {}
    if not bcfg.get("enabled"):
        return df
    x = cfg.get("x")
    if not x or x not in df.columns:
        return df
    if not pd.api.types.is_numeric_dtype(df[x]):
        return df  # silently skip — binning a non-numeric X makes no sense

    method = bcfg.get("method", "cut")
    n_bins = int(bcfg.get("n_bins", 10))
    label_mode = bcfg.get("labels", "range")

    out = df.copy()
    series = pd.to_numeric(out[x], errors="coerce")

    try:
        if method == "cut":
            edges = bcfg.get("edges")
            width = bcfg.get("width")
            if edges:
                # User-specified edges
                binned = pd.cut(series, bins=edges, include_lowest=True,
                                duplicates="drop")
            elif width and width > 0:
                # Equal-width bins of a chosen size
                lo = float(series.min())
                hi = float(series.max())
                # Build edges from lo to hi+width step width
                n_steps = int(np.ceil((hi - lo) / width)) if hi > lo else 1
                edges = [lo + i * width for i in range(n_steps + 1)]
                if edges[-1] < hi:
                    edges.append(hi + width)
                binned = pd.cut(series, bins=edges, include_lowest=True,
                                duplicates="drop")
            else:
                # Plain n equal-width bins
                binned = pd.cut(series, bins=n_bins, include_lowest=True,
                                duplicates="drop")
        elif method == "qcut":
            binned = pd.qcut(series, q=n_bins, duplicates="drop")
        else:
            return df  # unknown method — no-op
    except Exception:
        return df  # any binning failure leaves data unchanged

    # Convert interval labels to readable strings
    if label_mode == "ordinal":
        # "Bin 1", "Bin 2", … in sort order
        cats = binned.cat.categories
        mapping = {iv: f"Bin {i+1}" for i, iv in enumerate(cats)}
        out[x] = binned.map(mapping).astype("string")
    else:
        # "[lo, hi]" notation
        def _fmt(iv):
            if pd.isna(iv):
                return None
            return f"[{iv.left:.0f}, {iv.right:.0f}]"
        out[x] = binned.map(_fmt).astype("string")
    return out



def _get_top_n(cfg: dict) -> int | None:
    """
    Return a valid Top N integer from the chart config.

    Top N is intentionally applied only inside the visualization builders.
    It never mutates the dataframe shown in the data preview / dataframe
    fragment.
    """
    raw = cfg.get("top_n")
    if raw in (None, "", "All", "—"):
        return None
    try:
        top_n = int(raw)
    except (TypeError, ValueError):
        return None
    return top_n if top_n > 0 else None


def _apply_top_n_rows(df: pd.DataFrame, cfg: dict,
                      sort_candidates: list[str | None]) -> pd.DataFrame:
    """
    Sort row-level chart data by the first available numeric sort column,
    then keep only .head(top_n). Returns a copy so the source dataframe is
    not affected.
    """
    top_n = _get_top_n(cfg)
    if not top_n:
        return df

    for col in sort_candidates:
        if col and col in df.columns:
            sort_values = pd.to_numeric(df[col], errors="coerce")
            if sort_values.notna().any():
                out = df.copy()
                out["__top_n_sort__"] = sort_values
                return (
                    out.sort_values("__top_n_sort__", ascending=False,
                                    na_position="last")
                       .head(top_n)
                       .drop(columns="__top_n_sort__")
                )
    return df


def _apply_top_n_grouped(df: pd.DataFrame, cfg: dict,
                         value_col: str) -> pd.DataFrame:
    """
    Sort aggregated chart data by its numeric measure, then keep .head(top_n).
    Used by Bar / Pie where the plotted values are already grouped.
    """
    top_n = _get_top_n(cfg)
    if not top_n or value_col not in df.columns:
        return df

    sort_values = pd.to_numeric(df[value_col], errors="coerce")
    if not sort_values.notna().any():
        return df

    out = df.copy()
    out["__top_n_sort__"] = sort_values
    return (
        out.sort_values("__top_n_sort__", ascending=False, na_position="last")
           .head(top_n)
           .drop(columns="__top_n_sort__")
    )

def build_figure(df: pd.DataFrame, cfg: dict):
    """Build a plotly figure from a chart-config dict. Returns (fig, error)."""
    import plotly.express as px

    palette_map = {
        "Plotly": px.colors.qualitative.Plotly,
        "D3": px.colors.qualitative.D3,
        "Pastel": px.colors.qualitative.Pastel,
        "Bold": px.colors.qualitative.Bold,
        "Dark24": px.colors.qualitative.Dark24,
        "Set2": px.colors.qualitative.Set2,
        "Viridis": px.colors.sequential.Viridis,
    }
    seq = palette_map.get(cfg.get("palette"), px.colors.qualitative.Plotly)
    NONE = "—"

    ct = cfg["chart_type"]
    x, y, size = cfg.get("x"), cfg.get("y"), cfg.get("size")
    color_arg = None if cfg.get("color") in (None, NONE) else cfg["color"]
    facet_arg = None if cfg.get("facet") in (None, NONE) else cfg["facet"]
    template = cfg.get("template", "plotly_white")
    ttl = cfg.get("title") or None
    opacity = cfg.get("opacity", 0.8)
    trendline = cfg.get("trendline", False)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Apply binning to the X column if requested. We only bin for chart types
    # where it makes statistical sense — Bar / Box / Violin / Pie — to avoid
    # destroying axis info on Scatter/Line/Area/etc.
    if ct in ("Bar", "Box", "Violin", "Pie"):
        df = apply_x_binning(df, cfg)
    # Histogram already has built-in bins; skip our binning to avoid conflict.

    # Top N is visualization-only: these copies are used for plotting only,
    # while the filtered dataframe fragment remains unchanged.
    if ct == "Scatter":
        df = _apply_top_n_rows(df, cfg, [y])
    elif ct == "Bubble":
        df = _apply_top_n_rows(df, cfg, [size, y])
    elif ct in ("Line", "Area"):
        ys_for_top = y if isinstance(y, list) else [y]
        df = _apply_top_n_rows(df, cfg, [c for c in ys_for_top if c])
    elif ct in ("Histogram",):
        df = _apply_top_n_rows(df, cfg, [x])
    elif ct in ("Box", "Violin"):
        df = _apply_top_n_rows(df, cfg, [y])

    common = dict(template=template, color=color_arg,
                  color_discrete_sequence=seq, title=ttl)
    if facet_arg:
        common["facet_col"] = facet_arg

    try:
        if ct == "Scatter":
            fig = px.scatter(df, x=x, y=y, opacity=opacity,
                             trendline="ols" if trendline else None, **common)
            if cfg.get("marker_size"):
                fig.update_traces(marker=dict(size=cfg["marker_size"]))
        elif ct == "Bubble":
            fig = px.scatter(df, x=x, y=y, size=size, opacity=opacity,
                             size_max=40,
                             trendline="ols" if trendline else None, **common)
        elif ct in ("Line", "Area"):
            ys = [c for c in (y if isinstance(y, list) else [y]) if c]
            if not ys:
                return None, "Pick at least one Y column."
            # Sort by X before plotting — Plotly connects points in row order,
            # so an unsorted dataframe produces a "spaghetti" line that zig-zags
            # back and forth across the X axis. Drop rows where X is missing
            # so they don't create visual artefacts at the start/end. The
            # original dataframe is not mutated (we work on a local copy).
            plot_df = df.dropna(subset=[x]).sort_values(by=x, kind="mergesort")
            fn = px.line if ct == "Line" else px.area
            fig = fn(plot_df, x=x, y=ys, **common)
        elif ct == "Bar":
            extra = []
            if color_arg and color_arg != x:
                extra.append(color_arg)
            if facet_arg and facet_arg not in (x, color_arg):
                extra.append(facet_arg)
            bar_kwargs = dict(template=template, color_discrete_sequence=seq, title=ttl)
            if color_arg:
                bar_kwargs["color"] = color_arg
            if facet_arg:
                bar_kwargs["facet_col"] = facet_arg
            agg = cfg.get("agg")
            if y in (None, NONE) or agg == "count":
                grp = df.groupby([x] + extra).size().reset_index(name="count")
                grp = _apply_top_n_grouped(grp, cfg, "count")
                fig = px.bar(grp, x=x, y="count", **bar_kwargs)
            else:
                grp = df.groupby([x] + extra)[y].agg(agg).reset_index()
                grp = _apply_top_n_grouped(grp, cfg, y)
                fig = px.bar(grp, x=x, y=y, **bar_kwargs)
        elif ct == "Histogram":
            fig = px.histogram(df, x=x, nbins=cfg.get("bins", 30),
                               opacity=opacity, **common)
        elif ct == "Box":
            fig = px.box(df, y=y, x=None if x in (None, NONE) else x, **common)
        elif ct == "Violin":
            fig = px.violin(df, y=y, x=None if x in (None, NONE) else x,
                            box=True, **common)
        elif ct == "Pie":
            if y in (None, NONE):
                vc = df[x].value_counts().reset_index()
                vc.columns = [x, "count"]
                vc = _apply_top_n_grouped(vc, cfg, "count")
                fig = px.pie(vc, names=x, values="count", template=template,
                             title=ttl, color_discrete_sequence=seq)
            else:
                grp = df.groupby(x, dropna=False)[y].sum().reset_index()
                grp = _apply_top_n_grouped(grp, cfg, y)
                fig = px.pie(grp, names=x, values=y, template=template,
                             title=ttl, color_discrete_sequence=seq)
        else:  # Correlation heatmap
            if len(num_cols) < 2:
                return None, "Need at least two numeric columns."
            corr = df[num_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title=ttl or "Correlation heatmap", template=template)

        fig.update_layout(
            showlegend=cfg.get("show_legend", True),
            height=cfg.get("height", 500),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        if cfg.get("log_x"):
            fig.update_xaxes(type="log")
        if cfg.get("log_y"):
            fig.update_yaxes(type="log")
        return fig, None
    except Exception as e:
        return None, str(e)


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column names are unique. If duplicates exist (which makes
    pyarrow/Streamlit rendering fail), rename the 2nd+ occurrences with a
    numeric suffix. Returns the same object if already unique.
    """
    cols = list(df.columns)
    if len(set(cols)) == len(cols):
        return df
    seen: dict[str, int] = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 1
            new_cols.append(c)
    out = df.copy()
    out.columns = new_cols
    return out


def unique_column_name(base: str, existing) -> str:
    """Return `base`, or `base_2`, `base_3`, … so it doesn't clash."""
    taken = set(existing)
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def infer_better_dtype(s: pd.Series) -> pd.Series:
    """
    Try to upgrade a Series to the most useful dtype.

    Order of attempts:
        1. integer  — if every meaningful value parses as a whole number
        2. float    — if every meaningful value parses as a number
        3. datetime — if every meaningful value parses as a date AND the
                      column isn't already numeric (don't reinterpret
                      genuine numbers as Unix timestamps)
        4. fall back to the original Series unchanged

    Empty strings and whitespace-only strings are treated as missing, and
    leading/trailing whitespace is stripped before parsing. This is important
    after a column split: e.g. splitting "20 - 25" or "20-" can leave parts
    like " 25" or "" which must still be recognised as numeric (or empty)
    rather than blocking the whole column from converting.

    NaN values are preserved; rows where conversion would silently fail
    (i.e. produce NaN where the original had a real, non-empty value) cause
    the attempt to be rejected.
    """
    if s.empty:
        return s

    # Already a clean numeric or datetime dtype? leave it alone.
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s

    # Build a cleaned copy: strip whitespace, turn ""/whitespace into NaN.
    cleaned = s.copy()
    if cleaned.dtype == object or pd.api.types.is_string_dtype(cleaned):
        cleaned = cleaned.map(
            lambda v: v.strip() if isinstance(v, str) else v
        )
        cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    meaningful = cleaned.dropna()
    if meaningful.empty:
        # Whole column is blank/NaN — nothing to infer, but normalise the
        # blanks to NaN so downstream code sees a clean missing column.
        return cleaned

    # ---- numeric (int first, then float) ----
    as_num = pd.to_numeric(meaningful, errors="coerce")
    if as_num.notna().all():
        full = pd.to_numeric(cleaned, errors="coerce")
        # Prefer int only when there are no missing values (int64 can't hold
        # NaN) and every value is whole.
        if full.notna().all() and (full % 1 == 0).all():
            try:
                return full.astype("int64")
            except (ValueError, OverflowError):
                pass
        return full  # float (NaN-safe)

    # ---- datetime ----
    as_dt = pd.to_datetime(meaningful, errors="coerce")
    if as_dt.notna().all():
        return pd.to_datetime(cleaned, errors="coerce")

    return s


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Data Analysis Studio")
st.caption(
    "Upload data, explore it, clean it with a logged recipe, visualize, and export."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Upload & Overview", "🧹 Cleaning & Prep", "📈 Visualization", "📄 Report & Export"]
)


# ===========================================================================
# TAB 1 — UPLOAD & OVERVIEW
# ===========================================================================
with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Load your data")
        source = st.radio(
            "Source",
            ["Upload a file", "Google Sheets URL"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if source == "Upload a file":
            uploaded = st.file_uploader(
                "Drop a CSV, Excel or JSON file",
                type=["csv", "xlsx", "xls", "json"],
                help="Supported formats: CSV, Excel (.xlsx/.xls), JSON",
            )
            if uploaded is not None and st.button("Load file", type="primary"):
                try:
                    df = load_uploaded_file(uploaded)
                    st.session_state.df_original = df.copy()
                    st.session_state.df = df.copy()
                    st.session_state.file_name = uploaded.name
                    st.session_state.recipe = []
                    st.session_state.load_error = None
                except Exception as e:
                    st.session_state.load_error = str(e)
                    st.error(f"❌ Could not load the file. {e}")

        else:  # Google Sheets
            url = st.text_input(
                "Google Sheets URL",
                placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0",
                help="The sheet must be shared as 'Anyone with the link – Viewer'.",
            )
            if url and st.button("Load sheet", type="primary"):
                try:
                    df = load_google_sheet(url)
                    st.session_state.df_original = df.copy()
                    st.session_state.df = df.copy()
                    st.session_state.file_name = "google_sheet"
                    st.session_state.recipe = []
                    st.session_state.load_error = None
                except Exception as e:
                    st.session_state.load_error = str(e)
                    st.error(f"❌ Could not load the sheet. {e}")

    with right:
        st.subheader("Current dataframe")
        if st.session_state.df is not None:
            with st.container(border=True):
                st.caption("File name")
                st.markdown(f"**{st.session_state.file_name or '—'}**")

                if st.button("🔄 Reset Session", use_container_width=True):
                    reset_session()
                    st.rerun()

                undo_disabled = len(st.session_state.get("recipe") or []) == 0
                if st.button(
                    "↩️ Undo",
                    use_container_width=True,
                    disabled=undo_disabled,
                    help="Cancel the latest transformation and remove it from the recipe.",
                ):
                    ok, msg, errors = undo_last_recipe_action()
                    if ok:
                        if errors:
                            msg += " Replay warnings: " + "; ".join(errors[:3])
                        flash(msg, tab="tab1")
                    else:
                        st.warning(msg)
                    st.rerun()
        else:
            st.info("No dataset loaded yet.")

    # ---------- Overview ------------------------------------------------------
    if st.session_state.df is not None:
        st.divider()
        # Defensive repair of any duplicate column names before rendering.
        _healed = dedupe_columns(st.session_state.df)
        if _healed is not st.session_state.df:
            st.session_state.df = _healed
        df = st.session_state.df
        blocks = cached_overview_blocks(df, df_fingerprint(df))

        # Live status: original vs changed, refresh time, rows, and
        # percent of rows with ZERO missing values across all columns.
        # If a row has even one missing value, it doesn't count as complete.
        now_str = datetime.now().strftime("%H:%M:%S")
        steps_applied = len(st.session_state.recipe)
        status_word = "Original data" if steps_applied == 0 else "Changed"
        status_emoji = "🟢" if steps_applied == 0 else "🟡"
        if len(df):
            n_complete = int(df.notna().all(axis=1).sum())
            pct_complete = n_complete / len(df) * 100
        else:
            n_complete = 0; pct_complete = 0.0

        st.markdown(
            f"""
            <div style='
                background:linear-gradient(90deg,#065f46 0%,#10b981 100%);
                color:#fff; padding:12px 18px; border-radius:8px;
                margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between;
                            align-items:center; gap:18px; flex-wrap:wrap;'>
                    <div>
                        <div style='font-size:.78rem; opacity:.85;
                                    letter-spacing:.5px; text-transform:uppercase;'>
                            Status
                        </div>
                        <div style='font-size:1rem; font-weight:600;'>
                            {status_emoji} {status_word}
                            <span style='opacity:.8; font-weight:400;'>
                              · {steps_applied} step(s)
                            </span>
                        </div>
                    </div>
                    <div>
                        <div style='font-size:.78rem; opacity:.85;
                                    letter-spacing:.5px; text-transform:uppercase;'>
                            Rows
                        </div>
                        <div style='font-size:1rem; font-weight:600;'>
                            {len(df):,}
                        </div>
                    </div>
                    <div>
                        <div style='font-size:.78rem; opacity:.85;
                                    letter-spacing:.5px; text-transform:uppercase;'>
                            Complete rows
                        </div>
                        <div style='font-size:1rem; font-weight:600;'>
                            {pct_complete:.1f}%
                            <span style='opacity:.8; font-weight:400;'>
                              ({n_complete:,})
                            </span>
                        </div>
                    </div>
                    <div>
                        <div style='font-size:.78rem; opacity:.85;
                                    letter-spacing:.5px; text-transform:uppercase;'>
                            Refreshed
                        </div>
                        <div style='font-size:1rem; font-weight:600;'>
                            {now_str}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preview — comes immediately after status, no intermediate strip
        st.subheader(f"Preview · updated {now_str}")
        st.dataframe(
            st.session_state.df.head(50),
            use_container_width=True, height=280,
        )

        # Schema & nulls
        st.subheader("Schema, dtypes & missingness")
        st.dataframe(blocks["info"], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download schema as CSV",
            df_to_csv_bytes(blocks["info"]),
            file_name="schema.csv",
            mime="text/csv",
        )

        # ---- Rename columns ----
        with st.expander("✏️ Rename columns"):
            st.caption(
                "Edit the **new name** column to rename. Leave a name unchanged "
                "to keep it. Names must be unique and non-empty."
            )
            rename_tbl = pd.DataFrame({
                "current_name": list(df.columns),
                "new_name": list(df.columns),
            })
            edited_names = st.data_editor(
                rename_tbl,
                use_container_width=True,
                num_rows="fixed",
                disabled=["current_name"],
                hide_index=True,
                key="rename_editor",
                height=min(420, 40 + 35 * len(rename_tbl)),
            )

            mapping = {
                row["current_name"]: str(row["new_name"]).strip()
                for _, row in edited_names.iterrows()
                if str(row["new_name"]).strip() != str(row["current_name"])
            }

            # Validate
            problems = []
            if mapping:
                new_full = [
                    mapping.get(c, c) for c in df.columns
                ]
                if any(n == "" for n in mapping.values()):
                    problems.append("New names cannot be empty.")
                if len(set(new_full)) != len(new_full):
                    problems.append("Resulting column names must be unique.")

            st.caption(f"Pending renames: **{len(mapping)}**")
            for p in problems:
                st.warning(p)

            if st.button("Apply renames",
                         disabled=(len(mapping) == 0 or bool(problems))):
                before_cols = df.shape[1]
                df = df.rename(columns=mapping)
                st.session_state.df = df
                log_step(
                    "rename_columns",
                    {"mapping": mapping},
                    len(df), len(df),  # row count unchanged
                )
                flash(
                    f"Renamed {len(mapping)} column(s): "
                    + ", ".join(f"{k} → {v}" for k, v in mapping.items()),
                    tab="tab1",
                )
                st.rerun()

        # ---- Delete columns (with confirmation) ----
        with st.expander("🗑️ Delete columns"):
            st.caption(
                "Permanently remove columns from the working dataframe. "
                "You'll be asked to confirm before anything is deleted."
            )
            cols_to_delete = st.multiselect(
                "Select column(s) to delete",
                options=list(df.columns),
                key="delete_cols_select",
            )

            # Step 1: arm the deletion (does NOT delete yet)
            if st.button(
                "Delete selected columns",
                disabled=len(cols_to_delete) == 0,
            ):
                st.session_state.pending_delete = cols_to_delete
                st.rerun()

            # Step 2: confirmation prompt appears after arming
            pending = st.session_state.pending_delete
            if pending:
                st.warning(
                    f"⚠️ Are you sure you want to delete **{len(pending)}** "
                    f"column(s)? This cannot be undone (except via "
                    f"*Revert to original data* on the Cleaning tab).\n\n"
                    f"Columns: {', '.join(f'`{c}`' for c in pending)}"
                )
                col_yes, col_no = st.columns(2)
                if col_yes.button("✅ Yes, delete", type="primary",
                                  use_container_width=True):
                    before_cols = df.shape[1]
                    # Only drop columns that still exist (guard against
                    # stale selections after other edits).
                    to_drop = [c for c in pending if c in df.columns]

                    if len(to_drop) >= df.shape[1]:
                        st.error("You cannot delete all columns. Keep at least one column.")
                        st.stop()

                    df = df.drop(columns=to_drop)
                    st.session_state.df = df
                    log_step(
                        "delete_columns",
                        {"columns": to_drop,
                         "cols_before": before_cols,
                         "cols_after": df.shape[1]},
                        len(df), len(df),  # row count unchanged
                    )
                    st.session_state.pending_delete = None
                    flash(f"Deleted {len(to_drop)} column(s): "
                          f"{', '.join(to_drop)}.", tab="tab1")
                    st.rerun()
                if col_no.button("Cancel", use_container_width=True):
                    st.session_state.pending_delete = None
                    st.rerun()

        # Flash a confirmation of the most recent overview action
        show_flash("tab1")

        # Numeric summary
        if not blocks["numeric"].empty:
            st.subheader("Numeric summary")
            st.dataframe(blocks["numeric"], use_container_width=True)
            st.download_button(
                "⬇️ Download numeric summary",
                df_to_csv_bytes(blocks["numeric"].reset_index()),
                file_name="numeric_summary.csv",
                mime="text/csv",
            )

        # Categorical summary
        if not blocks["categorical"].empty:
            st.subheader("Categorical summary")
            st.dataframe(blocks["categorical"], use_container_width=True)
            st.download_button(
                "⬇️ Download categorical summary",
                df_to_csv_bytes(blocks["categorical"].reset_index()),
                file_name="categorical_summary.csv",
                mime="text/csv",
            )

        # Duplicates
        st.subheader("Duplicates")
        st.write(f"Full-row duplicates: **{int(df.duplicated().sum()):,}**")
        subset_cols = st.multiselect(
            "Check duplicates on a subset of columns",
            options=list(df.columns),
            help="Useful when only a few key columns identify a record.",
        )
        if subset_cols:
            n_dup_subset = int(df.duplicated(subset=subset_cols).sum())
            st.write(
                f"Duplicates considering only `{', '.join(subset_cols)}`: "
                f"**{n_dup_subset:,}**"
            )


# ===========================================================================
# TAB 2 — CLEANING & PREP
# ===========================================================================
with tab2:
    if st.session_state.df is None:
        st.info("📁 Load a dataset on the **Upload & Overview** tab first.")
        st.stop()

    # Defensive: if a past operation produced duplicate column names, fix them
    # now so the preview (and pyarrow) don't crash. Write the repair back.
    _healed = dedupe_columns(st.session_state.df)
    if _healed is not st.session_state.df:
        st.session_state.df = _healed
        st.warning(
            "Some columns had duplicate names and were automatically renamed "
            "with numeric suffixes so the table can render."
        )

    df: pd.DataFrame = st.session_state.df

    # Live status: always reflects the current session_state.df
    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div style='
            display:flex; justify-content:space-between; align-items:center;
            background:linear-gradient(90deg,#065f46 0%,#10b981 100%);
            color:#fff; padding:10px 16px; border-radius:8px;
            margin-bottom:10px; font-size:.9rem;'>
            <span>🟢 <b>Live preview</b> — {len(df):,} rows × {df.shape[1]} columns
                · {len(st.session_state.recipe)} step(s) applied</span>
            <span style='opacity:.85'>refreshed at {now_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Flash the result of the most recent action (survives the st.rerun)
    show_flash("tab2")

    st.subheader("Cleaning & transformation")
    st.caption(
        "Every action below is logged with parameters + timestamp into a JSON "
        "recipe you can export and replay later. The dataframe preview at the "
        "bottom and the status banner above always reflect the current state."
    )


    st.markdown('---')
    st.markdown("### ⭐ Common cleaning")
    st.caption("The operations used most often — filtering, handling missing values, and recoding values.")

    # ------------------------------------------------------------------ FILTERS
    with st.expander("🔍 Multi-level filtering", expanded=True):
        st.markdown(
            "Add filters and choose how each one combines with the previous: "
            "**AND** (both must be true), **OR** (either can be true), "
            "**NOT** (this condition must be false, combined with AND)."
        )
        n_filters = st.number_input("Number of filters", 0, 10, 0, 1)
        masks: list[pd.Series] = []
        connectors: list[str] = []      # connector applied to the i-th filter (i>=1)
        filter_log: list[dict] = []

        for i in range(int(n_filters)):
            # Connector row (shown for the 2nd filter onwards)
            if i > 0:
                conn = st.radio(
                    f"Connector before filter #{i+1}",
                    ["AND", "OR", "NOT"],
                    horizontal=True,
                    key=f"fconn_{i}",
                )
                connectors.append(conn)

            cols = st.columns([3, 2, 3])
            col = cols[0].selectbox(
                f"Column #{i+1}", df.columns, key=f"fcol_{i}"
            )
            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                op = cols[1].selectbox(
                    "Op", ["==", "!=", ">", ">=", "<", "<=", "between"],
                    key=f"fop_{i}",
                )
                if op == "between":
                    s_nonnull = series.dropna()
                    if s_nonnull.empty:
                        # Column is entirely null — between makes no sense; fall
                        # back to a manual numeric input so the UI keeps working.
                        cols[2].caption(
                            f"`{col}` has no non-null values — pick a number "
                            "to compare against."
                        )
                        val = cols[2].number_input("Value", key=f"fval_{i}")
                        mask = series == val
                    else:
                        lo = float(np.nanmin(series))
                        hi = float(np.nanmax(series))
                        if lo == hi:
                            # Constant column — st.slider would raise because
                            # min and max are equal. Switch to an exact-match
                            # comparison against that single value.
                            cols[2].caption(
                                f"`{col}` is constant at **{lo:g}** — "
                                "filter keeps rows equal to this value."
                            )
                            val = [lo, hi]  # tuple-like for series.between()
                            mask = series == lo
                        else:
                            val = cols[2].slider(
                                "Range", lo, hi, (lo, hi), key=f"fval_{i}"
                            )
                            mask = series.between(*val)
                else:
                    val = cols[2].number_input("Value", key=f"fval_{i}")
                    mask = {
                        "==": series == val, "!=": series != val,
                        ">": series > val, ">=": series >= val,
                        "<": series < val, "<=": series <= val,
                    }[op]
            else:
                op = cols[1].selectbox(
                    "Op", ["is in", "not in", "contains"], key=f"fop_{i}",
                )
                if op in ("is in", "not in"):
                    choices = cols[2].multiselect(
                        "Values",
                        sorted(series.dropna().astype(str).unique().tolist())[:5000],
                        key=f"fval_{i}",
                    )
                    s = series.astype(str)
                    mask = s.isin(choices) if op == "is in" else ~s.isin(choices)
                    val = choices
                else:  # contains
                    val = cols[2].text_input("Substring", key=f"fval_{i}")
                    mask = series.astype(str).str.contains(
                        val or "", case=False, na=False
                    )

            masks.append(mask.fillna(False))
            filter_log.append({"column": col, "op": op, "value": val})

        if st.button("Apply filters", type="primary", disabled=n_filters == 0):
            # Combine masks left-to-right using the chosen connectors.
            # NOT means: AND (NOT this_mask)
            combined = masks[0]
            for i, m in enumerate(masks[1:], start=0):
                conn = connectors[i] if i < len(connectors) else "AND"
                if conn == "AND":
                    combined = combined & m
                elif conn == "OR":
                    combined = combined | m
                else:  # NOT
                    combined = combined & (~m)

            before = len(df)
            df = df.loc[combined].reset_index(drop=True)
            st.session_state.df = df
            log_step(
                "filter",
                {"filters": filter_log, "connectors": connectors},
                before, len(df),
            )
            flash(f"Filtered: {before - len(df):,} rows removed.")
            st.rerun()

    # ------------------------------------------------------------------ NULLS
    with st.expander("🕳️ Missing values"):
        nulls = df.isna().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            st.success("No missing values 🎉")
        else:
            st.dataframe(
                nulls.rename("missing").to_frame().assign(
                    pct=lambda x: (x["missing"] / len(df) * 100).round(2)
                ),
                use_container_width=True,
            )

            strategy = st.selectbox(
                "Strategy",
                [
                    "Drop rows with any null",
                    "Drop rows null in selected columns",
                    "Fill with mean (numeric)",
                    "Fill with median (numeric)",
                    "Fill with mode (numeric)",
                    "Fill with most frequent value (categorical)",
                    "Fill with mean by group",
                    "Fill with median by group",
                    "Fill with custom value",
                    "Forward-fill (ffill)",
                    "Backward-fill (bfill)",
                ],
            )

            target_cols: list[str] = []
            group_col: str | None = None
            fill_value: Any = None
            sort_specs: list[dict] = []

            if strategy != "Drop rows with any null":
                # The pool of pickable columns depends on the strategy:
                # mode is only meaningful on numeric columns, most-frequent
                # on text/categorical. For other strategies, any column with
                # nulls is offered.
                cols_with_nulls = [c for c in df.columns if df[c].isna().any()]
                if strategy == "Fill with mode (numeric)":
                    pool = [c for c in cols_with_nulls
                            if pd.api.types.is_numeric_dtype(df[c])]
                    if not pool:
                        st.info("No numeric columns with nulls available.")
                elif strategy == "Fill with most frequent value (categorical)":
                    pool = [c for c in cols_with_nulls
                            if not pd.api.types.is_numeric_dtype(df[c])
                            and not pd.api.types.is_datetime64_any_dtype(df[c])]
                    if not pool:
                        st.info("No categorical/text columns with nulls available.")
                else:
                    pool = cols_with_nulls
                target_cols = st.multiselect("Apply to columns", pool)

            if "by group" in strategy:
                group_col = st.selectbox(
                    "Group by (category column)",
                    [c for c in df.columns if c not in target_cols],
                )

            if strategy == "Fill with custom value":
                fill_value = st.text_input("Value to insert", "")

            # --- Live preview of the exact replacement value, before applying.
            # For mode / most-frequent strategies we precompute the value per
            # column so the user sees what each null cell would become.
            if strategy in (
                "Fill with mode (numeric)",
                "Fill with most frequent value (categorical)",
            ) and target_cols:
                preview_rows = []
                for c in target_cols:
                    series = df[c].dropna()
                    if series.empty:
                        repl = None
                    else:
                        m = series.mode()
                        repl = m.iloc[0] if not m.empty else None
                    n_nulls = int(df[c].isna().sum())
                    preview_rows.append({
                        "column": c,
                        "value to insert": (repr(repl) if repl is not None
                                            else "— (no non-null values)"),
                        "nulls to fill": n_nulls,
                    })
                st.caption("**Value that will be inserted per column:**")
                st.dataframe(
                    pd.DataFrame(preview_rows),
                    use_container_width=True, hide_index=True,
                )

            # --- ffill / bfill: optional sort-order with tie-breaker ---
            if strategy in ("Forward-fill (ffill)", "Backward-fill (bfill)"):
                st.caption(
                    "Forward-fill and backward-fill propagate adjacent "
                    "non-null values into nulls. Choosing a sort order makes "
                    "the result deterministic — for time-series data, sort by "
                    "the date column; for non-date data, pick any column that "
                    "defines the order you want."
                )

                use_sort = st.checkbox(
                    "Sort rows before filling", value=True,
                    key="mv_use_sort",
                    help="When OFF, ffill/bfill propagates in the current row "
                         "order of the dataframe.",
                )
                if use_sort:
                    sort_cols_opts = [c for c in df.columns if c not in target_cols] \
                                     or list(df.columns)
                    s1c, s2c = st.columns([3, 2])
                    primary = s1c.selectbox(
                        "Primary sort column",
                        sort_cols_opts, key="mv_sort_primary",
                        help="Any column type works — date, numeric, or text.",
                    )
                    primary_order = s2c.radio(
                        "Order",
                        ["Ascending", "Descending"],
                        horizontal=True, key="mv_sort_primary_dir",
                    )
                    sort_specs.append({
                        "column": primary,
                        "ascending": primary_order == "Ascending",
                    })

                    use_tiebreaker = st.checkbox(
                        "Add a tie-breaker column (used when primary values "
                        "are equal)", value=False, key="mv_use_tie",
                    )
                    if use_tiebreaker:
                        tb_opts = [c for c in sort_cols_opts if c != primary]
                        if tb_opts:
                            t1c, t2c = st.columns([3, 2])
                            secondary = t1c.selectbox(
                                "Tie-breaker column",
                                tb_opts, key="mv_sort_secondary",
                            )
                            secondary_order = t2c.radio(
                                "Order ",
                                ["Ascending", "Descending"],
                                horizontal=True, key="mv_sort_secondary_dir",
                            )
                            sort_specs.append({
                                "column": secondary,
                                "ascending": secondary_order == "Ascending",
                            })
                        else:
                            st.info("No other columns available for tie-breaking.")

            if st.button("Apply missing-value strategy"):
                before = len(df)
                # Snapshot the columns we will touch so we can count
                # how many NaN values we actually filled.
                _cols_for_count = target_cols or list(df.columns)
                nan_before = int(df[_cols_for_count].isna().sum().sum()) \
                             if _cols_for_count else 0
                params: dict[str, Any] = {"strategy": strategy, "columns": target_cols}

                if strategy == "Drop rows with any null":
                    df = df.dropna().reset_index(drop=True)

                elif strategy == "Drop rows null in selected columns":
                    df = df.dropna(subset=target_cols).reset_index(drop=True)

                elif strategy in ("Fill with mean (numeric)", "Fill with median (numeric)"):
                    func = "mean" if "mean" in strategy else "median"
                    for c in target_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = df[c].fillna(getattr(df[c], func)())

                elif strategy == "Fill with mode (numeric)":
                    for c in target_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            m = df[c].dropna().mode()
                            if not m.empty:
                                df[c] = df[c].fillna(m.iloc[0])

                elif strategy == "Fill with most frequent value (categorical)":
                    for c in target_cols:
                        # Skip numerics + dates — the picker should already
                        # prevent these from being selected, but be defensive.
                        if (pd.api.types.is_numeric_dtype(df[c])
                                or pd.api.types.is_datetime64_any_dtype(df[c])):
                            continue
                        m = df[c].dropna().mode()
                        if not m.empty:
                            df[c] = df[c].fillna(m.iloc[0])

                elif "by group" in strategy:
                    func = "mean" if "mean" in strategy else "median"
                    params["group_by"] = group_col
                    for c in target_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = df.groupby(group_col)[c].transform(
                                lambda s: s.fillna(getattr(s, func)())
                            )

                elif strategy == "Fill with custom value":
                    params["value"] = fill_value
                    for c in target_cols:
                        try:
                            v = df[c].dtype.type(fill_value)
                        except Exception:
                            v = fill_value
                        df[c] = df[c].fillna(v)

                elif strategy in ("Forward-fill (ffill)", "Backward-fill (bfill)"):
                    method = "ffill" if "ffill" in strategy else "bfill"
                    params["method"] = method
                    params["sort_by"] = sort_specs  # may be empty
                    if sort_specs:
                        sort_keys = [s["column"] for s in sort_specs]
                        ascending = [s["ascending"] for s in sort_specs]
                        # Sort, fill, then restore original order
                        original_idx = df.index.copy()
                        df = df.sort_values(by=sort_keys, ascending=ascending,
                                            kind="mergesort")
                        for c in target_cols:
                            df[c] = getattr(df[c], method)()
                        df = df.loc[original_idx]
                    else:
                        # No sort — fill in current row order
                        for c in target_cols:
                            df[c] = getattr(df[c], method)()

                # Build a count-based message
                if strategy.startswith("Drop"):
                    affected = before - len(df)
                    msg = (f"{strategy} — {affected:,} row(s) dropped "
                           f"({before:,} → {len(df):,}).")
                else:
                    nan_after = int(df[_cols_for_count].isna().sum().sum()) \
                                if _cols_for_count else 0
                    filled = max(nan_before - nan_after, 0)
                    msg = (f"{strategy} — {filled:,} missing value(s) filled "
                           f"in {len(target_cols)} column(s).")

                st.session_state.df = df
                log_step("missing_values", params, before, len(df))
                flash(msg)
                st.rerun()

        # ---------------------------------------------------------------
        # Drop columns whose % of nulls exceeds a threshold.
        # ---------------------------------------------------------------
        render_drop_sparse_columns_tool(df)

    # ------------------------------------------------------------------ REPLACE VALUES
    with st.expander("🔁 Replace values (mapping table)"):
        st.caption(
            "Pick a column, then edit the **new value** column in the table "
            "below. Leave a row unchanged to keep its current value. "
            "Useful for fixing typos, merging categories, or recoding labels."
        )

        rv_target = st.selectbox(
            "Column to recode",
            list(df.columns),
            key="rv_col",
        )

        uniques = df[rv_target].dropna().unique().tolist()
        # Cap the table to a sane size so the UI stays responsive.
        MAX_VALUES = 1000
        if len(uniques) > MAX_VALUES:
            st.warning(
                f"This column has {len(uniques):,} unique values. "
                f"Only the first {MAX_VALUES} are shown — narrow down with "
                "filters first if you need to remap a long-tail value."
            )
            uniques = uniques[:MAX_VALUES]

        mapping_df = pd.DataFrame(
            {
                "original_value": [str(v) for v in uniques],
                "new_value": [str(v) for v in uniques],
            }
        )

        edited = st.data_editor(
            mapping_df,
            use_container_width=True,
            num_rows="fixed",
            disabled=["original_value"],
            key=f"rv_editor_{rv_target}",
            height=min(400, 40 + 35 * len(mapping_df)),
        )

        # Only build a mapping for rows that were actually changed.
        changes = {
            row["original_value"]: row["new_value"]
            for _, row in edited.iterrows()
            if str(row["original_value"]) != str(row["new_value"])
        }
        st.caption(f"Pending changes: **{len(changes)}**")

        if st.button("Apply value replacements", disabled=len(changes) == 0):
            before = len(df)
            col_series = df[rv_target]
            original_dtype = str(col_series.dtype)

            # Replace using string keys (matches what the editor showed)
            new_series = col_series.astype(str).replace(changes)

            # Restore NaNs that were originally NaN (astype(str) turned them
            # into the literal string "nan").
            new_series = new_series.mask(col_series.isna())

            # First, prefer to keep the original dtype if it still fits…
            try:
                cast = new_series.astype(col_series.dtype)
                final = cast
            except (ValueError, TypeError):
                final = new_series  # leave as object for now

            # …then try to UPGRADE the dtype: if all the new values look
            # numeric or all look like dates, promote the column. This is
            # how "21-25" → "21" turns the column into a real int column.
            promoted = infer_better_dtype(final)
            new_dtype = str(promoted.dtype)

            df[rv_target] = promoted
            st.session_state.df = df

            params = {
                "column": rv_target,
                "mapping": changes,
                "original_dtype": original_dtype,
                "new_dtype": new_dtype,
            }
            log_step("replace_values", params, before, len(df))

            msg = f"Replaced {len(changes)} value(s) in `{rv_target}`."
            if original_dtype != new_dtype:
                msg += f" Column dtype promoted: **{original_dtype} → {new_dtype}**."
            flash(msg)
            st.rerun()

    # ------------------------------------------------------------------ DUPLICATES
    with st.expander("🧬 Find & remove duplicates"):
        st.caption(
            "Count and remove duplicate rows — either full-row duplicates or "
            "duplicates within a subset of columns. Pick which occurrence to "
            "keep, preview every duplicate that would be affected, then apply."
        )

        # --- 1. Subset selection ---
        scope = st.radio(
            "Compare on",
            ["Full row (all columns)", "Subset of columns"],
            horizontal=True, key="dup_scope",
        )
        if scope == "Subset of columns":
            subset_cols = st.multiselect(
                "Columns that define a duplicate",
                list(df.columns),
                key="dup_subset_cols",
                help="Two rows count as duplicates when ALL selected columns "
                     "have the same values.",
            )
        else:
            subset_cols = list(df.columns)

        # --- 2. Which occurrence(s) to keep ---
        keep_choice = st.radio(
            "When duplicates are found, keep",
            ["First occurrence",
             "Last occurrence",
             "Drop all duplicates (keep none)"],
            horizontal=False, key="dup_keep",
            help="“First / Last” keeps one row per duplicate group. "
                 "“Drop all” removes every row that has any duplicate.",
        )
        keep_param: Any
        if keep_choice == "First occurrence":
            keep_param = "first"
        elif keep_choice == "Last occurrence":
            keep_param = "last"
        else:
            keep_param = False  # pandas: drop ALL occurrences

        # --- 3. Count + preview the duplicates ---
        scope_ok = bool(subset_cols)

        if not scope_ok:
            st.info("Pick at least one column to look for duplicates.")
        else:
            # All rows that have at least one duplicate in the chosen subset
            # (keep=False marks every member of every duplicate group).
            all_dup_mask = df.duplicated(subset=subset_cols, keep=False)
            n_total_dup_rows = int(all_dup_mask.sum())

            # Rows pandas would actually REMOVE under the current keep choice
            removed_mask = df.duplicated(subset=subset_cols, keep=keep_param)
            n_would_remove = int(removed_mask.sum())

            scope_label = ("all columns"
                           if scope == "Full row (all columns)"
                           else f"`{', '.join(subset_cols)}`")
            st.markdown(
                f"<div class='step-box'>"
                f"Comparing on: <b>{scope_label}</b><br>"
                f"<span class='rows'>{n_total_dup_rows:,} row(s) participate "
                f"in duplicate groups</span><br>"
                f"<span class='rows'>{n_would_remove:,} row(s) would be "
                f"removed</span> (keep: {keep_choice.lower()})<br>"
                f"{len(df) - n_would_remove:,} row(s) would remain"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Show the duplicates so the user can verify before deleting.
            if n_total_dup_rows > 0:
                show_cols = subset_cols if scope == "Subset of columns" else list(df.columns)
                dup_df = (df.loc[all_dup_mask, show_cols]
                           .sort_values(by=subset_cols, kind="mergesort"))
                # Mark which rows survive vs get removed under current keep choice
                dup_df = dup_df.copy()
                dup_df.insert(
                    0, "_status",
                    np.where(removed_mask.loc[dup_df.index],
                             "❌ would be removed", "✅ would be kept"),
                )
                st.caption(
                    f"Duplicate rows in scope (showing {min(len(dup_df), 500):,} "
                    f"of {len(dup_df):,}):"
                )
                st.dataframe(
                    dup_df.head(500),
                    use_container_width=True, hide_index=False, height=260,
                )

            # --- 4. Apply ---
            apply_disabled = (n_would_remove == 0)
            if st.button(
                "🗑️ Remove duplicates",
                disabled=apply_disabled,
                type="primary",
            ):
                before = len(df)
                df = df.drop_duplicates(
                    subset=subset_cols, keep=keep_param
                ).reset_index(drop=True)
                st.session_state.df = df
                # Translate keep_param back to a serialisable string for the log
                keep_str = ("first" if keep_param == "first"
                            else "last" if keep_param == "last"
                            else "none")
                params = {
                    "scope": ("full_row" if scope == "Full row (all columns)"
                              else "subset"),
                    "columns": subset_cols,
                    "keep": keep_str,
                    "rows_removed": before - len(df),
                }
                log_step("remove_duplicates", params, before, len(df))
                flash(
                    f"Removed {before - len(df):,} duplicate row(s) "
                    f"({before:,} → {len(df):,}) on {scope_label}, "
                    f"keep: {keep_choice.lower()}."
                )
                st.rerun()

    st.markdown('---')
    st.markdown("### 🔢 Numeric columns")
    st.caption("Operations that apply to numeric data.")

    # ------------------------------------------------------------------ NORMALIZE
    with st.expander("📐 Normalization & scaling", expanded=True):
        st.caption(
            "Rescale a numeric column. **Min-max** maps values to [0, 1]; "
            "**z-score** centers to mean 0 and scales to standard deviation 1; "
            "**none** leaves it unchanged."
        )
        norm_num_cols = df.select_dtypes(include="number").columns.tolist()
        if not norm_num_cols:
            st.info("No numeric columns available.")
        else:
            ncol = st.selectbox("Numeric column", norm_num_cols, key="norm_col")
            method = st.radio(
                "Method",
                ["Do nothing", "Min-max normalization (0–1)",
                 "Z-score standardization"],
                key="norm_method",
            )

            col_data = pd.to_numeric(df[ncol], errors="coerce")

            def _stat_table(s: pd.Series) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "min": [s.min()],
                        "max": [s.max()],
                        "mean": [s.mean()],
                        "std": [s.std()],
                        "median": [s.median()],
                    }
                ).round(4)

            # Compute the would-be result so we can show before/after side by side
            if method == "Min-max normalization (0–1)":
                rng = col_data.max() - col_data.min()
                preview = (col_data - col_data.min()) / rng if rng != 0 else col_data * 0.0
            elif method == "Z-score standardization":
                sd = col_data.std()
                preview = (col_data - col_data.mean()) / sd if sd != 0 else col_data * 0.0
            else:
                preview = col_data

            st.markdown("**Statistics — before**")
            st.dataframe(_stat_table(col_data), use_container_width=True, hide_index=True)
            st.markdown("**Statistics — after** (preview)")
            st.dataframe(_stat_table(preview), use_container_width=True, hide_index=True)

            if st.button("Apply scaling", disabled=(method == "Do nothing")):
                before = len(df)
                before_stats = _stat_table(col_data).to_dict("records")[0]
                ok = True
                if method == "Min-max normalization (0–1)":
                    rng = col_data.max() - col_data.min()
                    if rng == 0:
                        st.warning(f"`{ncol}` is constant — min-max is undefined. Nothing applied.")
                        ok = False
                    else:
                        df[ncol] = (col_data - col_data.min()) / rng
                        mkey = "min_max"
                else:  # z-score
                    sd = col_data.std()
                    if sd == 0:
                        st.warning(f"`{ncol}` has zero variance — z-score is undefined. Nothing applied.")
                        ok = False
                    else:
                        df[ncol] = (col_data - col_data.mean()) / sd
                        mkey = "z_score"

                if ok:
                    after_stats = _stat_table(df[ncol]).to_dict("records")[0]
                    st.session_state.df = df
                    log_step(
                        "scale",
                        {"column": ncol, "method": mkey,
                         "before": before_stats, "after": after_stats},
                        before, len(df),
                    )
                    flash(
                        f"Scaled `{ncol}` using {mkey.replace('_', '-')}. "
                        f"Range now [{df[ncol].min():.4g}, {df[ncol].max():.4g}], "
                        f"mean {df[ncol].mean():.4g}, std {df[ncol].std():.4g}."
                    )
                    st.rerun()

    # ------------------------------------------------------------------ OUTLIERS
    with st.expander("📉 Outlier handling (trim or winsorize)"):
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            target = st.selectbox("Numeric column", num_cols, key="out_col")
            col_num = pd.to_numeric(df[target], errors="coerce").dropna()

            # --- Permanent diagnostics: IQR and z-score bounds ---
            q1 = col_num.quantile(0.25)
            q3 = col_num.quantile(0.75)
            iqr = q3 - q1
            iqr_low, iqr_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_iqr_out = int(((col_num < iqr_low) | (col_num > iqr_high)).sum())

            mean_v, std_v = col_num.mean(), col_num.std()
            if std_v and not np.isnan(std_v):
                zscores = (col_num - mean_v) / std_v
                n_z_out = int((zscores.abs() > 3).sum())
            else:
                n_z_out = 0

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**IQR method**")
                st.markdown(
                    f"<div class='step-box'>"
                    f"Q1 = {q1:.4g} · Q3 = {q3:.4g} · IQR = {iqr:.4g}<br>"
                    f"Fences (1.5×IQR): [{iqr_low:.4g}, {iqr_high:.4g}]<br>"
                    f"<span class='rows'>Outliers beyond fences: {n_iqr_out:,}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown("**Z-score method**")
                st.markdown(
                    f"<div class='step-box'>"
                    f"mean = {mean_v:.4g} · std = {std_v:.4g}<br>"
                    f"|z| &gt; 3 bounds: "
                    f"[{mean_v - 3*std_v:.4g}, {mean_v + 3*std_v:.4g}]<br>"
                    f"<span class='rows'>Outliers with |z| &gt; 3: {n_z_out:,}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            action = st.radio(
                "Action",
                ["Trim (drop rows outside range)",
                 "Winsorize / cap (clip values to range)"],
                key="out_action",
            )
            low, high = st.slider(
                "Percentile range",
                0.0, 100.0, (1.0, 99.0), 0.5,
                key="out_slider",
                help="For trimming, rows outside this percentile range of the "
                     "column are dropped. For winsorizing, values outside the "
                     "range are clipped to the boundary instead of removed.",
            )

            if st.button("Apply outlier handling"):
                before = len(df)
                lo_v = df[target].quantile(low / 100)
                hi_v = df[target].quantile(high / 100)

                if action.startswith("Trim"):
                    df = df[df[target].between(lo_v, hi_v)].reset_index(drop=True)
                    st.session_state.df = df
                    removed = before - len(df)
                    log_step(
                        "outlier_trim",
                        {"column": target, "low_pct": low, "high_pct": high,
                         "low_value": float(lo_v), "high_value": float(hi_v)},
                        before, len(df),
                    )
                    flash(
                        f"Trimmed `{target}`: removed {removed:,} row(s) outside "
                        f"[{low}%, {high}%] → bounds [{lo_v:.4g}, {hi_v:.4g}]."
                    )
                    st.rerun()
                else:  # winsorize
                    n_capped = int(
                        ((df[target] < lo_v) | (df[target] > hi_v)).sum()
                    )
                    df[target] = df[target].clip(lower=lo_v, upper=hi_v)
                    st.session_state.df = df
                    log_step(
                        "outlier_winsorize",
                        {"column": target, "low_pct": low, "high_pct": high,
                         "low_value": float(lo_v), "high_value": float(hi_v),
                         "values_capped": n_capped},
                        before, len(df),
                    )
                    flash(
                        f"Winsorized `{target}`: capped {n_capped:,} value(s) to "
                        f"[{lo_v:.4g}, {hi_v:.4g}] (no rows removed)."
                    )
                    st.rerun()

    # ------------------------------------------------------------------ NUMERIC FORMAT
    with st.expander("🎨 Numeric value formatting"):
        st.caption("Adjust how numeric values display & are stored.")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            target = st.selectbox("Column", ["—"] + num_cols, key="fmt_num")
            mode = st.selectbox(
                "Format",
                ["Round to N decimals", "Convert to percentage (×100, suffix %)"],
            )
            decimals = st.number_input("Decimals", 0, 8, 2)
            if st.button("Apply numeric format") and target != "—":
                before = len(df)
                n_non_null = int(df[target].notna().sum())
                if mode == "Round to N decimals":
                    df[target] = df[target].round(int(decimals))
                    params = {"column": target, "mode": "round", "decimals": int(decimals)}
                    msg = (f"Rounded `{target}` to {decimals} decimal(s) — "
                           f"{n_non_null:,} value(s) reformatted.")
                else:
                    df[target] = (df[target] * 100).round(int(decimals)).astype(str) + "%"
                    params = {"column": target, "mode": "to_percent", "decimals": int(decimals)}
                    msg = (f"Converted `{target}` to percentages — "
                           f"{n_non_null:,} value(s) reformatted.")
                st.session_state.df = df
                log_step("numeric_format", params, before, len(df))
                flash(msg)
                st.rerun()

    st.markdown('---')
    st.markdown("### 🔤 Categorical & text columns")
    st.caption("Operations that apply to text, categorical, and date data.")

    # ------------------------------------------------------------------ TEXT OPS
    with st.expander("🔤 Text & categorical operations"):
        text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not text_cols:
            st.info("No text/categorical columns.")
        else:
            c1, c2 = st.columns(2)

            # --- Case change
            with c1:
                st.markdown("**Change case**")
                target = st.selectbox("Column", text_cols, key="case_col")
                case = st.radio(
                    "Case",
                    ["lower", "UPPER", "Title Case"],
                    horizontal=True,
                )
                if st.button("Apply case"):
                    before = len(df)
                    orig = df[target].astype(str).copy()
                    if case == "lower":
                        df[target] = orig.str.lower()
                    elif case == "UPPER":
                        df[target] = orig.str.upper()
                    else:
                        df[target] = orig.str.title()
                    changed = int((orig != df[target].astype(str)).sum())
                    st.session_state.df = df
                    log_step("case_change", {"column": target, "case": case}, before, len(df))
                    flash(f"Case changed to **{case}** in `{target}` — "
                          f"{changed:,} value(s) modified.")
                    st.rerun()

            # --- Split column
            with c2:
                st.markdown("**Split column by separator**")
                target = st.selectbox("Column", text_cols, key="split_col")
                preset = st.selectbox(
                    "Separator",
                    [".", " ", "-", ";", ",", "Custom…"],
                )
                sep = st.text_input("Custom separator", "") if preset == "Custom…" else preset
                max_splits = st.number_input("Max splits (-1 = all)", -1, 20, -1)
                if st.button("Split") and sep:
                    before = len(df)
                    parts = df[target].astype(str).str.split(
                        sep, n=int(max_splits), expand=True
                    )

                    # Build collision-free names. We accumulate into `taken`
                    # so the new names also don't clash with each other, and
                    # the existing dataframe columns are seeded in first. This
                    # prevents the "Duplicate column names" error when the same
                    # column is split more than once.
                    taken = list(df.columns)
                    new_names = []
                    for i in range(parts.shape[1]):
                        name = unique_column_name(f"{target}_part{i+1}", taken)
                        new_names.append(name)
                        taken.append(name)
                    parts.columns = new_names

                    # Auto-promote numeric / date parts so they're immediately
                    # usable in numeric formatting, outlier trimming, charts.
                    promoted_to: dict[str, str] = {}
                    for c in parts.columns:
                        before_dtype = str(parts[c].dtype)
                        parts[c] = infer_better_dtype(parts[c])
                        after_dtype = str(parts[c].dtype)
                        if before_dtype != after_dtype:
                            promoted_to[c] = after_dtype

                    df = pd.concat([df, parts], axis=1)
                    st.session_state.df = df
                    log_step(
                        "split_column",
                        {
                            "column": target,
                            "separator": sep,
                            "max_splits": int(max_splits),
                            "new_columns": parts.columns.tolist(),
                            "promoted_dtypes": promoted_to,
                        },
                        before, len(df),
                    )
                    msg = f"Created {parts.shape[1]} new columns."
                    if promoted_to:
                        promoted_str = ", ".join(
                            f"`{c}` → {d}" for c, d in promoted_to.items()
                        )
                        msg += f" Auto-promoted dtypes: {promoted_str}."
                    flash(msg)
                    st.rerun()

    # ------------------------------------------------------------------ CLEAN NUMERIC STRINGS
    with st.expander("🔢 Clean dirty numeric strings → numeric"):
        st.caption(
            "Strip thousands separators, percent signs, currency symbols, and "
            "other non-numeric characters, then parse the result as a number. "
            "Values that still can't be parsed become missing (NaN). "
            "Useful for columns like `\"1.232\"`, `\"1 232\"`, `\"$12.50\"`, "
            "`\"12%\"`, or `\"3,14\"`."
        )

        text_cols_cn = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if not text_cols_cn:
            st.info("No text/categorical columns available.")
        else:
            cn_targets = st.multiselect(
                "Columns to clean",
                text_cols_cn,
                key="cn_cols",
                help="Pick one or more columns. Each is cleaned independently.",
            )

            # Conversion logic shared by preview and apply.
            def _clean_to_numeric(s: pd.Series) -> tuple[pd.Series, int]:
                """
                Strip everything except digits, decimal point, and minus sign,
                then coerce to numeric. Returns (cleaned_series, new_na_count).
                """
                try:
                    as_str = s.astype(str)
                    # Keep only digits, dot, and minus.
                    stripped = (
                        as_str
                        .str.replace(r"[^\d.\-]", "", regex=True)
                        .replace({"": np.nan})
                    )
                    coerced = pd.to_numeric(stripped, errors="coerce")
                    # New NaN = NaN in result that wasn't NaN in input
                    new_na = int((coerced.isna() & s.notna()).sum())
                    return coerced, new_na
                except Exception:
                    return None, 0  # let caller show the error

            if cn_targets:
                st.markdown("**Preview — first 8 values per column**")
                for col in cn_targets:
                    sample = df[col].head(8)
                    cleaned_sample, _ = _clean_to_numeric(sample)
                    if cleaned_sample is None:
                        st.warning(f"`{col}` — preview unavailable.")
                        continue
                    # Full-column NaN diagnostics
                    _, full_new_na = _clean_to_numeric(df[col])
                    full_non_null = int(df[col].notna().sum())

                    prev_tbl = pd.DataFrame({
                        "before": sample.astype(str).values,
                        "after":  cleaned_sample.astype(str).values,
                    })
                    st.markdown(f"`{col}` → new dtype `{cleaned_sample.dtype}`")
                    st.dataframe(prev_tbl, use_container_width=True,
                                 hide_index=False,
                                 height=min(320, 40 + 35 * len(prev_tbl)))
                    if full_new_na:
                        st.caption(
                            f"⚠️ Applied to the full column, "
                            f"**{full_new_na:,} of {full_non_null:,}** "
                            f"non-null value(s) would become NaN "
                            f"because they can't be parsed as numbers."
                        )
                    else:
                        st.caption(
                            "✅ Every non-null value would parse successfully."
                        )

            if st.button(
                "Apply numeric cleaning",
                key="cn_apply",
                type="primary",
                disabled=not cn_targets,
            ):
                before = len(df)
                results: dict[str, dict] = {}
                failed: list[str] = []
                for col in cn_targets:
                    cleaned, new_na = _clean_to_numeric(df[col])
                    if cleaned is None:
                        failed.append(col)
                        continue
                    df[col] = cleaned
                    results[col] = {
                        "new_dtype": str(cleaned.dtype),
                        "values_to_na": new_na,
                    }

                st.session_state.df = df
                total_to_na = sum(r["values_to_na"] for r in results.values())
                log_step(
                    "clean_numeric_strings",
                    {"columns": list(results.keys()),
                     "values_to_na_per_column": {c: r["values_to_na"]
                                                 for c, r in results.items()},
                     "total_values_to_na": total_to_na},
                    before, len(df),
                )
                msg = (f"Cleaned {len(results)} column(s) to numeric "
                       f"({', '.join(results.keys())}). ")
                if total_to_na:
                    msg += f"{total_to_na:,} value(s) became NaN."
                else:
                    msg += "All values parsed successfully."
                if failed:
                    msg += f" Failed on: {', '.join(failed)}."
                flash(msg)
                st.rerun()

    # ------------------------------------------------------------------ TRIM WHITESPACE
    with st.expander("✂️ Trim leading/trailing whitespace"):
        st.caption(
            "Remove leading and/or trailing whitespace from every value in the "
            "selected text columns. Uses `lstrip()` / `rstrip()` and silently "
            "skips any values that can't be cast to a string."
        )

        text_cols_tr = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if not text_cols_tr:
            st.info("No text/categorical columns available.")
        else:
            tr_targets = st.multiselect(
                "Columns to trim",
                text_cols_tr,
                key="tr_cols",
            )
            tr_mode = st.radio(
                "Where to trim",
                ["Both sides (default)", "Left only (lstrip)",
                 "Right only (rstrip)"],
                key="tr_mode", horizontal=True,
            )

            def _safe_trim(value, mode: str):
                """Apply lstrip/rstrip safely. Non-string values pass through."""
                try:
                    if value is None or pd.isna(value):
                        return value
                    s = str(value)
                    if mode == "Left only (lstrip)":
                        return s.lstrip()
                    if mode == "Right only (rstrip)":
                        return s.rstrip()
                    return s.lstrip().rstrip()
                except Exception:
                    return value

            if tr_targets:
                st.markdown("**Preview — first 8 values per column**")
                for col in tr_targets:
                    sample = df[col].head(8)
                    try:
                        cleaned_sample = sample.map(
                            lambda v: _safe_trim(v, tr_mode)
                        )
                    except Exception:
                        st.warning(f"`{col}` — preview unavailable.")
                        continue
                    # Null-safe full-column change count
                    try:
                        full_clean = df[col].map(lambda v: _safe_trim(v, tr_mode))
                        both_present = df[col].notna() & full_clean.notna()
                        full_changed = int(
                            (both_present
                             & (df[col].astype(str) != full_clean.astype(str))
                            ).sum()
                        )
                    except Exception:
                        full_changed = 0

                    prev_tbl = pd.DataFrame({
                        "before": sample.astype(str).values,
                        "after":  cleaned_sample.astype(str).values,
                    })
                    st.markdown(f"`{col}`")
                    st.dataframe(prev_tbl, use_container_width=True,
                                 hide_index=False,
                                 height=min(320, 40 + 35 * len(prev_tbl)))
                    if full_changed:
                        st.caption(
                            f"**{full_changed:,}** value(s) in `{col}` would "
                            "be modified."
                        )
                    else:
                        st.caption(f"No leading/trailing whitespace found in `{col}`.")

            if st.button(
                "Apply whitespace trim",
                key="tr_apply",
                type="primary",
                disabled=not tr_targets,
            ):
                before = len(df)
                changes_per_col: dict[str, int] = {}
                failed: list[str] = []
                for col in tr_targets:
                    try:
                        new = df[col].map(lambda v: _safe_trim(v, tr_mode))
                        both_present = df[col].notna() & new.notna()
                        n_changed = int(
                            (both_present
                             & (df[col].astype(str) != new.astype(str))
                            ).sum()
                        )
                        df[col] = new
                        changes_per_col[col] = n_changed
                    except Exception:
                        failed.append(col)

                st.session_state.df = df
                total_changed = sum(changes_per_col.values())
                mode_short = ("both" if tr_mode.startswith("Both")
                              else "lstrip" if tr_mode.startswith("Left")
                              else "rstrip")
                log_step(
                    "trim_whitespace",
                    {"columns": list(changes_per_col.keys()),
                     "mode": mode_short,
                     "values_changed_per_column": changes_per_col,
                     "total_values_changed": total_changed},
                    before, len(df),
                )
                msg = (f"Trimmed whitespace ({mode_short}) on "
                       f"{len(changes_per_col)} column(s): "
                       f"{total_changed:,} value(s) modified.")
                if failed:
                    msg += f" Failed on: {', '.join(failed)}."
                flash(msg)
                st.rerun()

    # ------------------------------------------------------------------ RARE GROUPING
    with st.expander("🪣 Group rare categories into 'Other'"):
        st.caption(
            "Replace infrequent values in a categorical column with a single "
            "'Other' label. Useful when long-tail categories add noise without "
            "adding signal."
        )

        text_cols2 = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if not text_cols2:
            st.info("No text/categorical columns available.")
        else:
            rg_col = st.selectbox(
                "Column", text_cols2, key="rare_col",
            )
            vc = cached_value_counts(df, rg_col, df_fingerprint(df))

            mode = st.radio(
                "Threshold by",
                ["Absolute count", "Percentage of rows"],
                horizontal=True, key="rare_mode",
            )

            if mode == "Absolute count":
                cutoff_abs = st.number_input(
                    "Group values with frequency less than",
                    min_value=1, max_value=int(vc.max()) if len(vc) else 1,
                    value=min(10, int(vc.max()) if len(vc) else 1),
                    step=1, key="rare_cutoff_abs",
                    help="Any category occurring strictly fewer times than this "
                         "will be replaced with 'Other'.",
                )
                rare_mask = vc < cutoff_abs
                cutoff_repr = f"< {cutoff_abs} rows"
            else:
                cutoff_pct = st.slider(
                    "Group values with frequency less than (% of rows)",
                    0.0, 50.0, 1.0, 0.1, key="rare_cutoff_pct",
                )
                threshold_n = (cutoff_pct / 100.0) * len(df)
                rare_mask = vc < threshold_n
                cutoff_repr = f"< {cutoff_pct}% ({threshold_n:.1f} rows)"

            n_rare_vals = int(rare_mask.sum())
            n_kept_vals = int((~rare_mask).sum())
            n_rare_rows = int(vc[rare_mask].sum())

            other_label = st.text_input(
                "Replacement label", "Other", key="rare_other_label"
            )

            st.markdown(
                f"<div class='step-box'>"
                f"Threshold: <b>{cutoff_repr}</b><br>"
                f"<span class='rows'>{n_rare_vals:,} value(s) "
                f"({n_rare_rows:,} row(s)) would be merged into "
                f"`{other_label}`</span><br>"
                f"{n_kept_vals:,} value(s) kept as-is"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Preview a few of the rare values
            if n_rare_vals > 0:
                preview_rare = vc[rare_mask].head(15)
                st.caption("Rare values (sample):")
                st.dataframe(
                    preview_rare.rename("count").to_frame(),
                    use_container_width=True,
                )

            if st.button(
                "Apply rare-grouping",
                disabled=(n_rare_vals == 0 or not other_label),
            ):
                before = len(df)
                rare_set = set(vc[rare_mask].index.tolist())
                df[rg_col] = df[rg_col].where(~df[rg_col].isin(rare_set), other_label)
                st.session_state.df = df
                log_step(
                    "group_rare_categories",
                    {"column": rg_col, "threshold": cutoff_repr,
                     "other_label": other_label,
                     "values_merged": n_rare_vals,
                     "rows_relabelled": n_rare_rows},
                    before, len(df),
                )
                flash(
                    f"Merged {n_rare_vals:,} rare value(s) ({n_rare_rows:,} row(s)) "
                    f"in `{rg_col}` into `{other_label}`."
                )
                st.rerun()

    # ------------------------------------------------------------------ ONE-HOT ENCODING
    with st.expander("🟦 One-hot encoding"):
        st.caption(
            "Replace a categorical column with one 0/1 indicator column per "
            "unique value. The new columns are named `<column>_<value>`."
        )

        oh_text_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if not oh_text_cols:
            st.info("No text/categorical columns available.")
        else:
            oh_col = st.selectbox(
                "Column to encode", oh_text_cols, key="onehot_col",
            )
            n_uniques = int(df[oh_col].nunique(dropna=True))
            st.caption(
                f"`{oh_col}` has **{n_uniques:,}** unique value(s) — that's how "
                f"many indicator columns will be created."
            )

            oc1, oc2 = st.columns(2)
            drop_original = oc1.checkbox(
                "Drop original column after encoding", value=True,
                key="onehot_drop",
            )
            drop_first = oc2.checkbox(
                "Drop first category (avoid dummy-variable trap)",
                value=False, key="onehot_drop_first",
                help="Useful for regression models where you only need n-1 "
                     "indicator columns. Leave unchecked to keep all values.",
            )
            prefix_sep = st.text_input(
                "Prefix separator", "_", key="onehot_prefix_sep"
            )

            if n_uniques > 200:
                st.warning(
                    f"⚠️ {n_uniques} unique values is a lot — one-hot encoding "
                    f"would add that many columns. Consider grouping rare "
                    f"categories first."
                )

            if st.button(
                "Apply one-hot encoding",
                disabled=(n_uniques == 0 or n_uniques > 1000),
            ):
                before = len(df)
                before_cols = df.shape[1]
                dummies = pd.get_dummies(
                    df[oh_col],
                    prefix=oh_col,
                    prefix_sep=prefix_sep,
                    drop_first=drop_first,
                    dtype=int,
                )
                # Avoid clobbering existing columns of the same name.
                taken = list(df.columns)
                rename_map: dict = {}
                for c in dummies.columns:
                    new_name = unique_column_name(c, taken)
                    if new_name != c:
                        rename_map[c] = new_name
                    taken.append(new_name)
                if rename_map:
                    dummies = dummies.rename(columns=rename_map)

                df = pd.concat([df, dummies], axis=1)
                if drop_original:
                    df = df.drop(columns=[oh_col])
                st.session_state.df = df
                log_step(
                    "one_hot_encode",
                    {"column": oh_col,
                     "drop_original": drop_original,
                     "drop_first": drop_first,
                     "prefix_sep": prefix_sep,
                     "new_columns": dummies.columns.tolist(),
                     "cols_before": before_cols,
                     "cols_after": df.shape[1]},
                    before, len(df),
                )
                msg = (
                    f"One-hot encoded `{oh_col}`: added {dummies.shape[1]} "
                    f"indicator column(s)"
                )
                if drop_original:
                    msg += "; original column dropped"
                msg += "."
                flash(msg)
                st.rerun()

    # ------------------------------------------------------------------ DATE FORMAT
    with st.expander("📅 Date parsing & formatting"):
        st.caption("Parse a column as dates and choose how it displays.")

        # An eligible date column is one where ALL non-null values can be
        # parsed as datetimes. We exclude purely numeric columns because
        # pd.to_datetime would interpret them as Unix timestamps, which
        # almost never matches user intent. Cached so re-renders don't
        # re-scan every column on huge datasets.
        eligible_cols = cached_eligible_date_cols(df, df_fingerprint(df))

        if not eligible_cols:
            st.info(
                "No columns are eligible for date conversion "
                "(every non-null value in the column must be parsable as a date)."
            )
        else:
            d_target = st.selectbox("Column", eligible_cols, key="fmt_dt")

            fmt_presets = {
                "Full date (YYYY-MM-DD)": "%Y-%m-%d",
                "Full date with time (YYYY-MM-DD HH:MM:SS)": "%Y-%m-%d %H:%M:%S",
                "Day/Month/Year": "%d/%m/%Y",
                "Month/Day/Year": "%m/%d/%Y",
                "Year only": "%Y",
                "Month name + Year (Jan 2024)": "%b %Y",
                "Day name + date (Mon, 05 Jan 2024)": "%a, %d %b %Y",
                "Time only (HH:MM:SS)": "%H:%M:%S",
                "Custom…": None,
            }
            choice = st.selectbox("Output format", list(fmt_presets.keys()))
            if choice == "Custom…":
                d_fmt = st.text_input(
                    "Custom strftime",
                    "%Y-%m-%d",
                    help="Python strftime, e.g. %Y-%m-%d, %d/%m/%Y, %b %d %Y",
                )
            else:
                d_fmt = fmt_presets[choice]
                st.caption(f"Format string: `{d_fmt}`")

            if st.button("Parse & format as date"):
                before = len(df)
                try:
                    parsed = pd.to_datetime(df[d_target], errors="coerce")
                    df[d_target] = parsed.dt.strftime(d_fmt)
                    st.session_state.df = df
                    log_step(
                        "date_format",
                        {"column": d_target, "preset": choice, "format": d_fmt},
                        before, len(df),
                    )
                    flash(f"`{d_target}` reformatted as **{choice}**.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not parse dates: {e}")

    st.markdown('---')
    st.markdown("### 🧰 Other tools")
    st.caption("Less-frequently used utilities that aren't specific to a single column type.")

    # ------------------------------------------------------------------ DTYPE DETECT
    with st.expander("🧬 Change / re-detect column types"):
        st.markdown("**Manual conversion**")
        st.caption(
            "Convert a single column to a specific type. Every combination is "
            "available — numeric → string, string → numeric, float → integer, "
            "anything → datetime, etc. Values that can't be parsed for the "
            "chosen target become missing (NA)."
        )

        mc1, mc2 = st.columns([3, 2])
        conv_col = mc1.selectbox(
            "Column", list(df.columns), key="manual_dtype_col"
        )
        current = str(df[conv_col].dtype)
        mc1.caption(f"Current dtype: `{current}`")

        target_type = mc2.selectbox(
            "Convert to",
            ["String / text", "Integer", "Float", "Boolean",
             "Datetime", "Category"],
            key="manual_dtype_target",
        )

        def _convert(s: pd.Series, target: str):
            """Return (converted_series, error_or_None)."""
            try:
                if target == "String / text":
                    return s.astype("string"), None
                if target == "Integer":
                    num = pd.to_numeric(s, errors="coerce")
                    if num.isna().any():
                        return num.round().astype("Int64"), None  # nullable int
                    return num.round().astype("int64"), None
                if target == "Float":
                    return pd.to_numeric(s, errors="coerce").astype("float64"), None
                if target == "Boolean":
                    truthy = {"true", "1", "yes", "y", "t"}
                    falsy = {"false", "0", "no", "n", "f"}
                    def to_bool(v):
                        if pd.isna(v):
                            return pd.NA
                        t = str(v).strip().lower()
                        if t in truthy:
                            return True
                        if t in falsy:
                            return False
                        return pd.NA
                    return s.map(to_bool).astype("boolean"), None
                if target == "Datetime":
                    return pd.to_datetime(s, errors="coerce"), None
                return s.astype("category"), None  # Category
            except (ValueError, TypeError) as e:
                return None, str(e)

        # Live preview of the conversion result on the actual data
        preview_series, prev_err = _convert(df[conv_col], target_type)
        if prev_err:
            st.warning(f"Preview unavailable: {prev_err}")
        else:
            would_na = int(preview_series.isna().sum() - df[conv_col].isna().sum())
            prev_tbl = pd.DataFrame({
                "original": df[conv_col].head(8).astype(str).values,
                "converted": preview_series.head(8).astype(str).values,
            })
            st.caption(
                f"Preview → new dtype `{preview_series.dtype}`"
                + (f" · {would_na:,} value(s) would become missing"
                   if would_na > 0 else "")
            )
            st.dataframe(prev_tbl, use_container_width=True, hide_index=True)

        if st.button("Convert dtype"):
            before = len(df)
            converted, err = _convert(df[conv_col], target_type)
            if err:
                st.error(f"Could not convert `{conv_col}` to {target_type}: {err}")
            else:
                new_dtype = str(converted.dtype)
                new_nas = int(converted.isna().sum() - df[conv_col].isna().sum())
                df[conv_col] = converted
                st.session_state.df = df
                log_step(
                    "convert_dtype",
                    {"column": conv_col, "from": current, "to": new_dtype,
                     "target_choice": target_type,
                     "values_coerced_to_na": max(new_nas, 0)},
                    before, len(df),
                )
                msg = f"Converted `{conv_col}`: {current} → {new_dtype}."
                if new_nas > 0:
                    msg += f" {new_nas:,} value(s) couldn't be parsed and became missing."
                flash(msg)
                st.rerun()

        st.markdown("---")
        st.markdown("**Auto-detect (bulk promote)**")
        st.caption(
            "Scan every text column and promote it to a numeric or datetime "
            "type when *all* of its values qualify. Run this if you imported "
            "messy data, or after operations that produced numeric strings."
        )

        # Preview what would happen — cached so it doesn't re-scan the
        # whole dataframe on every render.
        proposals = cached_dtype_proposals(df, df_fingerprint(df))

        if not proposals:
            st.success("All columns already have the most useful dtype 🎉")
        else:
            preview_df = pd.DataFrame(
                [
                    {"column": c, "current_dtype": cur, "proposed_dtype": new}
                    for c, (cur, new) in proposals.items()
                ]
            )
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            if st.button("Promote dtypes", type="primary"):
                before = len(df)
                applied: dict[str, str] = {}
                for c, (cur, new) in proposals.items():
                    df[c] = infer_better_dtype(df[c])
                    applied[c] = f"{cur} → {new}"
                st.session_state.df = df
                log_step(
                    "redetect_dtypes",
                    {"changes": applied},
                    before, len(df),
                )
                flash(f"Promoted dtypes on {len(applied)} column(s).")
                st.rerun()

    # ------------------------------------------------------------------ NEW COLUMN
    with st.expander("➕ New column from formula"):
        st.caption(
            "Build a calculation column by adding pieces. Each piece is a "
            "column, a number, an operator, or a function. The formula is "
            "shown live and previewed below."
        )

        num_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = list(df.columns)

        # --- 1. New column name (always first) ---
        new_name = st.text_input(
            "New column name",
            value=st.session_state.get("formula_new_name", "new_column"),
            key="formula_new_name",
        )

        # The formula is stored as an ordered list of "parts". This avoids all
        # the widget-state issues of a free-text builder and makes the formula
        # easy to inspect & edit piece by piece.
        if "formula_parts" not in st.session_state:
            st.session_state.formula_parts = []
        parts: list[dict] = st.session_state.formula_parts

        # --- 2. Quick templates ---
        with st.popover("⚡ Quick templates"):
            st.caption("One-click formulas. Picking one replaces the current formula.")
            tpl = st.selectbox(
                "Template",
                ["— none —",
                 "mean of columns",
                 "sum of columns",
                 "std of columns",
                 "median of columns",
                 "min of columns",
                 "max of columns",
                 "ratio (a / b)",
                 "log of a column"],
                key="formula_template",
            )
            if tpl == "ratio (a / b)" and num_cols:
                tpl_a = st.selectbox("a (numerator)", num_cols, key="tpl_a")
                tpl_b = st.selectbox("b (denominator)", num_cols, key="tpl_b")
                if st.button("Apply template", key="apply_tpl_ratio"):
                    st.session_state.formula_parts = [
                        {"kind": "col", "name": tpl_a},
                        {"kind": "op", "value": "/"},
                        {"kind": "col", "name": tpl_b},
                    ]
                    st.rerun()
            elif tpl == "log of a column" and num_cols:
                tpl_c = st.selectbox("Column", num_cols, key="tpl_log_col")
                if st.button("Apply template", key="apply_tpl_log"):
                    st.session_state.formula_parts = [
                        {"kind": "fn", "name": "log", "args": [tpl_c]},
                    ]
                    st.rerun()
            elif tpl not in ("— none —", "ratio (a / b)", "log of a column") and num_cols:
                agg = tpl.split(" ", 1)[0]
                tpl_picks = st.multiselect("Columns", num_cols, key="tpl_picks")
                if st.button("Apply template", key="apply_tpl_agg") and tpl_picks:
                    st.session_state.formula_parts = [
                        {"kind": "fn", "name": agg, "args": list(tpl_picks)},
                    ]
                    st.rerun()

        st.markdown("---")
        st.markdown("**Add a piece to the formula**")

        # --- 3. Add piece controls (three modes via tabs for clarity) ---
        mode_col, mode_num, mode_op, mode_fn = st.tabs(
            ["📊 Column", "🔢 Number", "🧮 Operator", "ƒ Function"]
        )

        with mode_col:
            ac1, ac2 = st.columns([3, 1])
            pick = ac1.selectbox(
                "Column to add", all_cols, key="add_col_pick",
            )
            if ac2.button("Add", key="add_col_btn", use_container_width=True,
                          type="primary"):
                parts.append({"kind": "col", "name": pick})
                st.rerun()

        with mode_num:
            an1, an2 = st.columns([3, 1])
            nval = an1.number_input(
                "Number to add", value=0.0, format="%g",
                key="add_num_val",
            )
            if an2.button("Add", key="add_num_btn", use_container_width=True,
                          type="primary"):
                parts.append({"kind": "num", "value": float(nval)})
                st.rerun()

        with mode_op:
            ops = [
                ("+", "Addition"),
                ("-", "Subtraction"),
                ("*", "Multiplication"),
                ("/", "Division"),
                ("**", "Power"),
                ("(", "Open parenthesis"),
                (")", "Close parenthesis"),
            ]
            cols_op = st.columns(len(ops))
            for col, (sym, label) in zip(cols_op, ops):
                if col.button(sym, key=f"add_op_{sym}",
                              use_container_width=True, help=label):
                    parts.append({"kind": "op", "value": sym})
                    st.rerun()

        with mode_fn:
            # Two categories of functions:
            #   transforms — row-wise elementwise (log/log10/sqrt/abs)
            #   aggregates — return one number per column (or per group when
            #                a group-by category column is selected) and the
            #                value is broadcast to every row
            fn_kind = st.radio(
                "Function type",
                ["Aggregate (whole column or per group)",
                 "Transform (row-wise)"],
                key="add_fn_kind",
            )
            if fn_kind.startswith("Aggregate"):
                aggs = ["mean", "median", "std", "min", "max", "sum", "count"]
                af1, af2 = st.columns([2, 3])
                fn = af1.selectbox("Aggregate function", aggs, key="add_fn_agg_name")
                col_arg = af2.selectbox(
                    "Column to aggregate", num_cols, key="add_fn_agg_col",
                    help="The selected aggregate is computed across this whole "
                         "column (e.g. max gives the single highest value).",
                )
                # Optional group-by
                cat_or_str_cols = df.select_dtypes(
                    include=["object", "category", "string", "bool"]
                ).columns.tolist()
                group_choice = st.selectbox(
                    "Optionally group by (returns the aggregate per group)",
                    ["— no grouping (whole column) —"] + cat_or_str_cols,
                    key="add_fn_groupby",
                )
                group_by = None if group_choice.startswith("—") else group_choice
                if st.button("Add aggregate", key="add_fn_agg_btn", type="primary",
                             disabled=not col_arg):
                    parts.append({
                        "kind": "fn", "name": fn,
                        "col": col_arg, "group_by": group_by,
                        "agg_mode": "aggregate",
                    })
                    st.rerun()
            else:
                transforms = ["log", "log10", "sqrt", "abs"]
                af1, af2 = st.columns([2, 3])
                fn = af1.selectbox("Transform", transforms, key="add_fn_tr_name")
                col_arg = af2.selectbox(
                    "Column", num_cols, key="add_fn_tr_col",
                )
                if st.button("Add transform", key="add_fn_tr_btn", type="primary",
                             disabled=not col_arg):
                    parts.append({
                        "kind": "fn", "name": fn,
                        "col": col_arg, "group_by": None,
                        "agg_mode": "transform",
                    })
                    st.rerun()

        # --- 4. Display the assembled formula with per-piece delete buttons ---
        def part_label(p: dict) -> str:
            k = p["kind"]
            if k == "col":   return f"`{p['name']}`"
            if k == "num":   return f"{p['value']:g}"
            if k == "op":    return p["value"]
            if k == "fn":
                # Support both new (col/group_by) and legacy (args) structure
                if "col" in p:
                    label = f"{p['name']}(`{p['col']}`)"
                    if p.get("group_by"):
                        label += f" by `{p['group_by']}`"
                    return label
                return f"{p['name']}({', '.join('`'+a+'`' for a in p.get('args', []))})"
            return "?"

        st.markdown("---")
        st.markdown("**Current formula**")

        if not parts:
            st.info("No pieces yet. Add one above.")
        else:
            for i, p in enumerate(parts):
                pc1, pc2 = st.columns([6, 1])
                pc1.markdown(
                    f"<div style='background:#f1f5f9;color:#334155;"
                    f"padding:8px 12px;border-radius:6px;"
                    f"font-family:ui-monospace,Menlo,monospace;"
                    f"border-left:3px solid #3b82f6;'>"
                    f"<span style='color:#94a3b8;font-size:.8rem'>#{i+1}</span>"
                    f"  <span style='color:#334155;font-weight:500;'>"
                    f"{part_label(p)}</span></div>",
                    unsafe_allow_html=True,
                )
                if pc2.button("Remove", key=f"rm_part_{i}",
                              use_container_width=True):
                    parts.pop(i)
                    st.rerun()

            uc1, uc2 = st.columns([1, 1])
            if uc1.button("⌫ Remove last piece", use_container_width=True):
                parts.pop()
                st.rerun()
            if uc2.button("✖ Clear all pieces", use_container_width=True):
                st.session_state.formula_parts = []
                st.rerun()

        # --- 5. Evaluate the parts into a Series ---
        # We evaluate piece-by-piece into Series/scalars then combine using
        # Python `eval` on a token list. This handles aggregates that produce
        # column-wide scalars (broadcast to every row) cleanly.
        # --------------------------------------------------------------
        # Formula validators — run BEFORE evaluation to catch problems
        # the user can fix, and to auto-repair the small ones (unclosed
        # parentheses). Conflicts here block the "Create column" button
        # entirely; the auto-fix and the division-by-zero check are
        # warnings that don't block.
        # --------------------------------------------------------------
        def validate_formula_parts(parts: list[dict]) -> list[str]:
            """
            Scan the part sequence for arithmetic-grammar conflicts.

            Returns a list of human-readable problem messages. An empty
            list means the sequence is well-formed.

            Detects:
              - adjacent operators like  +*  /-  -/  (e.g. "2 + * 2")
              - operator at the very start (other than unary "-")
              - operator at the very end (any operator including minus)
              - "(" at the end with nothing after it
              - ")" with nothing before it
              - empty formula
            """
            problems: list[str] = []

            # Build a simplified token-kind sequence:
            #   "v"      — a value-producing token (col / num / fn)
            #   "+","-","*","/","**" — binary operators
            #   "(", ")" — parentheses
            kinds: list[str] = []
            for p in parts:
                k = p["kind"]
                if k in ("col", "num", "fn"):
                    kinds.append("v")
                elif k == "op":
                    kinds.append(p["value"])
                # any unknown kind: ignore for grammar purposes

            if not kinds:
                problems.append("The formula is empty.")
                return problems

            binary_ops = {"+", "-", "*", "/", "**"}

            # Leading token: only a value or "(" or unary "-" makes sense.
            first = kinds[0]
            if first in binary_ops and first != "-":
                problems.append(
                    f"The formula starts with `{first}`. Operators must sit "
                    "between two values."
                )

            # Trailing token: must be a value or ")".
            last = kinds[-1]
            if last in binary_ops:
                problems.append(
                    f"The formula ends with `{last}`. Add a value after it "
                    "(a column, a number, or a function)."
                )
            elif last == "(":
                problems.append(
                    "The formula ends with `(` and nothing inside it."
                )

            # Adjacency rules
            for i in range(1, len(kinds)):
                a, b = kinds[i - 1], kinds[i]

                # Two binary operators in a row, e.g. "2 + * 2"
                # Exception: a unary minus after another operator or `(`,
                # i.e. "2 * -3", "(-x)" — accept those.
                if a in binary_ops and b in binary_ops:
                    if b == "-" and a in binary_ops:
                        continue  # unary minus
                    problems.append(
                        f"Operators `{a}` and `{b}` appear back-to-back. "
                        "Remove one or put a value between them."
                    )

                # Two values in a row, e.g. "col col" or "5 col"
                if a == "v" and b == "v":
                    problems.append(
                        "Two values appear back-to-back with no operator. "
                        "Insert `+`, `-`, `*`, or `/` between them."
                    )

                # ")" immediately before "(" or a value, e.g. ")(" or ")5"
                if a == ")" and b in ("(", "v"):
                    problems.append(
                        "Closed parenthesis is followed by another value "
                        "without an operator in between."
                    )

                # Operator immediately before ")", e.g. "(2 +)"
                if a in binary_ops and b == ")":
                    problems.append(
                        f"`{a}` appears right before `)` — the parenthesis "
                        "is closing on an unfinished sub-expression."
                    )

                # "(" immediately followed by an operator (other than unary -)
                if a == "(" and b in binary_ops and b != "-":
                    problems.append(
                        f"`(` is followed by `{b}` with no value first."
                    )

            return problems


        def auto_close_parens(parts: list[dict]) -> tuple[list[dict], int]:
            """
            If the parts list has more "(" than ")", append enough ")" parts
            at the end to balance the count. Returns (new_parts, n_added).
            We never delete a stray ")", since that's a real conflict — the
            grammar validator above is responsible for flagging it.
            """
            n_open = sum(1 for p in parts
                         if p["kind"] == "op" and p["value"] == "(")
            n_close = sum(1 for p in parts
                          if p["kind"] == "op" and p["value"] == ")")
            missing = n_open - n_close
            if missing <= 0:
                return parts, 0
            patched = list(parts) + [{"kind": "op", "value": ")"}] * missing
            return patched, missing


        def check_division_by_zero(parts: list[dict],
                                   frame: pd.DataFrame) -> str | None:
            """
            Look at each `/` operator in the part sequence and inspect the
            token that immediately follows. If it's a literal number 0, that
            is a guaranteed division by zero — refuse. If it's a column or an
            aggregate function and the resolved value contains any zero on
            the sample frame, return a *warning* (the user can still proceed;
            pandas will produce inf / NaN there, not raise).

            Returns:
              - a "blocking" message (str) if literal 0 is the divisor
              - a "warning" message (str) if a column/scalar divisor contains zero
              - None if no division-by-zero risk found
            """
            warnings: list[str] = []
            for i, p in enumerate(parts):
                if not (p["kind"] == "op" and p["value"] == "/"):
                    continue
                if i + 1 >= len(parts):
                    continue  # validator above catches trailing-operator case
                nxt = parts[i + 1]

                # Literal 0  →  hard block
                if nxt["kind"] == "num" and float(nxt["value"]) == 0:
                    return "Division by literal 0 — please replace the divisor."

                # Column divisor — check actual values on the sample
                if nxt["kind"] == "col":
                    col = nxt["name"]
                    if col in frame.columns:
                        try:
                            series = pd.to_numeric(frame[col], errors="coerce")
                            if (series == 0).any():
                                warnings.append(
                                    f"`{col}` contains 0 — rows where it's 0 "
                                    "will produce inf or NaN."
                                )
                        except Exception:
                            pass

                # Aggregate divisor — compute the scalar and check
                if nxt["kind"] == "fn" and nxt.get("agg_mode") == "aggregate" \
                        and not nxt.get("group_by"):
                    col = nxt.get("col")
                    if col and col in frame.columns:
                        try:
                            scalar = frame[col].agg(nxt["name"])
                            if scalar == 0:
                                return (f"Divisor `{nxt['name']}({col})` "
                                        "evaluates to 0 — division would fail.")
                        except Exception:
                            pass

            if warnings:
                return "Possible division by zero: " + " ".join(warnings)
            return None


        def evaluate_parts(parts: list[dict], frame: pd.DataFrame) -> pd.Series:
            """
            Compute each token to a value (Series or scalar), then assemble
            a Python expression that references those values by index and
            evaluate it. This keeps operator precedence + parentheses correct.
            """
            placeholders: list = []  # the resolved values, indexed by position
            expr_tokens: list[str] = []
            n = len(frame)

            for p in parts:
                k = p["kind"]
                if k == "op":
                    expr_tokens.append(p["value"])
                    continue
                if k == "num":
                    placeholders.append(float(p["value"]))
                elif k == "col":
                    placeholders.append(frame[p["name"]])
                elif k == "fn":
                    name = p["name"]
                    # Legacy fallback for the old multi-column row-wise form
                    if "args" in p and "col" not in p:
                        cols = p["args"]
                        if name in ("mean", "std", "median", "min", "max"):
                            val = frame[cols].agg(name, axis=1)
                        elif name in ("log", "log10", "sqrt", "abs"):
                            val = {"log": np.log, "log10": np.log10,
                                   "sqrt": np.sqrt, "abs": np.abs}[name](frame[cols[0]])
                        else:
                            raise ValueError(f"Unknown function: {name}")
                    else:
                        col = p["col"]
                        group_by = p.get("group_by")
                        mode = p.get("agg_mode")
                        s = frame[col]
                        if mode == "transform" or name in ("log", "log10", "sqrt", "abs"):
                            val = {"log": np.log, "log10": np.log10,
                                   "sqrt": np.sqrt, "abs": np.abs}[name](s)
                        else:
                            # Aggregate: whole column or per group, broadcast back
                            if group_by:
                                val = frame.groupby(group_by)[col].transform(name)
                            else:
                                scalar = s.agg(name)
                                val = pd.Series([scalar] * n, index=frame.index)
                    placeholders.append(val)
                else:
                    raise ValueError(f"Unknown part kind: {k}")
                expr_tokens.append(f"__v[{len(placeholders) - 1}]")

            if not expr_tokens:
                raise ValueError("Empty formula.")

            # Assemble expression with sensible spacing
            out = ""
            for t in expr_tokens:
                if not out:
                    out = t
                elif out.endswith("(") or t == ")":
                    out += t
                else:
                    out += " " + t

            result = eval(out, {"__builtins__": {}}, {"__v": placeholders})
            # If the result is a plain scalar (e.g. only an aggregate), broadcast.
            if not isinstance(result, pd.Series):
                result = pd.Series([result] * n, index=frame.index)
            return result

        # Build a human-readable expression for display
        def assemble_human(parts: list[dict]) -> str:
            return " ".join(part_label(p) for p in parts)

        st.markdown("---")
        st.markdown("**Preview**")

        if parts:
            # --- 1. Grammar check (blocks the build entirely) ---
            conflicts = validate_formula_parts(parts)

            # --- 2. Auto-close unclosed parentheses ---
            # If there's a structural conflict already, we still try to close
            # parens so the human-readable expression in the preview doesn't
            # look misleading — but we leave conflicts in the message list.
            effective_parts, n_added = auto_close_parens(parts)
            if n_added:
                st.info(
                    f"ℹ️ Auto-closed {n_added} unclosed parenthesis"
                    f"{'es' if n_added > 1 else ''} at the end of the formula."
                )

            human = assemble_human(effective_parts)
            st.markdown(
                f"<div style='font-family:monospace;background:#0f172a;"
                f"color:#f1f5f9;padding:10px 14px;border-radius:6px;'>"
                f"{new_name or 'new_column'} = {human}</div>",
                unsafe_allow_html=True,
            )

            # If there are grammar conflicts, refuse to attempt evaluation.
            if conflicts:
                for msg in conflicts:
                    st.error(f"⛔ {msg}")
                st.caption(
                    "Fix the issues above to enable the preview and the "
                    "**Create column** button."
                )
                preview_ok = False
            else:
                # --- 3. Division-by-zero check — may block, may warn ---
                dz_msg = check_division_by_zero(effective_parts, df.head(50))
                dz_blocks = bool(dz_msg) and (
                    "literal 0" in dz_msg or "evaluates to 0" in dz_msg
                )
                if dz_msg:
                    if dz_blocks:
                        st.error(f"⛔ {dz_msg}")
                    else:
                        st.warning(f"⚠️ {dz_msg}")

                # --- 4. Try evaluation on a sample ---
                preview_ok = False
                preview_err = None
                if dz_blocks:
                    preview_ok = False
                else:
                    try:
                        sample = df.head(10)
                        preview = evaluate_parts(effective_parts, sample)
                        preview_df = pd.DataFrame({new_name or "result": preview})
                        st.caption("First 10 rows:")
                        st.dataframe(preview_df, use_container_width=True)
                        preview_ok = True
                    except SyntaxError:
                        preview_err = (
                            "The pieces don't form a valid expression. "
                            "Check parentheses and that operators sit "
                            "between values, not at the start or end."
                        )
                    except KeyError as e:
                        preview_err = (f"Column not found: {e}. "
                                       "Maybe it was renamed or removed?")
                    except ZeroDivisionError:
                        preview_err = "Division by zero in the formula."
                    except TypeError as e:
                        preview_err = (f"Wrong type for one of the pieces: {e}. "
                                       "Most functions need numeric columns.")
                    except Exception as e:
                        preview_err = f"Couldn't evaluate the formula: {e}"

                if preview_err:
                    st.warning(f"⚠️ {preview_err}")

            if st.button("Create column", type="primary",
                         disabled=(not preview_ok or not new_name)):
                try:
                    result = evaluate_parts(effective_parts, df)
                    before = len(df)
                    df[new_name] = result
                    st.session_state.df = df
                    log_step(
                        "new_column",
                        {"name": new_name, "mode": "formula",
                         "parts": effective_parts, "expression": human},
                        before, len(df),
                    )
                    st.session_state.formula_parts = []
                    flash(f"Created `{new_name}` from formula: `{human}` — "
                          f"{len(df):,} value(s) computed.")
                    st.rerun()
                except Exception as e:
                    st.warning(
                        f"⚠️ Could not apply the formula to the full data: {e}. "
                        "The preview worked on a sample but failed on the full "
                        "dataset — there may be bad values further down."
                    )
        else:
            st.info("Add some pieces above to see a preview.")

    # ------------------------------------------------------------------ BIN NUMERIC COLUMN
    with st.expander("📦 Bin numeric column into categories"):
        st.caption(
            "Group a numeric column's values into discrete categories. "
            "**Equal-width** (`pd.cut`) divides the value range into intervals "
            "of the same size; **Quantile** (`pd.qcut`) puts roughly the same "
            "number of rows in each bin. The result is stored as a new "
            "string-typed column so it can be used like any other category."
        )

        bn_num_cols = df.select_dtypes(include="number").columns.tolist()
        if not bn_num_cols:
            st.info("No numeric columns available to bin.")
        else:
            bc1, bc2 = st.columns(2)
            bn_col = bc1.selectbox(
                "Numeric column to bin", bn_num_cols, key="bn_col",
            )
            bn_new_name = bc2.text_input(
                "New column name",
                value=f"{bn_col}_binned" if bn_col else "binned",
                key="bn_new_name",
            ).strip()

            bm1, bm2 = st.columns(2)
            bn_method_label = bm1.radio(
                "Method",
                ["Equal-width (pd.cut)", "Quantile (pd.qcut)"],
                key="bn_method",
                help="Equal-width: each bin spans the same numeric range. "
                     "Quantile: each bin contains ~the same number of rows.",
            )
            bn_method = "cut" if bn_method_label.startswith("Equal") else "qcut"
            bn_n_bins = bm2.slider(
                "Number of bins", 2, 50, 5, 1, key="bn_n_bins",
            )

            bn_labels_mode = st.radio(
                "Bin labels",
                ["Range (e.g. `[10, 25]`)",
                 "Ordinal (Bin 1, Bin 2, …)"],
                key="bn_labels_mode", horizontal=True,
            )

            # Build a preview so the user sees the bin edges and counts
            # BEFORE applying. Any error (e.g. all-null column, single
            # unique value for qcut) is surfaced as a friendly message.
            preview_ok = False
            try:
                source = pd.to_numeric(df[bn_col], errors="coerce")
                if bn_method == "cut":
                    binned = pd.cut(source, bins=int(bn_n_bins),
                                    include_lowest=True, duplicates="drop")
                else:
                    binned = pd.qcut(source, q=int(bn_n_bins),
                                     duplicates="drop")

                if bn_labels_mode.startswith("Ordinal"):
                    cats = binned.cat.categories
                    mapping = {iv: f"Bin {i+1}" for i, iv in enumerate(cats)}
                    binned_str = binned.map(mapping).astype("string")
                else:
                    def _fmt(iv):
                        if pd.isna(iv):
                            return None
                        return f"[{iv.left:.0f}, {iv.right:.0f}]"
                    binned_str = binned.map(_fmt).astype("string")

                preview_vc = binned_str.value_counts(dropna=False).reset_index()
                preview_vc.columns = ["bin", "count"]
                st.caption(
                    f"Preview — {len(preview_vc)} bin(s) would be created:"
                )
                st.dataframe(preview_vc, use_container_width=True,
                             hide_index=True,
                             height=min(280, 40 + 35 * len(preview_vc)))
                preview_ok = True
            except Exception as e:
                st.warning(
                    f"Can't preview binning with these settings: {e}. "
                    "Try a different method or fewer bins."
                )

            name_taken = bn_new_name in df.columns and bn_new_name != ""
            if name_taken:
                st.warning(
                    f"⚠️ `{bn_new_name}` already exists. Applying will overwrite it."
                )

            if st.button(
                "Apply binning",
                type="primary",
                key="bn_apply",
                disabled=(not preview_ok or not bn_new_name),
            ):
                before = len(df)
                try:
                    source = pd.to_numeric(df[bn_col], errors="coerce")
                    if bn_method == "cut":
                        binned = pd.cut(source, bins=int(bn_n_bins),
                                        include_lowest=True, duplicates="drop")
                    else:
                        binned = pd.qcut(source, q=int(bn_n_bins),
                                         duplicates="drop")

                    if bn_labels_mode.startswith("Ordinal"):
                        cats = binned.cat.categories
                        mapping = {iv: f"Bin {i+1}" for i, iv in enumerate(cats)}
                        result = binned.map(mapping).astype("string")
                        labels_key = "ordinal"
                    else:
                        def _fmt(iv):
                            if pd.isna(iv):
                                return None
                            return f"[{iv.left:.0f}, {iv.right:.0f}]"
                        result = binned.map(_fmt).astype("string")
                        labels_key = "range"

                    df[bn_new_name] = result
                    st.session_state.df = df
                    n_unique_bins = int(result.dropna().nunique())
                    log_step(
                        "bin_numeric",
                        {"column": bn_col,
                         "new_name": bn_new_name,
                         "bins": int(bn_n_bins),
                         "method": bn_method,
                         "labels": labels_key},
                        before, len(df),
                    )
                    flash(
                        f"Created `{bn_new_name}` from `{bn_col}` using "
                        f"{bn_method} into {n_unique_bins} bin(s)."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not apply binning: {e}")


    # ------------------------------------------------------------------ VALIDATION
    st.markdown('---')
    st.markdown("### ✅ Data validation")
    st.caption(
        "Define rules to check the dataframe against. Each rule produces an "
        "**Info** severity if it passes, **Critical** if any row violates it. "
        "Violation tables can be exported as CSV or Excel."
    )

    rules: list[dict] = st.session_state.validation_rules

    # ---- Rule builder ----
    with st.expander("➕ Add a validation rule", expanded=False):
        rule_kind = st.radio(
            "Rule type",
            ["Range (min/max for numeric or date)",
             "Allowed values (categorical)",
             "Not null"],
            key="val_rule_kind", horizontal=False,
        )

        if rule_kind.startswith("Range"):
            num_dt_cols = (df.select_dtypes(include="number").columns.tolist()
                           + df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist())
            rng_col = st.selectbox(
                "Column", num_dt_cols, key="val_rng_col",
                help="Pick a numeric or datetime column.",
            )
            if rng_col:
                is_date = pd.api.types.is_datetime64_any_dtype(df[rng_col])
                rc1, rc2 = st.columns(2)
                if is_date:
                    dmin = pd.to_datetime(df[rng_col].min()).date() \
                           if df[rng_col].notna().any() else datetime.utcnow().date()
                    dmax = pd.to_datetime(df[rng_col].max()).date() \
                           if df[rng_col].notna().any() else datetime.utcnow().date()
                    use_min = rc1.checkbox("Set minimum", value=True, key="val_use_min")
                    min_val = rc1.date_input("Minimum", value=dmin, key="val_min_dt") if use_min else None
                    use_max = rc2.checkbox("Set maximum", value=True, key="val_use_max")
                    max_val = rc2.date_input("Maximum", value=dmax, key="val_max_dt") if use_max else None
                    min_val = str(min_val) if min_val else None
                    max_val = str(max_val) if max_val else None
                else:
                    use_min = rc1.checkbox("Set minimum", value=True, key="val_use_min")
                    min_val = rc1.number_input("Minimum", value=0.0, key="val_min_num") if use_min else None
                    use_max = rc2.checkbox("Set maximum", value=True, key="val_use_max")
                    max_val = rc2.number_input("Maximum", value=100.0, key="val_max_num") if use_max else None
                if st.button("Add range rule", type="primary", key="val_add_range",
                             disabled=(min_val is None and max_val is None)):
                    rules.append({
                        "kind": "range", "column": rng_col,
                        "min": min_val, "max": max_val,
                    })
                    st.session_state.validation_rules = rules
                    st.rerun()

        elif rule_kind.startswith("Allowed"):
            cat_cols2 = df.select_dtypes(
                include=["object", "category", "string"]).columns.tolist()
            mode = st.radio(
                "Column source",
                ["Pick from dataframe", "Type manually"],
                key="val_allowed_mode", horizontal=True,
            )
            if mode == "Pick from dataframe":
                if not cat_cols2:
                    st.info("No categorical columns in the data.")
                    a_col = None
                else:
                    a_col = st.selectbox(
                        "Column", cat_cols2, key="val_allowed_col_pick",
                    )
            else:
                a_col = st.text_input(
                    "Column name", "", key="val_allowed_col_type",
                    help="Type a column name (useful if you'll replay against "
                         "another dataframe that doesn't yet exist).",
                ).strip() or None

            # Allowed values input
            avals_text = st.text_area(
                "Allowed values (one per line)",
                value="",
                key="val_allowed_values",
                help="Each line is one allowed value.",
            )
            allowed = [v.strip() for v in avals_text.splitlines() if v.strip()]
            ci = st.checkbox("Case insensitive", value=False, key="val_allowed_ci")

            if st.button("Add allowed-values rule", type="primary",
                         key="val_add_allowed",
                         disabled=(not a_col or not allowed)):
                rules.append({
                    "kind": "allowed", "column": a_col,
                    "allowed": allowed, "case_insensitive": ci,
                })
                st.session_state.validation_rules = rules
                st.rerun()

        else:  # Not null
            nn_col = st.selectbox(
                "Column", df.columns.tolist(), key="val_nn_col",
            )
            if st.button("Add not-null rule", type="primary",
                         key="val_add_nn", disabled=not nn_col):
                rules.append({"kind": "not_null", "column": nn_col})
                st.session_state.validation_rules = rules
                st.rerun()

    # ---- Current rules list ----
    if rules:
        st.markdown(f"**Current rules ({len(rules)}):**")
        for i, rule in enumerate(rules):
            rc1, rc2 = st.columns([6, 1])
            kind = rule.get("kind", "?")
            col = rule.get("column", "?")
            if kind == "range":
                desc = f"`{col}` in [{rule.get('min', '−∞')}, {rule.get('max', '+∞')}]"
            elif kind == "allowed":
                allowed = rule.get("allowed", [])
                desc = (f"`{col}` ∈ {allowed[:5]}"
                        + ("…" if len(allowed) > 5 else "")
                        + (" (case-insensitive)" if rule.get("case_insensitive") else ""))
            elif kind == "not_null":
                desc = f"`{col}` must not be null"
            else:
                desc = f"{kind} on `{col}`"
            rc1.markdown(
                f"<div style='background:rgba(148,163,184,0.15);"
                f"padding:8px 12px;border-radius:6px;"
                f"font-family:ui-monospace,Menlo,monospace;'>"
                f"<span style='opacity:.6;font-size:.8rem;'>#{i+1}</span>"
                f"  <b>{kind}</b>  {desc}</div>",
                unsafe_allow_html=True,
            )
            if rc2.button("Remove", key=f"val_rm_{i}",
                          use_container_width=True):
                rules.pop(i)
                st.session_state.validation_rules = rules
                st.session_state.validation_results = None
                st.rerun()

        # ---- Run / clear / export buttons ----
        run_col, clear_col = st.columns(2)
        if run_col.button("▶️ Run validation", type="primary",
                          use_container_width=True):
            st.session_state.validation_results = run_validations(df, rules)
            st.rerun()
        if clear_col.button("Clear rules", use_container_width=True):
            st.session_state.validation_rules = []
            st.session_state.validation_results = None
            st.rerun()
    else:
        st.info("No rules defined yet. Add one above.")

    # ---- Results ----
    results = st.session_state.validation_results
    if results:
        overall = results["overall_severity"]
        summary_df = results["summary"]
        violations_df = results["violations"]

        if overall == "Info":
            st.success(
                f"🟢 **Severity: Info** — All {len(summary_df)} rule(s) passed. "
                "Dataframe components don't create error incidents."
            )
        else:
            n_failed = int((summary_df["n_violations"] > 0).sum()) if "n_violations" in summary_df.columns else 0
            n_total_viol = int(violations_df.shape[0]) if not violations_df.empty else 0
            st.error(
                f"🔴 **Severity: Critical** — {n_failed} rule(s) failed "
                f"with {n_total_viol:,} total violation(s)."
            )

        # Summary table
        st.markdown("**Per-rule summary:**")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Violations table
        if not violations_df.empty:
            st.markdown(f"**Violations table ({len(violations_df):,} row(s)):**")
            st.dataframe(violations_df, use_container_width=True, hide_index=True)

            # Export buttons
            ex_csv, ex_xlsx = st.columns(2)
            ex_csv.download_button(
                "⬇️ Export violations (CSV)",
                data=violations_df.to_csv(index=False).encode("utf-8"),
                file_name="violations.csv",
                mime="text/csv",
                use_container_width=True,
            )
            try:
                _buf = io.BytesIO()
                with pd.ExcelWriter(_buf, engine="openpyxl") as w:
                    violations_df.to_excel(w, index=False, sheet_name="violations")
                    summary_df.to_excel(w, index=False, sheet_name="summary")
                ex_xlsx.download_button(
                    "⬇️ Export violations (Excel)",
                    data=_buf.getvalue(),
                    file_name="violations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                ex_xlsx.caption(f"Excel export unavailable: {e}")


    # ------------------------------------------------------------------ RECIPE
    st.markdown('---')
    st.markdown("### 📋 Information & current state")
    st.caption("The logged recipe of every step, and a live view of the dataframe.")
    st.subheader("🧾 Recipe — every step you've taken")

    if not st.session_state.recipe:
        st.info("No steps yet. Apply a transformation to start the recipe.")
    else:
        import html as _html
        for i, step in enumerate(st.session_state.recipe, 1):
            # Support both new (operation/parameters) and legacy (action/params)
            op = step.get("operation") or step.get("action", "?")
            params = step.get("parameters") or step.get("params", {})
            cols = step.get("affected_columns", [])
            params_str = _html.escape(json.dumps(params, default=str))
            cols_str = ", ".join(f"`{c}`" for c in cols) if cols else "—"
            st.markdown(
                f"<div class='step-box'>"
                f"<b>#{i} · {op}</b>"
                f"<span class='ts'>{step['timestamp']}</span><br>"
                f"<span class='params'>columns: {cols_str}</span><br>"
                f"<span class='params'>parameters = {params_str}</span><br>"
                f"<span class='rows'>rows: {step.get('rows_before', 0):,} → "
                f"{step.get('rows_after', 0):,} ({step.get('rows_changed', 0):+,})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        c1, c2 = st.columns(2)
        recipe_json = json.dumps(
            {"file": st.session_state.file_name, "steps": st.session_state.recipe},
            indent=2, default=str,
        )
        c1.download_button(
            "⬇️ Export recipe (JSON)",
            recipe_json.encode("utf-8"),
            file_name="recipe.json",
            mime="application/json",
            use_container_width=True,
        )
        if c2.button("↩️ Revert to original data", use_container_width=True):
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.recipe = []
            st.rerun()

    # ------------------------------------------------------------------ IMPORT
    st.markdown('---')
    st.subheader("📥 Import & replay a recipe")
    st.caption(
        "Upload a pipeline recipe JSON to re-apply its operations on the "
        "current dataframe. Steps that reference columns no longer present "
        "are skipped with a note."
    )

    up_recipe = st.file_uploader(
        "Recipe JSON",
        type=["json"],
        key="recipe_upload",
        help="Use a file previously exported from the **Export** tab.",
    )

    if up_recipe is not None:
        try:
            payload = json.loads(up_recipe.getvalue().decode("utf-8"))
            steps_in, validation_errors, validation_warnings = validate_recipe_payload(payload)

            if validation_errors:
                st.error(
                    "This recipe file cannot be replayed because it does not "
                    "follow the script-readable recipe structure."
                )
                st.dataframe(
                    pd.DataFrame({"Problem reason": validation_errors}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                if validation_warnings:
                    st.warning(
                        "Recipe is readable, but some preview metadata is missing:\n\n- "
                        + "\n- ".join(validation_warnings[:10])
                        + ("\n- …" if len(validation_warnings) > 10 else "")
                    )

                source_name = payload.get("source_file") or payload.get("file") or "—"
                st.success(
                    f"Found a valid recipe with **{len(steps_in)}** step(s) "
                    f"from `{source_name}`."
                )
                # Preview the steps
                preview_rows = []
                for i, step in enumerate(steps_in, 1):
                    affected_columns = step.get("affected_columns", [])
                    if not isinstance(affected_columns, list):
                        affected_columns = []
                    preview_rows.append({
                        "#": i,
                        "operation": step.get("operation") or step.get("action", "?"),
                        "affected_columns": ", ".join(affected_columns) or "—",
                        "timestamp": step.get("timestamp", "—"),
                    })
                st.dataframe(pd.DataFrame(preview_rows),
                             use_container_width=True, hide_index=True)

                rep1, rep2 = st.columns(2)
                replay_from_original = rep1.toggle(
                    "Apply to original data (instead of current)",
                    value=True,
                    help="If on, start from the originally loaded dataframe "
                         "and replace any in-progress edits. If off, the "
                         "recipe is applied on top of the current state.",
                )
                if rep2.button("▶️ Replay recipe", type="primary",
                               use_container_width=True):
                    base = (st.session_state.df_original.copy()
                            if replay_from_original
                            else st.session_state.df.copy())
                    new_df, applied, errors = apply_recipe(base, steps_in)
                    st.session_state.df = new_df
                    if replay_from_original:
                        # Reset recipe and re-log the applied steps so the new
                        # state matches the imported pipeline.
                        st.session_state.recipe = list(steps_in)
                    else:
                        st.session_state.recipe.extend(steps_in)
                    flash(
                        f"Replayed {len(applied)} step(s)"
                        + (f"; {len(errors)} skipped"
                           if errors else "."),
                        tab="tab2",
                    )
                    if errors:
                        st.warning(
                            "Some steps couldn't be replayed:\n\n- "
                            + "\n- ".join(errors[:10])
                        )
                    st.rerun()
        except json.JSONDecodeError as e:
            st.error(
                "Could not read the recipe file: invalid JSON syntax. "
                f"Reason: {e.msg} at line {e.lineno}, column {e.colno}."
            )
        except UnicodeDecodeError:
            st.error(
                "Could not read the recipe file: file must be UTF-8 encoded JSON."
            )
        except Exception as e:
            st.error(f"Could not read the recipe file: {e}")

    # ------------------------------------------------------------------ PREVIEW
    st.divider()
    current_df = st.session_state.df
    preview_ts = datetime.now().strftime("%H:%M:%S")
    head, tail = st.columns([4, 1])
    head.subheader(
        f"Current dataframe · {len(current_df):,} × {current_df.shape[1]} "
        f"· updated {preview_ts}"
    )
    n_show = tail.selectbox("Rows to show", [25, 50, 100, 500], index=1,
                            label_visibility="collapsed")
    st.dataframe(
        current_df.head(int(n_show)),
        use_container_width=True, height=320,
    )


# ===========================================================================
# TAB 3 — VISUALIZATION
# ===========================================================================
with tab3:
    if st.session_state.df is None:
        st.info("\U0001F4C1 Load a dataset on the **Upload & Overview** tab first.")
        st.stop()
    if not PLOTLY_OK:
        st.warning("Install `plotly` to use this tab:  `pip install plotly`")
        st.stop()

    df = st.session_state.df
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    NONE = "\u2014"

    # ----------------------------------------------------------------------
    # AI SUGGESTIONS — toggle at the top, always visible
    # ----------------------------------------------------------------------
    ai_header = st.container()
    with ai_header:
        topL, topR = st.columns([4, 1])
        topL.subheader("Visualization")
        st.session_state.ai_enabled = topR.toggle(
            "🤖 AI suggestions",
            value=st.session_state.ai_enabled,
            help="When ON, the AI panel proposes ready-to-use chart ideas based "
                 "on your data. It only suggests — nothing is created until you "
                 "confirm.",
        )

    st.caption(
        "Configure a chart below, then click **Create chart**. Created charts "
        "appear underneath and stay until you remove them \u2014 you can build as "
        "many as you like."
    )

    if st.session_state.ai_enabled:
        with st.container(border=True):
            st.markdown("### 🤖 AI suggestions")
            st.caption(
                "Suggestions focus on chronological / time-series structure "
                "first, then correlations, distributions, and group comparisons. "
                "Each suggestion explains which columns it uses and what insight "
                "it would reveal. **Nothing is created without your confirmation.**"
            )

            # The local rule-based suggestions work no matter what — they
            # don't need the openai package, network, or any key. We expose
            # them first as a baseline, then layer in the AI-powered button
            # only when the prerequisites are met.

            local_only_mode = (not OPENAI_OK)
            api_key = ""
            key_source = None

            if not OPENAI_OK:
                st.warning(
                    "The `openai` package isn't installed, so AI-powered "
                    "suggestions are unavailable. You can still use the "
                    "**local rule-based suggestions** below."
                )
            else:
                # Read API key from .env / environment / Streamlit secrets only.
                # We deliberately do NOT show a text input — keys should live in
                # configuration files, not be pasted into the UI.
                import os as _os
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", "")
                    if api_key:
                        key_source = ".streamlit/secrets.toml"
                except Exception:
                    api_key = ""
                if not api_key:
                    api_key = _os.environ.get("GROQ_API_KEY", "")
                    if api_key:
                        key_source = ".env / environment"

                if api_key:
                    masked = api_key[:7] + "…" + api_key[-4:] if len(api_key) > 12 else "•••"
                    st.success(
                        f"🔑 Groq API key loaded from **{key_source}** "
                        f"(`{masked}`)"
                    )
                else:
                    local_only_mode = True
                    st.warning(
                        "⚠️ No Groq API key found, so AI-powered suggestions "
                        "are unavailable.\n\n"
                        "To enable them, add `GROQ_API_KEY=gsk-…` to a `.env` "
                        "file next to `app.py`, set the `GROQ_API_KEY` "
                        "environment variable, or add it to "
                        "`.streamlit/secrets.toml`. Get a free key at "
                        "[console.groq.com/keys](https://console.groq.com/keys).\n\n"
                        "You can still use the **local rule-based suggestions** "
                        "below — they always work, no API required."
                    )

            # Model is fixed to the "instant" version — fast and rate-limit friendly.
            model_choice = "llama-3.1-8b-instant"

            bcol1, bcol2, bcol3 = st.columns([2, 2, 1])
            ai_btn_disabled = local_only_mode
            if bcol1.button(
                "💡 Get AI suggestions",
                type="primary",
                disabled=ai_btn_disabled,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Asking the model…"):
                        summary = summarize_for_ai(df)
                        st.session_state.ai_suggestions = (
                            request_ai_suggestions(summary, api_key, model_choice)
                        )
                        st.session_state.ai_error = None
                        st.session_state.ai_source = "ai"
                except Exception as e:
                    title, detail = classify_ai_error(e)
                    st.session_state.ai_error = {"title": title, "detail": detail}
                    # Automatic fallback so the feature keeps working.
                    st.session_state.ai_suggestions = fallback_suggestions(df)
                    st.session_state.ai_source = "fallback"
                st.rerun()

            if bcol2.button(
                "🧭 Local suggestions only",
                use_container_width=True,
                help="Skip the AI call and use simple rule-based suggestions "
                     "based on column dtypes and statistics. Always works, "
                     "no API needed.",
            ):
                st.session_state.ai_suggestions = fallback_suggestions(df)
                st.session_state.ai_error = None
                st.session_state.ai_source = "fallback_manual"
                st.rerun()

            if bcol3.button(
                "Clear",
                use_container_width=True,
                disabled=not st.session_state.ai_suggestions,
            ):
                st.session_state.ai_suggestions = []
                st.session_state.ai_pending = None
                st.session_state.ai_error = None
                st.rerun()

            # Friendly classified error message + fallback notice
            if st.session_state.ai_error:
                err = st.session_state.ai_error
                if isinstance(err, dict):
                    title, detail = err.get("title", "Error"), err.get("detail", "")
                else:
                    title, detail = "AI error", str(err)
                st.warning(f"⚠️ **{title}** — {detail}")

            # Banner showing which source the current suggestions came from
            src = st.session_state.get("ai_source")
            if src == "fallback":
                st.info(
                    "🧭 Showing **local rule-based suggestions** because the "
                    "AI call failed. These work offline and are always "
                    "available."
                )
            elif src == "fallback_manual":
                st.info(
                    "🧭 Showing **local rule-based suggestions** (no AI used)."
                )

            # Render suggestions
            raw_suggestions = st.session_state.ai_suggestions or []
            if raw_suggestions:
                # First pass: validate each suggestion against the dataframe
                # AND try a dry-run build on a tiny sample to catch any errors
                # the static validator would miss (e.g. plotly-specific quirks
                # with categorical X axes). Only suggestions that pass both are
                # shown — failed ones are silently dropped, not displayed as
                # "can't be built" placeholders.
                thumb_df = df.sample(min(500, len(df)), random_state=0) \
                             if len(df) > 500 else df
                dryrun_df = df.head(20) if len(df) >= 20 else df

                valid: list[tuple[dict, dict]] = []  # (suggestion, cfg)
                for sug in raw_suggestions:
                    cfg_preview = suggestion_to_config(sug, df)
                    if cfg_preview is None:
                        continue
                    # Quick dry-run on a tiny sample so we never surface a
                    # suggestion that would fail at render time.
                    try:
                        dry_fig, dry_err = build_figure(dryrun_df, cfg_preview)
                        if dry_err or dry_fig is None:
                            continue
                    except Exception:
                        continue
                    valid.append((sug, cfg_preview))

                if not valid:
                    st.info(
                        "No suggestions were applicable to the current data "
                        "shape. Try a different model, or use **Local "
                        "suggestions only**."
                    )
                else:
                    st.markdown(f"**{len(valid)} suggestion(s):**")
                    for idx, (sug, cfg_preview) in enumerate(valid):
                        with st.container(border=True):
                            chart = sug.get("chart_type", "?")
                            purpose = sug.get("purpose", "")
                            cu_parts = []
                            for label in ("x", "y", "color", "size"):
                                v = sug.get(label)
                                if v:
                                    cu_parts.append(f"**{label}**=`{v}`")
                            if sug.get("agg"):
                                cu_parts.append(f"**agg**={sug['agg']}")
                            cols_used = " · ".join(cu_parts) if cu_parts else "—"

                            tcol, ccol = st.columns([1, 2], gap="medium")

                            # Thumbnail preview
                            with tcol:
                                thumb_cfg = dict(cfg_preview)
                                thumb_cfg.update({
                                    "title": None,
                                    "height": 180,
                                    "show_legend": False,
                                    "template": "plotly_white",
                                })
                                thumb_rendered = False
                                try:
                                    fig_t, err_t = build_figure(thumb_df, thumb_cfg)
                                    if fig_t is not None:
                                        fig_t.update_layout(
                                            margin=dict(l=4, r=4, t=4, b=4),
                                            xaxis_title=None, yaxis_title=None,
                                            font=dict(size=9),
                                        )
                                        st.plotly_chart(
                                            fig_t, use_container_width=True,
                                            key=f"ai_thumb_{idx}_{src or 'na'}",
                                            config={"displayModeBar": False},
                                        )
                                        thumb_rendered = True
                                except Exception:
                                    thumb_rendered = False
                                if not thumb_rendered:
                                    icon = CHART_ICONS.get(chart, CHART_ICONS["_default"])
                                    st.markdown(
                                        f"<div style='display:flex;align-items:center;"
                                        f"justify-content:center;height:180px;"
                                        f"background:#f8fafc;border-radius:6px;'>"
                                        f"{icon}</div>",
                                        unsafe_allow_html=True,
                                    )

                            with ccol:
                                st.markdown(f"#### {idx+1}. {chart}")
                                st.markdown(f"📌 *{purpose}*")
                                st.markdown(f"📊 {cols_used}")
                                use_key = f"ai_use_{idx}_{src or 'na'}"
                                if st.button(
                                    "Use this suggestion",
                                    key=use_key,
                                    use_container_width=True,
                                ):
                                    st.session_state.ai_pending = {
                                        "idx": idx,
                                        "cfg": cfg_preview,
                                        "purpose": purpose,
                                    }
                                    st.rerun()

            # Confirmation prompt for the pending suggestion
            pend = st.session_state.ai_pending
            if pend:
                st.markdown("---")
                st.warning(
                    f"⚠️ Confirm creation: a **{pend['cfg']['chart_type']}** "
                    f"chart will be added with the AI's settings.\n\n"
                    f"_Purpose:_ {pend['purpose']}"
                )
                cc1, cc2 = st.columns(2)
                if cc1.button(
                    "✅ Yes, create chart",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.chart_seq += 1
                    cfg = dict(pend["cfg"])
                    cfg["id"] = st.session_state.chart_seq
                    st.session_state.charts.append(cfg)
                    st.session_state.ai_pending = None
                    flash(
                        f"AI-suggested {cfg['chart_type']} chart created.",
                        tab="tab2",
                    )
                    st.rerun()
                if cc2.button("Cancel", use_container_width=True):
                    st.session_state.ai_pending = None
                    st.rerun()

    # ----------------------------------------------------------------------
    # CONFIGURATION FORM (nothing renders until "Create chart" is pressed)
    # ----------------------------------------------------------------------
    with st.expander("\u2795 Configure a new chart", expanded=True):
        # Chart type + rendering engine on the same row
        ct_col, eng_col = st.columns([2, 1])
        chart_type = ct_col.selectbox(
            "Chart type",
            ["Scatter", "Bubble", "Line", "Bar", "Histogram",
             "Box", "Violin", "Area", "Pie", "Correlation heatmap"],
            key="cfg_chart_type",
        )
        engine_opts = ["Plotly"]
        if MATPLOTLIB_OK:
            engine_opts.append("Matplotlib")
        engine = eng_col.selectbox(
            "Engine",
            engine_opts,
            key="cfg_engine",
            help="Plotly: interactive, hoverable charts. Matplotlib: static, "
                 "publication-style charts.",
        )

        # --- Minimalist preview wireframe (no data, just the chart shape) ---
        # Background is transparent so the wireframe blends into the Streamlit
        # theme (light or dark) instead of looking like a separate element.
        # The wireframe SVG strokes themselves keep their grey color.
        st.markdown(
            f"<details open><summary style='cursor:pointer;color:#94a3b8;"
            f"font-size:.85rem;margin:6px 0 4px;'>"
            f"📐 Preview shape (not based on data)</summary>"
            f"<div style='background:transparent;padding:6px;"
            f"max-width:360px;margin:6px 0;'>{wireframe_for(chart_type)}"
            f"</div></details>",
            unsafe_allow_html=True,
        )

        cfg: dict = {"chart_type": chart_type, "engine": engine}

        # --- Axes & encodings ---
        st.markdown("**Axes & encoding**")
        if chart_type in ("Scatter", "Bubble"):
            c1, c2 = st.columns(2)
            cfg["x"] = c1.selectbox("X axis", num_cols or all_cols, key="cfg_x")
            cfg["y"] = c2.selectbox("Y axis", num_cols or all_cols, key="cfg_y")
            if chart_type == "Bubble":
                cfg["size"] = st.selectbox("Bubble size", num_cols, key="cfg_size")
        elif chart_type in ("Line", "Area"):
            cfg["x"] = st.selectbox("X axis", all_cols, key="cfg_x")
            cfg["y"] = st.multiselect("Y axis (one or more)", num_cols,
                                      default=num_cols[:1], key="cfg_y_multi")
        elif chart_type == "Bar":
            c1, c2 = st.columns(2)
            cfg["x"] = c1.selectbox("X axis (category)", all_cols, key="cfg_x")
            cfg["y"] = c2.selectbox("Y axis (numeric, optional)",
                                    [NONE] + num_cols, key="cfg_y")
            if cfg["y"] != NONE:
                cfg["agg"] = st.selectbox(
                    "Aggregation", ["mean", "sum", "median", "max", "min", "count"],
                    key="cfg_agg")
        elif chart_type == "Histogram":
            cfg["x"] = st.selectbox("Column", num_cols or all_cols, key="cfg_x")
            cfg["bins"] = st.slider("Bins", 5, 100, 30, key="cfg_bins")
        elif chart_type in ("Box", "Violin"):
            c1, c2 = st.columns(2)
            cfg["y"] = c1.selectbox("Value (numeric)", num_cols, key="cfg_y")
            cfg["x"] = c2.selectbox("Group by (optional)", [NONE] + all_cols, key="cfg_x")
        elif chart_type == "Pie":
            c1, c2 = st.columns(2)
            cfg["x"] = c1.selectbox("Category (names)", cat_cols or all_cols, key="cfg_x")
            cfg["y"] = c2.selectbox("Value (optional, else counts)",
                                    [NONE] + num_cols, key="cfg_y")

        # --- Top N (visualization only) ---
        st.markdown("**Top N values**")
        top_n_available = (
            chart_type in ("Bar", "Pie")
            or (bool(num_cols) and chart_type != "Correlation heatmap")
        )
        cfg["top_n"] = None
        if top_n_available:
            top_n_enabled = st.checkbox(
                "Show only Top N values",
                value=False,
                key="cfg_top_n_enabled",
                help="Sorts the chart by its numeric measure and applies "
                     "`.head(N)` only inside the visualization. It does not "
                     "change the dataframe preview or stored data.",
            )
            if top_n_enabled:
                max_top_n = max(1, int(len(df)))
                default_top_n = min(10, max_top_n)
                cfg["top_n"] = st.slider(
                    "Top N",
                    min_value=1,
                    max_value=max_top_n,
                    value=default_top_n,
                    step=1,
                    key="cfg_top_n",
                )
        else:
            st.caption("Top N is available when the chart has a numeric value to sort by.")

        # --- Binning (only for chart types where it makes sense, and only
        #     when the X column is numeric — otherwise this is a no-op). ---
        if chart_type in ("Bar", "Box", "Violin", "Pie"):
            x_col_for_bin = cfg.get("x")
            x_is_numeric = (
                x_col_for_bin
                and x_col_for_bin != NONE
                and x_col_for_bin in df.columns
                and pd.api.types.is_numeric_dtype(df[x_col_for_bin])
            )
            if x_is_numeric:
                with st.expander(
                    "📦 Bin numeric X into groups (pd.cut / pd.qcut)",
                    expanded=False,
                ):
                    st.caption(
                        f"`{x_col_for_bin}` is numeric. Binning groups its "
                        "values into intervals so the chart shows discrete "
                        "categories instead of one bar per unique value."
                    )
                    bin_enabled = st.checkbox(
                        "Enable binning", value=False, key="cfg_bin_enabled",
                    )
                    if bin_enabled:
                        bm1, bm2 = st.columns(2)
                        method = bm1.radio(
                            "Method",
                            ["Equal-width (pd.cut)",
                             "Equal-frequency (pd.qcut)"],
                            key="cfg_bin_method",
                            help="Equal-width splits the value range into "
                                 "intervals of the same size. Equal-frequency "
                                 "puts roughly the same number of rows in "
                                 "each bin.",
                        )
                        method_key = "qcut" if "qcut" in method else "cut"
                        n_bins = bm2.slider(
                            "Number of bins", 2, 50, 10, 1, key="cfg_bin_n",
                        )

                        binning: dict = {
                            "enabled": True,
                            "method": method_key,
                            "n_bins": int(n_bins),
                        }

                        # Extra options for pd.cut: custom width or explicit edges
                        if method_key == "cut":
                            mode_extra = st.radio(
                                "Bin sizing",
                                ["Auto (use N bins)",
                                 "Custom width",
                                 "Custom edges"],
                                key="cfg_bin_extra",
                                horizontal=True,
                            )
                            if mode_extra == "Custom width":
                                col_min = float(pd.to_numeric(
                                    df[x_col_for_bin], errors="coerce").min())
                                col_max = float(pd.to_numeric(
                                    df[x_col_for_bin], errors="coerce").max())
                                span = max(col_max - col_min, 1e-9)
                                default_w = span / max(n_bins, 1)
                                width = st.number_input(
                                    "Bin width",
                                    min_value=1e-9,
                                    value=float(f"{default_w:.6g}"),
                                    key="cfg_bin_width",
                                    help="Each bin will span this much on the "
                                         "X axis. Bins fill from the column "
                                         "minimum upward.",
                                )
                                binning["width"] = float(width)
                            elif mode_extra == "Custom edges":
                                edge_text = st.text_input(
                                    "Edges (comma-separated)",
                                    "0, 10, 25, 50, 100",
                                    key="cfg_bin_edges",
                                    help="Each value defines a bin boundary, "
                                         "e.g. `0, 10, 25, 50, 100` gives "
                                         "four bins.",
                                )
                                try:
                                    edges = sorted(set(
                                        float(s.strip())
                                        for s in edge_text.split(",")
                                        if s.strip()))
                                    if len(edges) >= 2:
                                        binning["edges"] = edges
                                    else:
                                        st.warning(
                                            "Need at least two distinct edges.")
                                except ValueError:
                                    st.warning(
                                        "Edges must be numbers separated by "
                                        "commas, e.g. `0, 10, 25, 50`.")

                        binning["labels"] = "range" if st.radio(
                            "Bin labels",
                            ["Range (e.g. `[10, 25]`)",
                             "Ordinal (Bin 1, Bin 2, …)"],
                            key="cfg_bin_labels",
                            horizontal=True,
                        ).startswith("Range") else "ordinal"
                        cfg["binning"] = binning
                    else:
                        cfg["binning"] = {"enabled": False}

        # --- Grouping & color ---
        st.markdown("**Grouping & color**")
        g1, g2, g3 = st.columns(3)
        cfg["color"] = g1.selectbox("Color / group by", [NONE] + all_cols, key="cfg_color")
        palette_label = g2.selectbox(
            "Palette", PALETTE_LABELS, key="cfg_palette_label",
        )
        cfg["palette"] = PALETTE_LABEL_TO_KEY.get(palette_label, "Plotly")
        cfg["facet"] = g3.selectbox("Facet (sub-plots) by", [NONE] + cat_cols, key="cfg_facet")

        # --- Styling ---
        st.markdown("**Styling & display**")
        s1, s2, s3 = st.columns(3)
        cfg["title"] = s1.text_input("Title", "", key="cfg_title")
        cfg["template"] = s2.selectbox(
            "Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2",
                      "seaborn", "simple_white"], index=1, key="cfg_template")
        cfg["height"] = s3.slider("Height (px)", 300, 900, 500, 50, key="cfg_height")
        s4, s5, s6 = st.columns(3)
        cfg["show_legend"] = s4.checkbox("Show legend", value=True, key="cfg_legend")
        cfg["opacity"] = s5.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key="cfg_opacity")
        if chart_type == "Scatter":
            cfg["marker_size"] = s6.slider("Marker size", 2, 20, 7, key="cfg_msize")
        l1, l2, l3 = st.columns(3)
        cfg["log_x"] = l1.checkbox("Log X", key="cfg_logx")
        cfg["log_y"] = l2.checkbox("Log Y", key="cfg_logy")
        if chart_type in ("Scatter", "Bubble"):
            if STATSMODELS_OK:
                cfg["trendline"] = l3.checkbox("Trendline (OLS)", key="cfg_trend")
            else:
                l3.checkbox("Trendline (OLS)", value=False, disabled=True,
                            help="Install statsmodels to enable.", key="cfg_trend")
                cfg["trendline"] = False

        # --- Filters ---
        st.markdown("**Filter data (optional)**")
        filters: list[dict] = []
        n_vf = st.number_input("Number of filters", 0, 8, 0, 1, key="cfg_nf")
        for i in range(int(n_vf)):
            st.markdown(f"_Filter {i+1}_")
            fcol = st.selectbox("Column", all_cols, key=f"cfg_fcol_{i}")
            s = df[fcol]
            if pd.api.types.is_numeric_dtype(s):
                op = st.selectbox("Condition",
                                  [">", "<", "between", "==", "!=", ">=", "<="],
                                  key=f"cfg_fop_{i}")
                if op == "between":
                    s_nonnull = s.dropna()
                    if s_nonnull.empty:
                        st.caption(
                            f"`{fcol}` has no non-null values — pick a number."
                        )
                        v = st.number_input("Value", key=f"cfg_fval_{i}")
                        filters.append({"kind": "numeric", "col": fcol,
                                        "op": "==", "val": v})
                    else:
                        lo = float(np.nanmin(s)); hi = float(np.nanmax(s))
                        if lo == hi:
                            st.caption(
                                f"`{fcol}` is constant at **{lo:g}** — "
                                "filter keeps rows equal to this value."
                            )
                            filters.append({"kind": "numeric", "col": fcol,
                                            "op": "between",
                                            "val": [lo, hi]})
                        else:
                            rng = st.slider("Range", lo, hi, (lo, hi),
                                            key=f"cfg_fval_{i}")
                            filters.append({"kind": "numeric", "col": fcol,
                                            "op": op,
                                            "val": [rng[0], rng[1]]})
                else:
                    v = st.number_input("Value", key=f"cfg_fval_{i}")
                    filters.append({"kind": "numeric", "col": fcol, "op": op, "val": v})
            elif pd.api.types.is_datetime64_any_dtype(s):
                mode = st.radio("Date input", ["Slider", "Manual"],
                                horizontal=True, key=f"cfg_fdmode_{i}")
                dmin = pd.to_datetime(s.min()).to_pydatetime()
                dmax = pd.to_datetime(s.max()).to_pydatetime()
                if mode == "Slider":
                    start, end = st.slider("Date range", min_value=dmin, max_value=dmax,
                                           value=(dmin, dmax), key=f"cfg_fval_{i}")
                else:
                    cc1, cc2 = st.columns(2)
                    start = cc1.date_input("Start", dmin, key=f"cfg_fstart_{i}")
                    end = cc2.date_input("End", dmax, key=f"cfg_fend_{i}")
                filters.append({"kind": "date", "col": fcol, "op": "range",
                                "val": [str(pd.to_datetime(start)), str(pd.to_datetime(end))]})
            else:
                op = st.selectbox("Condition",
                                  ["contains", "equals", "starts with", "ends with",
                                   "not contains", "not equals", "is in"],
                                  key=f"cfg_fop_{i}")
                if op == "is in":
                    choices = st.multiselect(
                        "Values",
                        sorted(s.dropna().astype(str).unique().tolist())[:5000],
                        key=f"cfg_fval_{i}")
                    filters.append({"kind": "categorical", "col": fcol, "op": op, "val": choices})
                else:
                    v = st.text_input("Value", key=f"cfg_fval_{i}")
                    filters.append({"kind": "categorical", "col": fcol, "op": op, "val": v})

        cfg["filters"] = filters

        # --- Create button: ONLY now is a chart added ---
        if st.button("\u2705 Create chart", type="primary"):
            st.session_state.chart_seq += 1
            cfg["id"] = st.session_state.chart_seq
            st.session_state.charts.append(cfg)
            st.rerun()

    # ----------------------------------------------------------------------
    # RENDER created charts (each with chart on left, customize panel on right)
    # ----------------------------------------------------------------------
    if not st.session_state.charts:
        st.info("No charts yet. Configure one above and click **Create chart**.")
    else:
        head, clearcol = st.columns([4, 1])
        head.markdown(f"### Created charts ({len(st.session_state.charts)})")
        if clearcol.button("Remove all", use_container_width=True):
            st.session_state.charts = []
            st.rerun()

        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        NONE_ = "—"

        for idx, cfg in enumerate(list(st.session_state.charts)):
            label = cfg.get("title") or f"{cfg['chart_type']} #{cfg['id']}"
            bar1, bar2 = st.columns([5, 1])
            bar1.markdown(f"**{label}**")
            if bar2.button("🗑 Remove", key=f"rm_{cfg['id']}",
                           use_container_width=True):
                st.session_state.charts = [
                    c for c in st.session_state.charts if c["id"] != cfg["id"]
                ]
                st.rerun()

            # 3:1 split — chart on the left, customize panel on the right.
            chart_col, ctrl_col = st.columns([3, 1], gap="medium")

            # ---- CUSTOMIZE PANEL (right column) ----
            with ctrl_col:
                cid = cfg["id"]

                # Pull live values from session_state widget keys when present,
                # falling back to the stored config. Then write the resolved
                # values back to cfg so the chart picks them up.
                with st.expander("🎨 Style", expanded=False):
                    cfg["title"] = st.text_input(
                        "Title", value=cfg.get("title", "") or "",
                        key=f"ctrl_title_{cid}",
                    )
                    engine_opts_p = ["Plotly"]
                    if MATPLOTLIB_OK:
                        engine_opts_p.append("Matplotlib")
                    cfg["engine"] = st.selectbox(
                        "Engine",
                        engine_opts_p,
                        index=engine_opts_p.index(cfg.get("engine", "Plotly"))
                              if cfg.get("engine") in engine_opts_p else 0,
                        key=f"ctrl_engine_{cid}",
                    )
                    cfg["template"] = st.selectbox(
                        "Theme",
                        ["plotly", "plotly_white", "plotly_dark", "ggplot2",
                         "seaborn", "simple_white"],
                        index=["plotly", "plotly_white", "plotly_dark",
                               "ggplot2", "seaborn", "simple_white"].index(
                            cfg.get("template", "plotly_white")
                        ),
                        key=f"ctrl_template_{cid}",
                    )
                    current_label = PALETTE_KEY_TO_LABEL.get(
                        cfg.get("palette", "Plotly"), PALETTE_LABELS[0]
                    )
                    palette_label_ctrl = st.selectbox(
                        "Palette", PALETTE_LABELS,
                        index=PALETTE_LABELS.index(current_label),
                        key=f"ctrl_palette_{cid}",
                    )
                    cfg["palette"] = PALETTE_LABEL_TO_KEY.get(palette_label_ctrl, "Plotly")
                    cfg["show_legend"] = st.checkbox(
                        "Show legend", value=cfg.get("show_legend", True),
                        key=f"ctrl_legend_{cid}",
                    )
                    cfg["opacity"] = st.slider(
                        "Opacity", 0.1, 1.0,
                        float(cfg.get("opacity", 0.8)), 0.05,
                        key=f"ctrl_opacity_{cid}",
                    )
                    top_n_available_ctrl = (
                        cfg.get("chart_type") in ("Bar", "Pie")
                        or (bool(num_cols)
                            and cfg.get("chart_type") != "Correlation heatmap")
                    )
                    if top_n_available_ctrl:
                        current_top_n = cfg.get("top_n")
                        top_n_enabled_ctrl = st.checkbox(
                            "Show only Top N values",
                            value=current_top_n is not None,
                            key=f"ctrl_top_n_enabled_{cid}",
                            help="Visualization-only: sorts by the chart's "
                                 "numeric measure and applies `.head(N)` "
                                 "without changing the dataframe fragment.",
                        )
                        if top_n_enabled_ctrl:
                            max_top_n_ctrl = max(1, int(len(df)))
                            try:
                                current_top_n = int(current_top_n or min(10, max_top_n_ctrl))
                            except (TypeError, ValueError):
                                current_top_n = min(10, max_top_n_ctrl)
                            current_top_n = min(max(1, current_top_n), max_top_n_ctrl)
                            cfg["top_n"] = st.slider(
                                "Top N", 1, max_top_n_ctrl, current_top_n, 1,
                                key=f"ctrl_top_n_{cid}",
                            )
                        else:
                            cfg["top_n"] = None
                    else:
                        cfg["top_n"] = None
                    cfg["log_x"] = st.checkbox(
                        "Log X", value=cfg.get("log_x", False),
                        key=f"ctrl_logx_{cid}",
                    )
                    cfg["log_y"] = st.checkbox(
                        "Log Y", value=cfg.get("log_y", False),
                        key=f"ctrl_logy_{cid}",
                    )
                    if cfg["chart_type"] in ("Scatter", "Bubble"):
                        if STATSMODELS_OK:
                            cfg["trendline"] = st.checkbox(
                                "Trendline (OLS)",
                                value=cfg.get("trendline", False),
                                key=f"ctrl_trend_{cid}",
                            )

                with st.expander("🔀 Color & grouping", expanded=False):
                    color_choices = [NONE_] + all_cols
                    cur_color = cfg.get("color", NONE_)
                    cfg["color"] = st.selectbox(
                        "Color / group by", color_choices,
                        index=color_choices.index(cur_color) if cur_color in color_choices else 0,
                        key=f"ctrl_color_{cid}",
                    )
                    facet_choices = [NONE_] + cat_cols
                    cur_facet = cfg.get("facet", NONE_)
                    cfg["facet"] = st.selectbox(
                        "Facet by", facet_choices,
                        index=facet_choices.index(cur_facet) if cur_facet in facet_choices else 0,
                        key=f"ctrl_facet_{cid}",
                    )

                with st.expander("🔍 Filters", expanded=False):
                    st.caption(
                        "Filters apply only to this chart. "
                        "Edit and press **Apply** to update the chart."
                    )

                    # Each chart keeps a separate DRAFT filter list in session
                    # state so the user can edit without each keystroke
                    # re-rendering the chart. Only "Apply" commits the draft
                    # to cfg["filters"].
                    draft_key = f"draft_filters_{cid}"
                    if draft_key not in st.session_state:
                        st.session_state[draft_key] = list(cfg.get("filters", []) or [])
                    draft = st.session_state[draft_key]

                    n_f = st.number_input(
                        "Number of filters", 0, 8, len(draft), 1,
                        key=f"ctrl_nf_{cid}",
                    )

                    new_draft: list[dict] = []
                    for j in range(int(n_f)):
                        prev = draft[j] if j < len(draft) else {}
                        st.markdown(f"_Filter {j+1}_")
                        col_default = prev.get("col", all_cols[0]) if all_cols else None
                        fcol = st.selectbox(
                            "Column", all_cols,
                            index=all_cols.index(col_default) if col_default in all_cols else 0,
                            key=f"ctrl_fcol_{cid}_{j}",
                        )
                        s = df[fcol]

                        if pd.api.types.is_numeric_dtype(s):
                            ops = [">", "<", "between", "==", "!=", ">=", "<="]
                            op = st.selectbox(
                                "Condition", ops,
                                index=ops.index(prev.get("op")) if prev.get("op") in ops else 0,
                                key=f"ctrl_fop_{cid}_{j}",
                            )
                            if op == "between":
                                s_nonnull = s.dropna()
                                if s_nonnull.empty:
                                    st.caption(
                                        f"`{fcol}` has no non-null values — "
                                        "pick a number."
                                    )
                                    v = st.number_input(
                                        "Value",
                                        value=float(prev.get("val", 0.0))
                                              if isinstance(prev.get("val"), (int, float)) else 0.0,
                                        key=f"ctrl_fval_{cid}_{j}",
                                    )
                                    new_draft.append({"kind": "numeric",
                                                      "col": fcol, "op": "==",
                                                      "val": v})
                                else:
                                    lo = float(np.nanmin(s)); hi = float(np.nanmax(s))
                                    if lo == hi:
                                        st.caption(
                                            f"`{fcol}` is constant at "
                                            f"**{lo:g}** — filter keeps rows "
                                            "equal to this value."
                                        )
                                        new_draft.append({"kind": "numeric",
                                                          "col": fcol,
                                                          "op": "between",
                                                          "val": [lo, hi]})
                                    else:
                                        pv = prev.get("val") or [lo, hi]
                                        rng = st.slider(
                                            "Range", lo, hi,
                                            (float(pv[0]), float(pv[1])),
                                            key=f"ctrl_fval_{cid}_{j}",
                                        )
                                        new_draft.append({"kind": "numeric",
                                                          "col": fcol, "op": op,
                                                          "val": [rng[0], rng[1]]})
                            elif op in ("==", "!="):
                                # Equal/not-equal also gets a value picker
                                # built from the actual column values.
                                opts = sorted(s.dropna().unique().tolist())[:5000]
                                opt_strs = [str(v) for v in opts]
                                default_val = prev.get("val")
                                default_idx = (
                                    opt_strs.index(str(default_val))
                                    if str(default_val) in opt_strs else 0
                                )
                                pick = st.selectbox(
                                    "Value", opt_strs, index=default_idx,
                                    key=f"ctrl_fval_{cid}_{j}",
                                )
                                # Convert back to original dtype if numeric.
                                try:
                                    v = type(opts[opt_strs.index(pick)])(pick)
                                except Exception:
                                    v = pick
                                new_draft.append({"kind": "numeric", "col": fcol,
                                                  "op": op, "val": v})
                            else:
                                v = st.number_input(
                                    "Value", value=float(prev.get("val", 0.0))
                                            if isinstance(prev.get("val"), (int, float)) else 0.0,
                                    key=f"ctrl_fval_{cid}_{j}",
                                )
                                new_draft.append({"kind": "numeric", "col": fcol,
                                                  "op": op, "val": v})

                        elif pd.api.types.is_datetime64_any_dtype(s):
                            dmin = pd.to_datetime(s.min()).to_pydatetime()
                            dmax = pd.to_datetime(s.max()).to_pydatetime()
                            try:
                                start_default = pd.to_datetime(prev.get("val", [dmin, dmax])[0]).to_pydatetime()
                                end_default = pd.to_datetime(prev.get("val", [dmin, dmax])[1]).to_pydatetime()
                            except Exception:
                                start_default, end_default = dmin, dmax
                            start, end = st.slider(
                                "Date range", min_value=dmin, max_value=dmax,
                                value=(start_default, end_default),
                                key=f"ctrl_fval_{cid}_{j}",
                            )
                            new_draft.append({"kind": "date", "col": fcol,
                                              "op": "range",
                                              "val": [str(pd.to_datetime(start)),
                                                      str(pd.to_datetime(end))]})

                        else:
                            cat_ops = ["equals", "not equals", "is in", "not in",
                                       "contains", "starts with", "ends with",
                                       "not contains"]
                            op = st.selectbox(
                                "Condition", cat_ops,
                                index=cat_ops.index(prev.get("op")) if prev.get("op") in cat_ops else 0,
                                key=f"ctrl_fop_{cid}_{j}",
                            )
                            opts = sorted(s.dropna().astype(str).unique().tolist())[:5000]

                            if op in ("is in", "not in"):
                                default_vals = prev.get("val") if isinstance(prev.get("val"), list) else []
                                choices = st.multiselect(
                                    "Values", opts,
                                    default=[v for v in default_vals if v in opts],
                                    key=f"ctrl_fval_{cid}_{j}",
                                )
                                new_draft.append({"kind": "categorical", "col": fcol,
                                                  "op": op, "val": choices})
                            elif op in ("equals", "not equals"):
                                # Use a single-select picker (no manual typing).
                                default_val = prev.get("val") if not isinstance(prev.get("val"), list) else None
                                default_idx = opts.index(str(default_val)) if str(default_val) in opts else 0
                                if opts:
                                    pick = st.selectbox(
                                        "Value", opts, index=default_idx,
                                        key=f"ctrl_fval_{cid}_{j}",
                                    )
                                else:
                                    pick = ""
                                new_draft.append({"kind": "categorical", "col": fcol,
                                                  "op": op, "val": pick})
                            else:
                                # contains / starts with / ends with / not contains
                                # — a substring search. Offer a picker of full
                                # values OR a manual substring (radio toggle).
                                input_mode = st.radio(
                                    "How to specify the value?",
                                    ["Pick from list", "Type manually"],
                                    horizontal=True,
                                    key=f"ctrl_fmode_{cid}_{j}",
                                )
                                if input_mode == "Pick from list" and opts:
                                    pick = st.selectbox(
                                        "Pick a value", opts,
                                        index=(opts.index(str(prev.get("val", "")))
                                               if str(prev.get("val", "")) in opts else 0),
                                        key=f"ctrl_fval_{cid}_{j}",
                                    )
                                else:
                                    pick = st.text_input(
                                        "Substring",
                                        value=str(prev.get("val", "")) if not isinstance(prev.get("val"), list) else "",
                                        key=f"ctrl_fval_{cid}_{j}",
                                    )
                                new_draft.append({"kind": "categorical", "col": fcol,
                                                  "op": op, "val": pick})

                    # Keep the in-progress draft alive across reruns
                    st.session_state[draft_key] = new_draft

                    # Apply / Reset row
                    ap1, ap2 = st.columns(2)
                    if ap1.button(
                        "✅ Apply",
                        key=f"apply_filters_{cid}",
                        type="primary",
                        use_container_width=True,
                    ):
                        cfg["filters"] = list(new_draft)
                        # Persist back to the chart in session state
                        for k, c in enumerate(st.session_state.charts):
                            if c["id"] == cid:
                                st.session_state.charts[k] = cfg
                                break
                        st.rerun()
                    if ap2.button(
                        "Reset",
                        key=f"reset_filters_{cid}",
                        use_container_width=True,
                    ):
                        cfg["filters"] = []
                        st.session_state[draft_key] = []
                        for k, c in enumerate(st.session_state.charts):
                            if c["id"] == cid:
                                st.session_state.charts[k] = cfg
                                break
                        st.rerun()

                # Save the mutated cfg back into session state (for style/color
                # changes which apply live; filters are handled above).
                for k, c in enumerate(st.session_state.charts):
                    if c["id"] == cid:
                        st.session_state.charts[k] = cfg
                        break

            # ---- CHART (left column) — built with the current cfg ----
            with chart_col:
                fdf, flabels = apply_viz_filters(df, cfg.get("filters", []))
                if fdf.empty:
                    st.warning("No rows match this chart's filters.")
                else:
                    engine = cfg.get("engine", "Plotly")
                    if engine == "Matplotlib" and MATPLOTLIB_OK:
                        fig, err = build_matplotlib_figure(fdf, cfg)
                        if err:
                            st.error(f"Could not render chart: {err}")
                        else:
                            st.pyplot(fig, use_container_width=True)
                            import matplotlib.pyplot as _plt
                            _plt.close(fig)
                    else:
                        fig, err = build_figure(fdf, cfg)
                        if err:
                            st.error(f"Could not render chart: {err}")
                        else:
                            st.plotly_chart(fig, use_container_width=True,
                                            key=f"chart_{cfg['id']}")
                    meta = f"{len(fdf):,} rows · {engine}"
                    if flabels:
                        meta += " · filters: " + " · ".join(flabels)
                    st.caption(meta)

            st.divider()



# ===========================================================================
# TAB 4 — REPORT & EXPORT
# ===========================================================================
with tab4:
    if st.session_state.df is None:
        st.info("📁 Load a dataset on the **Upload & Overview** tab first.")
        st.stop()

    df = st.session_state.df
    fp = df_fingerprint(df)
    default_stem = (st.session_state.file_name or "data").rsplit(".", 1)[0] + "_cleaned"

    st.subheader("📦 Export")
    st.caption(
        "Pick a filename and download the cleaned dataframe as CSV or Excel. "
        "Also export the transformation log and the pipeline recipe as JSON "
        "for replay on another dataset."
    )

    # ===== Filename field (pre-filled, editable) =====
    file_stem = st.text_input(
        "Filename (without extension)",
        value=default_stem,
        key="export_stem",
        help="This name is used for every download below. Extensions are "
             "added automatically (.csv, .xlsx, .json).",
    ).strip()
    if not file_stem:
        file_stem = default_stem

    st.markdown("---")

    # ===== Cleaned dataframe =====
    st.markdown("### Cleaned dataframe")
    st.caption(f"Final shape: **{len(df):,} rows × {df.shape[1]} columns**")

    e1, e2 = st.columns(2)
    e1.download_button(
        "⬇️ Download CSV",
        data=cached_csv_bytes(df, fp),
        file_name=f"{file_stem}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    try:
        e2.download_button(
            "⬇️ Download Excel",
            data=cached_excel_bytes(df, fp),
            file_name=f"{file_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
    except Exception as e:
        e2.caption(f"Excel export unavailable: {e}. Install `openpyxl`.")

    st.markdown("---")

    # ===== Transformation log =====
    n_steps = len(st.session_state.recipe)
    st.markdown("### Transformation log")
    st.caption(
        f"Structured record of every cleaning step "
        f"(**{n_steps}** step(s)). "
        "Schema per step: `operation`, `parameters`, `affected_columns`, "
        "`timestamp`."
    )

    log_payload = {
        "source_file": st.session_state.file_name,
        "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "final_shape": {"rows": int(len(df)), "cols": int(df.shape[1])},
        "n_steps": n_steps,
        "steps": [
            {
                "operation": s.get("operation") or s.get("action", "?"),
                "parameters": s.get("parameters") or s.get("params", {}),
                "affected_columns": s.get("affected_columns", []),
                "timestamp": s.get("timestamp"),
            }
            for s in st.session_state.recipe
        ],
    }
    log_bytes = json.dumps(log_payload, indent=2, default=str).encode("utf-8")

    with st.expander("Preview transformation log", expanded=False):
        if n_steps == 0:
            st.info("No steps yet.")
        else:
            preview_rows = []
            for i, step in enumerate(log_payload["steps"], 1):
                preview_rows.append({
                    "#": i,
                    "operation": step["operation"],
                    "affected_columns": ", ".join(step["affected_columns"]) or "—",
                    "timestamp": step["timestamp"],
                })
            st.dataframe(pd.DataFrame(preview_rows),
                         use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download transformation log (JSON)",
        data=log_bytes,
        file_name=f"{file_stem}_log.json",
        mime="application/json",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    # ===== Pipeline recipe (replay format) =====
    st.markdown("### Pipeline recipe")
    st.caption(
        "Replay-ready format. Import this JSON on the **Cleaning & Prep** tab "
        "to re-apply the same operations to a fresh dataset."
    )

    pipeline_payload = {
        "source_file": st.session_state.file_name,
        "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_steps": n_steps,
        # The full step records (including row counts) so replay can verify.
        "steps": st.session_state.recipe,
    }
    pipeline_bytes = json.dumps(pipeline_payload, indent=2, default=str).encode("utf-8")

    with st.expander("Preview pipeline recipe (raw JSON)", expanded=False):
        if n_steps == 0:
            st.info("No steps yet.")
        else:
            st.json(pipeline_payload, expanded=False)

    st.download_button(
        "⬇️ Download pipeline recipe (JSON)",
        data=pipeline_bytes,
        file_name=f"{file_stem}_pipeline.json",
        mime="application/json",
        type="primary",
        use_container_width=True,
    )
