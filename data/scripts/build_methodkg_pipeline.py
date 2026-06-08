#!/usr/bin/env python3
"""
Build MethodKG-ready NSF award outputs, version 7.

Fixes included through v7:
  1. Improved Co-PI parsing, including comma-separated Co-PI fields.
  2. Program edges use ProgramElementCode(s) as the primary program-node source.
  3. Missing-abstract rows are excluded from the annotation sample.
  4. Hard negatives are split into qual-only, quant-only, and method-heavy background strata.
  5. Legacy NSF organizations are flagged, with optional filtering.
  6. Very strict proximity-based MMR integration detection to reduce generic integration false positives.
  7. All design-label candidates are forced into the annotation sample.
  8. PI/Co-PI identity cleaning strips embedded emails and creates person_id.
  9. Invalid negative durations are flagged rather than removed.
  10. Integration detection is proximity-based: integration verbs must occur near both qualitative and quantitative evidence, or near explicit MMR plus methodological result/context terms.
  11. Adds project_text_id clusters based on normalized title + abstract.
  12. Supports cluster-aware annotation sampling so duplicate title/abstract awards stay together.
  13. Can sample from a cleaned awards file via --input_is_cleaned.

Input:
  Full_DSAA_Awards.csv

Outputs:
  cleaned_nsf_awards_2000_2025.csv
  nsf_awards_with_methodology_flags.csv
  annotation_sample_2000_2025.csv
  award_pi_edges.csv
  pi_collaboration_edges.csv
  award_institution_edges.csv
  award_program_edges.csv
  data_quality_report.csv

Example:
  python build_methodkg_pipeline_fixed_v7.py --input Full_DSAA_Awards.csv --outdir methodkg_outputs_v7 --sample_size 2500

Optional, to remove legacy NSF organizations instead of just flagging them:
  python build_methodkg_pipeline_fixed_v7.py --input Full_DSAA_Awards.csv --outdir methodkg_outputs_v7 --drop_legacy_orgs
"""

import argparse
import ast
import hashlib
import itertools
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_NSF_ORGS = {"DUE", "DRL", "EES", "DGE", "EEC"}

# -----------------------------
# Utility functions
# -----------------------------

def clean_excel_wrapped_value(x):
    """Clean values such as =\"0555934\" created by spreadsheet exports."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    elif s.startswith("='") and s.endswith("'"):
        s = s[2:-1]
    return s.strip()


def normalize_whitespace(x):
    if pd.isna(x):
        return ""
    s = clean_excel_wrapped_value(x)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ascii_fold(s):
    s = "" if pd.isna(s) else str(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_for_id(x):
    s = ascii_fold(clean_excel_wrapped_value(x)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_id(prefix, value):
    value = normalize_for_id(value)
    if not value:
        value = "unknown"
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def parse_money(x):
    s = clean_excel_wrapped_value(x)
    if not s:
        return np.nan
    s = s.replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_date_series(series):
    cleaned = series.apply(clean_excel_wrapped_value)
    return pd.to_datetime(cleaned, errors="coerce")


def safe_word_count(text):
    """Fast approximate word count for already-cleaned text."""
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return 0
    return len(text.split())


def bool_to_int(x):
    return int(bool(x))


def compile_any(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)


def regex_any(text, pattern):
    if not text:
        return False
    return bool(pattern.search(text))


def sample_n(df, n, random_state=42):
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0].copy()
    n = min(n, len(df))
    return df.sample(n=n, random_state=random_state)


def person_token_count(name):
    name = normalize_whitespace(name)
    if not name:
        return 0
    # Count alphabetic chunks. This is only a heuristic for Co-PI splitting.
    return len(re.findall(r"[A-Za-z]+", name))


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def extract_emails(value):
    """Return normalized email addresses found in a string."""
    s = clean_excel_wrapped_value(value)
    if not s:
        return []
    emails = [m.group(0).lower() for m in EMAIL_RE.finditer(s)]
    seen = set()
    out = []
    for email in emails:
        if email not in seen:
            seen.add(email)
            out.append(email)
    return out


def extract_first_email(*values):
    """Return the first email found across one or more values."""
    for value in values:
        emails = extract_emails(value)
        if emails:
            return emails[0]
    return ""


def strip_emails(value):
    """Remove embedded email addresses from a name/string."""
    s = clean_excel_wrapped_value(value)
    if not s:
        return ""
    s = EMAIL_RE.sub(" ", s)
    return s


def clean_person_name(name):
    """Clean a PI/Co-PI name and strip embedded emails/tags."""
    name = strip_emails(name)
    name = normalize_whitespace(name)
    name = re.sub(r"^(dr|prof|professor)\.\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(email|e-mail|mail|phone|tel)\b\s*[:=]?", " ", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" ,;:()[]<>")
    return name


def make_person_id(name, email=""):
    """Use email as the most stable identity key when available; otherwise use normalized name."""
    email = extract_first_email(email)
    name = clean_person_name(name)
    if email:
        return make_id("person", f"email:{email}")
    return make_id("person", f"name:{name}")


def person_token_count(name):
    name = clean_person_name(name)
    if not name:
        return 0
    # Count alphabetic chunks. This is only a heuristic for Co-PI splitting.
    return len(re.findall(r"[A-Za-z]+", name))


def split_commas_if_likely_multiple_people(segment):
    """
    Split a Co-PI segment on commas only when commas are likely to separate people.

    Why heuristic? Some name formats use commas inside one name, e.g., "Smith, John".
    In the NSF Co-PI export, however, multiple Co-PIs are often stored as
    "First Last, First Last". This function splits when at least one post-comma
    chunk looks like a full person name with two or more alphabetic tokens.
    """
    segment = normalize_whitespace(segment)
    if not segment or "," not in segment:
        return [segment] if segment else []

    parts = [p.strip() for p in segment.split(",") if p.strip()]
    if len(parts) <= 1:
        return [segment]

    post_chunks_full_names = sum(person_token_count(p) >= 2 for p in parts[1:])
    all_chunks_short = all(person_token_count(p) <= 1 for p in parts)

    if post_chunks_full_names >= 1 and not all_chunks_short:
        return parts

    if person_token_count(parts[0]) >= 2 and len(parts) >= 3:
        return parts

    return [segment]


def split_email_boundaries(segment):
    """
    Split strings like 'Alice A a@x.edu Bob B b@y.edu' into person-like chunks.
    This helps when the NSF export collapses newline-separated Co-PIs into one cell.
    """
    segment = normalize_whitespace(segment)
    if not segment:
        return []

    matches = list(EMAIL_RE.finditer(segment))
    if len(matches) <= 1:
        return [segment]

    chunks = []
    start = 0
    for m in matches:
        end = m.end()
        chunk = segment[start:end].strip(" ,;")
        if chunk:
            chunks.append(chunk)
        start = end

    tail = segment[start:].strip(" ,;")
    if tail:
        chunks.append(tail)

    return chunks


def split_people_records(value):
    """
    Split a Co-PI field into records with name, email, and person_id.

    Handles:
      - semicolons, pipes, slashes, newlines
      - "and" between names
      - comma-separated Co-PIs of the form "First Last, First Last"
      - embedded emails in names
      - stringified Python lists from prior CSV exports
    """
    raw = clean_excel_wrapped_value(value)
    if not raw:
        return []

    s = raw.strip()

    # Handle list-like strings, e.g., "['A B', 'C D']".
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                records = []
                for item in parsed:
                    records.extend(split_people_records(item))
                seen = set()
                out = []
                for rec in records:
                    key = rec["person_id"]
                    if key and key not in seen:
                        seen.add(key)
                        out.append(rec)
                return out
        except Exception:
            pass

    # Preserve row-like separators before whitespace normalization.
    s = s.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
    s = re.sub(r"\s*\|\s*", ";", s)
    s = re.sub(r"\s*/\s*", ";", s)
    s = re.sub(r"\s+ and \s+", ";", s, flags=re.IGNORECASE)
    s = normalize_whitespace(s)

    semi_parts = [p.strip() for p in s.split(";") if p.strip()]

    raw_parts = []
    for seg in semi_parts:
        for email_chunk in split_email_boundaries(seg):
            raw_parts.extend(split_commas_if_likely_multiple_people(email_chunk))

    seen = set()
    out = []
    for part in raw_parts:
        email = extract_first_email(part)
        name = clean_person_name(part)
        if not name and not email:
            continue
        person_id = make_person_id(name, email)
        if person_id in seen:
            continue
        seen.add(person_id)
        out.append({
            "person_id": person_id,
            "name": name,
            "email": email,
        })
    return out


def split_people(value):
    """Backward-compatible helper that returns cleaned names only."""
    return [rec["name"] for rec in split_people_records(value) if rec.get("name")]


def split_program_codes(value):
    """Split NSF ProgramElementCode(s) or ProgramReferenceCode(s) into stable code tokens."""
    s = normalize_whitespace(value)
    if not s:
        return []
    # Codes can be comma, semicolon, pipe, slash, or whitespace separated.
    parts = re.split(r"[,;|/\s]+", s)
    codes = []
    seen = set()
    for p in parts:
        p = re.sub(r"[^A-Za-z0-9]", "", p).strip()
        if not p:
            continue
        key = p.upper()
        if key not in seen:
            seen.add(key)
            codes.append(key)
    return codes


def split_program_names_conservative(value):
    """
    Conservative program-name splitter. Avoids comma splitting because program
    names often contain commas. Used only as fallback if no program codes exist.
    """
    s = normalize_whitespace(value)
    if not s:
        return []
    s = re.sub(r"\s*\|\s*", ";", s)
    s = s.replace("\n", ";").replace("\r", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        key = normalize_for_id(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


# -----------------------------
# Methodology signal patterns
# -----------------------------

EXPLICIT_MMR_PATTERNS = compile_any([
    r"\bmixed[-\s]?methods?\b",
    r"\bmixed[-\s]?methodolog(?:y|ies)\b",
    r"\bmixed\s+research\b",
    r"\bmixed[-\s]?methods?\s+research\b",
    r"\bmulti[-\s]?methods?\b",
    r"\bmultimethods?\b",
    r"\bmultimethodolog(?:y|ies)\b",
    r"\bMMR\b",
])

QUAL_PATTERNS = compile_any([
    r"\bqualitative\b",
    r"\binterviews?\b",
    r"\bfocus\s+groups?\b",
    r"\bcase\s+stud(?:y|ies)\b",
    r"\bethnograph\w*\b",
    r"\bobservations?\b",
    r"\bclassroom\s+observations?\b",
    r"\bfield\s+notes?\b",
    r"\bopen[-\s]?ended\b",
    r"\bthematic\s+analysis\b",
    r"\bcontent\s+analysis\b",
    r"\bdiscourse\s+analysis\b",
    r"\bcoding\b",
    r"\bgrounded\s+theory\b",
    r"\bphenomenolog\w*\b",
    r"\bnarrative\s+analysis\b",
])

QUANT_PATTERNS = compile_any([
    r"\bquantitative\b",
    r"\bsurveys?\b",
    r"\bquestionnaires?\b",
    r"\bstatistical\b",
    r"\bstatistics\b",
    r"\bregression\b",
    r"\blogistic\s+regression\b",
    r"\blinear\s+model\b",
    r"\bANOVA\b",
    r"\bANCOVA\b",
    r"\bt[-\s]?test\b",
    r"\bchi[-\s]?square\b",
    r"\bexperiment\w*\b",
    r"\brandomi[sz]ed\b",
    r"\bquasi[-\s]?experimental\b",
    r"\bpre[-\s]?post\b",
    r"\bassessment\b",
    r"\bscale\b",
    r"\bfactor\s+analysis\b",
    r"\bstructural\s+equation\b",
    r"\bSEM\b",
    r"\bBayesian\b",
    r"\bdata\s+mining\b",
    r"\blearning\s+analytics\b",
    r"\bmodeling\b",
    r"\bmodelling\b",
])

IMPLICIT_MMR_PHRASE_PATTERNS = compile_any([
    r"\bqualitative\s+and\s+quantitative\b",
    r"\bquantitative\s+and\s+qualitative\b",
    r"\bboth\s+qualitative\s+and\s+quantitative\b",
    r"\bqualitative\s*/\s*quantitative\b",
    r"\bquantitative\s*/\s*qualitative\b",
    r"\binterviews?\s+and\s+surveys?\b",
    r"\bsurveys?\s+and\s+interviews?\b",
    r"\bfocus\s+groups?\s+and\s+surveys?\b",
    r"\bsurveys?\s+and\s+focus\s+groups?\b",
    r"\bcase\s+stud(?:y|ies)\s+and\s+statistical\b",
    r"\bthematic\s+analysis\s+and\s+regression\b",
])

DESIGN_PATTERNS = compile_any([
    r"\bconvergent\s+parallel\b",
    r"\bconvergent\s+design\b",
    r"\bexplanatory\s+sequential\b",
    r"\bsequential\s+explanatory\b",
    r"\bexploratory\s+sequential\b",
    r"\bsequential\s+exploratory\b",
    r"\bembedded\s+design\b",
    r"\btransformative\s+design\b",
    r"\bmultiphase\s+design\b",
    r"\btriangulation\s+design\b",
])

INTEGRATION_PATTERNS = compile_any([
    r"\bintegrat\w*.{0,80}\bqualitative\b.{0,80}\bquantitative\b",
    r"\bintegrat\w*.{0,80}\bquantitative\b.{0,80}\bqualitative\b",
    r"\bqualitative\b.{0,80}\bquantitative\b.{0,80}\bintegrat\w*",
    r"\bquantitative\b.{0,80}\bqualitative\b.{0,80}\bintegrat\w*",
    r"\bmerge\w*.{0,80}\bqualitative\b.{0,80}\bquantitative\b",
    r"\btriangulat\w*.{0,80}\bqualitative\b.{0,80}\bquantitative\b",
    r"\bcombine\w*.{0,80}\bqualitative\b.{0,80}\bquantitative\b",
    r"\bconnect\w*.{0,80}\bqualitative\b.{0,80}\bquantitative\b",
    r"\bbuilding\s+from\s+qualitative\s+to\s+quantitative\b",
    r"\bbuilding\s+from\s+quantitative\s+to\s+qualitative\b",
    r"\bmeta[-\s]?inference\b",
    r"\bjoint\s+display\b",
])

METHOD_HEAVY_PATTERNS = compile_any([
    r"\bmethodolog\w*\b",
    r"\bmethods?\b",
    r"\bdata\s+collection\b",
    r"\bdata\s+analysis\b",
    r"\bevaluation\b",
    r"\bassess\w*\b",
    r"\bmeasure\w*\b",
    r"\bsample\b",
    r"\bparticipants?\b",
    r"\binstruments?\b",
    r"\bprotocol\b",
    r"\bsurveys?\b",
    r"\binterviews?\b",
    r"\bobservations?\b",
    r"\bstatistical\b",
    r"\bregression\b",
    r"\bcase\s+stud(?:y|ies)\b",
])



# -----------------------------
# Project/text duplicate clusters
# -----------------------------

def make_project_text_id(title, abstract, award_id=""):
    """
    Stable cluster id for rows with the same normalized title + abstract.

    NSF records sometimes contain supplements, continuations, or collaborative
    awards with different AwardNumber values but identical project descriptions.
    Since MethodKG labels are assigned from title/abstract text, these rows should
    be auditable and kept together during sampling and splitting.
    """
    title_norm = normalize_for_id(title)
    abstract_norm = normalize_for_id(abstract)
    text_key = f"{title_norm} {abstract_norm}".strip()
    if not text_key:
        # Avoid merging all missing-text rows into one cluster.
        award_key = clean_excel_wrapped_value(award_id) or "unknown_award"
        text_key = f"missing_text_award:{award_key}"
    h = hashlib.sha1(text_key.encode("utf-8")).hexdigest()[:16]
    return f"projtxt_{h}"


def ensure_project_text_clusters(df):
    """Add project_text_id and duplicate-cluster audit columns."""
    out = df.copy()

    if "award_id" not in out.columns:
        if "AwardNumber" in out.columns:
            out["award_id"] = out["AwardNumber"].apply(clean_excel_wrapped_value)
            out["award_id"] = out["award_id"].astype(str).str.replace(r"\D", "", regex=True).str.strip()
        else:
            out["award_id"] = [f"row_{i}" for i in range(len(out))]

    if "title_clean" not in out.columns:
        if "Title" in out.columns:
            out["title_clean"] = out["Title"].apply(normalize_whitespace)
        else:
            out["title_clean"] = ""

    if "abstract_clean" not in out.columns:
        if "Abstract" in out.columns:
            out["abstract_clean"] = out["Abstract"].apply(normalize_whitespace)
        else:
            out["abstract_clean"] = ""

    if "abstract_word_count" not in out.columns:
        out["abstract_word_count"] = out["abstract_clean"].apply(safe_word_count)
    if "has_abstract" not in out.columns:
        out["has_abstract"] = (out["abstract_word_count"] > 0).astype(int)
    else:
        out["has_abstract"] = pd.to_numeric(out["has_abstract"], errors="coerce").fillna(0).astype(int)
    if "title_word_count" not in out.columns:
        out["title_word_count"] = out["title_clean"].apply(safe_word_count)
    else:
        out["title_word_count"] = pd.to_numeric(out["title_word_count"], errors="coerce").fillna(0).astype(int)
    out["abstract_word_count"] = pd.to_numeric(out["abstract_word_count"], errors="coerce").fillna(0).astype(int)

    out["project_text_id"] = out.apply(
        lambda r: make_project_text_id(r.get("title_clean", ""), r.get("abstract_clean", ""), r.get("award_id", "")),
        axis=1,
    )

    cluster_counts = out.groupby("project_text_id", dropna=False)["award_id"].nunique()
    out["project_text_cluster_size"] = out["project_text_id"].map(cluster_counts).fillna(1).astype(int)
    out["duplicate_project_text_flag"] = (out["project_text_cluster_size"] > 1).astype(int)

    return out


def build_project_text_cluster_report(df):
    """Create an audit table for duplicate title+abstract/project-text clusters."""
    if "project_text_id" not in df.columns:
        df = ensure_project_text_clusters(df)

    def join_unique(values, limit=40):
        vals = []
        seen = set()
        for v in values:
            if pd.isna(v):
                continue
            s = str(v)
            if not s or s in seen:
                continue
            seen.add(s)
            vals.append(s)
            if len(vals) >= limit:
                vals.append("...")
                break
        return "|".join(vals)

    agg_map = {
        "award_count": ("award_id", "nunique"),
        "row_count": ("award_id", "size"),
        "award_ids": ("award_id", lambda x: join_unique(x, limit=50)),
        "title_clean": ("title_clean", "first"),
    }
    if "abstract_word_count" in df.columns:
        agg_map["abstract_word_count"] = ("abstract_word_count", "first")
    if "start_year" in df.columns:
        agg_map["start_year_min"] = ("start_year", "min")
        agg_map["start_year_max"] = ("start_year", "max")

    optional_cols = [
        "candidate_stratum", "NSFDirectorate", "NSFOrganization", "ProgramElementCode(s)",
        "organization_clean", "person_id", "pi_clean"
    ]
    for col in optional_cols:
        if col in df.columns:
            agg_map[f"{col}_values"] = (col, lambda x, c=col: join_unique(x, limit=20))

    report = df.groupby("project_text_id", dropna=False).agg(**agg_map).reset_index()
    report["duplicate_project_text_flag"] = (report["award_count"] > 1).astype(int)
    report = report.sort_values(["award_count", "project_text_id"], ascending=[False, True]).reset_index(drop=True)
    return report

# -----------------------------
# Main cleaning
# -----------------------------

def clean_awards(df, start_year=2000, end_year=2025, drop_legacy_orgs=False):
    required_cols = [
        "AwardNumber",
        "Title",
        "NSFOrganization",
        "Program(s)",
        "StartDate",
        "LastAmendmentDate",
        "PrincipalInvestigator",
        "State",
        "Organization",
        "AwardInstrument",
        "ProgramManager",
        "EndDate",
        "AwardedAmountToDate",
        "Co-PIName(s)",
        "PIEmailAddress",
        "OrganizationStreet",
        "OrganizationCity",
        "OrganizationState",
        "OrganizationZip",
        "OrganizationPhone",
        "NSFDirectorate",
        "ProgramElementCode(s)",
        "ProgramReferenceCode(s)",
        "ARRAAmount",
        "Abstract",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(clean_excel_wrapped_value)

    out["award_id"] = out["AwardNumber"].apply(clean_excel_wrapped_value)
    out["award_id"] = out["award_id"].str.replace(r"\D", "", regex=True).str.strip()

    out["title_clean"] = out["Title"].apply(normalize_whitespace)
    out["abstract_clean"] = out["Abstract"].apply(normalize_whitespace)

    out["start_date"] = parse_date_series(out["StartDate"])
    out["end_date"] = parse_date_series(out["EndDate"])
    out["last_amendment_date"] = parse_date_series(out["LastAmendmentDate"])

    out["start_year"] = out["start_date"].dt.year
    out["end_year"] = out["end_date"].dt.year

    out["award_amount"] = out["AwardedAmountToDate"].apply(parse_money)
    out["arra_amount"] = out["ARRAAmount"].apply(parse_money)

    out["abstract_word_count"] = out["abstract_clean"].apply(safe_word_count)
    out["has_abstract"] = (out["abstract_word_count"] > 0).astype(int)
    out["title_word_count"] = out["title_clean"].apply(safe_word_count)

    out["organization_clean"] = out["Organization"].apply(normalize_whitespace)
    out["organization_norm"] = out["organization_clean"].apply(normalize_for_id)
    out["institution_id"] = out["organization_clean"].apply(lambda x: make_id("inst", x))

    out["pi_email_extracted"] = out.apply(
        lambda r: extract_first_email(r.get("PIEmailAddress", ""), r.get("PrincipalInvestigator", "")),
        axis=1,
    )
    out["pi_clean"] = out["PrincipalInvestigator"].apply(clean_person_name)
    out["pi_name_clean"] = out["pi_clean"]
    out["pi_norm"] = out["pi_clean"].apply(normalize_for_id)
    out["person_id"] = out.apply(lambda r: make_person_id(r["pi_clean"], r["pi_email_extracted"]), axis=1)
    # pi_id is retained as an alias for person_id for backward compatibility.
    out["pi_id"] = out["person_id"]

    out["co_pi_records"] = out["Co-PIName(s)"].apply(split_people_records)
    out["co_pi_list"] = out["co_pi_records"].apply(lambda records: [rec.get("name", "") for rec in records if rec.get("name", "")])
    out["num_co_pis"] = out["co_pi_records"].apply(len)
    out["team_size"] = out.apply(
        lambda r: int(bool(r["pi_clean"])) + int(r["num_co_pis"]),
        axis=1,
    )

    today = pd.Timestamp(datetime.today().date())
    out["award_status_derived"] = np.where(
        out["end_date"].isna(),
        "unknown",
        np.where(out["end_date"] >= today, "active", "expired"),
    )

    out["duration_days"] = (out["end_date"] - out["start_date"]).dt.days
    out["duration_months"] = out["duration_days"] / 30.4375
    out["invalid_duration_flag"] = (
        out["duration_days"].notna() & (out["duration_days"] < 0)
    ).astype(int)

    out["legacy_nsf_org_flag"] = (~out["NSFOrganization"].isin(TARGET_NSF_ORGS)).astype(int)
    out["nsf_org_scope"] = np.where(
        out["legacy_nsf_org_flag"] == 1,
        "legacy_or_other",
        "target_five_units",
    )

    before_filter_rows = len(out)
    out = out[(out["start_year"] >= start_year) & (out["start_year"] <= end_year)].copy()

    if drop_legacy_orgs:
        out = out[out["legacy_nsf_org_flag"] == 0].copy()

    # Drop duplicate award IDs. Keep latest amendment if available.
    out = out.sort_values(
        by=["award_id", "last_amendment_date"],
        ascending=[True, False],
        na_position="last",
    )
    out["duplicate_award_id_flag"] = out.duplicated(subset=["award_id"], keep="first").astype(int)
    out = out.drop_duplicates(subset=["award_id"], keep="first").copy()

    # Add project/text duplicate clusters after award-id deduplication.
    out = ensure_project_text_clusters(out)

    out.attrs["before_filter_rows"] = before_filter_rows

    preferred_order = [
        "award_id",
        "AwardNumber",
        "title_clean",
        "abstract_clean",
        "project_text_id",
        "project_text_cluster_size",
        "duplicate_project_text_flag",
        "has_abstract",
        "start_date",
        "end_date",
        "last_amendment_date",
        "start_year",
        "end_year",
        "award_status_derived",
        "duration_days",
        "duration_months",
        "invalid_duration_flag",
        "award_amount",
        "arra_amount",
        "NSFDirectorate",
        "NSFOrganization",
        "legacy_nsf_org_flag",
        "nsf_org_scope",
        "Program(s)",
        "ProgramElementCode(s)",
        "ProgramReferenceCode(s)",
        "AwardInstrument",
        "ProgramManager",
        "pi_clean",
        "pi_name_clean",
        "pi_email_extracted",
        "person_id",
        "pi_id",
        "Co-PIName(s)",
        "co_pi_records",
        "co_pi_list",
        "num_co_pis",
        "team_size",
        "PIEmailAddress",
        "organization_clean",
        "institution_id",
        "State",
        "OrganizationState",
        "OrganizationCity",
        "OrganizationZip",
        "abstract_word_count",
        "title_word_count",
    ]

    remaining = [c for c in out.columns if c not in preferred_order]
    out = out[preferred_order + remaining]

    return out


# -----------------------------
# Methodology flags and strata
# -----------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_search_text(text):
    """Normalize text for fast phrase/token matching. Avoids costly Unicode folding."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def prep_phrases(phrases):
    return [normalize_search_text(p) for p in phrases if normalize_search_text(p)]


def has_phrase(norm_text, phrases):
    if not norm_text:
        return False
    padded = f" {norm_text} "
    for phrase_norm in phrases:
        if f" {phrase_norm} " in padded:
            return True
    return False


def has_token_prefix(norm_text, prefixes):
    if not norm_text:
        return False
    tokens = norm_text.split()
    for tok in tokens:
        for pref in prefixes:
            if tok.startswith(pref):
                return True
    return False




def has_strict_method_integration(norm_text, strict_phrases, integration_prefixes, window_size=10):
    """
    Detect MMR-relevant integration language.

    Generic words such as integrate/combined/connection are counted only if they
    occur near qualitative, quantitative, mixed-methods, or methodology-context terms.
    This reduces false positives such as 'integrating data science into curriculum'.
    """
    if not norm_text:
        return False

    if has_phrase(norm_text, strict_phrases):
        return True

    tokens = norm_text.split()
    if not tokens:
        return False

    # Context terms that make an integration verb likely methodological rather than curricular/administrative.
    context_terms = {
        "qualitative", "quantitative", "mixed", "method", "methods", "methodology", "methodological",
        "mmr", "interview", "interviews", "survey", "surveys", "questionnaire", "questionnaires",
        "focus", "group", "groups", "observation", "observations", "statistical", "regression",
        "findings", "results", "strand", "strands", "analysis", "analyses", "thematic", "coding",
    }

    for i, tok in enumerate(tokens):
        if not any(tok.startswith(prefix) for prefix in integration_prefixes):
            continue
        left = max(0, i - window_size)
        right = min(len(tokens), i + window_size + 1)
        window = set(tokens[left:right])

        # Strongest evidence: qualitative and quantitative occur near the integration verb.
        if "qualitative" in window and "quantitative" in window:
            return True

        # Mixed-methods context near the integration verb.
        if "mixed" in window and ("method" in window or "methods" in window or "methodology" in window):
            return True

        # Other methodology context terms near the integration verb.
        if window.intersection(context_terms - {"mixed"}):
            return True

    return False


def _phrase_to_regex(phrase):
    phrase = normalize_search_text(phrase)
    if not phrase:
        return ""
    return r"\b" + r"\s+".join(re.escape(p) for p in phrase.split()) + r"\b"


def _compile_phrase_prefix_regex(phrases=None, prefixes=None):
    phrases = phrases or []
    prefixes = prefixes or []
    parts = []
    for phrase in phrases:
        pat = _phrase_to_regex(phrase)
        if pat:
            parts.append(pat)
    for prefix in prefixes:
        parts.append(r"\b" + re.escape(prefix) + r"\w*\b")
    if not parts:
        return None
    return re.compile("(?:" + "|".join(parts) + ")", flags=re.IGNORECASE)


def _regex_hit(pattern, text):
    return bool(pattern and text and pattern.search(text))


def _direct_strong_integration_text(norm_text):
    """
    High-precision mixed-methods integration phrases.

    These can count even when the abstract does not separately trigger the
    explicit/implicit MMR flags, because the phrases are specific to mixed-methods
    integration/reporting practice.
    """
    if not norm_text:
        return False

    padded = f" {norm_text} "
    direct_phrases = [
        " joint display ",
        " joint displays ",
        " meta inference ",
        " meta inferences ",
        " metainference ",
        " metainferences ",
        " building from qualitative to quantitative ",
        " building from quantitative to qualitative ",
    ]
    return any(p in padded for p in direct_phrases)


def _strict_integration_text(norm_text, strict_pattern=None, integration_verb_pattern=None, window_size=14):
    """
    v6 high-precision mixed-methods integration detector.

    This detector is intentionally conservative. It returns True only when an
    integration verb appears in a local window with clear mixed-methods evidence,
    or when the abstract contains a highly specific mixed-methods integration
    phrase such as "joint display" or "meta-inference".

    The main goal is to avoid generic false positives such as:
      - integrating computing into the curriculum
      - integrating mathematics and statistics into biology
      - integrated laboratory experiences
      - connecting school and work
      - combined optics / combined data sets
    """
    if not norm_text:
        return False

    if _direct_strong_integration_text(norm_text):
        return True

    tokens = norm_text.split()
    if not tokens:
        return False

    # Strong verbs are relatively plausible MMR-integration indicators, but only
    # with local qualitative+quantitative or explicit MMR context.
    strong_verb_prefixes = (
        "integrat",   # integrate, integrated, integrating, integration
        "merge",      # merge, merged, merges
        "merg",       # merging
        "triangulat", # triangulate, triangulation
    )

    # Weak verbs are very noisy in NSF abstracts. They are accepted only under
    # the strictest local conditions.
    weak_verb_prefixes = (
        "combin",     # combine, combined, combining
        "connect",    # connect, connected, connecting
    )

    # Keep qualitative evidence terms specific. Avoid broad terms like "case" or
    # "observation" because they often occur in non-methodological contexts.
    qual_prefixes = (
        "qualitative",
        "interview",
        "focus",       # focus group(s)
        "thematic",
        "coding",
        "coded",
        "ethnograph",
        "phenomenolog",
        "narrative",
    )

    # Keep quantitative evidence terms specific. Avoid broad words such as
    # "statistics" because they often refer to a discipline/course rather than
    # a quantitative method.
    quant_prefixes = (
        "quantitative",
        "survey",
        "questionnaire",
        "statistical",
        "regression",
        "anova",
        "ancova",
        "randomized",
        "randomised",
        "experimental",
        "experiment",
        "bayesian",
    )

    method_context_terms = {
        "data", "dataset", "datasets",
        "finding", "findings",
        "result", "results",
        "analysis", "analyses",
        "evidence",
        "strand", "strands",
        "phase", "phases",
        "inference", "inferences",
        "theme", "themes",
        "response", "responses",
    }

    explicit_mmr_local_phrases = (
        "mixed method",
        "mixed methods",
        "mixed methodology",
        "mixed methodologies",
        "multi method",
        "multi methods",
        "multimethod",
        "multimethods",
        "mmr",
    )

    # Very common curricular/science-integration contexts that should suppress
    # weak evidence unless explicit MMR/qual+quant is present locally.
    generic_integration_context = {
        "curriculum", "curricula", "course", "courses", "classroom",
        "module", "modules", "laboratory", "laboratories", "lab", "labs",
        "program", "programs", "degree", "degrees", "discipline", "disciplines",
        "disciplinary", "interdisciplinary", "computing", "computation",
        "computational", "mathematics", "statistics", "biology", "physics",
        "chemistry", "engineering", "ethics", "sustainability", "concept",
        "concepts", "content", "instruction", "instructional", "teaching",
        "learning", "workforce", "school", "schools", "community", "communities",
    }

    for i, tok in enumerate(tokens):
        is_strong_verb = tok.startswith(strong_verb_prefixes)
        is_weak_verb = tok.startswith(weak_verb_prefixes)
        if not (is_strong_verb or is_weak_verb):
            continue

        left = max(0, i - window_size)
        right = min(len(tokens), i + window_size + 1)
        window = tokens[left:right]
        window_set = set(window)
        window_text = " ".join(window)
        window_padded = f" {window_text} "

        has_qual = any(t.startswith(qual_prefixes) for t in window)
        has_quant = any(t.startswith(quant_prefixes) for t in window)
        has_method_context = bool(window_set & method_context_terms)
        has_local_mmr = any(f" {phrase} " in window_padded for phrase in explicit_mmr_local_phrases)
        has_generic_context = bool(window_set & generic_integration_context)

        # Highest precision: integration verb near explicit qualitative and
        # quantitative evidence. Require a method/results/data context so generic
        # phrases like "integrating qualitative ideas with quantitative reasoning"
        # are less likely to pass.
        if has_qual and has_quant and has_method_context:
            return True

        # Explicit MMR near an integration verb is acceptable only when the same
        # window also talks about data/findings/results/analysis/strands/phases.
        if is_strong_verb and has_local_mmr and has_method_context:
            return True

        # Weak verbs such as combine/connect are not accepted unless they satisfy
        # the strongest condition above. This prevents "combined optics" and
        # "connect school to work" from being marked as MMR integration.
        if is_weak_verb:
            continue

        # For strong verbs, allow local qualitative+quantitative evidence without
        # method_context only if there is no obvious generic curricular/science
        # integration context. This is a narrow fallback for terse abstracts.
        if has_qual and has_quant and not has_generic_context:
            return True

    return False

def add_methodology_flags(df):
    """
    Add methodology flags using predictable per-row regex matching.

    These flags are candidate-retrieval signals, not final human labels.
    v3 uses stricter integration detection to avoid generic curriculum/institutional
    phrases such as 'integrating data science' unless they are near methodology terms.
    """
    # Shallow copy avoids duplicating large object columns such as co_pi_records.
    out = df.copy(deep=False)
    raw_text = (out["title_clean"].fillna("") + " " + out["abstract_clean"].fillna("")).str.strip()
    norm_text = raw_text.apply(normalize_search_text).tolist()

    explicit_re = _compile_phrase_prefix_regex([
        "mixed method", "mixed methods", "mixed methodology", "mixed methodologies",
        "mixed research", "mixed methods research", "multi method", "multi methods",
        "multimethod", "multimethods", "multimethodology", "multimethodologies", "mmr"
    ])

    qual_re = _compile_phrase_prefix_regex([
        "qualitative", "interview", "interviews", "focus group", "focus groups",
        "case study", "case studies", "observation", "observations", "classroom observation",
        "classroom observations", "field note", "field notes", "open ended",
        "thematic analysis", "content analysis", "discourse analysis", "coding",
        "grounded theory", "narrative analysis"
    ], ["ethnograph", "phenomenolog"])

    quant_re = _compile_phrase_prefix_regex([
        "quantitative", "survey", "surveys", "questionnaire", "questionnaires",
        "statistical", "statistics", "regression", "logistic regression", "linear model",
        "anova", "ancova", "t test", "chi square", "experiment", "experiments",
        "experimental", "randomized", "randomised", "quasi experimental", "pre post",
        "assessment", "scale", "factor analysis", "structural equation", "sem",
        "bayesian", "data mining", "learning analytics", "modeling", "modelling"
    ], ["experiment"])

    implicit_re = _compile_phrase_prefix_regex([
        "qualitative and quantitative", "quantitative and qualitative",
        "both qualitative and quantitative", "qualitative quantitative", "quantitative qualitative",
        "interviews and surveys", "survey and interview", "surveys and interviews",
        "focus groups and surveys", "surveys and focus groups",
        "case study and statistical", "case studies and statistical",
        "thematic analysis and regression"
    ])

    design_re = _compile_phrase_prefix_regex([
        "convergent parallel", "convergent design", "explanatory sequential",
        "sequential explanatory", "exploratory sequential", "sequential exploratory",
        "embedded design", "transformative design", "multiphase design", "triangulation design"
    ])

    strict_integration_re = _compile_phrase_prefix_regex([
        "meta inference", "metainference", "joint display",
        "building from qualitative to quantitative", "building from quantitative to qualitative",
        "integrate qualitative quantitative", "integrate quantitative qualitative",
        "integrating qualitative quantitative", "integrating quantitative qualitative",
        "merge qualitative quantitative", "merge quantitative qualitative",
        "merged qualitative quantitative", "merged quantitative qualitative",
        "triangulate qualitative quantitative", "triangulate quantitative qualitative",
        "triangulating qualitative quantitative", "triangulating quantitative qualitative",
        "combined qualitative quantitative", "combined quantitative qualitative",
        "connect qualitative quantitative", "connect quantitative qualitative"
    ])
    integration_verb_re = _compile_phrase_prefix_regex([], [
        "integrat", "merge", "merg", "triangulat", "combin", "connect"
    ])

    method_heavy_re = _compile_phrase_prefix_regex([
        "method", "methods", "methodology", "methodological", "data collection", "data analysis",
        "evaluation", "assessment", "measure", "measures", "sample", "participants",
        "instrument", "instruments", "protocol", "survey", "surveys", "interview",
        "interviews", "observation", "observations", "statistical", "regression",
        "case study", "case studies"
    ], ["methodolog", "evaluat", "assess", "measur"])

    explicit = []
    qual = []
    quant = []
    implicit_phrase = []
    design = []
    integration_term = []
    integration_strict = []
    integration_direct_strong = []
    method_heavy = []

    for txt in norm_text:
        explicit.append(int(_regex_hit(explicit_re, txt)))
        qual.append(int(_regex_hit(qual_re, txt)))
        quant.append(int(_regex_hit(quant_re, txt)))
        implicit_phrase.append(int(_regex_hit(implicit_re, txt)))
        design.append(int(_regex_hit(design_re, txt)))
        # Broad diagnostic flag: any integration-like verb or strong phrase.
        integration_term.append(int(_regex_hit(strict_integration_re, txt) or _regex_hit(integration_verb_re, txt)))
        # Strong direct phrases such as joint display/meta-inference.
        integration_direct_strong.append(int(_direct_strong_integration_text(txt)))
        # Strict retrieval/sampling flag: local methodological integration evidence.
        integration_strict.append(int(_strict_integration_text(txt, strict_integration_re, integration_verb_re)))
        method_heavy.append(int(_regex_hit(method_heavy_re, txt)))

    out["explicit_mmr_candidate"] = explicit
    out["qual_signal_candidate"] = qual
    out["quant_signal_candidate"] = quant
    out["implicit_mmr_phrase_candidate"] = implicit_phrase
    out["design_label_candidate"] = design
    out["integration_term_candidate"] = integration_term
    out["integration_direct_strong_candidate"] = integration_direct_strong
    out["integration_strict_local_candidate"] = integration_strict
    out["method_heavy_candidate"] = method_heavy

    out["implicit_mmr_candidate"] = (
        ((out["qual_signal_candidate"] == 1) & (out["quant_signal_candidate"] == 1))
        | (out["implicit_mmr_phrase_candidate"] == 1)
    ).astype(int)

    # Final high-precision integration flag.
    # Local integration evidence must also have document-level explicit/implicit
    # MMR context, except for very strong phrases such as joint display or
    # meta-inference. This prevents generic curriculum/institutional integration
    # from entering the enriched stratum.
    out["integration_candidate"] = (
        (out["integration_strict_local_candidate"] == 1)
        & (
            (out["explicit_mmr_candidate"] == 1)
            | (out["implicit_mmr_candidate"] == 1)
            | (out["integration_direct_strong_candidate"] == 1)
        )
    ).astype(int)

    out["design_integration_candidate"] = (
        (out["design_label_candidate"] == 1) | (out["integration_candidate"] == 1)
    ).astype(int)

    out["any_mmr_like_candidate"] = (
        (out["explicit_mmr_candidate"] == 1)
        | (out["implicit_mmr_candidate"] == 1)
        | (out["design_integration_candidate"] == 1)
    ).astype(int)

    out["qual_only_hard_negative_candidate"] = (
        (out["qual_signal_candidate"] == 1)
        & (out["quant_signal_candidate"] == 0)
        & (out["any_mmr_like_candidate"] == 0)
    ).astype(int)

    out["quant_only_hard_negative_candidate"] = (
        (out["quant_signal_candidate"] == 1)
        & (out["qual_signal_candidate"] == 0)
        & (out["any_mmr_like_candidate"] == 0)
    ).astype(int)

    out["method_heavy_background_candidate"] = (
        (out["method_heavy_candidate"] == 1)
        & (out["any_mmr_like_candidate"] == 0)
        & (out["qual_only_hard_negative_candidate"] == 0)
        & (out["quant_only_hard_negative_candidate"] == 0)
    ).astype(int)

    out["hard_negative_candidate"] = (
        (out["qual_only_hard_negative_candidate"] == 1)
        | (out["quant_only_hard_negative_candidate"] == 1)
        | (out["method_heavy_background_candidate"] == 1)
    ).astype(int)

    out["random_background_candidate"] = (
        (out["any_mmr_like_candidate"] == 0)
        & (out["hard_negative_candidate"] == 0)
    ).astype(int)

    conditions = [
        out["design_integration_candidate"] == 1,
        out["explicit_mmr_candidate"] == 1,
        (out["implicit_mmr_candidate"] == 1) & (out["explicit_mmr_candidate"] == 0),
        out["qual_only_hard_negative_candidate"] == 1,
        out["quant_only_hard_negative_candidate"] == 1,
        out["method_heavy_background_candidate"] == 1,
        out["random_background_candidate"] == 1,
    ]
    choices = [
        "design_integration_enriched",
        "explicit_mmr",
        "implicit_mmr",
        "qual_only_hard_negative",
        "quant_only_hard_negative",
        "method_heavy_background",
        "random_background",
    ]
    out["candidate_stratum"] = np.select(conditions, choices, default="other_background")

    return out


# -----------------------------
# Annotation sample
# -----------------------------

def matched_background_sample(positive_df, background_df, n, random_state=42):
    """Sample background rows roughly matched to positives by start_year and NSFOrganization."""
    if n <= 0 or len(background_df) == 0:
        return background_df.iloc[0:0].copy()

    key_cols = ["start_year", "NSFOrganization"]
    pos_counts = positive_df.groupby(key_cols, dropna=False).size().reset_index(name="pos_count")
    total_pos = pos_counts["pos_count"].sum()

    selected = []
    used_ids = set()

    for _, row in pos_counts.iterrows():
        if total_pos == 0:
            break

        target = int(round((row["pos_count"] / total_pos) * n))
        if target <= 0:
            continue

        pool = background_df[
            (background_df["start_year"] == row["start_year"])
            & (background_df["NSFOrganization"] == row["NSFOrganization"])
            & (~background_df["award_id"].isin(used_ids))
        ]

        take = sample_n(pool, target, random_state=random_state)
        selected.append(take)
        used_ids.update(take["award_id"].tolist())

    if selected:
        out = pd.concat(selected, ignore_index=True)
    else:
        out = background_df.iloc[0:0].copy()

    if len(out) < n:
        remaining = background_df[~background_df["award_id"].isin(set(out["award_id"].tolist()))]
        fill = sample_n(remaining, n - len(out), random_state=random_state + 1)
        out = pd.concat([out, fill], ignore_index=True)

    return out.drop_duplicates(subset=["award_id"]).head(n).copy()


def make_quota(sample_size, weights):
    raw = {k: int(round(sample_size * v)) for k, v in weights.items()}
    diff = sample_size - sum(raw.values())
    keys = list(weights.keys())
    i = 0
    while diff != 0:
        k = keys[i % len(keys)]
        if diff > 0:
            raw[k] += 1
            diff -= 1
        elif raw[k] > 0:
            raw[k] -= 1
            diff += 1
        i += 1
    return raw


def _cluster_rows(pool_all, cluster_col, cluster_ids):
    if not cluster_ids:
        return pool_all.iloc[0:0].copy()
    return pool_all[pool_all[cluster_col].isin(set(cluster_ids))].copy()


def _sample_clusters_by_rows(pool_all, candidate_pool, target_rows, selected_clusters,
                             cluster_col, random_state, current_total=0,
                             max_total_rows=None, allow_overrun=False):
    """
    Sample cluster ids from candidate_pool until approximately target_rows rows are
    obtained after expanding each selected cluster to all rows in pool_all.
    """
    if target_rows <= 0 or len(candidate_pool) == 0:
        return [], selected_clusters

    cluster_ids = candidate_pool[cluster_col].dropna().drop_duplicates().tolist()
    rng = np.random.default_rng(random_state)
    rng.shuffle(cluster_ids)

    chosen = []
    chosen_rows = 0

    for cid in cluster_ids:
        if cid in selected_clusters:
            continue
        n_cluster = int((pool_all[cluster_col] == cid).sum())
        if n_cluster <= 0:
            continue

        would_total = current_total + chosen_rows + n_cluster
        would_quota = chosen_rows + n_cluster

        if max_total_rows is not None and would_total > max_total_rows:
            # Preserve cluster integrity. If overrun is not allowed, skip clusters
            # that would exceed the requested sample size.
            if not allow_overrun:
                continue

        chosen.append(cid)
        chosen_rows += n_cluster
        if chosen_rows >= target_rows and not allow_overrun:
            break
        if chosen_rows >= target_rows and allow_overrun:
            break

    selected_clusters.update(chosen)
    return chosen, selected_clusters


def _representative_per_cluster(df, cluster_col):
    """Choose one deterministic representative row per cluster."""
    sort_cols = []
    for c in [cluster_col, "start_year", "award_id"]:
        if c in df.columns:
            sort_cols.append(c)
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")
    return df.drop_duplicates(subset=[cluster_col], keep="first").copy()


def build_annotation_sample(df, sample_size=2500, random_state=42,
                            duplicate_cluster_mode="cluster_expand",
                            cluster_col="project_text_id",
                            allow_sample_size_overrun=False):
    """
    Build a stratified annotation sample.

    Duplicate-aware v7 behavior:
      - Missing-abstract rows are excluded.
      - project_text_id clusters are based on normalized title + abstract.
      - duplicate_cluster_mode="cluster_expand" samples clusters and includes all
        award rows in selected clusters, preserving different award IDs with the
        same title/abstract together.
      - duplicate_cluster_mode="cluster_representative" samples clusters but keeps
        one representative award row per cluster for annotation.
      - duplicate_cluster_mode="award" reproduces award-level sampling by treating
        each award_id as its own cluster.
    """
    if duplicate_cluster_mode not in {"award", "cluster_expand", "cluster_representative"}:
        raise ValueError("duplicate_cluster_mode must be one of: award, cluster_expand, cluster_representative")

    df = ensure_project_text_clusters(df)
    pool_all = df[df["has_abstract"] == 1].copy()

    if duplicate_cluster_mode == "award":
        cluster_col = "award_id"
    elif cluster_col not in pool_all.columns:
        raise ValueError(f"cluster_col {cluster_col!r} not found in dataframe")

    # Ensure cluster audit columns reflect the requested cluster_col if using project_text_id.
    selected_clusters = set()
    out_parts = []
    current_total = 0

    # Force all design-label candidate clusters, because design labels are rare and central.
    forced_pool = pool_all[pool_all["design_label_candidate"] == 1]
    forced_ids = forced_pool[cluster_col].dropna().drop_duplicates().tolist()

    if forced_ids:
        # If forced clusters alone exceed sample_size, sample forced clusters while
        # preserving cluster integrity. This should be rare.
        forced_rows = _cluster_rows(pool_all, cluster_col, forced_ids)
        if len(forced_rows) <= sample_size or allow_sample_size_overrun:
            selected_clusters.update(forced_ids)
            out_parts.append(forced_rows)
            current_total += len(forced_rows)
        else:
            chosen, selected_clusters = _sample_clusters_by_rows(
                pool_all=pool_all,
                candidate_pool=forced_pool,
                target_rows=sample_size,
                selected_clusters=selected_clusters,
                cluster_col=cluster_col,
                random_state=random_state,
                current_total=0,
                max_total_rows=sample_size,
                allow_overrun=False,
            )
            forced_rows = _cluster_rows(pool_all, cluster_col, chosen)
            out_parts.append(forced_rows)
            current_total += len(forced_rows)

    remaining_target = max(sample_size - current_total, 0)

    weights = {
        "design_integration_enriched": 0.12,
        "explicit_mmr": 0.18,
        "implicit_mmr": 0.18,
        "qual_only_hard_negative": 0.14,
        "quant_only_hard_negative": 0.14,
        "method_heavy_background": 0.12,
        "random_background": 0.12,
    }
    quotas = make_quota(remaining_target, weights)

    # Sample non-background strata first.
    for idx, s in enumerate([
        "design_integration_enriched",
        "explicit_mmr",
        "implicit_mmr",
        "qual_only_hard_negative",
        "quant_only_hard_negative",
        "method_heavy_background",
    ]):
        candidate_pool = pool_all[
            (pool_all["candidate_stratum"] == s)
            & (~pool_all[cluster_col].isin(selected_clusters))
        ]
        chosen, selected_clusters = _sample_clusters_by_rows(
            pool_all=pool_all,
            candidate_pool=candidate_pool,
            target_rows=quotas[s],
            selected_clusters=selected_clusters,
            cluster_col=cluster_col,
            random_state=random_state + idx + 10,
            current_total=current_total,
            max_total_rows=sample_size,
            allow_overrun=allow_sample_size_overrun,
        )
        take = _cluster_rows(pool_all, cluster_col, chosen)
        out_parts.append(take)
        current_total += len(take)

    # Sample random background clusters, roughly matched only in the award-level
    # legacy mode. For cluster-aware mode, preserve cluster integrity first.
    background_pool = pool_all[
        (pool_all["candidate_stratum"] == "random_background")
        & (~pool_all[cluster_col].isin(selected_clusters))
    ]
    chosen, selected_clusters = _sample_clusters_by_rows(
        pool_all=pool_all,
        candidate_pool=background_pool,
        target_rows=quotas["random_background"],
        selected_clusters=selected_clusters,
        cluster_col=cluster_col,
        random_state=random_state + 90,
        current_total=current_total,
        max_total_rows=sample_size,
        allow_overrun=allow_sample_size_overrun,
    )
    bg_take = _cluster_rows(pool_all, cluster_col, chosen)
    out_parts.append(bg_take)
    current_total += len(bg_take)

    sample = pd.concat(out_parts, ignore_index=True).drop_duplicates(subset=["award_id"])

    # Fill any remaining rows with unused clusters from the full pool.
    while len(sample) < sample_size:
        remaining = pool_all[~pool_all[cluster_col].isin(set(sample[cluster_col].tolist()))]
        if len(remaining) == 0:
            break
        chosen, selected_clusters = _sample_clusters_by_rows(
            pool_all=pool_all,
            candidate_pool=remaining,
            target_rows=sample_size - len(sample),
            selected_clusters=set(sample[cluster_col].tolist()),
            cluster_col=cluster_col,
            random_state=random_state + 99 + len(sample),
            current_total=len(sample),
            max_total_rows=sample_size,
            allow_overrun=allow_sample_size_overrun,
        )
        if not chosen:
            break
        fill = _cluster_rows(pool_all, cluster_col, chosen)
        sample = pd.concat([sample, fill], ignore_index=True).drop_duplicates(subset=["award_id"])

    # If the user requests a cluster representative benchmark, keep one row per
    # title+abstract cluster. Labels can then be copied to all cluster members later.
    if duplicate_cluster_mode == "cluster_representative":
        sample = _representative_per_cluster(sample, cluster_col)

    # If overrun is disabled, preserve cluster integrity and avoid trimming by row.
    # If exact sample_size cannot be reached because the next cluster would exceed
    # the target, the sample may be slightly smaller than sample_size. This is safer
    # than splitting duplicate title/abstract clusters.
    if len(sample) > sample_size and not allow_sample_size_overrun:
        # This can happen only in award mode or forced overlarge cases. In cluster
        # modes, use representative clusters to trim without splitting clusters.
        if duplicate_cluster_mode == "award":
            sample = sample_n(sample, sample_size, random_state=random_state + 123)
        else:
            reps = sample[[cluster_col]].drop_duplicates().sample(frac=1.0, random_state=random_state + 123)
            keep_clusters = []
            n_rows = 0
            for cid in reps[cluster_col].tolist():
                c_rows = sample[sample[cluster_col] == cid]
                if n_rows + len(c_rows) > sample_size:
                    continue
                keep_clusters.append(cid)
                n_rows += len(c_rows)
            sample = sample[sample[cluster_col].isin(keep_clusters)].copy()

    sample = ensure_project_text_clusters(sample)
    sample["annotation_duplicate_mode"] = duplicate_cluster_mode
    sample["annotation_cluster_col"] = cluster_col
    sample = sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sample.insert(0, "annotation_id", [f"ann_{i + 1:05d}" for i in range(len(sample))])

    label_cols = {
        "label_mmr_class": "",
        "label_qual_signal": "",
        "label_quant_signal": "",
        "label_design_present": "",
        "label_design_type": "",
        "label_integration_present": "",
        "label_integration_type": "",
        "label_reporting_completeness": "",
        "annotator_id": "",
        "annotation_notes": "",
    }
    for col, default in label_cols.items():
        sample[col] = default

    keep_cols = [
        "annotation_id",
        "award_id",
        "project_text_id",
        "project_text_cluster_size",
        "duplicate_project_text_flag",
        "annotation_duplicate_mode",
        "annotation_cluster_col",
        "title_clean",
        "abstract_clean",
        "abstract_word_count",
        "title_word_count",
        "start_year",
        "NSFDirectorate",
        "NSFOrganization",
        "legacy_nsf_org_flag",
        "nsf_org_scope",
        "Program(s)",
        "ProgramElementCode(s)",
        "AwardInstrument",
        "award_amount",
        "pi_clean",
        "person_id",
        "organization_clean",
        "State",
        "candidate_stratum",
        "explicit_mmr_candidate",
        "implicit_mmr_candidate",
        "qual_signal_candidate",
        "quant_signal_candidate",
        "design_label_candidate",
        "integration_term_candidate",
        "integration_direct_strong_candidate",
        "integration_strict_local_candidate",
        "integration_candidate",
        "design_integration_candidate",
        "qual_only_hard_negative_candidate",
        "quant_only_hard_negative_candidate",
        "method_heavy_background_candidate",
        "hard_negative_candidate",
        "random_background_candidate",
        "label_mmr_class",
        "label_qual_signal",
        "label_quant_signal",
        "label_design_present",
        "label_design_type",
        "label_integration_present",
        "label_integration_type",
        "label_reporting_completeness",
        "annotator_id",
        "annotation_notes",
    ]
    keep_cols = [c for c in keep_cols if c in sample.columns]
    return sample[keep_cols].copy()


# -----------------------------
# Graph edge construction
# -----------------------------

def build_award_pi_edges(df):
    rows = []

    for _, r in df.iterrows():
        award_id = r["award_id"]
        year = r["start_year"]

        lead_name = clean_person_name(r.get("pi_clean", ""))
        lead_email = extract_first_email(r.get("pi_email_extracted", ""), r.get("PIEmailAddress", ""), r.get("PrincipalInvestigator", ""))
        lead_person_id = make_person_id(lead_name, lead_email)

        if lead_name or lead_email:
            rows.append({
                "award_id": award_id,
                "person_id": lead_person_id,
                # pi_id is retained as an alias for compatibility with existing graph code.
                "pi_id": lead_person_id,
                "pi_name": lead_name,
                "pi_email": lead_email,
                "role": "PI",
                "start_year": year,
                "institution_id": r.get("institution_id", ""),
                "organization_clean": r.get("organization_clean", ""),
                "NSFDirectorate": r.get("NSFDirectorate", ""),
                "NSFOrganization": r.get("NSFOrganization", ""),
            })

        co_records = r.get("co_pi_records", [])
        if isinstance(co_records, str):
            # CSV reload may turn the list of dicts into a string.
            try:
                parsed = ast.literal_eval(co_records)
                if isinstance(parsed, list):
                    co_records = parsed
                else:
                    co_records = split_people_records(r.get("Co-PIName(s)", co_records))
            except Exception:
                co_records = split_people_records(r.get("Co-PIName(s)", co_records))

        if not isinstance(co_records, list):
            co_records = split_people_records(r.get("Co-PIName(s)", ""))

        for rec in co_records:
            if isinstance(rec, dict):
                co_name = clean_person_name(rec.get("name", ""))
                co_email = extract_first_email(rec.get("email", ""))
            else:
                co_name = clean_person_name(rec)
                co_email = extract_first_email(rec)

            if not co_name and not co_email:
                continue

            co_person_id = make_person_id(co_name, co_email)
            rows.append({
                "award_id": award_id,
                "person_id": co_person_id,
                "pi_id": co_person_id,
                "pi_name": co_name,
                "pi_email": co_email,
                "role": "Co-PI",
                "start_year": year,
                "institution_id": r.get("institution_id", ""),
                "organization_clean": r.get("organization_clean", ""),
                "NSFDirectorate": r.get("NSFDirectorate", ""),
                "NSFOrganization": r.get("NSFOrganization", ""),
            })

    edges = pd.DataFrame(rows)
    if len(edges) == 0:
        return pd.DataFrame(columns=[
            "award_id", "person_id", "pi_id", "pi_name", "pi_email", "role", "start_year",
            "institution_id", "organization_clean", "NSFDirectorate", "NSFOrganization"
        ])

    edges = edges.drop_duplicates(subset=["award_id", "person_id", "role"])
    return edges


def build_pi_collaboration_edges(award_pi_edges):
    rows = []

    for award_id, group in award_pi_edges.groupby("award_id"):
        pis = group[["pi_id", "pi_name", "start_year"]].drop_duplicates(subset=["pi_id"])
        if len(pis) < 2:
            continue

        year = pis["start_year"].dropna().iloc[0] if pis["start_year"].notna().any() else np.nan
        pi_records = pis[["pi_id", "pi_name"]].to_dict("records")

        for a, b in itertools.combinations(pi_records, 2):
            pi1, pi2 = sorted([a, b], key=lambda x: x["pi_id"])
            rows.append({
                "pi_id_1": pi1["pi_id"],
                "pi_name_1": pi1["pi_name"],
                "pi_id_2": pi2["pi_id"],
                "pi_name_2": pi2["pi_name"],
                "award_id": award_id,
                "start_year": year,
                "edge_weight": 1,
            })

    event_edges = pd.DataFrame(rows)
    if len(event_edges) == 0:
        return pd.DataFrame(columns=[
            "pi_id_1", "pi_name_1", "pi_id_2", "pi_name_2", "award_count",
            "first_year", "last_year", "award_years", "award_ids"
        ])

    agg = event_edges.groupby(
        ["pi_id_1", "pi_name_1", "pi_id_2", "pi_name_2"],
        dropna=False,
    ).agg(
        award_count=("award_id", "nunique"),
        first_year=("start_year", "min"),
        last_year=("start_year", "max"),
        award_years=("start_year", lambda x: "|".join(str(int(y)) for y in sorted(set(x.dropna())))),
        award_ids=("award_id", lambda x: "|".join(sorted(set(x.astype(str))))),
    ).reset_index()

    return agg


def build_award_institution_edges(df):
    cols = [
        "award_id",
        "institution_id",
        "organization_clean",
        "State",
        "OrganizationState",
        "OrganizationCity",
        "start_year",
        "NSFDirectorate",
        "NSFOrganization",
        "legacy_nsf_org_flag",
        "nsf_org_scope",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].drop_duplicates().copy()


def build_award_program_edges(df):
    """
    Build award-program edges using ProgramElementCode(s) as the primary source.
    Program names are kept as raw descriptive metadata but not used for splitting.
    """
    rows = []

    for _, r in df.iterrows():
        award_id = r["award_id"]
        year = r["start_year"]
        raw_program_names = normalize_whitespace(r.get("Program(s)", ""))
        element_codes = split_program_codes(r.get("ProgramElementCode(s)", ""))
        reference_codes = split_program_codes(r.get("ProgramReferenceCode(s)", ""))

        if element_codes:
            for code in element_codes:
                rows.append({
                    "award_id": award_id,
                    "program_node_source": "ProgramElementCode",
                    "program_id": make_id("program", f"element:{code}"),
                    "program_code": code,
                    "program_name_raw": raw_program_names,
                    "start_year": year,
                    "NSFDirectorate": r.get("NSFDirectorate", ""),
                    "NSFOrganization": r.get("NSFOrganization", ""),
                    "legacy_nsf_org_flag": r.get("legacy_nsf_org_flag", ""),
                    "nsf_org_scope": r.get("nsf_org_scope", ""),
                    "ProgramReferenceCode(s)": r.get("ProgramReferenceCode(s)", ""),
                })
        else:
            # Fallback to conservative names if element codes are missing.
            names = split_program_names_conservative(raw_program_names)
            if not names:
                names = ["unknown_program"]
            for name in names:
                rows.append({
                    "award_id": award_id,
                    "program_node_source": "ProgramNameFallback",
                    "program_id": make_id("program", f"name:{name}"),
                    "program_code": "",
                    "program_name_raw": name,
                    "start_year": year,
                    "NSFDirectorate": r.get("NSFDirectorate", ""),
                    "NSFOrganization": r.get("NSFOrganization", ""),
                    "legacy_nsf_org_flag": r.get("legacy_nsf_org_flag", ""),
                    "nsf_org_scope": r.get("nsf_org_scope", ""),
                    "ProgramReferenceCode(s)": r.get("ProgramReferenceCode(s)", ""),
                })

        # Optional reference-code edges could be modeled separately. Here we keep
        # reference codes as metadata to avoid mixing program and reference nodes.
        _ = reference_codes

    edges = pd.DataFrame(rows)
    return edges.drop_duplicates()



def _numeric_sum(series):
    """Sum a flag/count series robustly when CSV reload has string values like '0', '1', '0.0'."""
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def _coerce_cleaned_input_types(df):
    """
    Coerce key columns when --input_is_cleaned is used.

    The cleaned CSV is read with dtype=str to preserve identifiers, so numeric flag
    columns such as legacy_nsf_org_flag and has_abstract must be converted back to
    numeric values before filtering, sampling, and quality reporting.
    """
    out = df.copy()

    numeric_cols = [
        "start_year", "end_year", "award_amount", "arra_amount",
        "abstract_word_count", "title_word_count", "has_abstract",
        "duration_days", "duration_months", "invalid_duration_flag",
        "legacy_nsf_org_flag", "num_co_pis", "team_size",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    date_cols = ["start_date", "end_date", "last_amendment_date"]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # Recompute these if missing or malformed.
    if "abstract_clean" in out.columns:
        if "abstract_word_count" not in out.columns or out["abstract_word_count"].isna().all():
            out["abstract_word_count"] = out["abstract_clean"].apply(safe_word_count)
        if "has_abstract" not in out.columns or out["has_abstract"].isna().all():
            out["has_abstract"] = (out["abstract_word_count"].fillna(0) > 0).astype(int)
        else:
            out["has_abstract"] = out["has_abstract"].fillna((out["abstract_word_count"].fillna(0) > 0).astype(int)).astype(int)

    if "legacy_nsf_org_flag" not in out.columns and "NSFOrganization" in out.columns:
        out["legacy_nsf_org_flag"] = (~out["NSFOrganization"].isin(TARGET_NSF_ORGS)).astype(int)
    elif "legacy_nsf_org_flag" in out.columns:
        out["legacy_nsf_org_flag"] = out["legacy_nsf_org_flag"].fillna(0).astype(int)

    if "invalid_duration_flag" in out.columns:
        out["invalid_duration_flag"] = out["invalid_duration_flag"].fillna(0).astype(int)

    return out

# -----------------------------
# Data quality report
# -----------------------------

def build_data_quality_report(raw_df, cleaned_df, flagged_df, annotation_df,
                              award_pi_edges, pi_collab_edges,
                              award_inst_edges, award_program_edges,
                              drop_legacy_orgs=False):
    rows = []

    def add(section, metric, value):
        rows.append({"section": section, "metric": metric, "value": value})

    add("configuration", "drop_legacy_orgs", int(bool(drop_legacy_orgs)))
    add("raw_input", "raw_rows", len(raw_df))
    add("raw_input", "raw_columns", raw_df.shape[1])
    if "AwardNumber" in raw_df.columns:
        raw_award_series = raw_df["AwardNumber"].apply(clean_excel_wrapped_value)
    elif "award_id" in raw_df.columns:
        raw_award_series = raw_df["award_id"].apply(clean_excel_wrapped_value)
    else:
        raw_award_series = pd.Series([], dtype=str)
    add("raw_input", "raw_duplicate_award_numbers", int(raw_award_series.duplicated().sum()) if len(raw_award_series) else "")

    add("cleaned_awards", "cleaned_rows", len(cleaned_df))
    add("cleaned_awards", "unique_award_ids", cleaned_df["award_id"].nunique())
    if "project_text_id" in cleaned_df.columns:
        cleaned_cluster_counts = cleaned_df.groupby("project_text_id")["award_id"].nunique()
        add("cleaned_awards", "unique_project_text_clusters", int(cleaned_cluster_counts.shape[0]))
        add("cleaned_awards", "duplicate_project_text_clusters", int((cleaned_cluster_counts > 1).sum()))
        add("cleaned_awards", "rows_in_duplicate_project_text_clusters", int(cleaned_df[cleaned_df["project_text_id"].isin(cleaned_cluster_counts[cleaned_cluster_counts > 1].index)].shape[0]))
        add("cleaned_awards", "max_project_text_cluster_size", int(cleaned_cluster_counts.max()) if len(cleaned_cluster_counts) else 0)
    add("cleaned_awards", "missing_award_id", int((cleaned_df["award_id"] == "").sum()))
    add("cleaned_awards", "missing_title", int((cleaned_df["title_clean"] == "").sum()))
    add("cleaned_awards", "missing_abstract", int((cleaned_df["abstract_clean"] == "").sum()))
    add("cleaned_awards", "missing_start_date", int(cleaned_df["start_date"].isna().sum()))
    add("cleaned_awards", "missing_end_date", int(cleaned_df["end_date"].isna().sum()))
    add("cleaned_awards", "missing_award_amount", int(cleaned_df["award_amount"].isna().sum()))
    add("cleaned_awards", "missing_pi", int((cleaned_df["pi_clean"] == "").sum()))
    add("cleaned_awards", "missing_organization", int((cleaned_df["organization_clean"] == "").sum()))
    add("cleaned_awards", "legacy_nsf_org_rows", _numeric_sum(cleaned_df["legacy_nsf_org_flag"]))
    add("cleaned_awards", "invalid_duration_rows", _numeric_sum(cleaned_df["invalid_duration_flag"]) if "invalid_duration_flag" in cleaned_df.columns else 0)
    add("cleaned_awards", "lead_pi_email_extracted_rows", int((cleaned_df.get("pi_email_extracted", pd.Series(dtype=str)).fillna("") != "").sum()) if "pi_email_extracted" in cleaned_df.columns else 0)

    if cleaned_df["start_year"].notna().any():
        add("cleaned_awards", "start_year_min", int(cleaned_df["start_year"].min()))
        add("cleaned_awards", "start_year_max", int(cleaned_df["start_year"].max()))

    for status, count in cleaned_df["award_status_derived"].value_counts(dropna=False).items():
        add("award_status", str(status), int(count))

    for col in ["abstract_word_count", "title_word_count", "award_amount", "duration_months", "team_size", "num_co_pis"]:
        if col in cleaned_df.columns:
            series = pd.to_numeric(cleaned_df[col], errors="coerce")
            add(f"{col}_summary", "mean", round(float(series.mean()), 4) if series.notna().any() else "")
            add(f"{col}_summary", "median", round(float(series.median()), 4) if series.notna().any() else "")
            add(f"{col}_summary", "min", round(float(series.min()), 4) if series.notna().any() else "")
            add(f"{col}_summary", "max", round(float(series.max()), 4) if series.notna().any() else "")

    for year, count in cleaned_df["start_year"].value_counts(dropna=False).sort_index().items():
        add("year_distribution", str(year), int(count))

    for directorate, count in cleaned_df["NSFDirectorate"].value_counts(dropna=False).items():
        add("directorate_distribution", str(directorate), int(count))

    for org, count in cleaned_df["NSFOrganization"].value_counts(dropna=False).items():
        add("nsf_organization_distribution", str(org), int(count))

    for scope, count in cleaned_df["nsf_org_scope"].value_counts(dropna=False).items():
        add("nsf_org_scope_distribution", str(scope), int(count))

    flag_cols = [
        "explicit_mmr_candidate",
        "implicit_mmr_candidate",
        "qual_signal_candidate",
        "quant_signal_candidate",
        "design_label_candidate",
        "integration_term_candidate",
        "integration_direct_strong_candidate",
        "integration_strict_local_candidate",
        "integration_candidate",
        "design_integration_candidate",
        "qual_only_hard_negative_candidate",
        "quant_only_hard_negative_candidate",
        "method_heavy_background_candidate",
        "hard_negative_candidate",
        "random_background_candidate",
        "any_mmr_like_candidate",
    ]
    for col in flag_cols:
        if col in flagged_df.columns:
            add("methodology_flags", col, _numeric_sum(flagged_df[col]))

    if {"integration_candidate", "explicit_mmr_candidate", "implicit_mmr_candidate"}.issubset(flagged_df.columns):
        integration_without_mmr = (
            (flagged_df["integration_candidate"] == 1)
            & (flagged_df["explicit_mmr_candidate"] == 0)
            & (flagged_df["implicit_mmr_candidate"] == 0)
        )
        add("methodology_flags", "integration_candidate_without_explicit_or_implicit_mmr", int(integration_without_mmr.sum()))

    for stratum, count in flagged_df["candidate_stratum"].value_counts(dropna=False).items():
        add("candidate_strata", str(stratum), int(count))

    add("annotation_sample", "annotation_rows", len(annotation_df))
    if "project_text_id" in annotation_df.columns:
        ann_cluster_counts = annotation_df.groupby("project_text_id")["award_id"].nunique()
        add("annotation_sample", "unique_project_text_clusters", int(ann_cluster_counts.shape[0]))
        add("annotation_sample", "duplicate_project_text_clusters", int((ann_cluster_counts > 1).sum()))
        add("annotation_sample", "rows_in_duplicate_project_text_clusters", int(annotation_df[annotation_df["project_text_id"].isin(ann_cluster_counts[ann_cluster_counts > 1].index)].shape[0]))
        add("annotation_sample", "max_project_text_cluster_size", int(ann_cluster_counts.max()) if len(ann_cluster_counts) else 0)
    if "annotation_duplicate_mode" in annotation_df.columns:
        for mode, count in annotation_df["annotation_duplicate_mode"].value_counts(dropna=False).items():
            add("annotation_sample_duplicate_mode", str(mode), int(count))
    add("annotation_sample", "missing_abstract_rows", int((annotation_df["abstract_clean"].fillna("") == "").sum()))
    if "design_label_candidate" in annotation_df.columns:
        add("annotation_sample", "design_label_candidate_rows", _numeric_sum(annotation_df["design_label_candidate"]))
    if "integration_candidate" in annotation_df.columns:
        add("annotation_sample", "integration_candidate_rows", _numeric_sum(annotation_df["integration_candidate"]))
    for stratum, count in annotation_df["candidate_stratum"].value_counts(dropna=False).items():
        add("annotation_sample_strata", str(stratum), int(count))

    add("graph_outputs", "award_pi_edges", len(award_pi_edges))
    add("graph_outputs", "unique_pis", award_pi_edges["pi_id"].nunique() if len(award_pi_edges) else 0)
    add("graph_outputs", "unique_person_ids", award_pi_edges["person_id"].nunique() if len(award_pi_edges) and "person_id" in award_pi_edges.columns else 0)
    add("graph_outputs", "pi_edge_rows_with_email", int((award_pi_edges["pi_email"].fillna("") != "").sum()) if len(award_pi_edges) and "pi_email" in award_pi_edges.columns else 0)
    add("graph_outputs", "pi_name_rows_with_embedded_email", int(award_pi_edges["pi_name"].fillna("").str.contains(EMAIL_RE).sum()) if len(award_pi_edges) and "pi_name" in award_pi_edges.columns else 0)
    add("graph_outputs", "pi_collaboration_edges", len(pi_collab_edges))
    add("graph_outputs", "award_institution_edges", len(award_inst_edges))
    add("graph_outputs", "unique_institutions", award_inst_edges["institution_id"].nunique() if len(award_inst_edges) else 0)
    add("graph_outputs", "award_program_edges", len(award_program_edges))
    add("graph_outputs", "unique_programs", award_program_edges["program_id"].nunique() if len(award_program_edges) else 0)

    if len(award_program_edges):
        for source, count in award_program_edges["program_node_source"].value_counts(dropna=False).items():
            add("program_edge_source", str(source), int(count))

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

PROCESSED_OUTPUT_FILES = [
    "cleaned_nsf_awards_2000_2025.csv",
    "nsf_awards_with_methodology_flags.csv",
    "annotation_sample_2000_2025.csv",
    "project_text_cluster_report.csv",
    "annotation_project_text_cluster_report.csv",
    "data_quality_report.csv",
]

EDGE_OUTPUT_FILES = [
    "award_pi_edges.csv",
    "pi_collaboration_edges.csv",
    "award_institution_edges.csv",
    "award_program_edges.csv",
]


def resolve_path(path_value, repo_root: Path) -> Path:
    """Resolve CLI paths relative to the repo root unless absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def check_overwrite(paths, overwrite: bool):
    """Fail fast if any output already exists and --overwrite was not passed."""
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        shown = "\n".join(f"  - {p}" for p in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ... and {len(existing) - 20} more"
        raise FileExistsError(
            "Refusing to overwrite existing MethodKG pipeline outputs. "
            "Pass --overwrite if you intentionally want to replace them.\n"
            f"Existing outputs:\n{shown}{extra}"
        )


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build MethodKG processed award files and canonical graph edge files. "
            "Processed award/flag/sample/report files are written to --processed_dir, "
            "while graph edge CSVs are written to --edges_dir."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/Full_DSAA_Awards.csv",
        help="Input CSV file. Default: data/raw/Full_DSAA_Awards.csv",
    )
    parser.add_argument(
        "--repo_root",
        default=None,
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--processed_dir",
        default=None,
        help=(
            "Directory for processed award outputs. Default: "
            "data/processed/methodkg_outputs_v7_clustered_from_cleaned"
        ),
    )
    parser.add_argument(
        "--edges_dir",
        default=None,
        help="Directory for edge CSV outputs. Default: data/edges",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Deprecated alias for --processed_dir. Kept only for backward "
            "compatibility; edge files still go to --edges_dir."
        ),
    )
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--sample_size", type=int, default=2500)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--drop_legacy_orgs", action="store_true", help="Drop NSFOrganization values outside DUE, DRL, EES, DGE, and EEC")
    parser.add_argument("--input_is_cleaned", action="store_true", help="Treat --input as a previously cleaned MethodKG awards CSV and skip raw cleaning")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed and edge outputs")
    parser.add_argument(
        "--duplicate_cluster_mode",
        choices=["award", "cluster_expand", "cluster_representative"],
        default="cluster_expand",
        help="How annotation sampling handles duplicate title+abstract clusters. cluster_expand keeps all award IDs in selected duplicate clusters.",
    )
    parser.add_argument("--cluster_col", default="project_text_id", help="Cluster column for duplicate-aware sampling")
    parser.add_argument("--allow_sample_size_overrun", action="store_true", help="Allow sample to exceed --sample_size to avoid splitting a duplicate cluster")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    input_path = resolve_path(args.input, repo_root)

    # Default layout used by the cleaned MethodKG repository.
    processed_dir_value = args.processed_dir or args.outdir or "data/processed"
    edges_dir_value = args.edges_dir or "data/edges"
    processed_dir = resolve_path(processed_dir_value, repo_root)
    edges_dir = resolve_path(edges_dir_value, repo_root)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Pass --input explicitly if your raw/cleaned award file has a different name."
        )

    processed_paths = [processed_dir / name for name in PROCESSED_OUTPUT_FILES]
    edge_paths = [edges_dir / name for name in EDGE_OUTPUT_FILES]
    check_overwrite(processed_paths + edge_paths, overwrite=args.overwrite)

    processed_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    print("Resolved paths:")
    print(f"  repo_root: {repo_root}")
    print(f"  input: {input_path}")
    print(f"  processed_dir: {processed_dir}")
    print(f"  edges_dir: {edges_dir}")

    raw_df = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")

    if args.input_is_cleaned:
        cleaned_df = raw_df.copy()
        cleaned_df = _coerce_cleaned_input_types(cleaned_df)
        cleaned_df = ensure_project_text_clusters(cleaned_df)
        if "start_year" in cleaned_df.columns:
            cleaned_df = cleaned_df[(cleaned_df["start_year"] >= args.start_year) & (cleaned_df["start_year"] <= args.end_year)].copy()
        if args.drop_legacy_orgs and "legacy_nsf_org_flag" in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df["legacy_nsf_org_flag"].fillna(0).astype(int) == 0].copy()
    else:
        cleaned_df = clean_awards(
            raw_df,
            start_year=args.start_year,
            end_year=args.end_year,
            drop_legacy_orgs=args.drop_legacy_orgs,
        )

    flagged_df = add_methodology_flags(cleaned_df)
    flagged_df = ensure_project_text_clusters(flagged_df)

    annotation_df = build_annotation_sample(
        flagged_df,
        sample_size=args.sample_size,
        random_state=args.random_state,
        duplicate_cluster_mode=args.duplicate_cluster_mode,
        cluster_col=args.cluster_col,
        allow_sample_size_overrun=args.allow_sample_size_overrun,
    )

    award_pi_edges = build_award_pi_edges(flagged_df)
    pi_collab_edges = build_pi_collaboration_edges(award_pi_edges)
    award_inst_edges = build_award_institution_edges(flagged_df)
    award_program_edges = build_award_program_edges(flagged_df)

    project_text_cluster_report = build_project_text_cluster_report(flagged_df)
    annotation_project_text_cluster_report = build_project_text_cluster_report(annotation_df)

    report_df = build_data_quality_report(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        flagged_df=flagged_df,
        annotation_df=annotation_df,
        award_pi_edges=award_pi_edges,
        pi_collab_edges=pi_collab_edges,
        award_inst_edges=award_inst_edges,
        award_program_edges=award_program_edges,
        drop_legacy_orgs=args.drop_legacy_orgs,
    )

    processed_outputs = {
        "cleaned_nsf_awards_2000_2025.csv": cleaned_df,
        "nsf_awards_with_methodology_flags.csv": flagged_df,
        "annotation_sample_2000_2025.csv": annotation_df,
        "project_text_cluster_report.csv": project_text_cluster_report,
        "annotation_project_text_cluster_report.csv": annotation_project_text_cluster_report,
        "data_quality_report.csv": report_df,
    }
    edge_outputs = {
        "award_pi_edges.csv": award_pi_edges,
        "pi_collaboration_edges.csv": pi_collab_edges,
        "award_institution_edges.csv": award_inst_edges,
        "award_program_edges.csv": award_program_edges,
    }

    for name, frame in processed_outputs.items():
        write_csv(frame, processed_dir / name)
    for name, frame in edge_outputs.items():
        write_csv(frame, edges_dir / name)

    print("\nDone.")
    print("Processed files written to:", processed_dir.resolve())
    for name in PROCESSED_OUTPUT_FILES:
        path = processed_dir / name
        print(f"- {name}: {path.stat().st_size:,} bytes")

    print("\nEdge files written to:", edges_dir.resolve())
    for name in EDGE_OUTPUT_FILES:
        path = edges_dir / name
        print(f"- {name}: {path.stat().st_size:,} bytes")

    print("\nCanonical data layout:")
    print(f"  processed outputs: {processed_dir}")
    print(f"  graph edges:       {edges_dir}")


if __name__ == "__main__":
    main()
