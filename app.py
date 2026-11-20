"""TruthLayer AI — Flagship Executive SaaS Fact-Checking Web Application (Streamlit)."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cache import AuditReportCache
from src.config import AppSettings, get_detected_secret_names, load_settings
from src.demo_data import build_demo_trap_report
from src.models import (
    FactCheckReport,
    PageContent,
    ParsedDocument,
    TrapType,
    Verdict,
    VerificationResult,
)
from src.pdf_parser import parse_pdf
from src.report_exporter import export_csv, export_html, export_json, export_markdown
from src.search_client import extract_domain
from src.verifier import FactCheckEngine

SAMPLE_PDF_PATH = Path(__file__).parent / "sample_docs" / "sample_trap_document.pdf"

st.set_page_config(
    page_title="TruthLayer AI · Automated Fact-Check Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _get_shared_audit_cache() -> AuditReportCache:
    """Shared SHA-256 report cache across sessions to save API credits on repeat PDFs."""
    return AuditReportCache(max_entries=128)


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container {
        padding-top: 3.75rem !important;
        padding-bottom: 3rem;
        max-width: 1260px;
    }

    /* Top Executive Status Header */
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        padding: 0.8rem 1.25rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        margin-bottom: 1.15rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .brand-logo {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: -0.02em;
    }
    .brand-title {
        font-weight: 800;
        font-size: 1.06rem;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 0.76rem;
        color: #64748B;
    }
    .status-pills {
        display: flex;
        gap: 0.45rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        background: #F8FAFC;
        color: #334155;
        font-family: 'JetBrains Mono', monospace;
    }
    .pill-live {
        background: #ECFDF5;
        border-color: #A7F3D0;
        color: #065F46;
    }
    .pill-telemetry {
        background: #EFF6FF;
        border-color: #BFDBFE;
        color: #1E40AF;
    }
    .pill-cache {
        background: #F5F3FF;
        border-color: #DDD6FE;
        color: #5B21B6;
    }
    .dot-live {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        display: inline-block;
    }

    /* Hero Banner */
    .hero-card {
        background: radial-gradient(circle at 88% 18%, rgba(59, 130, 246, 0.28), transparent 45%),
                    radial-gradient(circle at 12% 90%, rgba(16, 185, 129, 0.16), transparent 42%),
                    linear-gradient(135deg, #090E1A 0%, #0F1D3A 58%, #162B54 100%);
        color: #FFFFFF;
        padding: 1.85rem 2.15rem;
        border-radius: 18px;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 10px 28px rgba(9, 14, 26, 0.14);
    }
    .hero-kicker {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #93C5FD;
        background: rgba(37, 99, 235, 0.24);
        border: 1px solid rgba(147, 197, 253, 0.28);
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.7rem;
    }
    .hero-card h1 {
        margin: 0 0 0.45rem 0;
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #FFFFFF;
        line-height: 1.18;
    }
    .hero-card p {
        margin: 0 0 1.2rem 0;
        color: #CBD5E1;
        font-size: 0.98rem;
        max-width: 820px;
        line-height: 1.55;
    }
    .hero-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
    }
    .hero-step {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 12px;
        padding: 0.78rem 0.95rem;
    }
    .hero-step-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .hero-step-desc {
        font-size: 0.77rem;
        color: #94A3B8;
        line-height: 1.42;
    }

    /* Risk Alert Banner */
    .risk-banner {
        padding: 1rem 1.3rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .risk-high { background: #FEF2F2; border: 1px solid #FECACA; color: #7F1D1D; }
    .risk-med  { background: #FFFBEB; border: 1px solid #FDE68A; color: #78350F; }
    .risk-low  { background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; }

    /* KPI Strip */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.15rem;
    }
    .kpi-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-val {
        font-size: 1.72rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Trap Summary Strip */
    .trap-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        padding: 0.65rem 1rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 0.8rem;
    }

    /* Claim Audit Cards */
    .audit-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.95rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.035);
    }
    .audit-card-verified { border-left: 4px solid #10B981; }
    .audit-card-inaccurate { border-left: 4px solid #F59E0B; }
    .audit-card-false { border-left: 4px solid #EF4444; }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 0.7rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.68rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-verified { background: #DCFCE7; color: #166534; }
    .badge-inaccurate { background: #FEF3C7; color: #92400E; }
    .badge-false { background: #FEE2E2; color: #991B1B; }

    .trap-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.71rem;
        letter-spacing: 0.03em;
        margin-left: 0.35rem;
    }
    .trap-outdated { background: #FFEDD5; color: #9A3412; border: 1px solid #FDBA74; }
    .trap-numeric  { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .trap-misattr  { background: #F3E8FF; color: #6B21A8; border: 1px solid #D8B4FE; }
    .trap-fab      { background: #FFE4E6; color: #9F1239; border: 1px solid #FDA4AF; }

    .meta-tag {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        font-size: 0.73rem;
        font-weight: 600;
        background: #F1F5F9;
        color: #475569;
        margin-left: 0.3rem;
    }

    /* Confidence Bar */
    .conf-wrap {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.76rem;
        font-weight: 600;
        color: #475569;
    }
    .conf-track {
        width: 74px;
        height: 7px;
        background: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 999px;
    }

    .claim-headline {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }
    .diff-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 0.8rem;
    }
    .diff-box {
        border-radius: 10px;
        padding: 0.75rem 0.95rem;
        font-size: 0.86rem;
        line-height: 1.48;
    }
    .diff-claimed {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #475569;
    }
    .diff-real-inaccurate {
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        color: #78350F;
    }
    .diff-real-false {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #7F1D1D;
    }
    .diff-real-verified {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
    }
    .diff-label {
        font-size: 0.69rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
        opacity: 0.85;
    }
    .analysis-text {
        font-size: 0.89rem;
        color: #334155;
        line-height: 1.52;
        margin-bottom: 0.75rem;
    }
    .source-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.7rem;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #1D4ED8 !important;
        text-decoration: none !important;
    }
    .source-chip:hover {
        background: #DBEAFE;
    }
    .source-chip-official {
        background: #ECFDF5;
        border-color: #A7F3D0;
        color: #065F46 !important;
    }
    .source-chip-official:hover {
        background: #D1FAE5;
    }

    /* PDF Document Inspector Highlights */
    .doc-viewer {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.45rem 1.7rem;
        line-height: 1.78;
        font-size: 0.94rem;
        color: #1E293B;
        white-space: pre-wrap;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .hl-verified { background: #DCFCE7; padding: 1px 5px; border-radius: 4px; border-bottom: 2px solid #16A34A; }
    .hl-inaccurate { background: #FEF3C7; padding: 1px 5px; border-radius: 4px; border-bottom: 2px solid #D97706; }
    .hl-false { background: #FEE2E2; padding: 1px 5px; border-radius: 4px; border-bottom: 2px solid #DC2626; }

    @media (max-width: 900px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-steps { grid-template-columns: 1fr; }
        .diff-grid { grid-template-columns: 1fr; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _trap_badge_html(trap_type: TrapType) -> str:
    if trap_type == TrapType.NONE:
        return ""
    cls_map = {
        TrapType.OUTDATED_STAT: "trap-outdated",
        TrapType.NUMERIC_DISTORTION: "trap-numeric",
        TrapType.MISATTRIBUTED_SOURCE: "trap-misattr",
        TrapType.FABRICATED_CLAIM: "trap-fab",
    }
    cls = cls_map.get(trap_type, "trap-numeric")
    return f'<span class="trap-badge {cls}">⚡ {html.escape(trap_type.value)}</span>'


def _build_trust_gauge(trust_score: float) -> go.Figure:
    bar_color = "#10B981" if trust_score >= 75 else ("#F59E0B" if trust_score >= 45 else "#EF4444")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=trust_score,
            number={"suffix": "%", "font": {"size": 34, "color": "#0F172A", "family": "Plus Jakarta Sans"}},
            title={"text": "Document Trust Score", "font": {"size": 13, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "#FEE2E2"},
                    {"range": [45, 75], "color": "#FEF3C7"},
                    {"range": [75, 100], "color": "#DCFCE7"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=18, r=18, t=42, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _build_verdict_donut(report: FactCheckReport) -> go.Figure:
    labels = ["Verified", "Inaccurate", "False"]
    values = [report.verified_count, report.inaccurate_count, report.false_count]
    colors = ["#10B981", "#F59E0B", "#EF4444"]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+value",
            textfont=dict(size=12, family="Plus Jakarta Sans"),
            hovertemplate="%{label}: %{value} claims (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Verdict Breakdown", font=dict(size=13, color="#475569"), x=0.5),
        showlegend=False,
        height=250,
        margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_category_bar(report: FactCheckReport) -> go.Figure:
    rows = [{"Category": r.claim.category.value, "Verdict": r.verdict.value} for r in report.results]
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()

    pivot = df.pivot_table(index="Category", columns="Verdict", aggfunc="size", fill_value=0)
    fig = go.Figure()
    color_map = {"Verified": "#10B981", "Inaccurate": "#F59E0B", "False": "#EF4444"}
    for verdict in ["Verified", "Inaccurate", "False"]:
        if verdict in pivot.columns:
            fig.add_trace(
                go.Bar(
                    name=verdict,
                    x=pivot.index.tolist(),
                    y=pivot[verdict].tolist(),
                    marker_color=color_map[verdict],
                )
            )
    fig.update_layout(
        barmode="stack",
        title=dict(text="Category Risk Profile", font=dict(size=13, color="#475569"), x=0.02),
        height=250,
        margin=dict(l=20, r=20, t=42, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(dtick=1, gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_sidebar() -> AppSettings:
    with st.sidebar:
        st.markdown("### Engine Configuration")
        st.caption(
            "Deployment secrets load automatically. Enter keys below only to override or test locally."
        )
        gemini_key = st.text_input(
            "Gemini API Key (Primary)",
            type="password",
            help="Free tier at aistudio.google.com/apikey",
        )
        openai_key = st.text_input(
            "OpenAI API Key (Failover)",
            type="password",
            help="Automatic fallback if Gemini is omitted or rate-limited.",
        )
        tavily_key = st.text_input(
            "Tavily Search Key (Optional)",
            type="password",
            help="Uses live DuckDuckGo fallback automatically when left blank.",
        )
        max_claims = st.slider("Max claims to audit", min_value=5, max_value=30, value=15, step=1)
        enable_cache = st.toggle(
            "Enable SHA-256 report cache",
            value=True,
            help="Repeat uploads of the same PDF return instantaneously with 0 API calls.",
        )

        settings = load_settings(
            gemini_override=gemini_key,
            openai_override=openai_key,
            tavily_override=tavily_key,
            max_claims=max_claims,
            enable_cache=enable_cache,
        )

        st.divider()
        st.markdown("### Verdict & Trap Taxonomy")
        st.markdown(
            """
            <div style="font-size:0.82rem;line-height:1.65;">
              <span class="badge badge-verified">Verified</span> Confirmed by authoritative live sources.<br/>
              <span class="badge badge-inaccurate">Inaccurate</span> Outdated stat, numeric distortion, or misattributed source.<br/>
              <span class="badge badge-false">False</span> Fabricated claim or zero credible evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        cache_count = len(_get_shared_audit_cache())
        st.caption(f"SHA-256 Cache: {cache_count} document(s) indexed")
        st.caption("TruthLayer AI · v3.0 Flagship Audit Engine")
        return settings


def _render_claim_card(res: VerificationResult) -> None:
    v_slug = res.verdict.value.lower()
    conf_pct = int(round(res.confidence * 100))
    conf_color = "#10B981" if res.verdict == Verdict.VERIFIED else ("#F59E0B" if res.verdict == Verdict.INACCURATE else "#EF4444")

    chips_parts: List[str] = []
    for s in res.sources[:3]:
        dom = s.domain or extract_domain(s.url)
        is_official = s.authority_score >= 8
        chip_cls = "source-chip source-chip-official" if is_official else "source-chip"
        prefix = "🏛️" if is_official else "🔗"
        date_tag = f" · {s.published_date[:10]}" if s.published_date else ""
        title_short = s.title[:38] + ("…" if len(s.title) > 38 else "")
        chips_parts.append(
            f'<a class="{chip_cls}" href="{html.escape(s.url)}" target="_blank" rel="noopener">'
            f'{prefix} <strong>{html.escape(dom)}</strong> · {html.escape(title_short)}{html.escape(date_tag)}'
            f"</a>"
        )
    chips = "".join(chips_parts) or (
        '<span style="font-size:0.8rem;color:#94A3B8;">No supporting live web sources found.</span>'
    )

    if res.corrected_fact:
        real_cls = "diff-real-false" if res.verdict == Verdict.FALSE else "diff-real-inaccurate"
        diff_block = f"""
        <div class="diff-grid">
            <div class="diff-box diff-claimed">
                <div class="diff-label">Document Claimed (Page {res.claim.page_number})</div>
                "{html.escape(res.claim.original_quote)}"
            </div>
            <div class="diff-box {real_cls}">
                <div class="diff-label">Verified Real Fact</div>
                {html.escape(res.corrected_fact)}
            </div>
        </div>
        """
    else:
        diff_block = f"""
        <div class="diff-grid">
            <div class="diff-box diff-claimed">
                <div class="diff-label">Document Quote (Page {res.claim.page_number})</div>
                "{html.escape(res.claim.original_quote)}"
            </div>
            <div class="diff-box diff-real-verified">
                <div class="diff-label">Verification Status</div>
                Matches authoritative retrieved live web evidence.
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="audit-card audit-card-{v_slug}">
            <div class="card-header">
                <div>
                    <span class="badge badge-{v_slug}">{res.verdict.value}</span>
                    {_trap_badge_html(res.trap_type)}
                    <span class="meta-tag">Claim #{res.claim.claim_id}</span>
                    <span class="meta-tag">Page {res.claim.page_number}</span>
                    <span class="meta-tag">{html.escape(res.claim.category.value)}</span>
                </div>
                <div class="conf-wrap">
                    <span>Confidence</span>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{conf_pct}%;background:{conf_color};"></div>
                    </div>
                    <span style="color:#0F172A;font-family:'JetBrains Mono',monospace;">{conf_pct}%</span>
                </div>
            </div>
            <div class="claim-headline">{html.escape(res.claim.claim_text)}</div>
            {diff_block}
            <div class="analysis-text">{html.escape(res.explanation)}</div>
            <div class="source-chips">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Inspect dual search queries, source authority & grounding — Claim #{res.claim.claim_id}"):
        q1, q2 = st.columns(2)
        q1.markdown("**Primary Authoritative Query**")
        q1.code(res.search_query_used or res.claim.primary_query, language="text")
        q2.markdown("**Adversarial / 2025–2026 Fact-Check Query**")
        q2.code(res.adversarial_query_used or res.claim.adversarial_query, language="text")

        meta_line = f"Claim latency: `{res.latency_sec:.2f}s` · Grounded citations: `{len(res.cited_urls)}`"
        if res.stripped_ungrounded_urls:
            meta_line += f" · Guardrail stripped `{len(res.stripped_ungrounded_urls)}` ungrounded URL(s)"
        st.caption(meta_line)

        for idx, src in enumerate(res.sources, start=1):
            pub = f" · Published `{src.published_date}`" if src.published_date else ""
            st.markdown(
                f"**{idx}. [{src.title}]({src.url})** — `{src.domain}` · *{src.authority_tier}* "
                f"(score `{src.authority_score}`, via `{src.provider}`, origin `{src.query_origin}`{pub})"
            )
            st.caption(src.snippet[:340] + ("..." if len(src.snippet) > 340 else ""))


def _highlight_quote_in_text(rendered_escaped: str, raw_quote: str, replacement_cls: str, tooltip: str) -> str:
    """Apply exact match first, falling back to whitespace-tolerant regex highlight."""
    quote_esc = html.escape(raw_quote.strip())
    if not quote_esc:
        return rendered_escaped

    span = f'<span class="{replacement_cls}" title="{html.escape(tooltip)}">{{match}}</span>'
    if quote_esc in rendered_escaped:
        return rendered_escaped.replace(quote_esc, span.format(match=quote_esc), 1)

    tokens = [re.escape(tok) for tok in quote_esc.split() if tok]
    if len(tokens) >= 4:
        pattern = re.compile(r"\s+".join(tokens), flags=re.IGNORECASE)
        match = pattern.search(rendered_escaped)
        if match:
            matched_text = match.group(0)
            return (
                rendered_escaped[: match.start()]
                + span.format(match=matched_text)
                + rendered_escaped[match.end() :]
            )
    return rendered_escaped


def _render_highlighted_document(pages: List[PageContent], report: FactCheckReport) -> None:
    if not pages:
        st.info("Upload a PDF or load the Sample Trap Document to inspect inline page highlights.")
        return

    verdict_cls = {
        Verdict.VERIFIED: "hl-verified",
        Verdict.INACCURATE: "hl-inaccurate",
        Verdict.FALSE: "hl-false",
    }

    st.caption(
        "Color-coded inline highlights map extracted quotes directly back to their verification verdict "
        "(hover any highlighted passage to preview its verdict and correction):"
    )
    st.markdown(
        '<span class="hl-verified">Verified</span> &nbsp; '
        '<span class="hl-inaccurate">Inaccurate / Outdated</span> &nbsp; '
        '<span class="hl-false">False / Fabricated</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    page_options = ["All Pages"] + [f"Page {p.page_number}" for p in pages]
    selected_view = st.selectbox("Page Selector", options=page_options, index=0)

    visible_pages = (
        pages
        if selected_view == "All Pages"
        else [p for p in pages if f"Page {p.page_number}" == selected_view]
    )

    for p in visible_pages:
        page_results = [r for r in report.results if r.claim.page_number == p.page_number]
        st.markdown(
            f"**Page {p.page_number}** · `{len(page_results)}` audited claim(s) · "
            f"extraction: `{p.extraction_method}`"
        )
        rendered = html.escape(p.text)
        for res in page_results:
            tooltip = f"Claim #{res.claim.claim_id} [{res.verdict.value}]"
            if res.trap_type != TrapType.NONE:
                tooltip += f" ({res.trap_type.value})"
            if res.corrected_fact:
                tooltip += f" — Real Fact: {res.corrected_fact[:160]}"
            rendered = _highlight_quote_in_text(
                rendered,
                res.claim.original_quote,
                verdict_cls[res.verdict],
                tooltip,
            )
        st.markdown(f'<div class="doc-viewer">{rendered}</div>', unsafe_allow_html=True)

        if p.tables_markdown:
            with st.expander(f"Structured Markdown Tables Extracted from Page {p.page_number} ({len(p.tables_markdown)})"):
                for t_idx, md_table in enumerate(p.tables_markdown, start=1):
                    st.markdown(f"**Table {t_idx}**")
                    st.markdown(md_table)
        st.write("")


def _record_audit_in_history(report: FactCheckReport, pages: List[PageContent]) -> None:
    history: List[Dict[str, Any]] = st.session_state.setdefault("audit_history", [])
    entry_id = f"{report.document_name} · {report.checked_at_utc} · {report.document_sha256[:8]}"
    # Replace if identical entry_id already exists
    history = [h for h in history if h["id"] != entry_id]
    history.insert(0, {"id": entry_id, "report": report, "pages": pages})
    st.session_state["audit_history"] = history[:15]


def _render_report_workspace(report: FactCheckReport, pages: List[PageContent]) -> None:
    flagged = report.flagged_count
    if report.trust_score < 50:
        r_cls = "risk-high"
        r_title = "Critical Integrity Alert — Adversarial Trap / High-Risk Document"
        r_msg = (
            f"{flagged} of {report.total_claims} audited claims "
            f"({round((flagged / max(report.total_claims, 1)) * 100)}%) were flagged as inaccurate or fabricated."
        )
    elif report.trust_score < 80:
        r_cls = "risk-med"
        r_title = "Moderate Verification Risk — Factual Corrections Required"
        r_msg = f"{flagged} claim(s) require authoritative updates prior to publication."
    else:
        r_cls = "risk-low"
        r_title = "High Document Integrity — Passed TruthLayer Verification"
        r_msg = f"{report.verified_count} of {report.total_claims} claims matched authoritative live web sources."

    sha_short = report.document_sha256[:12] if report.document_sha256 else "N/A"
    t = report.telemetry
    cache_label = "CACHE HIT" if t.cache_hit else f"{t.llm_calls} LLM · {t.search_calls} Search"

    st.markdown(
        f"""
        <div class="risk-banner {r_cls}">
            <div>
                <div style="font-weight:800;font-size:1.0rem;margin-bottom:0.15rem;">{r_title}</div>
                <div style="font-size:0.87rem;">{r_msg}</div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.76rem;opacity:0.88;text-align:right;">
                {html.escape(report.document_name)} · sha256:{html.escape(sha_short)}<br/>
                {html.escape(report.checked_at_utc)} · {t.total_latency_sec:.2f}s ({cache_label})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-tile">
                <div class="kpi-label">Trust Score</div>
                <div class="kpi-val">{report.trust_score}%</div>
                <div class="kpi-sub">{report.verified_count}/{report.total_claims} verified</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Claims Audited</div>
                <div class="kpi-val">{report.total_claims}</div>
                <div class="kpi-sub">{report.page_count} page(s) · {t.avg_claim_latency_sec:.2f}s/claim</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Verified</div>
                <div class="kpi-val" style="color:#059669;">{report.verified_count}</div>
                <div class="kpi-sub">Matches live web</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">Inaccurate</div>
                <div class="kpi-val" style="color:#D97706;">{report.inaccurate_count}</div>
                <div class="kpi-sub">Outdated / distorted</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-label">False</div>
                <div class="kpi-val" style="color:#DC2626;">{report.false_count}</div>
                <div class="kpi-sub">Fabricated / refuted</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_ledger, tab_table, tab_doc, tab_export = st.tabs(
        [
            "📊 Executive Analytics",
            "🛡️ Claim-by-Claim Audit Ledger",
            "📑 Evidence Matrix",
            "📄 PDF Highlight Inspector",
            "⬇️ Export & Audit History",
        ]
    )

    with tab_overview:
        c1, c2, c3 = st.columns([1.05, 1.05, 1.3])
        with c1:
            st.plotly_chart(_build_trust_gauge(report.trust_score), width="stretch")
        with c2:
            st.plotly_chart(_build_verdict_donut(report), width="stretch")
        with c3:
            st.plotly_chart(_build_category_bar(report), width="stretch")

        traps = report.trap_breakdown
        if traps:
            trap_pills = "".join(
                f'<span class="meta-tag" style="background:#FFF7ED;color:#9A3412;border:1px solid #FED7AA;">'
                f"{html.escape(k)}: <strong>{v}</strong></span>"
                for k, v in sorted(traps.items(), key=lambda kv: kv[1], reverse=True)
            )
            st.markdown(
                f'<div class="trap-strip"><strong>Detected Trap Patterns:</strong> {trap_pills}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Priority Corrections (Actionable Fixes)")
        priority = sorted(
            [r for r in report.results if r.verdict != Verdict.VERIFIED],
            key=lambda r: (0 if r.verdict == Verdict.FALSE else 1, r.claim.claim_id),
        )
        if not priority:
            st.success("No inaccurate or false claims found in this document.")
        else:
            for r in priority:
                _render_claim_card(r)

    with tab_ledger:
        f1, f2, f3, f4 = st.columns([1.4, 1.3, 1.4, 1.6])
        verdicts = f1.multiselect(
            "Verdict",
            options=[Verdict.FALSE.value, Verdict.INACCURATE.value, Verdict.VERIFIED.value],
            default=[Verdict.FALSE.value, Verdict.INACCURATE.value, Verdict.VERIFIED.value],
        )
        categories = sorted({r.claim.category.value for r in report.results})
        selected_cats = f2.multiselect("Category", options=categories, default=categories)
        trap_options = sorted({r.trap_type.value for r in report.results})
        selected_traps = f3.multiselect("Trap Type", options=trap_options, default=trap_options)
        query = f4.text_input("Keyword Search", placeholder="e.g., Apple, India, tax, 2024")

        filtered = [
            r
            for r in report.results
            if r.verdict.value in verdicts
            and r.claim.category.value in selected_cats
            and r.trap_type.value in selected_traps
            and (
                not query.strip()
                or query.lower() in r.claim.claim_text.lower()
                or query.lower() in r.explanation.lower()
                or query.lower() in r.trap_type.value.lower()
                or (r.corrected_fact and query.lower() in r.corrected_fact.lower())
            )
        ]
        st.caption(f"Showing {len(filtered)} of {report.total_claims} audited claims")
        for r in filtered:
            _render_claim_card(r)

    with tab_table:
        table_rows = [
            {
                "ID": r.claim.claim_id,
                "Page": r.claim.page_number,
                "Category": r.claim.category.value,
                "Verdict": r.verdict.value,
                "Trap Type": r.trap_type.value,
                "Confidence": round(r.confidence * 100.0, 1),
                "Document Claim": r.claim.claim_text,
                "Verified Real Fact": r.corrected_fact or "Confirmed accurate",
                "Sources": len(r.sources),
                "Primary Source": (
                    r.cited_urls[0]
                    if r.cited_urls
                    else (r.sources[0].url if r.sources else "")
                ),
            }
            for r in report.results
        ]
        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.0f%%", min_value=0.0, max_value=100.0
                ),
                "Primary Source": st.column_config.LinkColumn(
                    "Primary Source", display_text="Open Source ↗"
                ),
            },
        )

    with tab_doc:
        _render_highlighted_document(pages, report)

    with tab_export:
        st.markdown("#### Export Complete Audit Report")
        stem = Path(report.document_name).stem
        e1, e2, e3, e4 = st.columns(4)
        e1.download_button(
            "Download Styled HTML",
            data=export_html(report),
            file_name=f"{stem}_executive_audit.html",
            mime="text/html",
            width="stretch",
        )
        e2.download_button(
            "Download Markdown",
            data=export_markdown(report),
            file_name=f"{stem}_audit.md",
            mime="text/markdown",
            width="stretch",
        )
        e3.download_button(
            "Download CSV",
            data=export_csv(report),
            file_name=f"{stem}_audit.csv",
            mime="text/csv",
            width="stretch",
        )
        e4.download_button(
            "Download JSON",
            data=export_json(report),
            file_name=f"{stem}_audit.json",
            mime="application/json",
            width="stretch",
        )

        st.divider()
        st.markdown("#### Session Audit History & Comparison")
        history: List[Dict[str, Any]] = st.session_state.get("audit_history", [])
        if history:
            hist_df = pd.DataFrame(
                [
                    {
                        "Document": h["report"].document_name,
                        "Audited (UTC)": h["report"].checked_at_utc,
                        "Trust Score": h["report"].trust_score,
                        "Claims": h["report"].total_claims,
                        "Verified": h["report"].verified_count,
                        "Inaccurate": h["report"].inaccurate_count,
                        "False": h["report"].false_count,
                        "Latency (s)": h["report"].telemetry.total_latency_sec,
                        "Cache Hit": "Yes" if h["report"].telemetry.cache_hit else "No",
                    }
                    for h in history
                ]
            )
            st.dataframe(
                hist_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Trust Score": st.column_config.ProgressColumn(
                        "Trust Score", format="%.1f%%", min_value=0.0, max_value=100.0
                    ),
                },
            )
            if len(history) > 1:
                labels = [h["id"] for h in history]
                sel_col, btn_col = st.columns([2.6, 1.0])
                chosen = sel_col.selectbox("Switch active workspace to a previous audit", options=labels)
                if btn_col.button("Load Selected Audit", width="stretch"):
                    for h in history:
                        if h["id"] == chosen:
                            st.session_state["last_report"] = h["report"]
                            st.session_state["active_pages"] = h["pages"]
                            st.rerun()
        else:
            st.caption("Audited PDFs in this session will appear here for side-by-side comparison.")

        with st.expander("Preview Executive Markdown Report"):
            st.code(export_markdown(report), language="markdown")


def main() -> None:
    settings = _render_sidebar()
    shared_cache = _get_shared_audit_cache()

    active_report: Optional[FactCheckReport] = st.session_state.get("last_report")
    llm_badge = (
        f'<span class="pill pill-live"><span class="dot-live"></span>LLM: {html.escape(settings.active_llm_model)}</span>'
        if settings.active_llm_provider != "none"
        else '<span class="pill">LLM: Key Needed (or Demo Preview)</span>'
    )
    search_badge = (
        '<span class="pill pill-live">Search: Tavily + DDG</span>'
        if settings.active_search_provider == "tavily"
        else '<span class="pill">Search: DuckDuckGo (Live)</span>'
    )

    telemetry_badge = '<span class="pill pill-telemetry">Latency: Ready</span>'
    if active_report is not None:
        t = active_report.telemetry
        if t.cache_hit:
            telemetry_badge = f'<span class="pill pill-cache">⚡ Cache Hit: {t.total_latency_sec:.3f}s</span>'
        else:
            telemetry_badge = (
                f'<span class="pill pill-telemetry">'
                f"⚡ {t.total_latency_sec:.2f}s · {t.avg_claim_latency_sec:.2f}s/claim · "
                f"{t.llm_calls} LLM / {t.search_calls} Search</span>"
            )

    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand-left">
                <div class="brand-logo">TL</div>
                <div>
                    <div class="brand-title">TruthLayer AI</div>
                    <div class="brand-sub">Adversarial PDF Claim Verification &amp; Citation Grounding Platform</div>
                </div>
            </div>
            <div class="status-pills">
                {llm_badge}
                {search_badge}
                {telemetry_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">AI Truth Layer · Multi-Stage Adversarial Verification</div>
            <h1>Audit any PDF for outdated stats, distorted figures &amp; fabricated claims.</h1>
            <p>Upload a marketing brief, financial deck, or research whitepaper. TruthLayer extracts body text and Markdown tables, decomposes atomic claims with dual authoritative + adversarial queries, and enforces strict citation grounding against live web evidence.</p>
            <div class="hero-steps">
                <div class="hero-step">
                    <div class="hero-step-title">01 · Table-Aware Atomic Decomposition</div>
                    <div class="hero-step-desc">Extracts verbatim quotes, resolves pronouns, and parses Markdown tables with exact [Page N] provenance &amp; SHA-256 caching.</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-title">02 · Authority-Ranked Dual Search Fusion</div>
                    <div class="hero-step-desc">Runs official + adversarial 2025–2026 fact-check queries across Tavily &amp; DuckDuckGo, prioritizing .gov, .edu, SEC, UN &amp; Reuters.</div>
                </div>
                <div class="hero-step">
                    <div class="hero-step-title">03 · Grounded Trap Adjudication</div>
                    <div class="hero-step-desc">Classifies Verified, Inaccurate, or False with Trap-Type tags, mandatory Verified Real Facts, and zero hallucinated citations.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_actions = st.columns([2.1, 1.15])
    with col_up:
        uploaded = st.file_uploader(
            "Drop a PDF document here to audit",
            type=["pdf"],
            help="Supports multi-page marketing briefs, tables, pitch decks, and financial summaries (max 25 MB).",
        )
    with col_actions:
        st.markdown("**Instant Evaluation Controls**")
        b1 = st.button("Load Sample Trap Document PDF", width="stretch")
        b2 = st.button(
            "Preview Live Audit (Zero-Key Demo)",
            width="stretch",
            help="Explore the full 5-tab analytics dashboard immediately without consuming API credits.",
        )

    if b2:
        demo_pages: List[PageContent] = []
        demo_sha = "demo-preview-sha256-trap-doc"
        if SAMPLE_PDF_PATH.exists():
            parsed_demo = parse_pdf(SAMPLE_PDF_PATH.read_bytes())
            demo_pages = parsed_demo.pages
            demo_sha = parsed_demo.sha256_hash
        demo_report = build_demo_trap_report(document_sha256=demo_sha)
        st.session_state["active_pages"] = demo_pages
        st.session_state["last_report"] = demo_report
        _record_audit_in_history(demo_report, demo_pages)

    pdf_bytes: Optional[bytes] = None
    doc_name: Optional[str] = None

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        doc_name = uploaded.name
    elif b1 and SAMPLE_PDF_PATH.exists():
        pdf_bytes = SAMPLE_PDF_PATH.read_bytes()
        doc_name = SAMPLE_PDF_PATH.name
        st.session_state["sample_bytes"] = pdf_bytes
        st.session_state["sample_name"] = doc_name
    elif "sample_bytes" in st.session_state and uploaded is None:
        pdf_bytes = st.session_state["sample_bytes"]
        doc_name = st.session_state["sample_name"]

    if pdf_bytes and doc_name:
        try:
            parsed: ParsedDocument = parse_pdf(pdf_bytes)
        except Exception as exc:
            st.error(
                f"Could not read **{doc_name}** as a valid PDF ({exc}). "
                "Ensure the file is an unencrypted, non-corrupted PDF document."
            )
            parsed = ParsedDocument(pages=[], total_pages=0, sha256_hash="")

        st.session_state["active_pages"] = parsed.pages

        if parsed.is_scanned_or_image_only and parsed.warning_message:
            st.warning(parsed.warning_message)
        elif parsed.total_pages > 0:
            is_cached = settings.enable_cache and shared_cache.has(parsed.sha256_hash)
            cache_note = " · ⚡ Instant SHA-256 Cache Available" if is_cached else ""
            table_note = f" · {parsed.table_count} table(s) extracted" if parsed.table_count else ""

            info_col, opt_col, run_col = st.columns([2.0, 0.9, 1.0])
            with info_col:
                st.info(
                    f"Ready to audit **{doc_name}** — {parsed.total_pages} page(s){table_note} "
                    f"· `sha256:{parsed.sha256_hash[:12]}`{cache_note}"
                )
            with opt_col:
                force_refresh = st.checkbox(
                    "Bypass cache",
                    value=False,
                    disabled=not is_cached,
                    help="Force a fresh live LLM + web search audit instead of reusing the SHA-256 cached report.",
                )
            with run_col:
                run_clicked = st.button(
                    "Run Live Fact-Check Audit",
                    type="primary",
                    width="stretch",
                )

            if run_clicked:
                if not parsed.pages:
                    st.error(
                        "No selectable text was extracted from this PDF. "
                        "Please run OCR on scanned pages and re-upload."
                    )
                elif settings.active_llm_provider == "none" and not (is_cached and not force_refresh):
                    seen_keys = get_detected_secret_names()
                    diag = (
                        f"Detected Streamlit Secret key(s): `{', '.join(seen_keys)}` (expected `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`)."
                        if seen_keys
                        else "Streamlit Secrets is currently empty — open **Manage app → Settings → Secrets** on Streamlit Cloud and save `GEMINI_API_KEY = \"...\"`, OR paste your key in the **Gemini API Key** field in the left sidebar."
                    )
                    st.error(
                        "No LLM API key detected. "
                        f"{diag} (Or click **Preview Live Audit (Zero-Key Demo)**.)"
                    )
                else:
                    with st.status("Running TruthLayer Adversarial Verification Pipeline...", expanded=True) as status:
                        prog = st.progress(0.0)

                        def _on_progress(done: int, total: int, msg: str) -> None:
                            prog.progress(min(1.0, done / max(total, 1)))
                            status.write(f"• {msg}")

                        try:
                            # Serve cached report directly without constructing LLMClient if keys are absent
                            if is_cached and not force_refresh and settings.active_llm_provider == "none":
                                cached_rep = shared_cache.get(parsed.sha256_hash)
                                if cached_rep is not None:
                                    cached_rep.document_name = doc_name
                                    cached_rep.telemetry = cached_rep.telemetry.model_copy(
                                        update={"cache_hit": True, "total_latency_sec": 0.005, "llm_calls": 0, "search_calls": 0}
                                    )
                                    report = cached_rep
                                else:
                                    raise RuntimeError("Cache entry evicted; provide an LLM API key to re-run.")
                            else:
                                engine = FactCheckEngine(settings, report_cache=shared_cache)
                                report = engine.audit_document(
                                    document_name=doc_name,
                                    pages=parsed.pages,
                                    total_pages=parsed.total_pages,
                                    document_sha256=parsed.sha256_hash,
                                    bypass_cache=force_refresh,
                                    progress_cb=_on_progress,
                                )

                            prog.progress(1.0)
                            label = (
                                f"Instant cache hit ({report.telemetry.total_latency_sec:.3f}s)!"
                                if report.telemetry.cache_hit
                                else f"Verification complete in {report.telemetry.total_latency_sec:.2f}s!"
                            )
                            status.update(label=label, state="complete", expanded=False)
                            st.session_state["last_report"] = report
                            _record_audit_in_history(report, parsed.pages)
                            st.rerun()
                        except Exception as exc:
                            status.update(label="Pipeline error", state="error")
                            st.error(f"Verification failed: {exc}")

    if "last_report" in st.session_state:
        st.divider()
        _render_report_workspace(
            st.session_state["last_report"],
            st.session_state.get("active_pages", []),
        )


if __name__ == "__main__":
    main()
