"""Export helpers for FactCheckReport (JSON, CSV, Executive Markdown, and Standalone Styled HTML)."""

from __future__ import annotations

import csv
import html
import io

from src.models import FactCheckReport, TrapType, Verdict


def export_json(report: FactCheckReport) -> str:
    """Serialize the full audit report to pretty-printed JSON."""
    if hasattr(report, "model_dump_json"):
        return report.model_dump_json(indent=2)
    return report.json(indent=2)  # type: ignore[attr-defined]


def export_csv(report: FactCheckReport) -> str:
    """Flatten verification results into an Excel-friendly CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Claim ID",
            "Page",
            "Category",
            "Verdict",
            "Trap Type",
            "Confidence",
            "Claim",
            "Corrected Real Fact",
            "Explanation",
            "Primary Query",
            "Adversarial Query",
            "Latency (s)",
            "Primary Source URL",
        ]
    )
    for res in report.results:
        primary_url = res.cited_urls[0] if res.cited_urls else (res.sources[0].url if res.sources else "")
        writer.writerow(
            [
                res.claim.claim_id,
                res.claim.page_number,
                res.claim.category.value,
                res.verdict.value,
                res.trap_type.value,
                f"{int(res.confidence * 100)}%",
                res.claim.claim_text,
                res.corrected_fact or "",
                res.explanation,
                res.search_query_used or res.claim.primary_query,
                res.adversarial_query_used or res.claim.adversarial_query,
                f"{res.latency_sec:.2f}",
                primary_url,
            ]
        )
    return buf.getvalue()


def export_markdown(report: FactCheckReport) -> str:
    """Generate an executive Markdown audit report with trap tags, telemetry, and clickable citations."""
    t = report.telemetry
    model_str = f"{report.llm_provider} ({report.llm_model})" if report.llm_model else report.llm_provider
    lines = [
        f"# Fact-Check Audit Report: {report.document_name}",
        "",
        f"- **Audited:** {report.checked_at_utc}",
        f"- **Pages Analyzed:** {report.page_count}",
        f"- **SHA-256 Fingerprint:** `{report.document_sha256[:16]}`" if report.document_sha256 else "- **SHA-256 Fingerprint:** `N/A`",
        f"- **Trust Score:** {report.trust_score}% ({report.verified_count}/{report.total_claims} Verified)",
        f"- **Summary:** {report.verified_count} Verified | {report.inaccurate_count} Inaccurate | {report.false_count} False",
        f"- **Engine:** LLM `{model_str}` + Live Search `{report.search_provider}`",
        f"- **Telemetry:** {t.total_latency_sec:.2f}s total ({t.avg_claim_latency_sec:.2f}s/claim) | {t.llm_calls} LLM calls | {t.search_calls} search calls | Cache hit: `{t.cache_hit}`",
        "",
        "## Claim-by-Claim Findings",
        "",
    ]

    for res in report.results:
        trap_label = f" · Trap: {res.trap_type.value}" if res.trap_type != TrapType.NONE else ""
        lines.append(
            f"### #{res.claim.claim_id} [{res.verdict.value.upper()}] — Page {res.claim.page_number} ({res.claim.category.value}{trap_label})"
        )
        lines.append(f"- **Claim:** {res.claim.claim_text}")
        lines.append(f"- **Confidence:** {int(res.confidence * 100)}%")
        if res.corrected_fact:
            lines.append(f"- **Verified Real Fact:** {res.corrected_fact}")
        lines.append(f"- **Analysis:** {res.explanation}")
        if res.sources:
            src_links = ", ".join(
                f"[{s.domain or s.title} — {s.title}]({s.url})" for s in res.sources[:3]
            )
            lines.append(f"- **Sources:** {src_links}")
        lines.append("")

    return "\n".join(lines)


def export_html(report: FactCheckReport) -> str:
    """Generate a standalone, print-ready styled HTML executive audit report."""
    badge_styles = {
        Verdict.VERIFIED: "background:#DCFCE7;color:#166534;",
        Verdict.INACCURATE: "background:#FEF3C7;color:#92400E;",
        Verdict.FALSE: "background:#FEE2E2;color:#991B1B;",
    }
    cards_html = []
    for res in report.results:
        bstyle = badge_styles[res.verdict]
        trap_pill = ""
        if res.trap_type != TrapType.NONE:
            trap_pill = (
                f'<span style="margin-left:6px;padding:3px 9px;border-radius:999px;font-weight:700;'
                f'font-size:11px;background:#F1F5F9;color:#334155;border:1px solid #CBD5E1;">'
                f'TRAP: {html.escape(res.trap_type.value)}</span>'
            )

        diff_block = ""
        if res.corrected_fact:
            border = "#DC2626" if res.verdict == Verdict.FALSE else "#D97706"
            bg = "#FEF2F2" if res.verdict == Verdict.FALSE else "#FFFBEB"
            diff_block = f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;">
              <div style="background:#F8FAFC;border:1px solid #E2E8F0;padding:10px 12px;border-radius:8px;font-size:13px;color:#475569;">
                <div style="font-size:10px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;">Document Claimed (Page {res.claim.page_number})</div>
                "{html.escape(res.claim.original_quote)}"
              </div>
              <div style="background:{bg};border:1px solid {border};border-left:4px solid {border};padding:10px 12px;border-radius:8px;font-size:13px;">
                <div style="font-size:10px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;">Verified Real Fact</div>
                {html.escape(res.corrected_fact)}
              </div>
            </div>
            """

        src_html = "".join(
            f'<li><a href="{html.escape(s.url)}" target="_blank" rel="noopener">'
            f'<strong>{html.escape(s.domain)}</strong> · {html.escape(s.title)}</a> '
            f'<span style="color:#64748B;font-size:12px;">({html.escape(s.authority_tier)})</span></li>'
            for s in res.sources[:3]
        )
        cards_html.append(
            f"""
            <div style="border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;margin-bottom:14px;background:#fff;">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px;">
                <div>
                  <span style="padding:3px 10px;border-radius:999px;font-weight:700;font-size:12px;{bstyle}">{res.verdict.value.upper()}</span>
                  {trap_pill}
                  <strong style="margin-left:8px;">Claim #{res.claim.claim_id}</strong> · Page {res.claim.page_number} · {html.escape(res.claim.category.value)}
                </div>
                <span style="color:#64748B;font-size:13px;">Confidence: <strong>{int(res.confidence * 100)}%</strong></span>
              </div>
              <p style="font-size:15px;font-weight:700;margin:6px 0;color:#0F172A;">{html.escape(res.claim.claim_text)}</p>
              {diff_block}
              <p style="color:#334155;font-size:14px;margin:6px 0;line-height:1.5;">{html.escape(res.explanation)}</p>
              <ul style="margin:8px 0 0 18px;font-size:13px;color:#2563EB;">{src_html}</ul>
            </div>
            """
        )

    t = report.telemetry
    sha_short = report.document_sha256[:16] if report.document_sha256 else "N/A"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>TruthLayer Executive Audit — {html.escape(report.document_name)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#F8FAFC; color:#0F172A; margin:0; padding:32px; }}
  .wrap {{ max-width:960px; margin:0 auto; }}
  .header {{ background:linear-gradient(135deg,#0B1324,#172B52); color:#fff; padding:26px 30px; border-radius:16px; margin-bottom:22px; }}
  .meta {{ font-size:13px; color:#CBD5E1; margin-top:6px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:22px; }}
  .kpi {{ background:#fff; border:1px solid #E2E8F0; border-radius:12px; padding:14px; text-align:center; }}
  .kpi h2 {{ margin:0; font-size:26px; }}
  .kpi p {{ margin:4px 0 0; color:#64748B; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1 style="margin:0;">TruthLayer AI · Executive Fact-Check Audit</h1>
      <div class="meta">{html.escape(report.document_name)} · {html.escape(report.checked_at_utc)} · {report.page_count} page(s) · SHA-256: <code>{html.escape(sha_short)}</code></div>
      <div class="meta">Engine: {html.escape(report.llm_provider)} {html.escape(report.llm_model)} + {html.escape(report.search_provider)} · Latency: {t.total_latency_sec:.2f}s ({t.llm_calls} LLM / {t.search_calls} Search)</div>
    </div>
    <div class="kpis">
      <div class="kpi"><h2>{report.trust_score}%</h2><p>Trust Score</p></div>
      <div class="kpi"><h2 style="color:#166534;">{report.verified_count}</h2><p>Verified</p></div>
      <div class="kpi"><h2 style="color:#B45309;">{report.inaccurate_count}</h2><p>Inaccurate</p></div>
      <div class="kpi"><h2 style="color:#B91C1C;">{report.false_count}</h2><p>False</p></div>
    </div>
    {''.join(cards_html)}
  </div>
</body>
</html>"""
