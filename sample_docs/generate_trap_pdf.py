"""Generate a realistic 1-page Marketing 'Trap Document' PDF for testing and demo recording."""

from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_trap_pdf(output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0D1B2A"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=10,
    )

    story = [
        Paragraph("Q4 Global AI & Tech Market Brief (Sample Trap Document)", title_style),
        Paragraph(
            "Prepared for Automated Fact-Check Evaluation — Contains a deliberate mix of verified facts, "
            "outdated statistics, and fabricated claims.",
            subtitle_style,
        ),
        Paragraph("1. Technology & Search Milestones", heading_style),
        Paragraph(
            "OpenAI officially launched ChatGPT to the public on November 30, 2022, triggering the fastest "
            "consumer software adoption wave in history. Meanwhile, Google was originally founded in September 1998 "
            "by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
            body_style,
        ),
        Paragraph("2. Outdated Market Figures (Trap Claims)", heading_style),
        Paragraph(
            "As of today, India's population stands at 1.21 billion people, making it the second most populous "
            "country in the world behind China. In the smartphone market, Apple's total annual revenue in fiscal year "
            "2024 was $180 billion, down sharply from prior years.",
            body_style,
        ),
        Paragraph("3. Fabricated Claims (Intentional Lies)", heading_style),
        Paragraph(
            "According to a 2025 United Nations Digital Commerce Mandate, all European Union websites are legally "
            "required to pay a 15% 'AI Citation Tax' on organic search traffic starting January 2026. Additionally, "
            "NVIDIA acquired OpenAI in March 2025 for $420 billion in an all-cash transaction.",
            body_style,
        ),
    ]

    doc.build(story)


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_trap_document.pdf"
    build_trap_pdf(out)
    print(f"Generated {out.resolve()}")
