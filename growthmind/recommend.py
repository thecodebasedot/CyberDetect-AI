"""AI Recommendation Engine.

Combines model signals (SEO feature importances, per-page weaknesses, traffic
anomalies) with transparent domain rules to produce prioritized, actionable
recommendations — each with a problem, the reason, the fix, and an estimated
traffic/revenue impact. This is what turns GrowthMind from an analytics report
into a growth *advisor*.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Recommendation:
    area: str            # SEO | Performance | Content | Conversion | Anomaly
    problem: str
    reason: str
    action: str
    expected_gain: str   # human-readable impact estimate
    priority: int        # 1 (highest) .. 5 (lowest)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Page-level SEO recommendations
# --------------------------------------------------------------------------
def page_recommendations(pages: pd.DataFrame, limit: int = 10) -> list[Recommendation]:
    """Rule-based on-page fixes for the weakest high-opportunity pages.

    We target pages that already attract traffic but score poorly on SEO — the
    biggest, fastest wins.
    """
    recs: list[Recommendation] = []
    df = pages.copy()

    # Opportunity = traffic weighted by how far the page is below a good score.
    df["gap"] = (80 - df["seo_score"]).clip(lower=0)
    df["opportunity"] = df["gap"] * np.log1p(df["organic_traffic"])
    focus = df.sort_values("opportunity", ascending=False).head(limit)

    for _, row in focus.iterrows():
        issues = _page_issues(row)
        if not issues:
            continue
        problem, reason, action, gain, prio = issues[0]  # top issue for the page
        recs.append(Recommendation(
            area="SEO",
            problem=f"{row['url']}: {problem}",
            reason=reason,
            action=action,
            expected_gain=gain,
            priority=prio,
        ))
    return recs


def _page_issues(row: pd.Series):
    """Return a priority-ordered list of concrete issues for one page."""
    issues = []
    if row["word_count"] < 400:
        issues.append((
            "thin content",
            f"only {int(row['word_count'])} words — below the ~800-word depth "
            "that ranks for competitive queries",
            "expand the article with useful sections, examples and FAQs",
            "+8–15% organic traffic",
            1,
        ))
    if row["alt_ratio"] < 0.6:
        issues.append((
            "missing image alt text",
            f"only {row['alt_ratio']:.0%} of images have alt attributes",
            "add descriptive alt text to every image",
            "+3–6% (accessibility + image search)",
            3,
        ))
    if row["page_speed"] < 60:
        issues.append((
            "slow page",
            f"page-speed score is {row['page_speed']:.0f}/100",
            "compress images, defer JS and enable caching",
            "+5–12% (Core Web Vitals ranking factor)",
            2,
        ))
    if not row["has_schema"]:
        issues.append((
            "no structured data",
            "page has no schema markup, so it can't earn rich results",
            "add relevant JSON-LD schema (Article/Product/FAQ)",
            "+4–9% CTR from rich snippets",
            2,
        ))
    if row["internal_links"] < 5:
        issues.append((
            "weak internal linking",
            f"only {int(row['internal_links'])} internal links",
            "add 5–10 contextual internal links from related pages",
            "+4–8% (link equity + crawl depth)",
            3,
        ))
    if abs(row["title_length"] - 55) > 20:
        issues.append((
            "sub-optimal title length",
            f"title is {int(row['title_length'])} chars (ideal ≈ 50–60)",
            "rewrite the title tag to 50–60 characters with the target keyword",
            "+2–5% CTR",
            4,
        ))
    return sorted(issues, key=lambda x: x[4])


# --------------------------------------------------------------------------
# Site-level recommendations from trends + models
# --------------------------------------------------------------------------
def site_recommendations(
    daily: pd.DataFrame,
    sales_importance: pd.Series | None = None,
    anomalies: pd.DataFrame | None = None,
) -> list[Recommendation]:
    """Recommendations derived from recent trends, the sales model and anomalies."""
    recs: list[Recommendation] = []
    recent = daily.tail(30)
    prev = daily.tail(60).head(30)

    # Traffic trend.
    if len(prev) and recent["visitors"].mean() < prev["visitors"].mean() * 0.95:
        drop = 1 - recent["visitors"].mean() / prev["visitors"].mean()
        recs.append(Recommendation(
            area="SEO",
            problem="traffic is trending down",
            reason=f"last-30-day visitors are {drop:.0%} below the prior 30 days",
            action="audit recently-dropped keywords in Search Console and refresh "
                   "the pages that lost rankings",
            expected_gain="recover the lost traffic",
            priority=1,
        ))

    # Bounce rate.
    if recent["bounce_rate"].mean() > 0.6:
        recs.append(Recommendation(
            area="Conversion",
            problem="high bounce rate",
            reason=f"average bounce rate is {recent['bounce_rate'].mean():.0%}",
            action="improve above-the-fold relevance, page speed and internal linking",
            expected_gain="+5–10% engaged sessions",
            priority=2,
        ))

    # Page speed.
    if recent["page_speed"].mean() < 70:
        recs.append(Recommendation(
            area="Performance",
            problem="slow average page speed",
            reason=f"site-wide page speed is {recent['page_speed'].mean():.0f}/100",
            action="optimize images, adopt a CDN and reduce render-blocking resources",
            expected_gain="+5–12% traffic & conversions",
            priority=2,
        ))

    # Conversion lever from the sales model.
    if sales_importance is not None and len(sales_importance):
        top_lever = sales_importance.index[0]
        recs.append(Recommendation(
            area="Conversion",
            problem="revenue is most sensitive to one lever",
            reason=f"the sales model ranks '{top_lever}' as the biggest revenue driver",
            action=f"run experiments that improve '{top_lever}' first",
            expected_gain="highest expected revenue lift per unit effort",
            priority=2,
        ))

    # Anomalies.
    if anomalies is not None and len(anomalies):
        recent_anom = anomalies.head(3)
        recs.append(Recommendation(
            area="Anomaly",
            problem=f"{len(anomalies)} anomalous traffic day(s) detected",
            reason="the anomaly detector flagged unusual KPI patterns "
                   f"(e.g. {pd.to_datetime(recent_anom['date'].iloc[0]).date()})",
            action="investigate tracking, algorithm updates or campaign changes on "
                   "the flagged dates",
            expected_gain="prevent silent traffic loss",
            priority=1,
        ))

    return recs


def prioritize(recs: list[Recommendation]) -> list[Recommendation]:
    """Sort recommendations by priority (highest first)."""
    return sorted(recs, key=lambda r: r.priority)
