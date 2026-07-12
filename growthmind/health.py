"""Website Health Score.

Rolls up the site's SEO, performance, UX, security and content signals into a
single 0–100 score with per-dimension breakdown and a letter grade. Security is
proxied from configuration signals available in the dataset (HTTPS/schema/mobile
readiness); in a production deployment it would come from a real security scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HealthScore:
    overall: float
    grade: str
    dimensions: dict = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [f"Website Health Score: {self.overall:.0f}/100  (grade {self.grade})",
                 "-" * 40]
        for name, value in self.dimensions.items():
            bar = "█" * int(value / 5) + "·" * (20 - int(value / 5))
            lines.append(f"  {name:<12} {value:5.0f}  {bar}")
        return "\n".join(lines)


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_health(daily: pd.DataFrame, pages: pd.DataFrame) -> HealthScore:
    """Compute the composite health score from recent metrics and page snapshots."""
    recent = daily.tail(30)

    # SEO: blend of average position (lower is better) and mean page SEO score.
    pos = recent["avg_position"].mean()
    seo_pos_component = float(np.clip(100 - (pos - 1) * 4, 0, 100))
    seo = 0.5 * seo_pos_component + 0.5 * float(pages["seo_score"].mean())

    # Performance: average page speed.
    performance = float(recent["page_speed"].mean())

    # UX: derived from bounce rate and time on site.
    bounce = recent["bounce_rate"].mean()
    time_on_site = recent["avg_time_on_site"].mean()
    ux = float(np.clip((1 - bounce) * 100 * 0.6 + np.clip(time_on_site / 3, 0, 100) * 0.4, 0, 100))

    # Content: mean content score plus a depth bonus.
    depth_bonus = float(np.clip((pages["word_count"].mean() - 300) / 30, 0, 20))
    content = float(np.clip(recent["content_score"].mean() * 0.8 + depth_bonus, 0, 100))

    # Security proxy: HTTPS assumed, weighted by schema + mobile coverage.
    security = float(np.clip(
        70 + 15 * pages["has_schema"].mean() + 15 * pages["mobile_friendly"].mean(),
        0, 100,
    ))

    dimensions = {
        "SEO": round(seo, 1),
        "Performance": round(performance, 1),
        "UX": round(ux, 1),
        "Security": round(security, 1),
        "Content": round(content, 1),
    }
    weights = {"SEO": 0.30, "Performance": 0.20, "UX": 0.20, "Security": 0.15, "Content": 0.15}
    overall = round(sum(dimensions[k] * w for k, w in weights.items()), 1)

    return HealthScore(overall=overall, grade=_grade(overall), dimensions=dimensions)
