"""AI Growth Strategy.

Turns the raw recommendations into a time-boxed action plan — what to fix
*today*, *this week*, and *this month* — by bucketing recommendations by
priority. This is the one-click "what should I do next?" output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .recommend import Recommendation


@dataclass
class GrowthStrategy:
    today: list = field(default_factory=list)      # list[Recommendation]
    this_week: list = field(default_factory=list)
    this_month: list = field(default_factory=list)

    def pretty(self) -> str:
        def block(title, recs):
            if not recs:
                return f"\n{title}\n  (nothing pressing)"
            lines = [f"\n{title}"]
            for r in recs:
                lines.append(f"  • [{r.area}] {r.action}")
                lines.append(f"      ↳ {r.problem} — {r.expected_gain}")
            return "\n".join(lines)

        return (
            "AI Growth Strategy\n"
            "==================" +
            block("TODAY (do now)", self.today) +
            block("THIS WEEK", self.this_week) +
            block("THIS MONTH", self.this_month)
        )


def build_strategy(recommendations: list[Recommendation]) -> GrowthStrategy:
    """Bucket recommendations into today / this week / this month by priority."""
    ordered = sorted(recommendations, key=lambda r: r.priority)
    strategy = GrowthStrategy()
    for r in ordered:
        if r.priority <= 1:
            strategy.today.append(r)
        elif r.priority <= 2:
            strategy.this_week.append(r)
        else:
            strategy.this_month.append(r)
    return strategy
