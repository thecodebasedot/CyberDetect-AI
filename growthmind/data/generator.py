"""Synthetic website-analytics data generator.

GrowthMind AI ships without a bundled dataset so it can run fully offline. This
module fabricates three coherent, *causally linked* tables that mimic what you
would export from Google Analytics + Search Console + a site crawler:

* ``daily_metrics`` — a multi-year daily time-series of site KPIs with trend,
  weekly seasonality, a couple of injected traffic anomalies, and revenue that
  genuinely depends on traffic/CTR/conversion.
* ``pages``         — per-page SEO snapshot whose ``seo_score`` is a real
  function of the on-page features (so the SEO model has something to learn).
* ``users``         — per-visitor behaviour with latent segments (new,
  returning, potential buyer, buyer) so K-Means has real structure to recover.

Because everything is seeded, the whole dataset is reproducible.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import RANDOM_STATE


# --------------------------------------------------------------------------
# Daily site KPIs (time-series)
# --------------------------------------------------------------------------
def generate_daily_metrics(
    days: int = 730,
    start: date = date(2023, 1, 1),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a daily KPI time-series with trend, seasonality and anomalies."""
    rng = np.random.default_rng(random_state)
    t = np.arange(days)

    # Organic traffic: upward trend + weekly seasonality + slow SEO improvement.
    trend = 800 + 2.2 * t
    weekly = 120 * np.sin(2 * np.pi * t / 7)
    monthly = 200 * np.sin(2 * np.pi * t / 30)
    noise = rng.normal(0, 60, days)
    organic = np.clip(trend + weekly + monthly + noise, 50, None)

    # Paid traffic: campaign bursts on top of a modest baseline.
    paid = np.clip(250 + 80 * np.sin(2 * np.pi * t / 45) + rng.normal(0, 40, days), 0, None)

    # Inject a few realistic anomalies (an algorithm-update drop + a viral spike).
    organic[300:315] *= 0.55          # sudden ranking drop
    organic[500:505] *= 1.9           # viral / featured spike

    visitors = organic + paid
    sessions = visitors * rng.uniform(1.1, 1.4, days)

    # SEO signals slowly improve over time.
    avg_position = np.clip(18 - 0.012 * t + rng.normal(0, 0.8, days), 1, 50)
    ctr = np.clip(0.02 + 0.5 / avg_position + rng.normal(0, 0.004, days), 0.005, 0.4)
    backlinks = np.cumsum(rng.poisson(3, days)) + 500
    page_speed = np.clip(88 - 0.004 * t + rng.normal(0, 3, days), 40, 100)
    content_score = np.clip(60 + 0.02 * t + rng.normal(0, 5, days), 0, 100)

    bounce_rate = np.clip(0.62 - 0.00008 * t + rng.normal(0, 0.03, days), 0.2, 0.9)
    avg_time = np.clip(90 + 0.05 * t + rng.normal(0, 15, days), 20, 600)

    # Conversion depends (softly) on speed, content and bounce.
    conversion_rate = np.clip(
        0.012
        + 0.00015 * (page_speed - 70)
        + 0.0004 * (content_score - 60) / 10
        - 0.02 * (bounce_rate - 0.5)
        + rng.normal(0, 0.003, days),
        0.001,
        0.15,
    )

    sales = np.round(sessions * conversion_rate).astype(int)
    aov = rng.normal(48, 8, days).clip(10, None)      # average order value
    revenue = np.round(sales * aov, 2)

    dates = [start + timedelta(days=int(i)) for i in t]

    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "visitors": np.round(visitors).astype(int),
        "sessions": np.round(sessions).astype(int),
        "organic_traffic": np.round(organic).astype(int),
        "paid_traffic": np.round(paid).astype(int),
        "bounce_rate": bounce_rate.round(4),
        "avg_time_on_site": avg_time.round(1),
        "ctr": ctr.round(4),
        "avg_position": avg_position.round(2),
        "backlinks": backlinks.astype(int),
        "page_speed": page_speed.round(1),
        "content_score": content_score.round(1),
        "conversion_rate": conversion_rate.round(4),
        "sales": sales,
        "revenue": revenue,
    })


# --------------------------------------------------------------------------
# Per-page SEO snapshot
# --------------------------------------------------------------------------
def generate_pages(n: int = 400, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate per-page SEO features with a derived, learnable ``seo_score``."""
    rng = np.random.default_rng(random_state + 1)

    word_count = rng.lognormal(mean=6.6, sigma=0.6, size=n).clip(80, 8000)
    images = rng.poisson(6, n)
    alt_ratio = rng.beta(5, 2, n)                       # fraction of images with alt text
    internal_links = rng.poisson(12, n)
    external_links = rng.poisson(4, n)
    title_length = rng.normal(55, 12, n).clip(10, 90)
    meta_length = rng.normal(150, 30, n).clip(0, 320)
    h1_count = rng.choice([0, 1, 1, 1, 2, 3], size=n)
    page_speed = rng.normal(75, 15, n).clip(20, 100)
    has_schema = rng.binomial(1, 0.45, n)
    mobile_friendly = rng.binomial(1, 0.8, n)

    # A transparent scoring function the SEO model will try to recover.
    score = (
        0.020 * np.clip(word_count, 0, 2500)          # content depth (capped)
        + 6.0 * alt_ratio                              # accessibility
        + 0.8 * np.clip(internal_links, 0, 25)         # internal linking
        + 0.25 * page_speed                            # performance
        + 8.0 * has_schema                             # structured data
        + 6.0 * mobile_friendly                        # mobile
        - 6.0 * np.abs(title_length - 55) / 10         # title-length sweet spot
        - 4.0 * np.abs(meta_length - 155) / 30         # meta-length sweet spot
        - 5.0 * np.abs(h1_count - 1)                   # exactly one H1 is ideal
    )
    score = score + rng.normal(0, 4, n)
    # Squash to a 0–100 SEO score.
    seo_score = (100 * (score - score.min()) / (score.max() - score.min())).round(1)

    organic_traffic = np.round(
        np.clip(seo_score, 1, None) ** 1.6 * rng.uniform(0.4, 1.2, n)
    ).astype(int)

    return pd.DataFrame({
        "url": [f"/page/{i:04d}" for i in range(n)],
        "word_count": word_count.round().astype(int),
        "images": images,
        "alt_ratio": alt_ratio.round(3),
        "internal_links": internal_links,
        "external_links": external_links,
        "title_length": title_length.round().astype(int),
        "meta_length": meta_length.round().astype(int),
        "h1_count": h1_count,
        "page_speed": page_speed.round(1),
        "has_schema": has_schema,
        "mobile_friendly": mobile_friendly,
        "seo_score": seo_score,
        "organic_traffic": organic_traffic,
    })


# --------------------------------------------------------------------------
# Per-user behaviour
# --------------------------------------------------------------------------
def generate_users(n: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate per-visitor behaviour with four latent segments."""
    rng = np.random.default_rng(random_state + 2)

    # Latent mixture: new, returning, potential buyer, buyer.
    props = np.array([0.45, 0.30, 0.15, 0.10])
    seg = rng.choice(4, size=n, p=props)

    def per_seg(mapping):
        return np.array([mapping[s] for s in seg])

    sessions = np.maximum(1, rng.poisson(per_seg({0: 1, 1: 6, 2: 4, 3: 9}))).astype(int)
    pageviews = np.maximum(1, rng.poisson(per_seg({0: 2, 1: 15, 2: 12, 3: 30}))).astype(int)
    avg_time = np.clip(rng.normal(per_seg({0: 45, 1: 160, 2: 130, 3: 240}), 30), 5, 900)
    recency = np.clip(rng.normal(per_seg({0: 40, 1: 12, 2: 20, 3: 5}), 8), 0, 365).round().astype(int)
    num_orders = per_seg({0: 0, 1: 0, 2: 0, 3: 3}) + rng.poisson(per_seg({0: 0, 1: 0, 2: 0, 3: 2}))
    num_orders = num_orders.astype(int)
    total_spent = np.round(num_orders * rng.normal(52, 12, n).clip(10, None), 2)

    return pd.DataFrame({
        "user_id": [f"u{i:06d}" for i in range(n)],
        "sessions": sessions,
        "pageviews": pageviews,
        "avg_time_on_site": avg_time.round(1),
        "recency_days": recency,
        "num_orders": num_orders,
        "total_spent": total_spent,
        "_latent_segment": seg,   # kept for validation only; models never see it
    })


# --------------------------------------------------------------------------
# Convenience: generate + persist everything
# --------------------------------------------------------------------------
def generate_all(random_state: int = RANDOM_STATE, save: bool = True):
    """Generate all three tables and (optionally) write them to ``datasets/``."""
    from ..config import DAILY_METRICS_CSV, PAGES_CSV, USERS_CSV

    daily = generate_daily_metrics(random_state=random_state)
    pages = generate_pages(random_state=random_state)
    users = generate_users(random_state=random_state)

    if save:
        daily.to_csv(DAILY_METRICS_CSV, index=False)
        pages.to_csv(PAGES_CSV, index=False)
        users.to_csv(USERS_CSV, index=False)

    return daily, pages, users


if __name__ == "__main__":
    d, p, u = generate_all()
    print(f"daily_metrics: {d.shape}")
    print(f"pages        : {p.shape}")
    print(f"users        : {u.shape}")
