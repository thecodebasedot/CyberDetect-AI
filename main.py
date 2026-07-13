#!/usr/bin/env python3
"""GrowthMind AI — command-line interface.

Predict. Optimize. Grow.

An AI growth-intelligence engine: forecasts traffic & sales, scores SEO health,
segments users, detects traffic anomalies, and turns it all into a prioritized
growth strategy.

Examples
--------
  python main.py generate                 # build the synthetic datasets
  python main.py train                    # fit & save all 5 models
  python main.py analyze --horizon 30     # print the full growth report
  python main.py dashboard                # write dashboard/index.html
  python main.py demo                     # generate -> train -> analyze -> dashboard
  python main.py forecast-eval            # backtest XGBoost vs Holt-Winters vs Prophet
  python main.py agent                     # run one Autonomous Growth Agent cycle
  python main.py serve                    # launch the FastAPI backend
"""

from __future__ import annotations

import argparse
import sys


def cmd_generate(args: argparse.Namespace) -> None:
    from growthmind.data import generate_all
    daily, pages, users = generate_all(save=True)
    print("Generated synthetic datasets in datasets/:")
    print(f"  daily_metrics.csv : {daily.shape[0]:>5} rows x {daily.shape[1]} cols")
    print(f"  pages.csv         : {pages.shape[0]:>5} rows x {pages.shape[1]} cols")
    print(f"  users.csv         : {users.shape[0]:>5} rows x {users.shape[1]} cols")


def cmd_train(args: argparse.Namespace) -> None:
    from growthmind.pipeline import train_all
    print("Training GrowthMind AI models ...\n")
    report = train_all(regenerate=args.regenerate)
    print(report.pretty())
    print("\nModels saved to models/.")


def cmd_analyze(args: argparse.Namespace) -> None:
    from growthmind.pipeline import analyze
    insights = analyze(horizon=args.horizon)
    _print_insights(insights)


def cmd_dashboard(args: argparse.Namespace) -> None:
    from growthmind.pipeline import analyze
    from growthmind.report import write_dashboard
    insights = analyze(horizon=args.horizon)
    path = write_dashboard(insights)
    print(f"Dashboard written to {path}")
    print("Open it in a browser to explore the report.")


def cmd_demo(args: argparse.Namespace) -> None:
    from growthmind.data import generate_all
    from growthmind.pipeline import analyze, train_all
    from growthmind.report import write_dashboard

    print("=" * 60)
    print(" GrowthMind AI — end-to-end demo")
    print("=" * 60)
    generate_all(save=True)
    print("[1/3] datasets generated")
    report = train_all(regenerate=False)
    print("[2/3] models trained\n")
    print(report.pretty())
    print("\n[3/3] analysis\n")
    insights = analyze(horizon=args.horizon)
    _print_insights(insights)
    path = write_dashboard(insights)
    print(f"\nDashboard: {path}")


def cmd_customers(args: argparse.Namespace) -> None:
    from growthmind.models import CustomerIntelligence
    from growthmind.pipeline import load_datasets

    _, _, users = load_datasets()
    ci = CustomerIntelligence.load()
    summary = ci.summary(users)

    print("Customer Intelligence")
    print("---------------------")
    print(f"  Customers                : {summary['customers']:,}")
    print(f"  Avg purchase propensity  : {summary['avg_purchase_prob']:.1%}")
    print(f"  Avg churn risk           : {summary['avg_churn_risk']:.1%}")
    print(f"  High-churn customers     : {summary['high_churn_customers']:,}")
    print(f"  Revenue at risk          : ${summary['revenue_at_risk']:,.0f}")
    print(f"  Avg predicted CLV        : ${summary['avg_predicted_clv']:,.2f}")
    print(f"  Conversion opportunities : {summary['conversion_opportunities']:,}")

    scored = ci.score(users)
    print("\nTop 10 at-risk high-value customers:")
    top = (scored.sort_values(["churn_risk", "predicted_clv"], ascending=False)
           .head(10)[["user_id", "purchase_prob", "churn_risk", "predicted_clv"]])
    print(top.to_string(index=False))


def cmd_forecast_eval(args: argparse.Namespace) -> None:
    from growthmind.forecasting import compare_forecasters
    from growthmind.pipeline import load_datasets

    daily, _, _ = load_datasets()
    print(f"Rolling-origin backtest — horizon={args.horizon}d, folds={args.folds}\n"
          "(training window grows each fold; lower is better)\n")
    table = compare_forecasters(daily, horizon=args.horizon, folds=args.folds)
    print(table.to_string(index=False))
    best = table.iloc[0]["model"]
    print(f"\nBest model by MAE: {best}")


def cmd_agent(args: argparse.Namespace) -> None:
    from growthmind.agent import AutonomousGrowthAgent
    from growthmind.connectors import available_sources, get_connector

    if args.source not in available_sources():
        print(f"Unknown source '{args.source}'. Available: {', '.join(available_sources())}")
        raise SystemExit(2)

    connector = get_connector(args.source)
    if not connector.is_available():
        print(f"Source '{args.source}' is not available (missing credentials). "
              "Falling back to --source local.")
        connector = get_connector("local")

    agent = AutonomousGrowthAgent(connector=connector, autonomy=args.autonomy)
    print(f"Running Autonomous Growth Agent (source={connector.name}, "
          f"autonomy={args.autonomy}) ...\n")
    report = agent.run_cycle(horizon=args.horizon)
    print(report.markdown())
    print(f"\nReport saved to {report.report_path}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    print(f"Starting GrowthMind AI API on http://{args.host}:{args.port} ...")
    print("Docs at /docs")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


def _print_insights(insights) -> None:
    k = insights.kpis
    print("Key metrics")
    print("-----------")
    print(f"  Health score           : {k['health_score']:.0f}/100 (grade {k['health_grade']})")
    print(f"  Avg daily visitors     : {k['avg_daily_visitors']:,}")
    print(f"  {k['forecast_horizon_days']}-day forecast (avg)    : "
          f"{k['forecast_avg_visitors']:,} "
          f"({k['predicted_traffic_change_pct']:+.1f}%)")
    print(f"  Monthly revenue        : ${k['monthly_revenue']:,.0f}")
    print(f"  Conversion rate        : {k['avg_conversion_rate']:.2%}")
    print(f"  Anomalous days flagged : {k['n_anomalies']}")
    if "revenue_at_risk" in k:
        print(f"  Revenue at churn risk  : ${k['revenue_at_risk']:,.0f} "
              f"({k['high_churn_customers']} customers)")
        print(f"  Conversion opportunities: {k['conversion_opportunities']}")
    print()
    print(insights.health.pretty())
    print()
    print("User segments")
    print("-------------")
    print(insights.segments.to_string(index=False))
    print()
    print(insights.strategy.pretty())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="growthmind",
        description="GrowthMind AI — Predict. Optimize. Grow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate synthetic datasets")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="train and save all models")
    p.add_argument("--regenerate", action="store_true", help="regenerate datasets first")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("analyze", help="print the full growth intelligence report")
    p.add_argument("--horizon", type=int, default=30, help="forecast horizon in days")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("dashboard", help="write the HTML dashboard")
    p.add_argument("--horizon", type=int, default=30)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("demo", help="run generate -> train -> analyze -> dashboard")
    p.add_argument("--horizon", type=int, default=30)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("customers",
                       help="customer intelligence: purchase / churn / CLV predictions")
    p.set_defaults(func=cmd_customers)

    p = sub.add_parser("forecast-eval",
                       help="backtest & compare traffic forecasters (XGBoost/Holt-Winters/Prophet)")
    p.add_argument("--horizon", type=int, default=14, help="forecast horizon per fold")
    p.add_argument("--folds", type=int, default=3, help="number of walk-forward folds")
    p.set_defaults(func=cmd_forecast_eval)

    p = sub.add_parser("agent", help="run one Autonomous Growth Agent cycle")
    p.add_argument("--source", default="local",
                   help="data source: local | gsc | ga4 (default: local)")
    p.add_argument("--autonomy", choices=["propose", "auto"], default="propose",
                   help="'propose' (human approves) or 'auto' (simulate low-risk actions)")
    p.add_argument("--horizon", type=int, default=30)
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("serve", help="launch the FastAPI backend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
