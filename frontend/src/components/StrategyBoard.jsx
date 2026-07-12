const COLUMNS = [
  { key: 'today', title: 'Today', accent: 'border-red-500' },
  { key: 'this_week', title: 'This Week', accent: 'border-orange-500' },
  { key: 'this_month', title: 'This Month', accent: 'border-blue-500' },
]

export default function StrategyBoard({ data }) {
  const plan = data || { today: [], this_week: [], this_month: [] }
  return (
    <div className="bg-panel border border-line rounded-xl p-4">
      <div className="text-sm text-gray-300 mb-3">AI Growth Strategy</div>
      <div className="grid gap-4 md:grid-cols-3">
        {COLUMNS.map((col) => (
          <div key={col.key} className={`border-l-2 ${col.accent} pl-3`}>
            <div className="font-semibold mb-2">{col.title}</div>
            {(plan[col.key] || []).length === 0 && (
              <div className="text-gray-500 text-sm">Nothing pressing</div>
            )}
            {(plan[col.key] || []).map((r, i) => (
              <div key={i} className="mb-3 text-sm">
                <div>
                  <span className="text-accent">[{r.area}]</span> {r.action}
                </div>
                <div className="text-gray-400 text-xs">
                  {r.problem} — <span className="text-green-400">{r.expected_gain}</span>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
