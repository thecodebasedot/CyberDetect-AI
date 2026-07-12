const PILL = {
  1: 'bg-red-500',
  2: 'bg-orange-500',
  3: 'bg-yellow-400 text-black',
  4: 'bg-blue-500',
  5: 'bg-gray-500',
}

export default function Recommendations({ data }) {
  const recs = data || []
  return (
    <div className="bg-panel border border-line rounded-xl p-4">
      <div className="text-sm text-gray-300 mb-3">AI Recommendations</div>
      {recs.length === 0 && <div className="text-gray-400 text-sm">No issues found.</div>}
      {recs.map((r, i) => (
        <div key={i} className="flex gap-3 py-3 border-t border-line first:border-t-0">
          <span
            className={`h-5 px-2 rounded-full text-xs font-bold inline-flex items-center ${
              PILL[Math.min(r.priority, 5)] || PILL[5]
            }`}
          >
            P{r.priority}
          </span>
          <div className="min-w-0">
            <div className="font-semibold">
              {r.area} — {r.problem}
            </div>
            <div className="text-gray-400 text-sm">{r.reason}</div>
            <div className="text-sm">
              ➜ {r.action} <span className="text-green-400 text-xs">{r.expected_gain}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
