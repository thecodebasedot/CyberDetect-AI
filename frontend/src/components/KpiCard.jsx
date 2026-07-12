export default function KpiCard({ label, value, sub, accent }) {
  return (
    <div className="bg-panel border border-line rounded-xl p-4">
      <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
      <div className="text-2xl font-bold mt-1.5" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  )
}
