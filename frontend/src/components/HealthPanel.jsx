export default function HealthPanel({ data }) {
  const dims = data?.dimensions || {}
  return (
    <div className="bg-panel border border-line rounded-xl p-4">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-sm text-gray-300">Website Health</div>
        <div className="text-sm">
          <b>{data?.overall ?? '—'}</b>/100 · grade {data?.grade ?? '—'}
        </div>
      </div>
      {Object.entries(dims).map(([name, val]) => (
        <div key={name} className="flex items-center gap-3 my-2 text-sm">
          <span className="w-24 text-gray-400">{name}</span>
          <div className="flex-1 h-2.5 bg-ink rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{ width: `${val}%`, background: 'linear-gradient(90deg,#4f8cff,#2ecc71)' }}
            />
          </div>
          <b className="w-8 text-right">{Math.round(val)}</b>
        </div>
      ))}
    </div>
  )
}
