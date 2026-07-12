import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export default function ForecastChart({ data }) {
  const rows = (data?.forecast || []).map((d) => ({
    date: d.date,
    visitors: d.predicted_visitors,
  }))

  return (
    <div className="bg-panel border border-line rounded-xl p-4">
      <div className="text-sm text-gray-300 mb-3">
        Predicted daily visitors — next {data?.horizon || rows.length} days
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4f8cff" stopOpacity={0.5} />
              <stop offset="100%" stopColor="#4f8cff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#232a34" vertical={false} />
          <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 11 }} minTickGap={40} />
          <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} width={48} />
          <Tooltip
            contentStyle={{ background: '#161b22', border: '1px solid #232a34', borderRadius: 8, color: '#e6edf3' }}
          />
          <Area type="monotone" dataKey="visitors" stroke="#4f8cff" strokeWidth={2.5} fill="url(#g)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
