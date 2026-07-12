export default function SegmentsTable({ data }) {
  const rows = data || []
  return (
    <div className="bg-panel border border-line rounded-xl p-4">
      <div className="text-sm text-gray-300 mb-3">User Segments</div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 text-left">
              <th className="py-1.5 pr-3">Segment</th>
              <th className="py-1.5 pr-3">Users</th>
              <th className="py-1.5 pr-3">Avg spend</th>
              <th className="py-1.5">Est. CLV</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.segment_label} className="border-t border-line">
                <td className="py-1.5 pr-3">{r.segment_label}</td>
                <td className="py-1.5 pr-3">{r.users}</td>
                <td className="py-1.5 pr-3">${Number(r.avg_spent).toFixed(0)}</td>
                <td className="py-1.5 font-semibold">${Number(r.est_clv).toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
