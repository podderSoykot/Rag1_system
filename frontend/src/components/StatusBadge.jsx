export default function StatusBadge({ status, backend, reason, ingestRunning }) {
  const indexing = status === 'indexing' || ingestRunning
  const ready = status === 'ready' && !indexing
  const needsIngestion = status === 'needs_ingestion'

  let label = 'Index ready'
  let style = 'bg-mint/10 text-mint ring-1 ring-mint/30'
  let dot = 'bg-mint animate-pulse-soft'

  if (indexing) {
    label = 'Indexing…'
    style = 'bg-accent/10 text-accent-light ring-1 ring-accent/30'
    dot = 'bg-accent-light animate-pulse-soft'
  } else if (needsIngestion) {
    label = 'Needs ingestion'
    style = 'bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/30'
    dot = 'bg-amber-400'
  }

  return (
    <div className="flex items-center gap-3">
      <div className={`flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${style}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        {label}
      </div>
      {backend && (
        <span className="hidden text-xs text-ink-500 sm:inline">{backend}</span>
      )}
      {needsIngestion && reason && (
        <span className="hidden max-w-xs truncate text-xs text-ink-500 lg:inline" title={reason}>
          {reason}
        </span>
      )}
    </div>
  )
}
