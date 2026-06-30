export default function InputBar({ value, onChange, onSubmit, disabled, topK, onTopKChange }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <div className="glass rounded-2xl p-2 shadow-card ring-1 ring-ink-700/40">
      <div className="flex items-end gap-2">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Ask anything about your documents…"
          rows={1}
          className="max-h-32 min-h-[44px] flex-1 resize-none bg-transparent px-3 py-2.5 text-sm text-ink-200 placeholder:text-ink-600 focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled || !value.trim()}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent text-white transition hover:bg-accent-light disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Send"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>
      <div className="mt-1 flex items-center justify-between px-3 pb-1">
        <span className="text-[11px] text-ink-600">Enter to send · Shift+Enter for new line</span>
        <label className="flex items-center gap-2 text-[11px] text-ink-500">
          Chunks
          <select
            value={topK}
            onChange={(e) => onTopKChange(Number(e.target.value))}
            disabled={disabled}
            className="rounded-md border border-ink-700 bg-ink-800 px-2 py-0.5 text-ink-300 focus:outline-none focus:ring-1 focus:ring-accent"
          >
            {[1, 2, 3, 5, 7, 10].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
      </div>
    </div>
  )
}
