import ReactMarkdown from 'react-markdown'

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-2">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 rounded-full bg-accent animate-typing"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  )
}

export default function Message({ role, content, sources, timing, cacheHit, loading }) {
  const isUser = role === 'user'

  return (
    <div
      className={`flex animate-fade-up ${isUser ? 'justify-end' : 'justify-start'}`}
      style={{ animationDelay: '0.05s' }}
    >
      <div className={`max-w-[85%] sm:max-w-[75%] ${isUser ? 'order-1' : ''}`}>
        {!isUser && (
          <div className="mb-1.5 flex items-center gap-2 px-1">
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-accent/20 ring-1 ring-accent/30">
              <svg className="h-3.5 w-3.5 text-accent-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <span className="text-xs font-medium text-ink-500">RAG Assistant</span>
            {cacheHit && (
              <span className="rounded bg-ink-800 px-1.5 py-0.5 text-[10px] text-ink-400">cached</span>
            )}
          </div>
        )}

        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-accent text-white shadow-glow'
              : 'glass shadow-card'
          }`}
        >
          {loading ? (
            <TypingIndicator />
          ) : isUser ? (
            <p className="text-sm leading-relaxed">{content}</p>
          ) : (
            <div className="prose-rag text-sm">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          )}
        </div>

        {!isUser && !loading && timing?.total_ms != null && (
          <p className="mt-1.5 px-1 text-[11px] text-ink-600">
            {Math.round(timing.total_ms)}ms
            {timing.retrieval_ms != null && ` · retrieval ${Math.round(timing.retrieval_ms)}ms`}
            {timing.generation_ms != null && ` · generation ${Math.round(timing.generation_ms)}ms`}
          </p>
        )}

        {!isUser && !loading && sources?.length > 0 && (
          <details className="mt-2 group">
            <summary className="cursor-pointer list-none px-1 text-xs text-accent-light hover:text-accent">
              <span className="inline-flex items-center gap-1">
                <svg className="h-3 w-3 transition group-open:rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                {sources.length} source{sources.length !== 1 ? 's' : ''}
              </span>
            </summary>
            <div className="mt-2 space-y-2">
              {sources.map((src, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-ink-700/50 bg-ink-900/50 p-3 text-xs leading-relaxed text-ink-400"
                >
                  <span className="mb-1 block font-medium text-ink-500">Chunk {i + 1}</span>
                  {src.length > 400 ? `${src.slice(0, 400)}…` : src}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}
