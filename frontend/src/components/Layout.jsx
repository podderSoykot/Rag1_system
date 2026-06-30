import { NavLink, Outlet } from 'react-router-dom'
import StatusBadge from './StatusBadge'
import { useApp } from '../context/AppContext'

const navClass = ({ isActive }) =>
  `rounded-lg px-3 py-1.5 text-sm font-medium transition ${
    isActive
      ? 'bg-accent/20 text-accent-light ring-1 ring-accent/30'
      : 'text-ink-400 hover:bg-ink-800 hover:text-ink-200'
  }`

export default function Layout() {
  const {
    status,
    error,
    ingestRunning,
    ingestMessage,
    documents,
    handleIngest,
  } = useApp()

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-ink-800/80 glass">
        <div className="mx-auto flex max-w-4xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <NavLink to="/" className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-glow shadow-glow">
                <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div className="hidden sm:block">
                <h1 className="font-display text-lg font-semibold tracking-tight text-white">
                  RAG System
                </h1>
                <p className="text-xs text-ink-500">Document intelligence</p>
              </div>
            </NavLink>
          </div>

          <nav className="flex items-center gap-1">
            <NavLink to="/" end className={navClass}>
              Chat
            </NavLink>
            <NavLink to="/documents" className={navClass}>
              Documents
            </NavLink>
          </nav>

          <div className="flex items-center gap-2 sm:gap-3">
            {status && (
              <StatusBadge
                status={status.status}
                backend={status.generation_backend}
                reason={status.reason}
                ingestRunning={ingestRunning}
              />
            )}
            {status?.status === 'needs_ingestion' && !ingestRunning && documents.length > 0 && (
              <button
                type="button"
                onClick={handleIngest}
                className="hidden rounded-lg bg-amber-500/15 px-3 py-1.5 text-xs font-medium text-amber-400 ring-1 ring-amber-500/30 transition hover:bg-amber-500/25 sm:inline-block"
              >
                Re-index
              </button>
            )}
          </div>
        </div>
      </header>

      {error && (
        <div className="border-b border-red-500/20 bg-red-500/10 px-4 py-2 text-center text-sm text-red-300">
          {error}
        </div>
      )}

      <Outlet />

      <footer className="border-t border-ink-800/50 py-3 text-center text-[11px] text-ink-600">
        Upload · Index · Ask · LangGraph · MCP ready
      </footer>
    </div>
  )
}
