import { useCallback, useRef, useState } from 'react'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function UploadPanel({
  documents,
  onUpload,
  uploading,
  ingestRunning,
  ingestMessage,
  onRefresh,
  expanded = false,
}) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleFiles = useCallback(
    (fileList) => {
      const pdfs = Array.from(fileList).filter((f) =>
        f.name.toLowerCase().endsWith('.pdf'),
      )
      if (pdfs.length) onUpload(pdfs)
    },
    [onUpload],
  )

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div className="glass animate-fade-up rounded-2xl p-5 shadow-card">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="font-display text-base font-semibold text-white">Documents</h3>
          <p className="text-xs text-ink-500">Upload PDFs — they are indexed automatically</p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="rounded-lg p-2 text-ink-500 transition hover:bg-ink-800 hover:text-ink-300"
          aria-label="Refresh documents"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed px-6 text-center transition ${
          expanded ? 'py-14' : 'py-8'
        } ${
          dragOver
            ? 'border-accent bg-accent/10'
            : 'border-ink-700/80 bg-ink-900/40 hover:border-accent/40 hover:bg-ink-800/40'
        } ${uploading ? 'pointer-events-none opacity-60' : ''}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-accent/15 ring-1 ring-accent/25">
          {uploading ? (
            <svg className="h-6 w-6 animate-spin text-accent-light" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="h-6 w-6 text-accent-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          )}
        </div>
        <p className="text-sm font-medium text-ink-200">
          {uploading ? 'Uploading…' : 'Drop PDFs here or click to browse'}
        </p>
        <p className="mt-1 text-xs text-ink-500">Multiple files supported</p>
      </div>

      {/* Ingestion progress */}
      {ingestRunning && (
        <div className="mt-4 flex items-center gap-3 rounded-xl bg-accent/10 px-4 py-3 ring-1 ring-accent/20">
          <svg className="h-4 w-4 shrink-0 animate-spin text-accent-light" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className="text-xs text-accent-light">{ingestMessage || 'Indexing documents…'}</p>
        </div>
      )}

      {/* Document list */}
      {documents.length > 0 && (
        <ul className={`mt-4 space-y-2 overflow-y-auto ${expanded ? 'max-h-96' : 'max-h-40'}`}>
          {documents.map((doc) => (
            <li
              key={doc.name}
              className="flex items-center gap-3 rounded-lg bg-ink-900/60 px-3 py-2 ring-1 ring-ink-700/40"
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-red-500/10 text-red-400">
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 2l5 5h-5V4zM8 12h8v2H8v-2zm0 4h5v2H8v-2z" />
                </svg>
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-ink-200">{doc.name}</p>
                <p className="text-[11px] text-ink-500">{formatSize(doc.size_bytes)}</p>
              </div>
            </li>
          ))}
        </ul>
      )}

      {documents.length === 0 && !uploading && (
        <p className="mt-4 text-center text-xs text-ink-600">No documents yet — upload a PDF to get started</p>
      )}
    </div>
  )
}
