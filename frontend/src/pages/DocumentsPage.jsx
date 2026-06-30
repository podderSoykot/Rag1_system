import { Link } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import UploadPanel from '../components/UploadPanel'

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function DocumentsPage() {
  const {
    documents,
    uploading,
    ingestRunning,
    ingestMessage,
    uploadNotice,
    indexUpToDate,
    chatAvailable,
    handleUpload,
    handleIngest,
    refresh,
  } = useApp()

  const totalBytes = documents.reduce((sum, d) => sum + d.size_bytes, 0)

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-8 sm:px-6">
      <div className="mb-8 animate-fade-up">
        <h2 className="font-display text-3xl font-semibold tracking-tight text-white">
          Document Library
        </h2>
        <p className="mt-2 max-w-lg text-sm text-ink-400">
          Upload PDF textbooks and papers. Files are saved and indexed automatically so you can query them in Chat.
        </p>
      </div>

      {/* Stats */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="glass rounded-xl p-4 shadow-card">
          <p className="text-2xl font-semibold text-white">{documents.length}</p>
          <p className="text-xs text-ink-500">Documents</p>
        </div>
        <div className="glass rounded-xl p-4 shadow-card">
          <p className="text-2xl font-semibold text-white">{formatSize(totalBytes)}</p>
          <p className="text-xs text-ink-500">Total size</p>
        </div>
        <div className="glass col-span-2 rounded-xl p-4 shadow-card sm:col-span-1">
          <p className={`text-2xl font-semibold ${
            indexUpToDate ? 'text-mint' : ingestRunning ? 'text-accent-light' : chatAvailable ? 'text-amber-300' : 'text-amber-400'
          }`}>
            {ingestRunning ? 'Indexing' : indexUpToDate ? 'Ready' : chatAvailable ? 'Updating' : 'Pending'}
          </p>
          <p className="text-xs text-ink-500">Index status</p>
        </div>
      </div>

      {uploadNotice && (
        <div className="mb-6 animate-fade-up rounded-xl border border-mint/20 bg-mint/10 px-4 py-3 text-sm text-mint">
          Uploaded <strong>{uploadNotice.count}</strong> file(s): {uploadNotice.files.join(', ')}.
          {uploadNotice.indexing && ' Indexing started.'}
        </div>
      )}

      <UploadPanel
        documents={documents}
        onUpload={handleUpload}
        uploading={uploading}
        ingestRunning={ingestRunning}
        ingestMessage={ingestMessage}
        onRefresh={refresh}
        expanded
      />

      <div className="mt-6 flex flex-wrap items-center gap-3">
        {!ingestRunning && documents.length > 0 && !indexUpToDate && (
          <button
            type="button"
            onClick={handleIngest}
            className="rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-white transition hover:bg-accent-light"
          >
            Run indexing
          </button>
        )}
        {chatAvailable && (
          <Link
            to="/"
            className="inline-flex items-center gap-2 rounded-xl bg-mint/15 px-4 py-2.5 text-sm font-medium text-mint ring-1 ring-mint/30 transition hover:bg-mint/25"
          >
            {ingestRunning ? 'Chat (current index)' : 'Go to Chat'}
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        )}
      </div>

      {/* How it works */}
      <div className="mt-10 glass rounded-2xl p-6 shadow-card">
        <h3 className="font-display text-base font-semibold text-white">How it works</h3>
        <ol className="mt-4 space-y-3 text-sm text-ink-400">
          <li className="flex gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/20 text-xs font-medium text-accent-light">1</span>
            Upload one or more PDF files using the drop zone above
          </li>
          <li className="flex gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/20 text-xs font-medium text-accent-light">2</span>
            The system extracts text, creates chunks, and builds vector indexes
          </li>
          <li className="flex gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/20 text-xs font-medium text-accent-light">3</span>
            When the index shows <span className="text-mint">Ready</span>, all documents are searchable. You can chat during indexing if an index already exists.
          </li>
        </ol>
      </div>
    </main>
  )
}
