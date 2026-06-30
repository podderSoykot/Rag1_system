import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { sendQuery, sendResearch } from '../api/client'
import { useApp } from '../context/AppContext'
import InputBar from '../components/InputBar'
import Message from '../components/Message'
import SuggestedQuestions from '../components/SuggestedQuestions'

export default function ChatPage() {
  const { chatAvailable, ingestRunning, ingestMessage } = useApp()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [topK, setTopK] = useState(3)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const isResearchCommand = (text) => {
    const t = text.trim().toLowerCase()
    return (
      t.startsWith('/research') ||
      t.startsWith('research on ') ||
      t.startsWith('research about ')
    )
  }

  const handleSubmit = async (text) => {
    const raw = (text ?? input).trim()
    if (!raw || loading) return

    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: raw }])
    setLoading(true)

    try {
      if (isResearchCommand(raw)) {
        const result = await sendResearch(raw)
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `## Research: ${result.topic}\n\n${result.answer}`,
            sources: result.sources,
            timing: result.timing,
          },
        ])
      } else {
        const result = await sendQuery(raw, topK)
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: result.answer || 'No answer generated.',
            sources: result.sources,
            timing: result.timing,
            cacheHit: result.cache_hit,
          },
        ])
      }
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `**Error:** ${e.message}`,
          sources: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const showHero = messages.length === 0 && !loading
  const chatDisabled = loading || !chatAvailable

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-4 py-6 sm:px-6">
      {ingestRunning && chatAvailable && (
        <div className="mb-4 flex items-center gap-3 rounded-xl border border-accent/20 bg-accent/10 px-4 py-3 text-sm text-accent-light">
          <svg className="h-4 w-4 shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span>
            {ingestMessage || 'Indexing new documents…'} You can keep chatting — answers use the current index until indexing finishes.
          </span>
        </div>
      )}

      {showHero && (
        <div className="mb-8 animate-fade-up text-center">
          <h2 className="font-display text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Ask your documents
            <span className="block bg-gradient-to-r from-accent-light to-mint bg-clip-text text-transparent">
              anything.
            </span>
          </h2>
          <p className="mx-auto mt-4 max-w-md text-sm leading-relaxed text-ink-400">
            Powered by LangGraph retrieval. Use <code className="text-accent-light">/research topic</code> for a deep multi-angle report.
          </p>

          {!chatAvailable && (
            <div className="mx-auto mt-6 max-w-sm rounded-xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              {ingestRunning ? (
                'Building index for the first time… chat will unlock when indexing completes.'
              ) : (
                <>
                  No index yet.{' '}
                  <Link to="/documents" className="font-medium text-amber-200 underline hover:text-white">
                    Upload documents
                  </Link>{' '}
                  to get started.
                </>
              )}
            </div>
          )}

          {chatAvailable && (
            <div className="mt-8">
              <SuggestedQuestions onSelect={handleSubmit} disabled={chatDisabled} />
            </div>
          )}
        </div>
      )}

      <div className="flex-1 space-y-6 pb-4">
        {messages.map((msg, i) => (
          <Message key={i} {...msg} />
        ))}
        {loading && <Message role="assistant" loading />}
        <div ref={bottomRef} />
      </div>

      <div className="sticky bottom-0 pb-4 pt-2">
        <InputBar
          value={input}
          onChange={setInput}
          onSubmit={() => handleSubmit()}
          disabled={chatDisabled}
          topK={topK}
          onTopKChange={setTopK}
        />
        {chatDisabled && !loading && (
          <p className="mt-2 text-center text-[11px] text-ink-600">
            {ingestRunning && !chatAvailable
              ? 'First-time indexing in progress…'
              : !chatAvailable && (
                  <>
                    Upload PDFs on the{' '}
                    <Link to="/documents" className="text-accent-light hover:underline">
                      Documents
                    </Link>{' '}
                    page first
                  </>
                )}
          </p>
        )}
      </div>
    </main>
  )
}
