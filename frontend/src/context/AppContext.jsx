import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import {
  fetchDocuments,
  fetchIngestStatus,
  fetchStatus,
  startIngestion,
  uploadPdfs,
} from '../api/client'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [status, setStatus] = useState(null)
  const [documents, setDocuments] = useState([])
  const [error, setError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [ingestRunning, setIngestRunning] = useState(false)
  const [ingestMessage, setIngestMessage] = useState('')
  const [uploadNotice, setUploadNotice] = useState(null)
  const pollRef = useRef(null)

  const loadStatus = useCallback(async () => {
    try {
      const data = await fetchStatus()
      setStatus(data)
      setError(null)
      return data
    } catch (e) {
      setError('Cannot reach API. Start the backend: uvicorn api:app --reload')
      setStatus(null)
      return null
    }
  }, [])

  const loadDocuments = useCallback(async () => {
    try {
      const data = await fetchDocuments()
      setDocuments(data.documents || [])
      return data.documents || []
    } catch {
      return []
    }
  }, [])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback(() => {
    stopPolling()
    setIngestRunning(true)
    pollRef.current = setInterval(async () => {
      try {
        const ingest = await fetchIngestStatus()
        setIngestMessage(ingest.message || '')
        if (!ingest.running) {
          stopPolling()
          setIngestRunning(false)
          if (ingest.error) {
            setError(ingest.error)
          } else {
            await loadStatus()
            await loadDocuments()
          }
        }
      } catch {
        stopPolling()
        setIngestRunning(false)
      }
    }, 2000)
  }, [loadDocuments, loadStatus, stopPolling])

  useEffect(() => {
    loadStatus()
    loadDocuments()
    fetchIngestStatus()
      .then((s) => {
        if (s.running) {
          setIngestMessage(s.message)
          setIngestRunning(true)
          startPolling()
        }
      })
      .catch(() => {})
    return stopPolling
  }, [loadStatus, loadDocuments, startPolling, stopPolling])

  // Refresh status periodically while ingesting (chat stays enabled)
  useEffect(() => {
    if (!ingestRunning) return undefined
    const id = setInterval(loadStatus, 5000)
    return () => clearInterval(id)
  }, [ingestRunning, loadStatus])

  const handleUpload = async (files) => {
    setUploading(true)
    setError(null)
    setUploadNotice(null)
    try {
      const result = await uploadPdfs(files, { ingest: true })
      await loadDocuments()
      setUploadNotice({
        count: result.count,
        files: result.uploaded,
        indexing: result.ingestion_started,
      })
      if (result.ingestion_started) {
        startPolling()
      } else {
        await loadStatus()
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const handleIngest = async () => {
    setError(null)
    try {
      await startIngestion(false)
      startPolling()
    } catch (e) {
      setError(e.message)
    }
  }

  const refresh = () => {
    loadStatus()
    loadDocuments()
  }

  const chatAvailable = Boolean(status?.chat_available)
  const indexUpToDate = status?.status === 'ready' && !ingestRunning

  return (
    <AppContext.Provider
      value={{
        status,
        documents,
        error,
        setError,
        uploading,
        ingestRunning,
        ingestMessage,
        uploadNotice,
        setUploadNotice,
        chatAvailable,
        indexUpToDate,
        loadStatus,
        loadDocuments,
        handleUpload,
        handleIngest,
        refresh,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
