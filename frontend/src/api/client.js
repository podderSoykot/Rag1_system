const API_BASE = '/api'

async function request(path, options = {}) {
  const headers = { ...options.headers }
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    const detail = data.detail
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg).join(', ')
          : data.message || `Request failed (${res.status})`
    throw new Error(message)
  }

  return data
}

export function fetchStatus() {
  return request('/status')
}

export function fetchDocuments() {
  return request('/documents')
}

export function fetchIngestStatus() {
  return request('/ingest/status')
}

export function sendQuery(query, topK = 3) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  })
}

export function startIngestion(force = false) {
  return request('/ingest', {
    method: 'POST',
    body: JSON.stringify({ force }),
  })
}

export function uploadPdfs(files, { ingest = true } = {}) {
  const form = new FormData()
  for (const file of files) {
    form.append('files', file)
  }
  const qs = ingest ? '?ingest=true' : '?ingest=false'
  return request(`/upload${qs}`, {
    method: 'POST',
    body: form,
  })
}
