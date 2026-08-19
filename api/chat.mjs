/**
 * Insightbase /api/chat — retrieval-augmented Q&A over user-provided docs.
 * LLM mode activates when LLM_API_KEY, LLM_BASE_URL and LLM_MODEL env vars are set
 * (OpenAI-compatible /chat/completions). Otherwise a built-in retrieval-only
 * demo mode returns cited excerpts from the most relevant chunks.
 */

export const maxDuration = 30

function ok(res, payload) {
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.status(200).json(payload)
}

function bad(res, message, status = 400) {
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.status(status).json({ ok: false, error: message })
}

function splitSentences(text) {
  return String(text || '')
    .replace(/([.!?。！？…])\s*/g, '$1\u0001')
    .split('\u0001')
    .map((s) => s.trim())
    .filter(Boolean)
}

function chunkText(text, maxLen = 900) {
  const chunks = []
  let current = ''
  for (const sentence of splitSentences(text)) {
    if ((current + ' ' + sentence).length > maxLen && current) {
      chunks.push(current.trim())
      current = sentence
    } else {
      current = current ? current + ' ' + sentence : sentence
    }
  }
  if (current.trim()) chunks.push(current.trim())
  return chunks
}

function tokenize(text) {
  const t = String(text || '').toLowerCase()
  const tokens = []
  const words = t.match(/[a-z0-9][a-z0-9_.+-]*/g)
  if (words) tokens.push(...words.filter((w) => w.length > 1 && !/^\d+$/.test(w)))
  for (const run of t.match(/[\u4e00-\u9fff]+/g) || []) {
    if (run.length > 0) {
      if (run.length === 1) tokens.push(run)
      else {
        for (let i = 0; i < run.length - 1; i++) tokens.push(run.slice(i, i + 2))
        tokens.push(run)
      }
    }
  }
  return tokens
}

function scoreChunks(query, chunks) {
  const qTokens = tokenize(query)
  if (qTokens.length === 0) return chunks.map((c, i) => ({ chunk: c, score: 1 / (1 + i), terms: [] }))
  const qf = Object.create(null)
  for (const tok of qTokens) qf[tok] = (qf[tok] || 0) + 1
  const avgLen = Math.max(1, chunks.reduce((n, c) => n + tokenize(c.text).length, 0) / Math.max(1, chunks.length))
  const boostTitle = 1.6
  return chunks.map((c) => {
    const titleTokens = tokenize(c.title || '')
    const bodyTokens = tokenize(c.text || '')
    let score = 0
    const terms = []
    for (const tok of Object.keys(qf)) {
      const inTitle = tok.length > 1 && titleTokens.filter((x) => x === tok).length
      const inBody = tok.length > 1 ? bodyTokens.filter((x) => x === tok).length : 0
      if (inTitle > 0 || inBody > 0) {
        const count = inBody + inTitle * boostTitle
        const idf = 1 + Math.log((chunks.length + 1) / (1 + chunks.filter((o) => tokenize(o.text).includes(tok)).length))
        score += idf * (count * 2.2) / (count + 1.2 * (bodyTokens.length / Math.max(1, avgLen)))
        terms.push(tok)
      }
    }
    return { chunk: c, score, terms }
  }).sort((a, b) => b.score - a.score)
}

function compactExcerpts(scored, maxTotal = 2400) {
  const out = []
  let budget = maxTotal
  for (const item of scored.slice(0, 4)) {
    if (budget <= 0) break
    const text = item.chunk.text.slice(0, Math.max(120, Math.min(800, budget)))
    if (text.trim()) {
      out.push(`### ${item.chunk.title}\n${text}`)
      budget -= text.length
    }
  }
  return out.join('\n\n')
}

function builtInAnswer(query, scored) {
  const top = scored.slice(0, 3).filter((x) => x.score > 0)
  if (top.length === 0) {
    return {
      answer: "No sufficiently relevant chunk was found for that query. Try pasting more specific documents into the library, or ask in English/Chinese with matching terminology.",
      citations: []
    }
  }
  const lines = []
  lines.push('Based on the knowledge base, the most relevant findings are:')
  top.forEach((item, i) => {
    const excerpt = item.chunk.text.slice(0, 360)
    lines.push(`\n${i + 1}. ${item.chunk.title}`)
    lines.push(`   ${excerpt}${excerpt.length < item.chunk.text.length ? '…' : ''}`)
  })
  lines.push('\nUse the "Add knowledge" panel to add your own documents and re-ask this question.')
  return {
    answer: lines.join('\n'),
    citations: top.map((item) => ({ title: item.chunk.title, score: Math.min(0.99, item.score) }))
  }
}

async function callLlm(query, context, history) {
  const base = String(process.env.LLM_BASE_URL || '').replace(/\/+$/, '')
  const url = base.endsWith('/chat/completions') ? base : base + '/chat/completions'
  const messages = [
    {
      role: 'system',
      content: 'You are Insightbase, a retrieval-augmented knowledge assistant. Answer only from the provided context. Cite document titles in brackets. If the context is insufficient, say so. Respond in the same language as the user question.'
    },
    ...(Array.isArray(history) ? history.slice(-6) : []),
    {
      role: 'user',
      content: `Knowledge base context:\n\n${context}\n\nQuestion: ${query}`
    }
  ]
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: 'Bearer ' + String(process.env.LLM_API_KEY || '')
    },
    body: JSON.stringify({
      model: process.env.LLM_MODEL || 'deepseek-chat',
      messages,
      temperature: 0.35,
      max_tokens: 700,
      stream: false
    }),
    signal: AbortSignal.timeout(25000)
  })
  if (!res.ok) throw new Error('upstream LLM HTTP ' + res.status)
  const data = await res.json()
  const answer = data && data.choices && data.choices[0] && data.choices[0].message
  if (!answer || !answer.content) throw new Error('upstream LLM returned an empty response')
  return answer.content.trim()
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.setHeader('access-control-allow-origin', '*')
    res.setHeader('access-control-allow-headers', 'content-type')
    res.setHeader('access-control-allow-methods', 'POST, OPTIONS')
    return res.status(204).end()
  }
  if (req.method !== 'POST') return bad(res, 'POST only', 405)
  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {})
    const query = String(body.query || body.q || '').trim()
    if (!query) return bad(res, 'query is required')
    const docs = Array.isArray(body.docs) && body.docs.length > 0
      ? body.docs.map((d) => ({ title: String(d.title || 'Untitled').slice(0, 140), text: String(d.text || d.content || '').slice(0, 30000) }))
      : []
    if (docs.length === 0) return bad(res, 'no knowledge base documents')
    const chunks = []
    for (const doc of docs) {
      for (const chunk of chunkText(doc.text)) {
        if (chunk.trim().length > 30) chunks.push({ title: doc.title, text: chunk.trim() })
      }
    }
    if (chunks.length === 0) return bad(res, 'no parseable text found in documents')
    const scored = scoreChunks(query, chunks)
    const context = compactExcerpts(scored)
    const citations = scored.slice(0, 4).map((item) => ({ title: item.chunk.title, score: Math.min(0.99, item.score) })).filter((c) => c.score > 0)

    const configured = !!(process.env.LLM_API_KEY && process.env.LLM_BASE_URL && process.env.LLM_MODEL)
    if (configured) {
      try {
        const answer = await callLlm(query, context, body.history || [])
        return ok(res, { ok: true, mode: 'llm', answer, citations })
      } catch (cause) {
        // Fall back to retrieval-only rather than failing the request.
        return ok(res, { ok: true, mode: 'demo', ...builtInAnswer(query, scored), citations, fallback: true, llmError: String(cause && cause.message ? cause.message : cause) })
      }
    }
    return ok(res, { ok: true, mode: 'demo', ...builtInAnswer(query, scored), citations })
  } catch (cause) {
    return bad(res, String(cause && cause.message ? cause.message : cause), 500)
  }
}