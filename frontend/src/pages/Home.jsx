import { useState } from 'react'
import { predictSentiment, extractAspects } from '../api/client'
import ResultCard from '../components/ResultCard'

export default function Home() {
  const [text,    setText]    = useState('')
  const [model,   setModel]   = useState('tfidf')
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [aspects, setAspects] = useState(null)

  const handleAnalyze = async () => {
  if (!text.trim()) return
  setLoading(true)
  setError(null)
  setResult(null)
  setAspects(null)
  try {
    const data = await predictSentiment(text, model)
    setResult(data)
    try {
      const aspectData = await extractAspects(text)
      console.log('aspects:', aspectData)
      setAspects(aspectData)
    } catch (aspErr) {
      console.error('Aspects error:', aspErr)
    }
  } catch (err) {
    setError('Failed to connect to API. Make sure the server is running.')
  } finally {
    setLoading(false)
  }
}

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">

      <h1 className="text-3xl font-bold text-gray-800 mb-2">
        Sentiment Analyzer
      </h1>
      <p className="text-gray-500 mb-8">
        Analyze product or service reviews — Positive, Neutral, or Negative.
      </p>

      {/* model selector */}
      <div className="flex gap-3 mb-4">
        {['tfidf', 'bert'].map(m => (
          <button
            key={m}
            onClick={() => { setModel(m); setResult(null) }}
            className={`px-4 py-2 rounded-full text-sm font-medium border transition-all
              ${model === m
                ? 'bg-gray-800 text-white border-gray-800'
                : 'bg-white text-gray-600 border-gray-300 hover:border-gray-500'
              }`}
          >
            {m === 'bert' ? 'BERT' : 'TF-IDF'}
          </button>
        ))}
      </div>

      {/* input */}
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Paste a product or service review here..."
        rows={5}
        className="w-full border border-gray-200 rounded-xl p-4 text-sm
                   focus:outline-none focus:ring-2 focus:ring-gray-300
                   resize-none mb-4 bg-white"
      />

      {/* example buttons */}
      <div className="flex gap-2 flex-wrap mb-4">
        {[
          { label: 'Positive', text: 'Absolutely loved it, best experience ever!' },
          { label: 'Negative', text: 'Terrible service, never coming back again.' },
          { label: 'Neutral',  text: 'It was okay, nothing special honestly.' }
        ].map(ex => (
          <button
            key={ex.label}
            onClick={() => { setText(ex.text); setResult(null) }}
            className="text-xs px-3 py-1.5 rounded-full border border-gray-200
                       text-gray-500 hover:border-gray-400 transition-all bg-white"
          >
            Try {ex.label}
          </button>
        ))}
      </div>

      {/* analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={loading || !text.trim()}
        className="w-full py-3 rounded-xl bg-gray-800 text-white font-medium
                   hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed
                   transition-all mb-6"
      >
        {loading ? 'Analyzing...' : 'Analyze Sentiment'}
      </button>

      {/* error */}
      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200
                        rounded-xl px-4 py-3 mb-4">
          {error}
        </div>
      )}

      {/* result */}
      {result && <ResultCard result={result} model={model} />}
      {aspects && aspects.detected && (
  <div className="mt-4 bg-white border border-gray-200 rounded-2xl p-6">
    <p className="text-sm font-semibold text-gray-700 mb-4">
      Aspects Detected
    </p>
    <div className="flex flex-col gap-3">
      {Object.entries(aspects.aspects).map(([aspect, words]) => (
        <div key={aspect} className="flex items-start gap-3">
          <span className="text-sm font-medium text-gray-600 w-24 shrink-0">
            {aspect}
          </span>
          <div className="flex flex-wrap gap-2">
            {words.map(word => (
              <span
                key={word}
                className="text-xs px-2.5 py-1 rounded-full bg-gray-100
                           text-gray-600 border border-gray-200"
              >
                {word}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
)}

    </div>
  )
}