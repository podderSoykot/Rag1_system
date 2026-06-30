const SUGGESTIONS = [
  'What is backpropagation?',
  'Explain gradient descent in neural networks',
  'What are the main chapters in the textbook?',
  'How does a perceptron work?',
  'What is overfitting and how to prevent it?',
]

export default function SuggestedQuestions({ onSelect, disabled }) {
  return (
    <div className="flex flex-wrap justify-center gap-2 px-4">
      {SUGGESTIONS.map((q) => (
        <button
          key={q}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(q)}
          className="rounded-full border border-ink-700/80 bg-ink-800/50 px-4 py-2 text-sm text-ink-300 transition hover:border-accent/50 hover:bg-accent/10 hover:text-white disabled:opacity-40"
        >
          {q}
        </button>
      ))}
    </div>
  )
}
