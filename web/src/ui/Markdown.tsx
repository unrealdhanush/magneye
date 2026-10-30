import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

// The guide (Muse Wonder) answers in Markdown. Raw HTML is never rendered; links open in a new tab.
const components: Components = {
  p: ({ children }) => <p className="text-body text-ink">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  ul: ({ children }) => <ul className="list-disc space-y-px ps-2.5 text-body text-ink marker:text-muted">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal space-y-px ps-2.5 text-body text-ink marker:text-muted">{children}</ol>,
  li: ({ children }) => <li className="ps-px">{children}</li>,
  h1: ({ children }) => <p className="text-headline text-ink">{children}</p>,
  h2: ({ children }) => <p className="text-headline text-ink">{children}</p>,
  h3: ({ children }) => <p className="text-subheadline font-semibold text-ink">{children}</p>,
  a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent-text underline">{children}</a>,
  code: ({ children }) => <code className="rounded-sm bg-ink/5 px-px font-mono text-callout">{children}</code>,
  pre: ({ children }) => <pre className="overflow-x-auto rounded-sm bg-ink/5 p-1 font-mono text-footnote">{children}</pre>,
  blockquote: ({ children }) => <blockquote className="border-s-2 border-ink/20 ps-1.5 text-muted">{children}</blockquote>,
  table: ({ children }) => <div className="overflow-x-auto"><table className="w-full text-footnote">{children}</table></div>,
  th: ({ children }) => <th className="border-b border-ink/10 px-px py-px text-start font-semibold">{children}</th>,
  td: ({ children }) => <td className="border-b border-ink/5 px-px py-px align-top">{children}</td>,
}

export function Markdown({ text }: { text: string }) {
  return <div className="space-y-1">{<ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{text}</ReactMarkdown>}</div>
}
