import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { Send, Bot, User, RefreshCw, Copy, Check, Sparkles, Code, Lightbulb, BookOpen, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
// Use light build with only required languages (reduces bundle ~600KB -> ~50KB)
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
// Import only the languages commonly used in coding problems
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript';
import java from 'react-syntax-highlighter/dist/esm/languages/prism/java';
import c from 'react-syntax-highlighter/dist/esm/languages/prism/c';
import cpp from 'react-syntax-highlighter/dist/esm/languages/prism/cpp';
import go from 'react-syntax-highlighter/dist/esm/languages/prism/go';
import rust from 'react-syntax-highlighter/dist/esm/languages/prism/rust';
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql';
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash';
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json';
import { tutorApi } from '@/api/tutor';

// Register languages
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('py', python);
SyntaxHighlighter.registerLanguage('javascript', javascript);
SyntaxHighlighter.registerLanguage('js', javascript);
SyntaxHighlighter.registerLanguage('typescript', typescript);
SyntaxHighlighter.registerLanguage('ts', typescript);
SyntaxHighlighter.registerLanguage('java', java);
SyntaxHighlighter.registerLanguage('c', c);
SyntaxHighlighter.registerLanguage('cpp', cpp);
SyntaxHighlighter.registerLanguage('c++', cpp);
SyntaxHighlighter.registerLanguage('go', go);
SyntaxHighlighter.registerLanguage('rust', rust);
SyntaxHighlighter.registerLanguage('sql', sql);
SyntaxHighlighter.registerLanguage('bash', bash);
SyntaxHighlighter.registerLanguage('sh', bash);
SyntaxHighlighter.registerLanguage('json', json);

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// Quick suggestion buttons
const quickSuggestions = [
  { icon: Code, text: '시간복잡도 설명해줘', color: 'bg-blue-500' },
  { icon: Lightbulb, text: 'Two Pointers 패턴 알려줘', color: 'bg-amber-500' },
  { icon: BookOpen, text: 'DP 문제 풀이 힌트 줘', color: 'bg-green-500' },
  { icon: Sparkles, text: '코드 리뷰해줘', color: 'bg-purple-500' },
];

// Code block component with copy functionality
function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-3">
      <div className="absolute right-2 top-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={handleCopy}
          className="p-1.5 rounded bg-neutral-100 dark:bg-slate-700 hover:bg-neutral-200 dark:hover:bg-slate-600 text-neutral-600 dark:text-slate-300 transition-colors"
          title="코드 복사"
        >
          {copied ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
        </button>
      </div>
      <div className="text-xs text-neutral-500 dark:text-slate-400 bg-neutral-100 dark:bg-slate-800 px-3 py-1 rounded-t border-b border-neutral-200 dark:border-slate-700">
        {language || 'code'}
      </div>
      <SyntaxHighlighter
        language={language || 'text'}
        style={oneDark}
        customStyle={{
          margin: 0,
          borderTopLeftRadius: 0,
          borderTopRightRadius: 0,
          fontSize: '0.875rem',
        }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}

// Markdown message renderer
function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          const isInline = !match && !className;

          if (isInline) {
            return (
              <code className="bg-slate-200 dark:bg-slate-700 px-1.5 py-0.5 rounded text-sm font-mono text-pink-600 dark:text-pink-400" {...props}>
                {children}
              </code>
            );
          }

          return (
            <CodeBlock language={match ? match[1] : ''}>
              {String(children).replace(/\n$/, '')}
            </CodeBlock>
          );
        },
        p({ children }) {
          return <p className="mb-2 last:mb-0">{children}</p>;
        },
        ul({ children }) {
          return <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>;
        },
        li({ children }) {
          return <li className="ml-2">{children}</li>;
        },
        h1({ children }) {
          return <h1 className="text-xl font-bold mb-2 mt-4 first:mt-0">{children}</h1>;
        },
        h2({ children }) {
          return <h2 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h2>;
        },
        h3({ children }) {
          return <h3 className="text-base font-bold mb-2 mt-2 first:mt-0">{children}</h3>;
        },
        blockquote({ children }) {
          return (
            <blockquote className="border-l-4 border-blue-500 pl-4 my-2 text-slate-600 dark:text-slate-400 italic">
              {children}
            </blockquote>
          );
        },
        a({ href, children }) {
          return (
            <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
              {children}
            </a>
          );
        },
        table({ children }) {
          return (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full border border-slate-300 dark:border-slate-600">{children}</table>
            </div>
          );
        },
        th({ children }) {
          return <th className="border border-slate-300 dark:border-slate-600 px-3 py-1 bg-slate-100 dark:bg-slate-700 font-semibold">{children}</th>;
        },
        td({ children }) {
          return <td className="border border-slate-300 dark:border-slate-600 px-3 py-1">{children}</td>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

// Typing indicator animation
function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center shadow-lg">
        <Bot className="h-5 w-5 text-white" />
      </div>
      <div className="bg-white dark:bg-slate-700 rounded-2xl rounded-tl-none px-5 py-3 shadow-sm border border-slate-100 dark:border-slate-600">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          <span className="ml-2 text-sm text-slate-500 dark:text-slate-400">AI가 생각 중...</span>
        </div>
      </div>
    </div>
  );
}

// Message timestamp formatter
function formatTime(date: Date): string {
  return date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
}

// Single message component
function ChatMessage({ message, onCopy }: { message: Message; onCopy: (content: string) => void }) {
  const [showCopied, setShowCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    onCopy(message.content);
    setShowCopied(true);
    setTimeout(() => setShowCopied(false), 2000);
  };

  return (
    <div className={`flex gap-3 group animate-fade-in ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      {isUser ? (
        <div className="w-10 h-10 bg-gradient-to-br from-slate-600 to-slate-800 rounded-full flex items-center justify-center shadow-lg flex-shrink-0">
          <User className="h-5 w-5 text-white" />
        </div>
      ) : (
        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center shadow-lg flex-shrink-0">
          <Bot className="h-5 w-5 text-white" />
        </div>
      )}

      {/* Message bubble */}
      <div className={`max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm ${
            isUser
              ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-tr-none'
              : 'bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 rounded-tl-none border border-slate-100 dark:border-slate-600'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownMessage content={message.content} />
          )}
        </div>

        {/* Timestamp & Actions */}
        <div className={`flex items-center gap-2 mt-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
          <span className="text-xs text-slate-400 dark:text-slate-500">{formatTime(message.timestamp)}</span>
          {!isUser && (
            <button
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
              title="메시지 복사"
            >
              {showCopied ? (
                <Check className="h-3.5 w-3.5 text-green-500" />
              ) : (
                <Copy className="h-3.5 w-3.5 text-slate-400" />
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function ChatPage() {
  const [searchParams] = useSearchParams();
  const problemId = searchParams.get('problem');

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: problemId
        ? '안녕하세요! 이 문제에 대해 도움이 필요하시면 질문해주세요. 힌트, 접근법, 코드 리뷰 등 다양한 방식으로 도움을 드릴 수 있어요.'
        : '안녕하세요! 알고리즘 학습을 도와드리는 **AI 튜터**입니다.\n\n다음과 같은 도움을 드릴 수 있어요:\n- 알고리즘 개념 설명\n- 문제 풀이 힌트\n- 코드 리뷰 및 최적화\n- 시간/공간 복잡도 분석\n\n무엇을 도와드릴까요?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 150) + 'px';
    }
  }, [input]);

  const handleCopyMessage = useCallback(async (content: string) => {
    await navigator.clipboard.writeText(content);
  }, []);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }

    try {
      const response = await tutorApi.chat({
        message: input,
        conversation_id: conversationId || undefined,
        problem_id: problemId || undefined,
      });

      if (response.is_new_conversation) {
        setConversationId(response.conversation_id);
      }

      const aiMessage: Message = {
        id: response.message.id || (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.message.content,
        timestamp: new Date(response.message.created_at || Date.now()),
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestionClick = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  const handleNewChat = () => {
    setConversationId(null);
    setMessages([
      {
        id: '1',
        role: 'assistant',
        content: '안녕하세요! 새로운 대화를 시작합니다. 무엇을 도와드릴까요?',
        timestamp: new Date(),
      },
    ]);
  };

  const showSuggestions = messages.length <= 1 && !isLoading;

  return (
    <div className="container mx-auto px-4 py-6 h-[calc(100vh-6rem)]">
      <div className="bg-slate-50 dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-700 h-full flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <Bot className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                AI 튜터
                <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400 rounded-full">
                  온라인
                </span>
              </h1>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                알고리즘 질문, 힌트, 코드 리뷰를 도와드려요
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {problemId && (
              <Link
                to={`/problems/${problemId}/solve`}
                className="flex items-center gap-2 px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                문제로 이동
              </Link>
            )}
            <button
              onClick={handleNewChat}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              새 대화
            </button>
          </div>
        </div>

        {/* Problem context banner */}
        {problemId && (
          <div className="px-6 py-2 bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-900/50">
            <p className="text-sm text-blue-700 dark:text-blue-300 flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              현재 문제와 연결되어 있습니다. 문제에 대한 맞춤 힌트를 받을 수 있어요!
            </p>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} onCopy={handleCopyMessage} />
          ))}

          {isLoading && <TypingIndicator />}

          {/* Quick suggestions */}
          {showSuggestions && (
            <div className="mt-8 animate-fade-in">
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-3 text-center">
                빠르게 시작하기
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {quickSuggestions.map((suggestion, idx) => {
                  const Icon = suggestion.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => handleSuggestionClick(suggestion.text)}
                      className="flex items-center gap-2 px-4 py-2.5 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl hover:border-blue-300 dark:hover:border-blue-500 hover:shadow-md transition-all text-sm text-slate-700 dark:text-slate-200"
                    >
                      <span className={`p-1 rounded ${suggestion.color}`}>
                        <Icon className="h-3.5 w-3.5 text-white" />
                      </span>
                      {suggestion.text}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
          <form onSubmit={handleSubmit} className="flex gap-3 items-end">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="메시지를 입력하세요... (Enter로 전송, Shift+Enter로 줄바꿈)"
                rows={1}
                className="w-full px-4 py-3 pr-12 border border-slate-300 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-slate-50 dark:bg-slate-700 dark:text-white dark:placeholder-slate-400 resize-none transition-all"
                style={{ minHeight: '48px', maxHeight: '150px' }}
              />
              <div className="absolute right-3 bottom-3 text-xs text-slate-400">
                {input.length > 0 && `${input.length}자`}
              </div>
            </div>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-5 py-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl disabled:shadow-none flex items-center gap-2"
            >
              <Send className="h-5 w-5" />
              <span className="hidden sm:inline">전송</span>
            </button>
          </form>
          <p className="text-xs text-slate-400 dark:text-slate-500 mt-2 text-center">
            AI 튜터는 때때로 부정확한 정보를 제공할 수 있습니다. 중요한 내용은 항상 확인하세요.
          </p>
        </div>
      </div>

      {/* Styles */}
      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
}
