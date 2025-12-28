import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: '안녕하세요! 알고리즘 학습을 도와드리는 AI 튜터입니다. 무엇을 도와드릴까요?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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

    // Simulate AI response (will be replaced with actual API call)
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: getSimulatedResponse(input),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="container mx-auto px-4 py-8 h-[calc(100vh-8rem)]">
      <div className="bg-white rounded-xl shadow-sm border border-neutral-200 h-full flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-neutral-200">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Bot className="h-6 w-6 text-blue-600" />
            AI Tutor Chat
          </h1>
          <p className="text-sm text-neutral-600 mt-1">
            Ask questions about algorithms, get hints, or request code reviews
          </p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : ''}`}
            >
              {message.role === 'assistant' && (
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="h-5 w-5 text-blue-600" />
                </div>
              )}
              <div
                className={`max-w-[70%] rounded-2xl px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-neutral-100 text-neutral-900'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
              {message.role === 'user' && (
                <div className="w-8 h-8 bg-neutral-200 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="h-5 w-5 text-neutral-600" />
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <Bot className="h-5 w-5 text-blue-600" />
              </div>
              <div className="bg-neutral-100 rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" />
                  <span className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <span className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-neutral-200">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 px-4 py-3 border border-neutral-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function getSimulatedResponse(input: string): string {
  const lowerInput = input.toLowerCase();

  if (lowerInput.includes('two sum') || lowerInput.includes('배열')) {
    return `Two Sum 문제를 도와드릴게요!

**문제 이해**
두 수의 합이 target이 되는 인덱스 쌍을 찾는 문제입니다.

**힌트**
1. 브루트 포스: O(n²) - 모든 쌍을 검사
2. 해시맵: O(n) - 각 숫자의 보수(target - num)를 저장

더 자세한 설명이 필요하시면 말씀해주세요!`;
  }

  if (lowerInput.includes('dp') || lowerInput.includes('동적')) {
    return `동적 프로그래밍(DP)은 복잡한 문제를 작은 하위 문제로 나누어 해결하는 기법입니다.

**핵심 개념**
1. 최적 부분 구조: 큰 문제의 최적해가 작은 문제의 최적해로 구성
2. 중복 부분 문제: 같은 하위 문제가 여러 번 계산됨
3. 메모이제이션: 계산 결과를 저장해 재사용

어떤 DP 문제를 풀고 계신가요?`;
  }

  return `좋은 질문입니다! 알고리즘 학습에서 중요한 부분이에요.

다음과 같은 도움을 드릴 수 있습니다:
- 알고리즘 문제 풀이 힌트
- 코드 리뷰 및 개선 제안
- 알고리즘/자료구조 개념 설명

구체적인 질문을 해주시면 더 자세히 도와드릴게요!`;
}
