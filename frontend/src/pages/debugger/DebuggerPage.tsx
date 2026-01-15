/**
 * Debugger Page - Enhanced with modern design
 */

import { useState, useCallback } from 'react';
import {
  Bug,
  BookOpen,
  ChevronDown,
  Code2,
  Terminal,
  Sparkles,
  Zap,
  GitBranch,
} from 'lucide-react';
import { DebuggerPanel } from '../../components/debugger';
import CodeViewer from '../../components/debugger/CodeViewer';

const SAMPLE_CODE = `# 피보나치 수열 계산
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 테스트
for i in range(7):
    result = fibonacci(i)
    print(f"fib({i}) = {result}")
`;

const SAMPLE_CODES = [
  {
    name: '피보나치 수열',
    description: '재귀 호출',
    code: SAMPLE_CODE,
  },
  {
    name: '버블 정렬',
    description: '중첩 루프',
    code: `# 버블 정렬 알고리즘
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# 테스트
numbers = [64, 34, 25, 12, 22, 11, 90]
print("정렬 전:", numbers)
sorted_numbers = bubble_sort(numbers.copy())
print("정렬 후:", sorted_numbers)
`,
  },
  {
    name: '팩토리얼',
    description: '재귀 함수',
    code: `# 팩토리얼 계산
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# 테스트
for i in range(6):
    result = factorial(i)
    print(f"{i}! = {result}")
`,
  },
  {
    name: '이진 탐색',
    description: '분할 정복',
    code: `# 이진 탐색 알고리즘
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# 테스트
sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
result = binary_search(sorted_list, target)
print(f"{target}의 인덱스: {result}")
`,
  },
];

export default function DebuggerPage() {
  const [code, setCode] = useState(SAMPLE_CODE);
  const [input, setInput] = useState('');
  const [breakpoints, setBreakpoints] = useState<number[]>([]);
  const [highlightedLine, setHighlightedLine] = useState<number | undefined>();
  const [showSamples, setShowSamples] = useState(false);

  const handleToggleBreakpoint = useCallback((lineNumber: number) => {
    setBreakpoints((prev) =>
      prev.includes(lineNumber)
        ? prev.filter((n) => n !== lineNumber)
        : [...prev, lineNumber]
    );
  }, []);

  const handleSelectSample = (sampleCode: string) => {
    setCode(sampleCode);
    setBreakpoints([]);
    setShowSamples(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-600 via-violet-600 to-purple-700 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-violet-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Bug className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <GitBranch className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Zap className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-7xl mx-auto px-6 py-10 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Sparkles className="w-4 h-4" />
                단계별 코드 실행
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Bug className="w-10 h-10 text-purple-200" />
                단계별 디버거
              </h1>
              <p className="text-purple-100 text-lg max-w-md">
                코드를 한 줄씩 실행하며 변수 변화와 호출 스택을 추적하세요
              </p>
            </div>

            {/* Sample Code Selector */}
            <div className="relative">
              <button
                onClick={() => setShowSamples(!showSamples)}
                className="flex items-center gap-2 px-5 py-2.5 bg-white/20 hover:bg-white/30 backdrop-blur-sm text-white rounded-xl transition-all"
              >
                <BookOpen className="w-5 h-5" />
                예제 코드
                <ChevronDown className={`w-4 h-4 transition-transform ${showSamples ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown */}
              {showSamples && (
                <div className="absolute top-full mt-2 right-0 w-72 bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden z-50">
                  {SAMPLE_CODES.map((sample) => (
                    <button
                      key={sample.name}
                      onClick={() => handleSelectSample(sample.code)}
                      className="w-full px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-center justify-between group"
                    >
                      <div>
                        <span className="text-sm font-medium text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white">
                          {sample.name}
                        </span>
                        <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                          {sample.description}
                        </p>
                      </div>
                      <Code2 className="w-4 h-4 text-slate-400 dark:text-slate-500" />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 -mt-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Code Editor */}
          <div className="space-y-4">
            {/* Code Input */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
              <div className="px-5 py-3 bg-gradient-to-r from-slate-100 to-slate-50 dark:from-slate-700 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                  </div>
                  <div className="flex items-center gap-2">
                    <Code2 className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                    <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">코드 편집기</span>
                  </div>
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-400">python</span>
              </div>
              <div className="p-4">
                <textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="Python 코드를 입력하세요..."
                  className="w-full h-64 font-mono text-sm p-4 bg-neutral-100 dark:bg-slate-900 text-neutral-900 dark:text-slate-100 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-purple-500"
                  spellCheck={false}
                />
              </div>
            </div>

            {/* Code Viewer with Line Numbers */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
              <div className="px-5 py-3 bg-gradient-to-r from-slate-100 to-slate-50 dark:from-slate-700 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Bug className="w-4 h-4 text-purple-500 dark:text-purple-400" />
                  <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">코드 뷰어</span>
                </div>
                {breakpoints.length > 0 && (
                  <span className="text-xs px-2 py-0.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg">
                    브레이크포인트: {breakpoints.length}개
                  </span>
                )}
              </div>
              <CodeViewer
                code={code}
                currentLine={highlightedLine}
                breakpoints={breakpoints}
                onToggleBreakpoint={handleToggleBreakpoint}
                className="max-h-80"
              />
            </div>

            {/* Input */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
              <div className="px-5 py-3 bg-gradient-to-r from-slate-100 to-slate-50 dark:from-slate-700 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">입력 (선택사항)</span>
              </div>
              <div className="p-4">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="프로그램 입력값 (줄바꿈으로 구분)"
                  className="w-full h-20 font-mono text-sm p-3 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 border border-slate-200 dark:border-slate-700"
                />
              </div>
            </div>
          </div>

          {/* Right: Debugger Panel */}
          <div className="lg:h-[calc(100vh-14rem)]">
            <DebuggerPanel
              code={code}
              input={input}
              onLineHighlight={setHighlightedLine}
            />
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-gradient-to-r from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-2xl p-6 border border-purple-200 dark:border-purple-800">
          <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-3 flex items-center gap-2">
            <Bug className="w-5 h-5" />
            사용 방법
          </h3>
          <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-2">
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600 dark:text-purple-400 min-w-[1.5rem]">1.</span>
              <span>코드 편집기에 Python 코드를 작성하거나 예제 코드를 선택하세요.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600 dark:text-purple-400 min-w-[1.5rem]">2.</span>
              <span>코드 뷰어에서 줄 번호 왼쪽을 클릭하여 브레이크포인트를 설정할 수 있습니다.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600 dark:text-purple-400 min-w-[1.5rem]">3.</span>
              <span>"디버그 시작" 버튼을 클릭하여 디버깅을 시작하세요.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600 dark:text-purple-400 min-w-[1.5rem]">4.</span>
              <span>재생 컨트롤을 사용하여 코드 실행을 단계별로 확인하세요.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-purple-600 dark:text-purple-400 min-w-[1.5rem]">5.</span>
              <span>각 단계에서 변수 값과 호출 스택을 확인할 수 있습니다.</span>
            </li>
          </ul>
        </div>
      </main>

      {/* Styles */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(-5deg); }
        }
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float-delayed 5s ease-in-out infinite;
          animation-delay: 1s;
        }
      `}</style>
    </div>
  );
}
