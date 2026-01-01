/**
 * Debugger Page
 * Standalone page for step-by-step code debugging
 */

import { useState, useCallback } from 'react';
import { Bug, BookOpen } from 'lucide-react';
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
    code: SAMPLE_CODE,
  },
  {
    name: '버블 정렬',
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
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bug className="w-6 h-6 text-purple-600" />
            <h1 className="text-xl font-bold text-gray-800">단계별 디버거</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSamples(!showSamples)}
              className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 flex items-center gap-1"
            >
              <BookOpen className="w-4 h-4" />
              예제 코드
            </button>
          </div>
        </div>
      </header>

      {/* Sample Code Selector */}
      {showSamples && (
        <div className="bg-white border-b shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex flex-wrap gap-2">
              {SAMPLE_CODES.map((sample) => (
                <button
                  key={sample.name}
                  onClick={() => handleSelectSample(sample.code)}
                  className="px-3 py-1.5 text-sm bg-purple-50 text-purple-700 rounded hover:bg-purple-100"
                >
                  {sample.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Code Editor */}
          <div className="space-y-4">
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-50 border-b flex items-center justify-between">
                <span className="font-medium text-gray-700 text-sm">코드 편집기</span>
                <span className="text-xs text-gray-500">
                  {breakpoints.length > 0 && `브레이크포인트: ${breakpoints.length}개`}
                </span>
              </div>
              <div className="p-4">
                <textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="Python 코드를 입력하세요..."
                  className="w-full h-64 font-mono text-sm p-3 bg-gray-900 text-gray-100 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-purple-500"
                  spellCheck={false}
                />
              </div>
            </div>

            {/* Code Viewer with Line Numbers */}
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-50 border-b">
                <span className="font-medium text-gray-700 text-sm">코드 뷰어</span>
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
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-50 border-b">
                <span className="font-medium text-gray-700 text-sm">입력 (선택사항)</span>
              </div>
              <div className="p-4">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="프로그램 입력값 (줄바꿈으로 구분)"
                  className="w-full h-20 font-mono text-sm p-3 bg-gray-50 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>
          </div>

          {/* Right: Debugger Panel */}
          <div className="lg:h-[calc(100vh-12rem)]">
            <DebuggerPanel
              code={code}
              input={input}
              onLineHighlight={setHighlightedLine}
            />
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-purple-50 rounded-lg p-4">
          <h3 className="font-medium text-purple-800 mb-2">사용 방법</h3>
          <ul className="text-sm text-purple-700 space-y-1">
            <li>1. 코드 편집기에 Python 코드를 작성하거나 예제 코드를 선택하세요.</li>
            <li>2. 코드 뷰어에서 줄 번호 왼쪽을 클릭하여 브레이크포인트를 설정할 수 있습니다.</li>
            <li>3. "디버그 시작" 버튼을 클릭하여 디버깅을 시작하세요.</li>
            <li>4. 재생 컨트롤을 사용하여 코드 실행을 단계별로 확인하세요.</li>
            <li>5. 각 단계에서 변수 값과 호출 스택을 확인할 수 있습니다.</li>
          </ul>
        </div>
      </main>
    </div>
  );
}
