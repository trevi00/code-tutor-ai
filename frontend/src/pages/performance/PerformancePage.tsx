/**
 * Performance Analysis Page
 */

import { useState } from 'react';
import { Activity, BookOpen } from 'lucide-react';
import { PerformancePanel } from '../../components/performance';

const SAMPLE_CODES = [
  {
    name: 'O(n) - 선형 탐색',
    code: `def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# 테스트
numbers = list(range(100))
result = linear_search(numbers, 50)
print(f"찾은 인덱스: {result}")`,
  },
  {
    name: 'O(n²) - 버블 정렬',
    code: `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# 테스트
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_nums = bubble_sort(numbers.copy())
print("정렬 결과:", sorted_nums)`,
  },
  {
    name: 'O(2^n) - 피보나치 재귀',
    code: `def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# 테스트 (작은 n 사용)
for i in range(10):
    print(f"fib({i}) = {fib(i)}")`,
  },
  {
    name: 'O(n log n) - 병합 정렬',
    code: `def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 테스트
arr = [38, 27, 43, 3, 9, 82, 10]
print("정렬 전:", arr)
print("정렬 후:", merge_sort(arr))`,
  },
  {
    name: 'O(log n) - 이진 탐색',
    code: `def binary_search(arr, target):
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
sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 11
result = binary_search(sorted_arr, target)
print(f"{target} 위치: {result}")`,
  },
];

export default function PerformancePage() {
  const [code, setCode] = useState(SAMPLE_CODES[0].code);
  const [input, setInput] = useState('');
  const [showSamples, setShowSamples] = useState(false);

  const handleSelectSample = (sampleCode: string) => {
    setCode(sampleCode);
    setShowSamples(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-blue-600" />
            <h1 className="text-xl font-bold text-gray-800">성능 분석</h1>
          </div>
          <button
            onClick={() => setShowSamples(!showSamples)}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 flex items-center gap-1"
          >
            <BookOpen className="w-4 h-4" />
            예제 코드
          </button>
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
                  className="px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded hover:bg-blue-100"
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
              <div className="px-4 py-2 bg-gray-50 border-b">
                <span className="font-medium text-gray-700 text-sm">코드 입력</span>
              </div>
              <div className="p-4">
                <textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="Python 코드를 입력하세요..."
                  className="w-full h-80 font-mono text-sm p-3 bg-gray-900 text-gray-100 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                  spellCheck={false}
                />
              </div>
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
                  className="w-full h-20 font-mono text-sm p-3 bg-gray-50 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>

          {/* Right: Performance Panel */}
          <div className="lg:h-[calc(100vh-12rem)]">
            <PerformancePanel code={code} input={input} />
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-blue-50 rounded-lg p-4">
          <h3 className="font-medium text-blue-800 mb-2">성능 분석 가이드</h3>
          <ul className="text-sm text-blue-700 space-y-1">
            <li><strong>빠른 분석:</strong> 시간/공간 복잡도만 정적 분석 (코드 실행 없음)</li>
            <li><strong>전체 분석:</strong> 실제 코드 실행 + 런타임 프로파일링 + 메모리 분석</li>
            <li><strong>최적화 점수:</strong> 복잡도, 중첩 깊이, 성능 이슈를 종합한 점수 (0-100)</li>
          </ul>
          <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div className="bg-green-100 text-green-700 px-2 py-1 rounded">O(1), O(log n) - 효율적</div>
            <div className="bg-blue-100 text-blue-700 px-2 py-1 rounded">O(n), O(n log n) - 양호</div>
            <div className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded">O(n²) - 주의 필요</div>
            <div className="bg-red-100 text-red-700 px-2 py-1 rounded">O(2^n), O(n!) - 비효율적</div>
          </div>
        </div>
      </main>
    </div>
  );
}
