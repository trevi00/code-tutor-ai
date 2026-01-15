/**
 * Performance Analysis Page - Enhanced with modern design
 */

import { useState } from 'react';
import {
  Activity,
  BookOpen,
  Sparkles,
  Zap,
  Clock,
  TrendingUp,
  ChevronDown,
  Code2,
  Terminal,
} from 'lucide-react';
import { PerformancePanel } from '../../components/performance';

const SAMPLE_CODES = [
  {
    name: 'O(n) - 선형 탐색',
    complexity: 'O(n)',
    color: 'emerald',
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
    complexity: 'O(n²)',
    color: 'amber',
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
    complexity: 'O(2^n)',
    color: 'rose',
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
    complexity: 'O(n log n)',
    color: 'blue',
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
    complexity: 'O(log n)',
    color: 'green',
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

const COMPLEXITY_BADGES = [
  { label: 'O(1), O(log n)', desc: '효율적', color: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400' },
  { label: 'O(n), O(n log n)', desc: '양호', color: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' },
  { label: 'O(n²)', desc: '주의 필요', color: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400' },
  { label: 'O(2^n), O(n!)', desc: '비효율적', color: 'bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-400' },
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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-rose-600 via-pink-600 to-rose-700 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Activity className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Zap className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <TrendingUp className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-7xl mx-auto px-6 py-10 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Sparkles className="w-4 h-4" />
                코드 성능 분석
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Activity className="w-10 h-10 text-rose-200" />
                성능 분석
              </h1>
              <p className="text-rose-100 text-lg max-w-md">
                코드의 시간/공간 복잡도를 분석하고 최적화 포인트를 찾아보세요
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
                <div className="absolute top-full mt-2 right-0 w-64 bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden z-50">
                  {SAMPLE_CODES.map((sample) => (
                    <button
                      key={sample.name}
                      onClick={() => handleSelectSample(sample.code)}
                      className="w-full px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-center justify-between group"
                    >
                      <span className="text-sm text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white">
                        {sample.name}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full bg-${sample.color}-100 dark:bg-${sample.color}-900/30 text-${sample.color}-700 dark:text-${sample.color}-400`}>
                        {sample.complexity}
                      </span>
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
                    <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">코드 입력</span>
                  </div>
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-400">python</span>
              </div>
              <div className="p-4">
                <textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="Python 코드를 입력하세요..."
                  className="w-full h-80 font-mono text-sm p-4 bg-neutral-100 dark:bg-slate-900 text-neutral-900 dark:text-slate-100 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-rose-500"
                  spellCheck={false}
                />
              </div>
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
                  className="w-full h-20 font-mono text-sm p-3 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-rose-500 border border-slate-200 dark:border-slate-700"
                />
              </div>
            </div>
          </div>

          {/* Right: Performance Panel */}
          <div className="lg:h-[calc(100vh-14rem)]">
            <PerformancePanel code={code} input={input} />
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-2xl p-6 border border-rose-200 dark:border-rose-800">
          <h3 className="font-bold text-rose-800 dark:text-rose-200 mb-3 flex items-center gap-2">
            <Clock className="w-5 h-5" />
            성능 분석 가이드
          </h3>
          <ul className="text-sm text-rose-700 dark:text-rose-300 space-y-2">
            <li className="flex items-start gap-2">
              <span className="font-bold text-rose-600 dark:text-rose-400">빠른 분석:</span>
              <span>시간/공간 복잡도만 정적 분석 (코드 실행 없음)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-rose-600 dark:text-rose-400">전체 분석:</span>
              <span>실제 코드 실행 + 런타임 프로파일링 + 메모리 분석</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold text-rose-600 dark:text-rose-400">최적화 점수:</span>
              <span>복잡도, 중첩 깊이, 성능 이슈를 종합한 점수 (0-100)</span>
            </li>
          </ul>

          {/* Complexity Badges */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
            {COMPLEXITY_BADGES.map((badge) => (
              <div
                key={badge.label}
                className={`px-3 py-2 rounded-xl text-center ${badge.color}`}
              >
                <div className="font-mono font-bold text-sm">{badge.label}</div>
                <div className="text-xs mt-0.5 opacity-75">{badge.desc}</div>
              </div>
            ))}
          </div>
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
