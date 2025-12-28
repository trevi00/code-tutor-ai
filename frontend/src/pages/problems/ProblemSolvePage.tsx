import { useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import Editor from '@monaco-editor/react';
import {
  Play,
  Send,
  MessageSquare,
  ChevronLeft,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
} from 'lucide-react';
import type { Problem, ExecutionStatus, SubmissionStatus } from '@/types';

// Mock problem data - will be replaced with API
const MOCK_PROBLEM: Problem = {
  id: '1',
  title: 'Two Sum',
  description: `## 문제 설명

정수 배열 \`nums\`와 정수 \`target\`이 주어집니다.
배열에서 두 수의 합이 \`target\`이 되는 두 수의 인덱스를 반환하세요.

각 입력에는 정확히 하나의 해답이 있으며, 같은 요소를 두 번 사용할 수 없습니다.

## 예제

**입력:** nums = [2, 7, 11, 15], target = 9
**출력:** [0, 1]
**설명:** nums[0] + nums[1] = 2 + 7 = 9이므로 [0, 1]을 반환합니다.

## 제한사항

- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- 정확히 하나의 유효한 답이 존재합니다.`,
  difficulty: 'easy',
  category: 'array',
  constraints: '2 <= nums.length <= 10^4',
  hints: [
    '브루트 포스로 O(n²)에 풀 수 있습니다.',
    '해시맵을 사용하면 O(n)에 풀 수 있습니다.',
    '각 숫자의 보수(target - num)를 저장해보세요.',
  ],
  solution_template: `def two_sum(nums: list[int], target: int) -> list[int]:
    """
    두 수의 합이 target이 되는 인덱스를 찾습니다.

    Args:
        nums: 정수 배열
        target: 목표 합

    Returns:
        두 수의 인덱스 리스트
    """
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
    print(two_sum([3, 2, 4], 6))       # [1, 2]
    print(two_sum([3, 3], 6))          # [0, 1]
`,
  time_limit_ms: 1000,
  memory_limit_mb: 256,
  is_published: true,
  test_cases: [
    { id: '1', input_data: '[2,7,11,15]\n9', expected_output: '[0, 1]', is_sample: true },
    { id: '2', input_data: '[3,2,4]\n6', expected_output: '[1, 2]', is_sample: true },
  ],
  created_at: new Date().toISOString(),
};

interface ExecutionResult {
  status: ExecutionStatus;
  stdout: string;
  stderr: string;
  execution_time_ms: number;
}

interface SubmissionResult {
  status: SubmissionStatus;
  passed_tests: number;
  total_tests: number;
  results: Array<{
    is_passed: boolean;
    actual_output: string;
    expected_output: string;
  }>;
}

export function ProblemSolvePage() {
  const { id } = useParams<{ id: string }>();
  const [code, setCode] = useState(MOCK_PROBLEM.solution_template);
  const [activeTab, setActiveTab] = useState<'description' | 'hints'>('description');
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [submissionResult, setSubmissionResult] = useState<SubmissionResult | null>(null);
  const [showHintIndex, setShowHintIndex] = useState(-1);

  const handleEditorChange = useCallback((value: string | undefined) => {
    if (value !== undefined) {
      setCode(value);
    }
  }, []);

  const handleRun = async () => {
    setIsRunning(true);
    setExecutionResult(null);
    setSubmissionResult(null);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Mock execution result
    setExecutionResult({
      status: 'success',
      stdout: '[0, 1]\n[1, 2]\n[0, 1]\n',
      stderr: '',
      execution_time_ms: 45,
    });

    setIsRunning(false);
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setExecutionResult(null);
    setSubmissionResult(null);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Mock submission result
    setSubmissionResult({
      status: 'accepted',
      passed_tests: 5,
      total_tests: 5,
      results: [
        { is_passed: true, actual_output: '[0, 1]', expected_output: '[0, 1]' },
        { is_passed: true, actual_output: '[1, 2]', expected_output: '[1, 2]' },
      ],
    });

    setIsSubmitting(false);
  };

  const problem = MOCK_PROBLEM;

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-white border-b border-neutral-200">
        <div className="flex items-center gap-4">
          <Link
            to="/problems"
            className="flex items-center gap-1 text-neutral-600 hover:text-blue-600"
          >
            <ChevronLeft className="h-5 w-5" />
            <span>Back</span>
          </Link>
          <h1 className="text-lg font-bold">{problem.title}</h1>
          <span
            className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${
              problem.difficulty === 'easy'
                ? 'bg-green-100 text-green-700'
                : problem.difficulty === 'medium'
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-red-100 text-red-700'
            }`}
          >
            {problem.difficulty}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to={`/chat?problem=${id}`}
            className="flex items-center gap-2 px-3 py-1.5 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
          >
            <MessageSquare className="h-4 w-4" />
            <span>AI 도움</span>
          </Link>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - Problem description */}
        <div className="w-1/2 flex flex-col border-r border-neutral-200">
          {/* Tabs */}
          <div className="flex border-b border-neutral-200">
            <button
              onClick={() => setActiveTab('description')}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === 'description'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-neutral-600 hover:text-neutral-900'
              }`}
            >
              문제 설명
            </button>
            <button
              onClick={() => setActiveTab('hints')}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === 'hints'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-neutral-600 hover:text-neutral-900'
              }`}
            >
              힌트 ({problem.hints.length})
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === 'description' ? (
              <div className="prose prose-neutral max-w-none">
                <div
                  dangerouslySetInnerHTML={{
                    __html: problem.description
                      .replace(/## (.*)/g, '<h2>$1</h2>')
                      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                      .replace(/`([^`]+)`/g, '<code>$1</code>')
                      .replace(/\n/g, '<br />'),
                  }}
                />
              </div>
            ) : (
              <div className="space-y-4">
                {problem.hints.map((hint, index) => (
                  <div
                    key={index}
                    className="border border-neutral-200 rounded-lg overflow-hidden"
                  >
                    <button
                      onClick={() =>
                        setShowHintIndex(showHintIndex === index ? -1 : index)
                      }
                      className="w-full px-4 py-3 flex items-center justify-between bg-neutral-50 hover:bg-neutral-100 transition-colors"
                    >
                      <span className="font-medium">힌트 {index + 1}</span>
                      <span className="text-sm text-neutral-500">
                        {showHintIndex === index ? '숨기기' : '보기'}
                      </span>
                    </button>
                    {showHintIndex === index && (
                      <div className="px-4 py-3 bg-white">{hint}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right panel - Code editor */}
        <div className="w-1/2 flex flex-col">
          {/* Editor */}
          <div className="flex-1">
            <Editor
              height="100%"
              defaultLanguage="python"
              value={code}
              onChange={handleEditorChange}
              theme="vs-dark"
              options={{
                fontSize: 14,
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                wordWrap: 'on',
              }}
            />
          </div>

          {/* Output panel */}
          <div className="h-48 border-t border-neutral-700 bg-neutral-900 flex flex-col">
            {/* Action buttons */}
            <div className="flex items-center gap-2 px-4 py-2 border-b border-neutral-700">
              <button
                onClick={handleRun}
                disabled={isRunning || isSubmitting}
                className="flex items-center gap-2 px-4 py-1.5 bg-neutral-700 text-white rounded hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isRunning ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                <span>실행</span>
              </button>
              <button
                onClick={handleSubmit}
                disabled={isRunning || isSubmitting}
                className="flex items-center gap-2 px-4 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isSubmitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
                <span>제출</span>
              </button>
              {(executionResult || submissionResult) && (
                <div className="ml-auto flex items-center gap-2 text-sm">
                  {submissionResult && (
                    <>
                      {submissionResult.status === 'accepted' ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <XCircle className="h-5 w-5 text-red-500" />
                      )}
                      <span
                        className={
                          submissionResult.status === 'accepted'
                            ? 'text-green-500'
                            : 'text-red-500'
                        }
                      >
                        {submissionResult.passed_tests}/{submissionResult.total_tests} 통과
                      </span>
                    </>
                  )}
                  {executionResult && (
                    <span className="text-neutral-400 flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      {executionResult.execution_time_ms}ms
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Output content */}
            <div className="flex-1 overflow-y-auto p-4 font-mono text-sm">
              {isRunning && (
                <div className="text-neutral-400">실행 중...</div>
              )}
              {isSubmitting && (
                <div className="text-neutral-400">채점 중...</div>
              )}
              {executionResult && (
                <div>
                  {executionResult.stdout && (
                    <pre className="text-green-400 whitespace-pre-wrap">
                      {executionResult.stdout}
                    </pre>
                  )}
                  {executionResult.stderr && (
                    <pre className="text-red-400 whitespace-pre-wrap">
                      {executionResult.stderr}
                    </pre>
                  )}
                </div>
              )}
              {submissionResult && (
                <div className="space-y-2">
                  {submissionResult.results.map((result, index) => (
                    <div
                      key={index}
                      className={`p-2 rounded ${
                        result.is_passed ? 'bg-green-900/30' : 'bg-red-900/30'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        {result.is_passed ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <XCircle className="h-4 w-4 text-red-500" />
                        )}
                        <span className="text-neutral-300">
                          테스트 케이스 {index + 1}
                        </span>
                      </div>
                      {!result.is_passed && (
                        <div className="text-xs text-neutral-400 ml-6">
                          <div>예상: {result.expected_output}</div>
                          <div>실제: {result.actual_output}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
              {!isRunning && !isSubmitting && !executionResult && !submissionResult && (
                <div className="text-neutral-500">
                  '실행' 버튼으로 코드를 테스트하거나, '제출' 버튼으로 채점을 받으세요.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
