import { useState, useCallback, useEffect } from 'react';
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
import { problemsApi } from '@/api/problems';
import { executionApi } from '@/api/execution';


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
    expected_output?: string;
  }>;
}

export function ProblemSolvePage() {
  const { id } = useParams<{ id: string }>();
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [code, setCode] = useState('');
  const [activeTab, setActiveTab] = useState<'description' | 'hints'>('description');
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [submissionResult, setSubmissionResult] = useState<SubmissionResult | null>(null);
  const [showHintIndex, setShowHintIndex] = useState(-1);

  useEffect(() => {
    const fetchProblem = async () => {
      if (!id) return;
      setLoading(true);
      setError(null);
      try {
        const data = await problemsApi.get(id);
        setProblem(data);
        setCode(data.solution_template || '# Write your solution here\n');
      } catch (err) {
        setError('Failed to load problem. Please try again.');
        console.error('Error fetching problem:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchProblem();
  }, [id]);

  const handleEditorChange = useCallback((value: string | undefined) => {
    if (value !== undefined) {
      setCode(value);
    }
  }, []);

  const handleRun = async () => {
    if (!id) return;
    setIsRunning(true);
    setExecutionResult(null);
    setSubmissionResult(null);

    try {
      const result = await executionApi.execute({
        problem_id: id,
        code,
        language: 'python',
      });
      setExecutionResult({
        status: result.status,
        stdout: result.stdout || '',
        stderr: result.stderr || '',
        execution_time_ms: result.execution_time_ms || 0,
      });
    } catch (err) {
      setExecutionResult({
        status: 'runtime_error',
        stdout: '',
        stderr: 'Failed to execute code. Please try again.',
        execution_time_ms: 0,
      });
      console.error('Execution error:', err);
    } finally {
      setIsRunning(false);
    }
  };

  const handleSubmit = async () => {
    if (!id) return;
    setIsSubmitting(true);
    setExecutionResult(null);
    setSubmissionResult(null);

    try {
      // /submit endpoint returns evaluated result immediately
      const result = await executionApi.submit({
        problem_id: id,
        code,
        language: 'python',
      });

      setSubmissionResult({
        status: result.status,
        passed_tests: result.passed_tests || 0,
        total_tests: result.total_tests || 0,
        results: result.test_results?.map((tr) => ({
          is_passed: tr.is_passed,
          actual_output: tr.actual_output || '',
        })) || [],
      });
    } catch (err) {
      setSubmissionResult({
        status: 'runtime_error',
        passed_tests: 0,
        total_tests: 0,
        results: [],
      });
      console.error('Submission error:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="h-[calc(100vh-4rem)] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto" />
          <p className="mt-2 text-neutral-500">Loading problem...</p>
        </div>
      </div>
    );
  }

  if (error || !problem) {
    return (
      <div className="h-[calc(100vh-4rem)] flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error || 'Problem not found'}</p>
          <Link
            to="/problems"
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            <ChevronLeft className="h-4 w-4" />
            Back to Problems
          </Link>
        </div>
      </div>
    );
  }

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
              힌트 ({(problem.hints || []).length})
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
                {(problem.hints || []).map((hint, index) => (
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
                          {result.expected_output && <div>예상: {result.expected_output}</div>}
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
