import { useState, useCallback, useEffect, useRef } from 'react';
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
  Sparkles,
  Lightbulb,
  Eye,
  EyeOff,
  Code2,
  Box,
  Terminal,
  ListChecks,
  ChevronDown,
  ChevronRight,
  Trophy,
  Keyboard,
  GripVertical,
  RotateCcw,
  Copy,
  Check,
} from 'lucide-react';
import type { Problem, ExecutionStatus, SubmissionStatus } from '@/types';
import { problemsApi } from '@/api/problems';
import { executionApi } from '@/api/execution';

// Pattern ID to Korean name mapping
const PATTERN_NAMES: Record<string, string> = {
  'two-pointers': '투 포인터',
  'sliding-window': '슬라이딩 윈도우',
  'binary-search': '이진 탐색',
  'bfs': 'BFS',
  'dfs': 'DFS',
  'dp': '동적 프로그래밍',
  'greedy': '그리디',
  'backtracking': '백트래킹',
  'monotonic-stack': '모노토닉 스택',
  'merge-intervals': '구간 병합',
  'matrix-traversal': '행렬 탐색',
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
    expected_output?: string;
  }>;
}

// Confetti component for success celebration
function Confetti() {
  const colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
  const confettiPieces = Array.from({ length: 50 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    delay: Math.random() * 0.5,
    duration: 1 + Math.random() * 1,
    color: colors[Math.floor(Math.random() * colors.length)],
    rotation: Math.random() * 360,
  }));

  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      {confettiPieces.map((piece) => (
        <div
          key={piece.id}
          className="absolute w-3 h-3 animate-confetti"
          style={{
            left: `${piece.left}%`,
            top: '-20px',
            backgroundColor: piece.color,
            animationDelay: `${piece.delay}s`,
            animationDuration: `${piece.duration}s`,
            transform: `rotate(${piece.rotation}deg)`,
          }}
        />
      ))}
      <style>{`
        @keyframes confetti {
          0% { transform: translateY(0) rotate(0deg); opacity: 1; }
          100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
        }
        .animate-confetti {
          animation: confetti linear forwards;
        }
      `}</style>
    </div>
  );
}

// Success Modal component
function SuccessModal({ onClose, passedTests, totalTests }: {
  onClose: () => void;
  passedTests: number;
  totalTests: number;
}) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-40" onClick={onClose}>
      <div
        className="bg-white dark:bg-slate-800 rounded-2xl p-8 max-w-md mx-4 text-center transform animate-bounce-in shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center">
          <Trophy className="w-10 h-10 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">축하합니다! 🎉</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          모든 테스트 케이스를 통과했습니다!
        </p>
        <div className="flex items-center justify-center gap-2 text-lg font-semibold text-green-600 dark:text-green-400 mb-6">
          <CheckCircle className="w-6 h-6" />
          {passedTests}/{totalTests} 통과
        </div>
        <div className="flex gap-3 justify-center">
          <Link
            to="/problems"
            className="px-6 py-2 bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors"
          >
            문제 목록
          </Link>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-lg hover:from-green-600 hover:to-emerald-600 transition-colors"
          >
            계속 풀기
          </button>
        </div>
      </div>
      <style>{`
        @keyframes bounce-in {
          0% { transform: scale(0.5); opacity: 0; }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); opacity: 1; }
        }
        .animate-bounce-in {
          animation: bounce-in 0.4s ease-out forwards;
        }
      `}</style>
    </div>
  );
}

// Keyboard shortcut tooltip
function ShortcutTooltip() {
  const [show, setShow] = useState(false);

  return (
    <div className="relative">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="p-1.5 text-neutral-400 hover:text-neutral-300 transition-colors"
        title="키보드 단축키"
      >
        <Keyboard className="w-4 h-4" />
      </button>
      {show && (
        <div className="absolute bottom-full right-0 mb-2 bg-neutral-800 border border-neutral-700 rounded-lg p-3 text-xs whitespace-nowrap z-50 shadow-xl">
          <div className="font-medium text-neutral-300 mb-2">키보드 단축키</div>
          <div className="space-y-1.5 text-neutral-400">
            <div className="flex justify-between gap-4">
              <span>코드 실행</span>
              <kbd className="px-1.5 py-0.5 bg-neutral-700 rounded text-neutral-300">Ctrl + Enter</kbd>
            </div>
            <div className="flex justify-between gap-4">
              <span>코드 제출</span>
              <kbd className="px-1.5 py-0.5 bg-neutral-700 rounded text-neutral-300">Ctrl + Shift + Enter</kbd>
            </div>
            <div className="flex justify-between gap-4">
              <span>코드 초기화</span>
              <kbd className="px-1.5 py-0.5 bg-neutral-700 rounded text-neutral-300">Ctrl + Shift + R</kbd>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function ProblemSolvePage() {
  const { id } = useParams<{ id: string }>();
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [code, setCode] = useState('');
  const [initialCode, setInitialCode] = useState('');
  const [activeTab, setActiveTab] = useState<'description' | 'approach' | 'hints'>('description');
  const [outputTab, setOutputTab] = useState<'console' | 'testcases'>('console');
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [submissionResult, setSubmissionResult] = useState<SubmissionResult | null>(null);
  const [showHintIndex, setShowHintIndex] = useState(-1);
  const [showSolution, setShowSolution] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [copied, setCopied] = useState(false);

  // Resizable panel state
  const [leftPanelWidth, setLeftPanelWidth] = useState(50);
  const [outputPanelHeight, setOutputPanelHeight] = useState(200);
  const [isResizingH, setIsResizingH] = useState(false);
  const [isResizingV, setIsResizingV] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchProblem = async () => {
      if (!id) return;
      setLoading(true);
      setError(null);
      try {
        const data = await problemsApi.get(id);
        setProblem(data);
        const template = data.solution_template || '# Write your solution here\n';
        setCode(template);
        setInitialCode(template);
      } catch (err) {
        setError('Failed to load problem. Please try again.');
        console.error('Error fetching problem:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchProblem();
  }, [id]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'Enter') {
        e.preventDefault();
        handleSubmit();
      } else if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleRun();
      } else if (e.ctrlKey && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        handleReset();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [code, id]);

  // Horizontal resize handler
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizingH || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const newWidth = ((e.clientX - rect.left) / rect.width) * 100;
      setLeftPanelWidth(Math.min(Math.max(newWidth, 25), 75));
    };
    const handleMouseUp = () => setIsResizingH(false);

    if (isResizingH) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizingH]);

  // Vertical resize handler
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizingV || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const editorBottom = rect.bottom;
      const newHeight = editorBottom - e.clientY;
      setOutputPanelHeight(Math.min(Math.max(newHeight, 100), 400));
    };
    const handleMouseUp = () => setIsResizingV(false);

    if (isResizingV) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizingV]);

  const handleEditorChange = useCallback((value: string | undefined) => {
    if (value !== undefined) {
      setCode(value);
    }
  }, []);

  const handleReset = () => {
    if (window.confirm('코드를 초기 상태로 되돌리시겠습니까?')) {
      setCode(initialCode);
    }
  };

  const handleCopyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRun = async () => {
    if (!id || isRunning || isSubmitting) return;
    setIsRunning(true);
    setExecutionResult(null);
    setSubmissionResult(null);
    setOutputTab('console');

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
    if (!id || isRunning || isSubmitting) return;
    setIsSubmitting(true);
    setExecutionResult(null);
    setSubmissionResult(null);
    setOutputTab('testcases');

    try {
      const result = await executionApi.submit({
        problem_id: id,
        code,
        language: 'python',
      });

      const submissionData = {
        status: result.status,
        passed_tests: result.passed_tests || 0,
        total_tests: result.total_tests || 0,
        results: result.test_results?.map((tr) => ({
          is_passed: tr.is_passed,
          actual_output: tr.actual_output || '',
          expected_output: tr.expected_output || '',
        })) || [],
      };

      setSubmissionResult(submissionData);

      // Show celebration for accepted submissions
      if (result.status === 'accepted') {
        setShowConfetti(true);
        setShowSuccessModal(true);
        setTimeout(() => setShowConfetti(false), 3000);
      }
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
      <div className="h-[calc(100vh-4rem)] flex items-center justify-center bg-neutral-50 dark:bg-slate-900">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 border-4 border-blue-200 dark:border-blue-900 rounded-full"></div>
            <div className="absolute inset-0 border-4 border-transparent border-t-blue-500 rounded-full animate-spin"></div>
          </div>
          <p className="text-neutral-600 dark:text-neutral-400 font-medium">문제를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !problem) {
    return (
      <div className="h-[calc(100vh-4rem)] flex items-center justify-center bg-neutral-50 dark:bg-slate-900">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
            <XCircle className="w-8 h-8 text-red-500" />
          </div>
          <p className="text-red-500 dark:text-red-400 mb-4 font-medium">{error || '문제를 찾을 수 없습니다'}</p>
          <Link
            to="/problems"
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
            문제 목록으로
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col" ref={containerRef}>
      {/* Confetti effect */}
      {showConfetti && <Confetti />}

      {/* Success modal */}
      {showSuccessModal && submissionResult && (
        <SuccessModal
          onClose={() => setShowSuccessModal(false)}
          passedTests={submissionResult.passed_tests}
          totalTests={submissionResult.total_tests}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-white dark:bg-slate-800 border-b border-neutral-200 dark:border-slate-700">
        <div className="flex items-center gap-4">
          <Link
            to="/problems"
            className="flex items-center gap-1 text-neutral-600 dark:text-neutral-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            <ChevronLeft className="h-5 w-5" />
            <span className="hidden sm:inline">목록</span>
          </Link>
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold dark:text-white truncate max-w-[300px]">{problem.title}</h1>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-semibold capitalize ${
                problem.difficulty === 'easy'
                  ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300'
                  : problem.difficulty === 'medium'
                  ? 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300'
                  : 'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300'
              }`}
            >
              {problem.difficulty === 'easy' ? '쉬움' : problem.difficulty === 'medium' ? '보통' : '어려움'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to={`/chat?problem=${id}`}
            className="flex items-center gap-2 px-3 py-1.5 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors"
          >
            <MessageSquare className="h-4 w-4" />
            <span className="hidden sm:inline">AI 도움</span>
          </Link>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - Problem description */}
        <div
          className="flex flex-col border-r border-neutral-200 dark:border-slate-700 bg-white dark:bg-slate-800"
          style={{ width: `${leftPanelWidth}%` }}
        >
          {/* Tabs */}
          <div className="flex border-b border-neutral-200 dark:border-slate-700">
            <button
              onClick={() => setActiveTab('description')}
              className={`px-4 py-2.5 font-medium transition-colors ${
                activeTab === 'description'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-blue-50/50 dark:bg-blue-900/20'
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-50 dark:hover:bg-slate-700/50'
              }`}
            >
              문제 설명
            </button>
            <button
              onClick={() => setActiveTab('approach')}
              className={`px-4 py-2.5 font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'approach'
                  ? 'text-purple-600 dark:text-purple-400 border-b-2 border-purple-600 dark:border-purple-400 bg-purple-50/50 dark:bg-purple-900/20'
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-50 dark:hover:bg-slate-700/50'
              }`}
            >
              <Sparkles className="h-4 w-4" />
              접근법
            </button>
            <button
              onClick={() => setActiveTab('hints')}
              className={`px-4 py-2.5 font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'hints'
                  ? 'text-amber-600 dark:text-amber-400 border-b-2 border-amber-600 dark:border-amber-400 bg-amber-50/50 dark:bg-amber-900/20'
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-50 dark:hover:bg-slate-700/50'
              }`}
            >
              <Lightbulb className="h-4 w-4" />
              힌트 ({(problem.hints || []).length})
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === 'description' && (
              <div className="animate-fade-in">
                {/* Pattern badges */}
                {problem.pattern_ids && problem.pattern_ids.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-4">
                    {problem.pattern_ids.map((patternId) => (
                      <Link
                        key={patternId}
                        to={`/patterns/${patternId}`}
                        className="inline-flex items-center gap-1 px-3 py-1 bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-full text-sm font-medium hover:bg-purple-200 dark:hover:bg-purple-900/70 transition-colors"
                      >
                        <Sparkles className="h-3 w-3" />
                        {PATTERN_NAMES[patternId] || patternId}
                      </Link>
                    ))}
                  </div>
                )}
                <div className="prose prose-neutral dark:prose-invert max-w-none">
                  <div
                    dangerouslySetInnerHTML={{
                      __html: problem.description
                        .replace(/## (.*)/g, '<h2 class="text-lg font-bold mt-6 mb-3">$1</h2>')
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 bg-neutral-100 dark:bg-slate-700 rounded text-sm">$1</code>')
                        .replace(/\n/g, '<br />'),
                    }}
                  />
                </div>
              </div>
            )}
            {activeTab === 'approach' && (
              <div className="space-y-6 animate-fade-in">
                {/* Patterns */}
                {problem.pattern_ids && problem.pattern_ids.length > 0 && (
                  <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/30 dark:to-indigo-900/30 rounded-xl p-4 border border-purple-100 dark:border-purple-800/50">
                    <h3 className="font-bold text-purple-900 dark:text-purple-300 mb-3 flex items-center gap-2">
                      <Sparkles className="h-5 w-5" />
                      적용 패턴
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {problem.pattern_ids.map((patternId) => (
                        <Link
                          key={patternId}
                          to={`/patterns/${patternId}`}
                          className="inline-flex items-center gap-1 px-3 py-1.5 bg-white dark:bg-slate-700 text-purple-700 dark:text-purple-300 rounded-lg text-sm font-medium hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors border border-purple-200 dark:border-purple-700 shadow-sm"
                        >
                          {PATTERN_NAMES[patternId] || patternId}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pattern explanation */}
                {problem.pattern_explanation && (
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/30 dark:to-cyan-900/30 rounded-xl p-4 border border-blue-100 dark:border-blue-800/50">
                    <h3 className="font-bold text-blue-900 dark:text-blue-300 mb-2 flex items-center gap-2">
                      <Lightbulb className="h-5 w-5" />
                      패턴 적용 방법
                    </h3>
                    <p className="text-blue-800 dark:text-blue-200 leading-relaxed">{problem.pattern_explanation}</p>
                  </div>
                )}

                {/* Approach hint */}
                {problem.approach_hint && (
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 rounded-xl p-4 border border-green-100 dark:border-green-800/50">
                    <h3 className="font-bold text-green-900 dark:text-green-300 mb-2">접근법 힌트</h3>
                    <p className="text-green-800 dark:text-green-200 leading-relaxed">{problem.approach_hint}</p>
                  </div>
                )}

                {/* Complexity hints */}
                <div className="grid grid-cols-2 gap-4">
                  {problem.time_complexity_hint && (
                    <div className="bg-neutral-50 dark:bg-slate-700/50 rounded-xl p-4 border border-neutral-200 dark:border-slate-600">
                      <h3 className="font-bold text-neutral-700 dark:text-neutral-300 mb-2 flex items-center gap-2 text-sm">
                        <Clock className="h-4 w-4 text-blue-500" />
                        목표 시간복잡도
                      </h3>
                      <p className="text-xl font-mono font-bold text-neutral-900 dark:text-white">
                        {problem.time_complexity_hint}
                      </p>
                    </div>
                  )}
                  {problem.space_complexity_hint && (
                    <div className="bg-neutral-50 dark:bg-slate-700/50 rounded-xl p-4 border border-neutral-200 dark:border-slate-600">
                      <h3 className="font-bold text-neutral-700 dark:text-neutral-300 mb-2 flex items-center gap-2 text-sm">
                        <Box className="h-4 w-4 text-purple-500" />
                        목표 공간복잡도
                      </h3>
                      <p className="text-xl font-mono font-bold text-neutral-900 dark:text-white">
                        {problem.space_complexity_hint}
                      </p>
                    </div>
                  )}
                </div>

                {/* Link to pattern pages */}
                {problem.pattern_ids && problem.pattern_ids.length > 0 && (
                  <div className="text-center pt-4 border-t border-neutral-200 dark:border-slate-600">
                    <Link
                      to="/patterns"
                      className="inline-flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 font-medium transition-colors"
                    >
                      패턴 학습 페이지에서 더 자세히 알아보기
                      <ChevronRight className="h-4 w-4" />
                    </Link>
                  </div>
                )}
              </div>
            )}
            {activeTab === 'hints' && (
              <div className="space-y-3 animate-fade-in">
                {/* Hints */}
                {(problem.hints || []).map((hint, index) => (
                  <div
                    key={index}
                    className="border border-neutral-200 dark:border-slate-600 rounded-xl overflow-hidden transition-all duration-200"
                  >
                    <button
                      onClick={() =>
                        setShowHintIndex(showHintIndex === index ? -1 : index)
                      }
                      className="w-full px-4 py-3 flex items-center justify-between bg-neutral-50 dark:bg-slate-700/50 hover:bg-neutral-100 dark:hover:bg-slate-700 transition-colors"
                    >
                      <span className="font-medium dark:text-white flex items-center gap-2">
                        <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                          showHintIndex === index
                            ? 'bg-amber-500 text-white'
                            : 'bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300'
                        }`}>
                          {index + 1}
                        </span>
                        힌트 {index + 1}
                      </span>
                      <ChevronDown className={`h-5 w-5 text-neutral-500 transition-transform duration-200 ${
                        showHintIndex === index ? 'rotate-180' : ''
                      }`} />
                    </button>
                    <div className={`overflow-hidden transition-all duration-200 ${
                      showHintIndex === index ? 'max-h-96' : 'max-h-0'
                    }`}>
                      <div className="px-4 py-3 bg-white dark:bg-slate-700 dark:text-neutral-200 border-t border-neutral-200 dark:border-slate-600">
                        {hint}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Solution */}
                {problem.reference_solution && (
                  <div className="border border-orange-200 dark:border-orange-800 rounded-xl overflow-hidden mt-6">
                    <button
                      onClick={() => setShowSolution(!showSolution)}
                      className="w-full px-4 py-3 flex items-center justify-between bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/30 dark:to-amber-900/30 hover:from-orange-100 hover:to-amber-100 dark:hover:from-orange-900/50 dark:hover:to-amber-900/50 transition-colors"
                    >
                      <span className="font-medium text-orange-800 dark:text-orange-300 flex items-center gap-2">
                        <Code2 className="h-4 w-4" />
                        정답 코드 보기
                      </span>
                      <span className="flex items-center gap-1 text-sm text-orange-600 dark:text-orange-400">
                        {showSolution ? (
                          <>
                            <EyeOff className="h-4 w-4" />
                            숨기기
                          </>
                        ) : (
                          <>
                            <Eye className="h-4 w-4" />
                            보기
                          </>
                        )}
                      </span>
                    </button>
                    <div className={`overflow-hidden transition-all duration-300 ${
                      showSolution ? 'max-h-[1000px]' : 'max-h-0'
                    }`}>
                      <div className="bg-neutral-900 p-4 overflow-x-auto">
                        <pre className="text-sm text-neutral-100 font-mono whitespace-pre">
                          {problem.reference_solution}
                        </pre>
                      </div>
                    </div>
                  </div>
                )}

                {/* Warning message */}
                {problem.reference_solution && !showSolution && (
                  <p className="text-sm text-neutral-500 dark:text-neutral-400 text-center mt-4 flex items-center justify-center gap-2">
                    <Lightbulb className="h-4 w-4" />
                    먼저 힌트를 활용해 직접 풀어보세요!
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Horizontal resize handle */}
        <div
          className="w-1 bg-neutral-200 dark:bg-slate-700 hover:bg-blue-400 dark:hover:bg-blue-500 cursor-col-resize flex items-center justify-center group transition-colors"
          onMouseDown={() => setIsResizingH(true)}
        >
          <GripVertical className="h-6 w-6 text-neutral-400 group-hover:text-white opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>

        {/* Right panel - Code editor */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Editor toolbar */}
          <div className="flex items-center justify-between px-3 py-1.5 bg-neutral-800 border-b border-neutral-700">
            <div className="flex items-center gap-2 text-sm text-neutral-400">
              <Code2 className="h-4 w-4" />
              <span>Python 3</span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={handleCopyCode}
                className="p-1.5 text-neutral-400 hover:text-white transition-colors"
                title="코드 복사"
              >
                {copied ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
              </button>
              <button
                onClick={handleReset}
                className="p-1.5 text-neutral-400 hover:text-white transition-colors"
                title="코드 초기화"
              >
                <RotateCcw className="h-4 w-4" />
              </button>
              <ShortcutTooltip />
            </div>
          </div>

          {/* Editor */}
          <div className="flex-1" style={{ height: `calc(100% - ${outputPanelHeight}px - 36px)` }}>
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
                padding: { top: 16 },
                lineNumbersMinChars: 3,
                renderLineHighlight: 'line',
                cursorBlinking: 'smooth',
                smoothScrolling: true,
              }}
            />
          </div>

          {/* Vertical resize handle */}
          <div
            className="h-1 bg-neutral-700 hover:bg-blue-500 cursor-row-resize transition-colors"
            onMouseDown={() => setIsResizingV(true)}
          />

          {/* Output panel */}
          <div
            className="bg-neutral-900 flex flex-col border-t border-neutral-700"
            style={{ height: outputPanelHeight }}
          >
            {/* Output tabs and action buttons */}
            <div className="flex items-center justify-between border-b border-neutral-700">
              <div className="flex">
                <button
                  onClick={() => setOutputTab('console')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
                    outputTab === 'console'
                      ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-900/20'
                      : 'text-neutral-400 hover:text-neutral-200'
                  }`}
                >
                  <Terminal className="h-4 w-4" />
                  콘솔
                </button>
                <button
                  onClick={() => setOutputTab('testcases')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium transition-colors ${
                    outputTab === 'testcases'
                      ? 'text-green-400 border-b-2 border-green-400 bg-green-900/20'
                      : 'text-neutral-400 hover:text-neutral-200'
                  }`}
                >
                  <ListChecks className="h-4 w-4" />
                  테스트 결과
                  {submissionResult && (
                    <span className={`ml-1 px-1.5 py-0.5 rounded text-xs ${
                      submissionResult.status === 'accepted'
                        ? 'bg-green-900/50 text-green-400'
                        : 'bg-red-900/50 text-red-400'
                    }`}>
                      {submissionResult.passed_tests}/{submissionResult.total_tests}
                    </span>
                  )}
                </button>
              </div>
              <div className="flex items-center gap-2 px-2">
                <button
                  onClick={handleRun}
                  disabled={isRunning || isSubmitting}
                  className="flex items-center gap-2 px-3 py-1.5 bg-neutral-700 text-white rounded hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
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
                  className="flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all text-sm shadow-lg shadow-green-900/30"
                >
                  {isSubmitting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                  <span>제출</span>
                </button>
              </div>
            </div>

            {/* Output content */}
            <div className="flex-1 overflow-y-auto p-4 font-mono text-sm">
              {outputTab === 'console' && (
                <div className="animate-fade-in">
                  {isRunning && (
                    <div className="flex items-center gap-2 text-neutral-400">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      실행 중...
                    </div>
                  )}
                  {executionResult && (
                    <div>
                      {executionResult.stdout && (
                        <pre className="text-green-400 whitespace-pre-wrap mb-2">
                          {executionResult.stdout}
                        </pre>
                      )}
                      {executionResult.stderr && (
                        <pre className="text-red-400 whitespace-pre-wrap">
                          {executionResult.stderr}
                        </pre>
                      )}
                      <div className="mt-3 pt-3 border-t border-neutral-700 text-neutral-500 flex items-center gap-4">
                        <span className="flex items-center gap-1">
                          <Clock className="h-4 w-4" />
                          {executionResult.execution_time_ms}ms
                        </span>
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          executionResult.status === 'success'
                            ? 'bg-green-900/50 text-green-400'
                            : 'bg-red-900/50 text-red-400'
                        }`}>
                          {executionResult.status === 'success' ? '성공' : '오류'}
                        </span>
                      </div>
                    </div>
                  )}
                  {!isRunning && !executionResult && (
                    <div className="text-neutral-500 flex items-center gap-2">
                      <Terminal className="h-4 w-4" />
                      '실행' 버튼을 눌러 코드를 테스트하세요 (Ctrl + Enter)
                    </div>
                  )}
                </div>
              )}

              {outputTab === 'testcases' && (
                <div className="animate-fade-in">
                  {isSubmitting && (
                    <div className="flex items-center gap-2 text-neutral-400">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      채점 중...
                    </div>
                  )}
                  {submissionResult && (
                    <div className="space-y-2">
                      {/* Summary */}
                      <div className={`p-3 rounded-lg mb-4 ${
                        submissionResult.status === 'accepted'
                          ? 'bg-green-900/30 border border-green-800'
                          : 'bg-red-900/30 border border-red-800'
                      }`}>
                        <div className="flex items-center gap-2">
                          {submissionResult.status === 'accepted' ? (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                          ) : (
                            <XCircle className="h-5 w-5 text-red-500" />
                          )}
                          <span className={`font-bold ${
                            submissionResult.status === 'accepted' ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {submissionResult.status === 'accepted' ? '정답입니다!' : '오답입니다'}
                          </span>
                          <span className="text-neutral-400 ml-auto">
                            {submissionResult.passed_tests}/{submissionResult.total_tests} 테스트 통과
                          </span>
                        </div>
                      </div>

                      {/* Test case results */}
                      {submissionResult.results.map((result, index) => (
                        <div
                          key={index}
                          className={`p-3 rounded-lg border transition-all ${
                            result.is_passed
                              ? 'bg-green-900/20 border-green-800/50'
                              : 'bg-red-900/20 border-red-800/50'
                          }`}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            {result.is_passed ? (
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            ) : (
                              <XCircle className="h-4 w-4 text-red-500" />
                            )}
                            <span className={`font-medium ${
                              result.is_passed ? 'text-green-400' : 'text-red-400'
                            }`}>
                              테스트 케이스 {index + 1}
                            </span>
                          </div>
                          {!result.is_passed && (
                            <div className="text-xs space-y-1 ml-6 mt-2">
                              {result.expected_output && (
                                <div className="flex gap-2">
                                  <span className="text-neutral-500 w-12">예상:</span>
                                  <code className="text-green-400 bg-green-900/30 px-2 py-0.5 rounded">
                                    {result.expected_output}
                                  </code>
                                </div>
                              )}
                              <div className="flex gap-2">
                                <span className="text-neutral-500 w-12">실제:</span>
                                <code className="text-red-400 bg-red-900/30 px-2 py-0.5 rounded">
                                  {result.actual_output || '(출력 없음)'}
                                </code>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {!isSubmitting && !submissionResult && (
                    <div className="text-neutral-500 flex items-center gap-2">
                      <ListChecks className="h-4 w-4" />
                      '제출' 버튼을 눌러 채점을 받으세요 (Ctrl + Shift + Enter)
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Global styles for animations */}
      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.2s ease-out forwards;
        }
      `}</style>
    </div>
  );
}
