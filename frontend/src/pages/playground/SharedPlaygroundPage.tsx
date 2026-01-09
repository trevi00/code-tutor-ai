/**
 * Shared Playground Page - Enhanced with modern design
 * View playground by share code
 */

import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  GitFork,
  Loader2,
  Terminal,
  FileInput,
  FileOutput,
  Code2,
  Clock,
  AlertCircle,
  CheckCircle,
  Share2,
  Copy,
  Check,
  LogIn,
} from 'lucide-react';
import {
  executePlayground,
  forkPlayground,
  getLanguages,
  getPlaygroundByShareCode,
} from '../../api/playground';
import type {
  ExecutionResponse,
  LanguageInfo,
  PlaygroundDetailResponse,
} from '../../api/playground';
import { useAuthStore } from '../../store/authStore';

// Language icons
const LANGUAGE_ICONS: Record<string, string> = {
  python: '🐍',
  javascript: '⚡',
  typescript: '📘',
  java: '☕',
  cpp: '⚙️',
  c: '🔧',
  go: '🐹',
  rust: '🦀',
  ruby: '💎',
};

function getLanguageIcon(langId: string) {
  return LANGUAGE_ICONS[langId] || '📄';
}

export default function SharedPlaygroundPage() {
  const { shareCode } = useParams<{ shareCode: string }>();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();

  const [playground, setPlayground] = useState<PlaygroundDetailResponse | null>(null);
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [code, setCode] = useState('');
  const [stdin, setStdin] = useState('');
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (shareCode) {
      loadPlayground();
      loadLanguages();
    }
  }, [shareCode]);

  const loadPlayground = async () => {
    if (!shareCode) return;

    try {
      setLoading(true);
      setError(null);
      const data = await getPlaygroundByShareCode(shareCode);
      setPlayground(data);
      setCode(data.code);
      setStdin(data.stdin || '');
    } catch (err) {
      console.error('Failed to load playground:', err);
      setError('플레이그라운드를 찾을 수 없거나 접근할 수 없습니다');
    } finally {
      setLoading(false);
    }
  };

  const loadLanguages = async () => {
    try {
      const data = await getLanguages();
      setLanguages(data.languages);
    } catch (err) {
      console.error('Failed to load languages:', err);
    }
  };

  const handleExecute = async () => {
    if (!playground) return;

    try {
      setExecuting(true);
      setExecutionResult(null);
      const result = await executePlayground(playground.id, { code, stdin });
      setExecutionResult(result);
    } catch (err) {
      console.error('Failed to execute:', err);
      setError('코드 실행에 실패했습니다');
    } finally {
      setExecuting(false);
    }
  };

  const handleFork = async () => {
    if (!playground) return;

    try {
      const forked = await forkPlayground(playground.id);
      navigate(`/playground/${forked.id}`);
    } catch (err) {
      console.error('Failed to fork:', err);
      setError('포크에 실패했습니다');
    }
  };

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLanguageDisplayName = (langId: string) => {
    const lang = languages.find((l) => l.id === langId);
    return lang?.display_name || langId;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-slate-400">공유된 플레이그라운드 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !playground) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertCircle className="w-10 h-10 text-red-400" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">플레이그라운드를 찾을 수 없습니다</h2>
          <p className="text-red-400 mb-6">{error || '링크가 올바르지 않거나 삭제되었을 수 있습니다'}</p>
          <button
            onClick={() => navigate('/playground')}
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl transition-all shadow-lg"
          >
            <ArrowLeft className="w-5 h-5" />
            플레이그라운드 둘러보기
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-slate-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/playground')}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/25">
              <span className="text-lg">{getLanguageIcon(playground.language)}</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-white">{playground.title}</h1>
                <span className="flex items-center gap-1 px-2.5 py-0.5 text-xs bg-emerald-500/20 text-emerald-400 rounded-full font-medium">
                  <Share2 className="w-3 h-3" />
                  공유됨
                </span>
                {playground.is_forked && (
                  <span className="flex items-center gap-1 px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full">
                    <GitFork className="w-3 h-3" />
                    Forked
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                <span className="px-2 py-0.5 bg-slate-700 rounded">{getLanguageDisplayName(playground.language)}</span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <Play className="w-3 h-3" />
                  {playground.run_count} 실행
                </span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <GitFork className="w-3 h-3" />
                  {playground.fork_count} 포크
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Run button */}
          <button
            onClick={handleExecute}
            disabled={executing}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-sm font-medium text-white shadow-lg shadow-emerald-500/25 transition-all"
          >
            {executing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                실행 중...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                실행
              </>
            )}
          </button>

          {/* Fork button */}
          {isAuthenticated ? (
            <button
              onClick={handleFork}
              className="flex items-center gap-2 px-4 py-2.5 bg-purple-600 hover:bg-purple-500 rounded-xl text-sm font-medium text-white transition-colors"
            >
              <GitFork className="w-4 h-4" />
              Fork하여 편집
            </button>
          ) : (
            <button
              onClick={() => navigate('/login')}
              className="flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm text-white transition-colors"
            >
              <LogIn className="w-4 h-4" />
              로그인하여 Fork
            </button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code panel */}
        <div className="flex-1 flex flex-col border-r border-slate-700">
          <div className="flex items-center justify-between px-4 py-2.5 bg-slate-800 border-b border-slate-700">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Code2 className="w-4 h-4 text-emerald-400" />
              <span className="font-medium">코드</span>
              <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded">읽기 전용</span>
            </div>
            <button
              onClick={copyCode}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? '복사됨!' : '코드 복사'}
            </button>
          </div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="flex-1 p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none focus:outline-none placeholder-slate-600"
            spellCheck={false}
          />
        </div>

        {/* Right panel */}
        <div className="w-[400px] flex flex-col bg-slate-850">
          {/* Input panel */}
          <div className="flex-1 flex flex-col border-b border-slate-700">
            <div className="flex items-center gap-2 px-4 py-2.5 bg-slate-800 border-b border-slate-700 text-sm text-slate-400">
              <FileInput className="w-4 h-4 text-blue-400" />
              <span className="font-medium">입력 (stdin)</span>
            </div>
            <textarea
              value={stdin}
              onChange={(e) => setStdin(e.target.value)}
              className="flex-1 p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none focus:outline-none placeholder-slate-600"
              placeholder="프로그램에 전달할 입력값을 입력하세요..."
              spellCheck={false}
            />
          </div>

          {/* Output panel */}
          <div className="flex-1 flex flex-col">
            <div className="flex items-center justify-between px-4 py-2.5 bg-slate-800 border-b border-slate-700">
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <FileOutput className="w-4 h-4 text-purple-400" />
                <span className="font-medium">출력</span>
              </div>
              {executionResult && (
                <div className={`flex items-center gap-2 px-2.5 py-1 rounded-lg text-xs ${
                  executionResult.is_success
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {executionResult.is_success ? (
                    <CheckCircle className="w-3.5 h-3.5" />
                  ) : (
                    <AlertCircle className="w-3.5 h-3.5" />
                  )}
                  {executionResult.is_success ? '성공' : `Exit: ${executionResult.exit_code}`}
                  <span className="flex items-center gap-1 text-slate-400 ml-1">
                    <Clock className="w-3 h-3" />
                    {executionResult.execution_time_ms}ms
                  </span>
                </div>
              )}
            </div>
            <div className="flex-1 p-4 font-mono text-sm overflow-auto bg-slate-900">
              {executing ? (
                <div className="flex items-center gap-3 text-slate-400">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>코드 실행 중...</span>
                </div>
              ) : executionResult ? (
                <div>
                  {executionResult.stdout && (
                    <pre className="whitespace-pre-wrap text-slate-100">{executionResult.stdout}</pre>
                  )}
                  {executionResult.stderr && (
                    <pre className="text-red-400 whitespace-pre-wrap mt-2 p-3 bg-red-500/10 rounded-lg">
                      {executionResult.stderr}
                    </pre>
                  )}
                  {!executionResult.stdout && !executionResult.stderr && (
                    <span className="text-slate-500">(출력 없음)</span>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-500">
                  <Terminal className="w-10 h-10 mb-3 opacity-50" />
                  <p className="text-center">
                    <span className="text-emerald-400">"실행"</span> 버튼을 눌러<br />코드를 실행하세요
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Description bar */}
      {playground.description && (
        <div className="px-5 py-3 bg-slate-800 border-t border-slate-700">
          <p className="text-sm text-slate-400">
            <span className="text-slate-500 mr-2">설명:</span>
            {playground.description}
          </p>
        </div>
      )}

      {/* Stats bar */}
      <div className="px-5 py-2 bg-slate-800/50 border-t border-slate-700 text-xs text-slate-500 flex items-center gap-6">
        <span className="flex items-center gap-1.5">
          <Play className="w-3.5 h-3.5" />
          {playground.run_count} 실행
        </span>
        <span className="flex items-center gap-1.5">
          <GitFork className="w-3.5 h-3.5" />
          {playground.fork_count} 포크
        </span>
        <span className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5" />
          수정일: {new Date(playground.updated_at).toLocaleString('ko-KR')}
        </span>
      </div>
    </div>
  );
}
