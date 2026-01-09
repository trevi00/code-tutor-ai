/**
 * Playground Editor Page - Enhanced with modern design
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Save,
  Share2,
  Settings,
  GitFork,
  Copy,
  Check,
  Loader2,
  RefreshCw,
  Trash2,
  X,
  Clock,
  Terminal,
  FileInput,
  FileOutput,
  Globe,
  Lock,
  Link2,
  ExternalLink,
  Code2,
  Zap,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import {
  deletePlayground,
  executePlayground,
  forkPlayground,
  getDefaultCode,
  getLanguages,
  getPlayground,
  regenerateShareCode,
  updatePlayground,
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

export default function PlaygroundEditorPage() {
  const { playgroundId } = useParams<{ playgroundId: string }>();
  const navigate = useNavigate();
  const { user } = useAuthStore();

  const [playground, setPlayground] = useState<PlaygroundDetailResponse | null>(null);
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [code, setCode] = useState('');
  const [stdin, setStdin] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [copied, setCopied] = useState(false);

  // Share modal state
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [regenerating, setRegenerating] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);

  // Settings panel state
  const [showSettings, setShowSettings] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editVisibility, setEditVisibility] = useState<'private' | 'unlisted' | 'public'>('private');

  const isOwner = playground && user && playground.owner_id === user.id;

  useEffect(() => {
    if (playgroundId) {
      loadPlayground();
      loadLanguages();
    }
  }, [playgroundId]);

  useEffect(() => {
    if (playground) {
      setShareUrl(`${window.location.origin}/playground/share/${playground.share_code}`);
    }
  }, [playground?.share_code]);

  const loadPlayground = async () => {
    if (!playgroundId) return;

    try {
      setLoading(true);
      setError(null);
      const data = await getPlayground(playgroundId);
      setPlayground(data);
      setCode(data.code);
      setStdin(data.stdin || '');
      setEditTitle(data.title);
      setEditDescription(data.description || '');
      setEditVisibility(data.visibility);
    } catch (err) {
      console.error('Failed to load playground:', err);
      setError('플레이그라운드를 불러오는데 실패했습니다');
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

  const handleCodeChange = (newCode: string) => {
    setCode(newCode);
    setHasUnsavedChanges(true);
  };

  const handleStdinChange = (newStdin: string) => {
    setStdin(newStdin);
    setHasUnsavedChanges(true);
  };

  const handleSave = useCallback(async () => {
    if (!playgroundId || !isOwner) return;

    try {
      setSaving(true);
      await updatePlayground(playgroundId, { code, stdin });
      setHasUnsavedChanges(false);
    } catch (err) {
      console.error('Failed to save:', err);
      setError('저장에 실패했습니다');
    } finally {
      setSaving(false);
    }
  }, [playgroundId, code, stdin, isOwner]);

  // Auto-save on Ctrl+S
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        if (isOwner && hasUnsavedChanges) {
          handleSave();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSave, isOwner, hasUnsavedChanges]);

  const handleExecute = async () => {
    if (!playgroundId) return;

    try {
      setExecuting(true);
      setExecutionResult(null);
      const result = await executePlayground(playgroundId, { code, stdin });
      setExecutionResult(result);
    } catch (err) {
      console.error('Failed to execute:', err);
      setError('코드 실행에 실패했습니다');
    } finally {
      setExecuting(false);
    }
  };

  const handleFork = async () => {
    if (!playgroundId) return;

    try {
      const forked = await forkPlayground(playgroundId);
      navigate(`/playground/${forked.id}`);
    } catch (err) {
      console.error('Failed to fork:', err);
      setError('포크에 실패했습니다');
    }
  };

  const handleLanguageChange = async (newLanguage: string) => {
    if (!playgroundId || !isOwner) return;

    try {
      const { code: defaultCode } = await getDefaultCode(newLanguage);
      await updatePlayground(playgroundId, { language: newLanguage, code: defaultCode });
      setCode(defaultCode);
      setPlayground((prev) => prev ? { ...prev, language: newLanguage, code: defaultCode } : null);
    } catch (err) {
      console.error('Failed to change language:', err);
    }
  };

  const handleSaveSettings = async () => {
    if (!playgroundId || !isOwner) return;

    try {
      setSaving(true);
      const updated = await updatePlayground(playgroundId, {
        title: editTitle,
        description: editDescription,
        visibility: editVisibility,
      });
      setPlayground(updated);
      setShowSettings(false);
    } catch (err) {
      console.error('Failed to save settings:', err);
      setError('설정 저장에 실패했습니다');
    } finally {
      setSaving(false);
    }
  };

  const handleRegenerateShareCode = async () => {
    if (!playgroundId || !isOwner) return;

    try {
      setRegenerating(true);
      const { share_code } = await regenerateShareCode(playgroundId);
      setPlayground((prev) => prev ? { ...prev, share_code } : null);
    } catch (err) {
      console.error('Failed to regenerate share code:', err);
    } finally {
      setRegenerating(false);
    }
  };

  const handleDelete = async () => {
    if (!playgroundId || !isOwner) return;

    if (!confirm('정말 이 플레이그라운드를 삭제하시겠습니까?')) return;

    try {
      await deletePlayground(playgroundId);
      navigate('/playground');
    } catch (err) {
      console.error('Failed to delete:', err);
      setError('삭제에 실패했습니다');
    }
  };

  const copyShareUrl = () => {
    navigator.clipboard.writeText(shareUrl);
    setShareCopied(true);
    setTimeout(() => setShareCopied(false), 2000);
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

  const getVisibilityIcon = (visibility: string) => {
    switch (visibility) {
      case 'public': return <Globe className="w-4 h-4" />;
      case 'unlisted': return <Link2 className="w-4 h-4" />;
      default: return <Lock className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-slate-400">플레이그라운드 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error && !playground) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-400" />
          </div>
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => navigate('/playground')}
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            플레이그라운드 목록으로
          </button>
        </div>
      </div>
    );
  }

  if (!playground) {
    return null;
  }

  return (
    <div className="flex flex-col h-screen bg-slate-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700 relative z-[60]">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/playground')}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 flex items-center justify-center">
              <span className="text-base">{getLanguageIcon(playground.language)}</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-semibold text-white">{playground.title}</h1>
                {playground.is_forked && (
                  <span className="flex items-center gap-1 px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full">
                    <GitFork className="w-3 h-3" />
                    Forked
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span>{getLanguageDisplayName(playground.language)}</span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  {getVisibilityIcon(playground.visibility)}
                  {playground.visibility}
                </span>
              </div>
            </div>
          </div>

          {hasUnsavedChanges && (
            <span className="flex items-center gap-1 px-2 py-1 text-xs bg-yellow-500/20 text-yellow-400 rounded-full">
              <div className="w-1.5 h-1.5 bg-yellow-400 rounded-full animate-pulse" />
              저장되지 않은 변경사항
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Language selector */}
          {isOwner && (
            <select
              value={playground.language}
              onChange={(e) => handleLanguageChange(e.target.value)}
              className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-sm text-white focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            >
              {languages.map((lang) => (
                <option key={lang.id} value={lang.id}>
                  {lang.display_name}
                </option>
              ))}
            </select>
          )}

          {/* Run button */}
          <button
            onClick={handleExecute}
            disabled={executing}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium text-white shadow-lg shadow-emerald-500/25 transition-all"
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

          {/* Save button */}
          {isOwner && (
            <button
              onClick={handleSave}
              disabled={saving || !hasUnsavedChanges}
              className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm text-white transition-colors"
            >
              {saving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              저장
            </button>
          )}

          {/* Fork button */}
          {user && !isOwner && (
            <button
              onClick={handleFork}
              className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-white transition-colors"
            >
              <GitFork className="w-4 h-4" />
              Fork
            </button>
          )}

          {/* Share button */}
          <button
            onClick={() => setShowShareModal(true)}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-white transition-colors"
          >
            <Share2 className="w-4 h-4" />
            공유
          </button>

          {/* Settings button */}
          {isOwner && (
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-white transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code editor panel */}
        <div className="flex-1 flex flex-col border-r border-slate-700">
          <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Code2 className="w-4 h-4" />
              코드
            </div>
            <button
              onClick={copyCode}
              className="flex items-center gap-1.5 px-2 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? '복사됨' : '복사'}
            </button>
          </div>
          <textarea
            value={code}
            onChange={(e) => handleCodeChange(e.target.value)}
            className="flex-1 p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none focus:outline-none placeholder-slate-600"
            placeholder="여기에 코드를 작성하세요..."
            spellCheck={false}
            readOnly={!isOwner}
          />
        </div>

        {/* Right panel */}
        <div className="w-[400px] flex flex-col bg-slate-850">
          {/* Input panel */}
          <div className="flex-1 flex flex-col border-b border-slate-700">
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 border-b border-slate-700 text-sm text-slate-400">
              <FileInput className="w-4 h-4" />
              입력 (stdin)
            </div>
            <textarea
              value={stdin}
              onChange={(e) => handleStdinChange(e.target.value)}
              className="flex-1 p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none focus:outline-none placeholder-slate-600"
              placeholder="프로그램에 전달할 입력값..."
              spellCheck={false}
              readOnly={!isOwner}
            />
          </div>

          {/* Output panel */}
          <div className="flex-1 flex flex-col">
            <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <FileOutput className="w-4 h-4" />
                출력
              </div>
              {executionResult && (
                <div className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${
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
                  <span className="flex items-center gap-1 text-slate-400">
                    <Clock className="w-3 h-3" />
                    {executionResult.execution_time_ms}ms
                  </span>
                </div>
              )}
            </div>
            <div className="flex-1 p-4 font-mono text-sm overflow-auto bg-slate-900">
              {executing ? (
                <div className="flex items-center gap-2 text-slate-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  실행 중...
                </div>
              ) : executionResult ? (
                <div>
                  {executionResult.stdout && (
                    <pre className="whitespace-pre-wrap text-slate-100">{executionResult.stdout}</pre>
                  )}
                  {executionResult.stderr && (
                    <pre className="text-red-400 whitespace-pre-wrap mt-2">
                      {executionResult.stderr}
                    </pre>
                  )}
                  {!executionResult.stdout && !executionResult.stderr && (
                    <span className="text-slate-500">(출력 없음)</span>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-500">
                  <Terminal className="w-8 h-8 mb-2 opacity-50" />
                  <p>"실행" 버튼을 눌러 코드를 실행하세요</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div className="px-4 py-2 bg-slate-800 border-t border-slate-700 text-xs text-slate-400 flex items-center gap-6">
        <span className="flex items-center gap-1">
          <Play className="w-3.5 h-3.5" />
          {playground.run_count} runs
        </span>
        <span className="flex items-center gap-1">
          <GitFork className="w-3.5 h-3.5" />
          {playground.fork_count} forks
        </span>
        <span className="flex items-center gap-1">
          <Clock className="w-3.5 h-3.5" />
          마지막 수정: {new Date(playground.updated_at).toLocaleString('ko-KR')}
        </span>
      </div>

      {/* Share Modal */}
      {showShareModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-2xl p-6 w-full max-w-md shadow-2xl border border-slate-700 animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 flex items-center justify-center">
                  <Share2 className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-xl font-bold text-white">플레이그라운드 공유</h2>
              </div>
              <button
                onClick={() => setShowShareModal(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  공유 URL
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={shareUrl}
                    readOnly
                    className="flex-1 px-4 py-2.5 bg-slate-700 border border-slate-600 rounded-xl text-sm text-slate-200"
                  />
                  <button
                    onClick={copyShareUrl}
                    className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-sm font-medium transition-colors flex items-center gap-2"
                  >
                    {shareCopied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                    {shareCopied ? '복사됨' : '복사'}
                  </button>
                </div>
              </div>

              {isOwner && (
                <div className="flex items-center justify-between pt-4 border-t border-slate-700">
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    {getVisibilityIcon(playground.visibility)}
                    {playground.visibility === 'public' && '모든 사용자가 볼 수 있음'}
                    {playground.visibility === 'unlisted' && '링크가 있는 사용자만'}
                    {playground.visibility === 'private' && '나만 볼 수 있음'}
                  </div>
                  <button
                    onClick={handleRegenerateShareCode}
                    disabled={regenerating}
                    className="flex items-center gap-1.5 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
                  >
                    <RefreshCw className={`w-4 h-4 ${regenerating ? 'animate-spin' : ''}`} />
                    {regenerating ? 'URL 재생성 중...' : 'URL 재생성'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-2xl p-6 w-full max-w-md shadow-2xl border border-slate-700 animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-slate-600 to-slate-500 flex items-center justify-center">
                  <Settings className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-xl font-bold text-white">플레이그라운드 설정</h2>
              </div>
              <button
                onClick={() => setShowSettings(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  제목
                </label>
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  className="w-full px-4 py-2.5 bg-slate-700 border border-slate-600 rounded-xl text-white focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  설명
                </label>
                <textarea
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  rows={3}
                  className="w-full px-4 py-2.5 bg-slate-700 border border-slate-600 rounded-xl text-white focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  공개 설정
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'private', label: 'Private', desc: '나만 볼 수 있음', icon: Lock },
                    { value: 'unlisted', label: 'Unlisted', desc: '링크가 있는 사용자만', icon: Link2 },
                    { value: 'public', label: 'Public', desc: '모든 사용자가 볼 수 있음', icon: Globe },
                  ].map((option) => {
                    const Icon = option.icon;
                    return (
                      <button
                        key={option.value}
                        onClick={() => setEditVisibility(option.value as 'private' | 'unlisted' | 'public')}
                        className={`w-full flex items-center gap-3 p-3 rounded-xl border-2 transition-all ${
                          editVisibility === option.value
                            ? 'border-emerald-500 bg-emerald-500/10'
                            : 'border-slate-600 hover:border-slate-500'
                        }`}
                      >
                        <Icon className={`w-5 h-5 ${editVisibility === option.value ? 'text-emerald-400' : 'text-slate-400'}`} />
                        <div className="text-left">
                          <div className={`text-sm font-medium ${editVisibility === option.value ? 'text-emerald-400' : 'text-white'}`}>
                            {option.label}
                          </div>
                          <div className="text-xs text-slate-400">{option.desc}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="pt-4 border-t border-slate-700">
                <button
                  onClick={handleDelete}
                  className="flex items-center gap-2 text-sm text-red-400 hover:text-red-300 transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  플레이그라운드 삭제
                </button>
              </div>
            </div>

            <div className="flex gap-3 mt-8">
              <button
                onClick={() => setShowSettings(false)}
                className="flex-1 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl font-medium transition-colors"
              >
                취소
              </button>
              <button
                onClick={handleSaveSettings}
                disabled={saving}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-medium disabled:opacity-50 transition-all flex items-center justify-center gap-2"
              >
                {saving ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    저장 중...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    저장
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Styles */}
      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        .animate-fade-in {
          animation: fade-in 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}
