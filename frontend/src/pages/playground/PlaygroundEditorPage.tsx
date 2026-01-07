/**
 * Playground Editor Page - Code editing and execution
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
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

  // Share modal state
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [regenerating, setRegenerating] = useState(false);

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
      setError('Failed to load playground');
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
      setError('Failed to save changes');
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
      setError('Failed to execute code');
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
      setError('Failed to fork playground');
    }
  };

  const handleLanguageChange = async (newLanguage: string) => {
    if (!playgroundId || !isOwner) return;

    try {
      // Get default code for new language
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
      setError('Failed to save settings');
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

    if (!confirm('Are you sure you want to delete this playground?')) return;

    try {
      await deletePlayground(playgroundId);
      navigate('/playground');
    } catch (err) {
      console.error('Failed to delete:', err);
      setError('Failed to delete playground');
    }
  };

  const copyShareUrl = () => {
    navigator.clipboard.writeText(shareUrl);
  };

  const getLanguageDisplayName = (langId: string) => {
    const lang = languages.find((l) => l.id === langId);
    return lang?.display_name || langId;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error && !playground) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <p className="text-red-500 mb-4">{error}</p>
        <button
          onClick={() => navigate('/playground')}
          className="text-indigo-600 hover:text-indigo-700"
        >
          Back to Playgrounds
        </button>
      </div>
    );
  }

  if (!playground) {
    return null;
  }

  return (
    <div className="flex flex-col min-h-[calc(100vh-8rem)]">
      {/* Header - z-[60] to be above the sticky nav (z-50) */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 text-white relative z-[60]">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/playground')}
            className="text-gray-400 hover:text-white"
          >
            ← Back
          </button>
          <h1 className="font-medium">{playground.title}</h1>
          {playground.is_forked && (
            <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">Forked</span>
          )}
          {hasUnsavedChanges && (
            <span className="text-xs text-yellow-400">Unsaved changes</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Language selector */}
          {isOwner && (
            <select
              value={playground.language}
              onChange={(e) => handleLanguageChange(e.target.value)}
              className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
            >
              {languages.map((lang) => (
                <option key={lang.id} value={lang.id}>
                  {lang.display_name}
                </option>
              ))}
            </select>
          )}

          {!isOwner && (
            <span className="px-2 py-1 bg-gray-700 rounded text-sm">
              {getLanguageDisplayName(playground.language)}
            </span>
          )}

          {/* Run button */}
          <button
            onClick={handleExecute}
            disabled={executing}
            className="px-4 py-1 bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded text-sm font-medium flex items-center gap-2"
          >
            {executing ? (
              <>
                <span className="animate-spin">⟳</span> Running...
              </>
            ) : (
              <>▶ Run</>
            )}
          </button>

          {/* Save button */}
          {isOwner && (
            <button
              onClick={handleSave}
              disabled={saving || !hasUnsavedChanges}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm"
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          )}

          {/* Fork button */}
          {user && !isOwner && (
            <button
              onClick={handleFork}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Fork
            </button>
          )}

          {/* Share button */}
          <button
            onClick={() => setShowShareModal(true)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            Share
          </button>

          {/* Settings button */}
          {isOwner && (
            <button
              onClick={() => setShowSettings(true)}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              ⚙️
            </button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Code editor panel */}
        <div className="flex-1 flex flex-col border-r border-gray-300">
          <div className="px-3 py-1 bg-gray-100 border-b text-sm text-gray-600">
            Code
          </div>
          <textarea
            value={code}
            onChange={(e) => handleCodeChange(e.target.value)}
            className="flex-1 p-4 font-mono text-sm resize-none focus:outline-none"
            placeholder="Write your code here..."
            spellCheck={false}
            readOnly={!isOwner}
          />
        </div>

        {/* Right panel */}
        <div className="w-96 flex flex-col">
          {/* Input panel */}
          <div className="flex-1 flex flex-col border-b">
            <div className="px-3 py-1 bg-gray-100 border-b text-sm text-gray-600">
              Input (stdin)
            </div>
            <textarea
              value={stdin}
              onChange={(e) => handleStdinChange(e.target.value)}
              className="flex-1 p-4 font-mono text-sm resize-none focus:outline-none"
              placeholder="Enter input for your program..."
              spellCheck={false}
              readOnly={!isOwner}
            />
          </div>

          {/* Output panel */}
          <div className="flex-1 flex flex-col">
            <div className="px-3 py-1 bg-gray-100 border-b text-sm text-gray-600 flex justify-between items-center">
              <span>Output</span>
              {executionResult && (
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    executionResult.is_success
                      ? 'bg-green-100 text-green-700'
                      : 'bg-red-100 text-red-700'
                  }`}
                >
                  {executionResult.is_success ? 'Success' : `Exit: ${executionResult.exit_code}`}
                  {' · '}
                  {executionResult.execution_time_ms}ms
                </span>
              )}
            </div>
            <div className="flex-1 p-4 font-mono text-sm overflow-auto bg-gray-50">
              {executing ? (
                <div className="text-gray-500">Running...</div>
              ) : executionResult ? (
                <div>
                  {executionResult.stdout && (
                    <pre className="whitespace-pre-wrap">{executionResult.stdout}</pre>
                  )}
                  {executionResult.stderr && (
                    <pre className="text-red-600 whitespace-pre-wrap mt-2">
                      {executionResult.stderr}
                    </pre>
                  )}
                  {!executionResult.stdout && !executionResult.stderr && (
                    <span className="text-gray-500">(No output)</span>
                  )}
                </div>
              ) : (
                <span className="text-gray-400">Click "Run" to execute your code</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div className="px-4 py-1 bg-gray-100 border-t text-xs text-gray-500 flex gap-4">
        <span>{playground.run_count} runs</span>
        <span>{playground.fork_count} forks</span>
        <span>Updated: {new Date(playground.updated_at).toLocaleString()}</span>
      </div>

      {/* Share Modal */}
      {showShareModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Share Playground</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Share URL
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={shareUrl}
                    readOnly
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-sm"
                  />
                  <button
                    onClick={copyShareUrl}
                    className="px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 text-sm"
                  >
                    Copy
                  </button>
                </div>
              </div>

              {isOwner && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">
                    Visibility: {playground.visibility}
                  </span>
                  <button
                    onClick={handleRegenerateShareCode}
                    disabled={regenerating}
                    className="text-sm text-indigo-600 hover:text-indigo-700"
                  >
                    {regenerating ? 'Regenerating...' : 'Regenerate URL'}
                  </button>
                </div>
              )}
            </div>

            <div className="flex justify-end mt-6">
              <button
                onClick={() => setShowShareModal(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Playground Settings</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Title
                </label>
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Visibility
                </label>
                <select
                  value={editVisibility}
                  onChange={(e) => setEditVisibility(e.target.value as 'private' | 'unlisted' | 'public')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                >
                  <option value="private">Private - Only you can see</option>
                  <option value="unlisted">Unlisted - Anyone with link</option>
                  <option value="public">Public - Listed publicly</option>
                </select>
              </div>

              <div className="pt-4 border-t">
                <button
                  onClick={handleDelete}
                  className="text-red-600 hover:text-red-700 text-sm"
                >
                  Delete Playground
                </button>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveSettings}
                disabled={saving}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
