/**
 * Shared Playground Page - View playground by share code
 */

import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
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
      setError('Playground not found or not accessible');
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
      setError('Failed to execute code');
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
      setError('Failed to fork playground');
    }
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

  if (error || !playground) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <p className="text-red-500 mb-4">{error || 'Playground not found'}</p>
        <button
          onClick={() => navigate('/playground')}
          className="text-indigo-600 hover:text-indigo-700"
        >
          Browse Playgrounds
        </button>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 text-white">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/playground')}
            className="text-gray-400 hover:text-white"
          >
            ← Browse
          </button>
          <h1 className="font-medium">{playground.title}</h1>
          <span className="text-xs bg-blue-600 px-2 py-0.5 rounded">Shared</span>
          {playground.is_forked && (
            <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">Forked</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2 py-1 bg-gray-700 rounded text-sm">
            {getLanguageDisplayName(playground.language)}
          </span>

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

          {/* Fork button */}
          {isAuthenticated && (
            <button
              onClick={handleFork}
              className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 rounded text-sm"
            >
              Fork to Edit
            </button>
          )}

          {!isAuthenticated && (
            <button
              onClick={() => navigate('/login')}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Login to Fork
            </button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Code editor panel (read-only) */}
        <div className="flex-1 flex flex-col border-r border-gray-300">
          <div className="px-3 py-1 bg-gray-100 border-b text-sm text-gray-600">
            Code (Read Only)
          </div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="flex-1 p-4 font-mono text-sm resize-none focus:outline-none bg-gray-50"
            spellCheck={false}
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
              onChange={(e) => setStdin(e.target.value)}
              className="flex-1 p-4 font-mono text-sm resize-none focus:outline-none"
              placeholder="Enter input for your program..."
              spellCheck={false}
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
                <span className="text-gray-400">Click "Run" to execute the code</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Description bar */}
      {playground.description && (
        <div className="px-4 py-2 bg-gray-50 border-t text-sm text-gray-600">
          {playground.description}
        </div>
      )}

      {/* Stats bar */}
      <div className="px-4 py-1 bg-gray-100 border-t text-xs text-gray-500 flex gap-4">
        <span>{playground.run_count} runs</span>
        <span>{playground.fork_count} forks</span>
      </div>
    </div>
  );
}
