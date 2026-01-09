/**
 * Real-time Collaboration Session Page - Enhanced with modern design
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  Users,
  MessageSquare,
  Send,
  Wifi,
  WifiOff,
  Crown,
  Code2,
  Loader2,
  AlertCircle,
  Circle,
} from 'lucide-react';
import { getSession, joinSession, leaveSession } from '../../api/collaboration';
import { useCollaboration } from '../../hooks/useCollaboration';
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
};

export default function CollaborationPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const [sessionTitle, setSessionTitle] = useState('');
  const [hostId, setHostId] = useState('');
  const [language, setLanguage] = useState('python');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [localCode, setLocalCode] = useState('');

  const {
    isConnected,
    codeContent,
    version,
    participants,
    chatMessages,
    sendCodeChange,
    sendCursorUpdate,
    sendChatMessage,
  } = useCollaboration({
    sessionId: sessionId || '',
    onCodeChange: (content) => {
      setLocalCode(content);
    },
    onError: (err) => {
      setError(err);
    },
  });

  // Load session and join
  useEffect(() => {
    if (!sessionId) return;

    const initSession = async () => {
      try {
        setLoading(true);
        await joinSession(sessionId);
        const session = await getSession(sessionId);
        setSessionTitle(session.title);
        setHostId(session.host_id);
        setLanguage(session.language);
        setLocalCode(session.code_content);
      } catch (err) {
        setError('세션 참가에 실패했습니다');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    initSession();

    return () => {
      leaveSession(sessionId).catch(console.error);
    };
  }, [sessionId]);

  // Sync local code with collaboration state
  useEffect(() => {
    setLocalCode(codeContent);
  }, [codeContent]);

  // Auto-scroll chat
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // Handle code change
  const handleCodeChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newValue = e.target.value;
      const oldValue = localCode;

      // Simple diff - find what changed
      if (newValue.length > oldValue.length) {
        // Insert
        const diffPos = findDiffPosition(oldValue, newValue);
        const inserted = newValue.slice(diffPos, diffPos + (newValue.length - oldValue.length));
        sendCodeChange('insert', diffPos, inserted);
      } else if (newValue.length < oldValue.length) {
        // Delete
        const diffPos = findDiffPosition(newValue, oldValue);
        const deletedLength = oldValue.length - newValue.length;
        sendCodeChange('delete', diffPos, '', deletedLength);
      } else {
        // Replace (same length but different content)
        const diffPos = findDiffPosition(oldValue, newValue);
        if (diffPos < newValue.length) {
          sendCodeChange('replace', diffPos, newValue[diffPos], 1);
        }
      }

      setLocalCode(newValue);
    },
    [localCode, sendCodeChange]
  );

  // Handle cursor movement
  const handleCursorChange = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const { selectionStart } = textarea;
    const lines = localCode.slice(0, selectionStart).split('\n');
    const line = lines.length - 1;
    const column = lines[lines.length - 1].length;

    sendCursorUpdate(line, column);
  }, [localCode, sendCursorUpdate]);

  // Handle chat submit
  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    sendChatMessage(chatInput);
    setChatInput('');
  };

  // Handle leave session
  const handleLeave = async () => {
    if (sessionId) {
      await leaveSession(sessionId);
    }
    navigate('/collaboration');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-violet-500 to-purple-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-slate-400">세션에 참가하는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertCircle className="w-10 h-10 text-red-400" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">세션 참가 실패</h2>
          <p className="text-red-400 mb-6">{error}</p>
          <button
            onClick={() => navigate('/collaboration')}
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white rounded-xl transition-all shadow-lg"
          >
            <ArrowLeft className="w-5 h-5" />
            세션 목록으로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  const activeParticipants = participants.filter((p) => p.isActive);
  const isHost = user?.id === hostId;

  return (
    <div className="h-screen flex flex-col bg-slate-900">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={handleLeave}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-violet-500 to-purple-500 flex items-center justify-center shadow-lg shadow-violet-500/25">
                <span className="text-lg">{LANGUAGE_ICONS[language] || '📄'}</span>
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="font-bold text-white">{sessionTitle}</h1>
                  {isHost && (
                    <span className="flex items-center gap-1 px-2 py-0.5 text-xs bg-amber-500/20 text-amber-400 rounded-full">
                      <Crown className="w-3 h-3" />
                      호스트
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                  <span className="px-2 py-0.5 bg-slate-700 rounded capitalize">{language}</span>
                  <span>•</span>
                  <span>v{version}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Connection Status */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
              isConnected
                ? 'bg-emerald-500/20 text-emerald-400'
                : 'bg-red-500/20 text-red-400'
            }`}>
              {isConnected ? (
                <>
                  <Wifi className="w-4 h-4" />
                  연결됨
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4" />
                  연결 끊김
                </>
              )}
            </div>

            {/* Participant avatars */}
            <div className="flex items-center -space-x-2">
              {activeParticipants.slice(0, 4).map((p) => (
                <div
                  key={p.userId}
                  className="w-8 h-8 rounded-full flex items-center justify-center text-xs text-white font-medium border-2 border-slate-800"
                  style={{ backgroundColor: p.color }}
                  title={p.username}
                >
                  {p.username.charAt(0).toUpperCase()}
                </div>
              ))}
              {activeParticipants.length > 4 && (
                <div className="w-8 h-8 rounded-full bg-slate-600 flex items-center justify-center text-xs text-slate-300 font-medium border-2 border-slate-800">
                  +{activeParticipants.length - 4}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code Editor */}
        <div className="flex-1 flex flex-col">
          <div className="px-4 py-2.5 bg-slate-800 border-b border-slate-700 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Code2 className="w-4 h-4 text-violet-400" />
              <span className="font-medium">코드 에디터</span>
              <span className="text-xs px-2 py-0.5 bg-violet-500/20 text-violet-400 rounded">실시간 공동 편집</span>
            </div>
            <div className="flex items-center gap-2">
              {activeParticipants.slice(0, 3).map((p) => (
                <div key={p.userId} className="flex items-center gap-1.5 px-2 py-1 bg-slate-700/50 rounded text-xs">
                  <Circle className="w-2 h-2" style={{ fill: p.color, color: p.color }} />
                  <span className="text-slate-300">{p.username}</span>
                  {p.cursorLine !== null && (
                    <span className="text-slate-500">L{p.cursorLine + 1}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
          <textarea
            ref={textareaRef}
            value={localCode}
            onChange={handleCodeChange}
            onKeyUp={handleCursorChange}
            onClick={handleCursorChange}
            className="flex-1 p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none focus:outline-none placeholder-slate-600"
            placeholder="여기에 코드를 작성하세요..."
            spellCheck={false}
          />
        </div>

        {/* Right Panel */}
        <div className="w-80 bg-slate-850 border-l border-slate-700 flex flex-col">
          {/* Participants */}
          <div className="p-4 border-b border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <Users className="w-4 h-4 text-violet-400" />
              <h2 className="text-sm font-semibold text-white">
                참가자 ({activeParticipants.length})
              </h2>
            </div>
            <div className="space-y-2">
              {activeParticipants.map((p) => (
                <div
                  key={p.userId}
                  className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors"
                >
                  <div
                    className="w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-medium shadow-lg"
                    style={{ backgroundColor: p.color }}
                  >
                    {p.username.charAt(0).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-white truncate">
                        {p.username}
                      </p>
                      {p.userId === hostId && (
                        <Crown className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                      )}
                    </div>
                    {p.cursorLine !== null && (
                      <p className="text-xs text-slate-500">
                        Line {p.cursorLine + 1}, Col {(p.cursorColumn || 0) + 1}
                      </p>
                    )}
                  </div>
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: p.color }}
                  />
                </div>
              ))}
              {activeParticipants.length === 0 && (
                <p className="text-sm text-slate-500 text-center py-4">
                  참가자가 없습니다
                </p>
              )}
            </div>
          </div>

          {/* Chat */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="px-4 py-3 border-b border-slate-700">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-violet-400" />
                <h2 className="text-sm font-semibold text-white">채팅</h2>
              </div>
            </div>
            <div
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto p-4 space-y-3"
            >
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className="group"
                >
                  <div className="flex items-start gap-2">
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center text-xs text-white flex-shrink-0 bg-violet-500"
                    >
                      {msg.username.charAt(0).toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-sm font-medium text-white">{msg.username}</span>
                        <span className="text-xs text-slate-500">
                          {new Date(msg.timestamp).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300 break-words">{msg.message}</p>
                    </div>
                  </div>
                </div>
              ))}
              {chatMessages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-slate-500">
                  <MessageSquare className="w-10 h-10 mb-3 opacity-50" />
                  <p className="text-sm">아직 메시지가 없습니다</p>
                  <p className="text-xs mt-1">첫 메시지를 보내보세요!</p>
                </div>
              )}
            </div>
            <form onSubmit={handleChatSubmit} className="p-4 border-t border-slate-700">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="메시지를 입력하세요..."
                  className="flex-1 px-3 py-2 bg-slate-800 border border-slate-600 text-white rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent placeholder-slate-500"
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim()}
                  className="px-4 py-2 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-all shadow-lg shadow-violet-500/25"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to find where two strings differ
function findDiffPosition(a: string, b: string): number {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) {
    i++;
  }
  return i;
}
