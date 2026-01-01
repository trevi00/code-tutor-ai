/**
 * Real-time Collaboration Session Page
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getSession, joinSession, leaveSession } from '../../api/collaboration';
import { useCollaboration } from '../../hooks/useCollaboration';
import { useAuthStore } from '../../store/authStore';

export default function CollaborationPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
        setError('Failed to join session');
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
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <p className="text-red-500 mb-4">{error}</p>
        <button
          onClick={() => navigate('/collaboration')}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg"
        >
          Back to Sessions
        </button>
      </div>
    );
  }

  const activeParticipants = participants.filter((p) => p.isActive);
  const isHost = user?.id === hostId;

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={handleLeave}
            className="text-gray-600 hover:text-gray-900"
          >
            &larr; Leave
          </button>
          <h1 className="text-lg font-semibold text-gray-900">{sessionTitle}</h1>
          <span
            className={`px-2 py-0.5 text-xs rounded-full ${
              isConnected
                ? 'bg-green-100 text-green-800'
                : 'bg-red-100 text-red-800'
            }`}
          >
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">v{version}</span>
          {isHost && (
            <span className="px-2 py-0.5 text-xs bg-indigo-100 text-indigo-800 rounded-full">
              Host
            </span>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code Editor */}
        <div className="flex-1 flex flex-col bg-gray-900">
          <div className="px-4 py-2 bg-gray-800 flex items-center justify-between">
            <span className="text-sm text-gray-400">
              {language.charAt(0).toUpperCase() + language.slice(1)}
            </span>
            <div className="flex items-center gap-2">
              {activeParticipants.slice(0, 5).map((p) => (
                <div
                  key={p.userId}
                  className="w-6 h-6 rounded-full flex items-center justify-center text-xs text-white"
                  style={{ backgroundColor: p.color }}
                  title={p.username}
                >
                  {p.username.charAt(0).toUpperCase()}
                </div>
              ))}
              {activeParticipants.length > 5 && (
                <span className="text-xs text-gray-400">
                  +{activeParticipants.length - 5}
                </span>
              )}
            </div>
          </div>
          <textarea
            ref={textareaRef}
            value={localCode}
            onChange={handleCodeChange}
            onKeyUp={handleCursorChange}
            onClick={handleCursorChange}
            className="flex-1 p-4 bg-gray-900 text-gray-100 font-mono text-sm resize-none focus:outline-none"
            placeholder="Start typing your code here..."
            spellCheck={false}
          />
        </div>

        {/* Right Panel */}
        <div className="w-80 bg-white border-l flex flex-col">
          {/* Participants */}
          <div className="p-4 border-b">
            <h2 className="text-sm font-semibold text-gray-700 mb-3">
              Participants ({activeParticipants.length})
            </h2>
            <div className="space-y-2">
              {activeParticipants.map((p) => (
                <div key={p.userId} className="flex items-center gap-2">
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
                    style={{ backgroundColor: p.color }}
                  >
                    {p.username.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {p.username}
                      {p.userId === hostId && (
                        <span className="ml-1 text-xs text-indigo-600">(Host)</span>
                      )}
                    </p>
                    {p.cursorLine !== null && (
                      <p className="text-xs text-gray-500">
                        Line {p.cursorLine + 1}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="p-4 border-b">
              <h2 className="text-sm font-semibold text-gray-700">Chat</h2>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {chatMessages.map((msg, idx) => (
                <div key={idx} className="text-sm">
                  <span className="font-medium text-gray-900">{msg.username}:</span>{' '}
                  <span className="text-gray-600">{msg.message}</span>
                </div>
              ))}
              {chatMessages.length === 0 && (
                <p className="text-sm text-gray-400 text-center">
                  No messages yet
                </p>
              )}
            </div>
            <form onSubmit={handleChatSubmit} className="p-4 border-t">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type a message..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700"
                >
                  Send
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
