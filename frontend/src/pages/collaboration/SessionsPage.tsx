/**
 * Collaboration Sessions List Page
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  createSession,
  listActiveSessions,
  listSessions,
} from '../../api/collaboration';
import type { SessionResponse } from '../../api/collaboration';

export default function SessionsPage() {
  const navigate = useNavigate();
  const [mySessions, setMySessions] = useState<SessionResponse[]>([]);
  const [activeSessions, setActiveSessions] = useState<SessionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newSessionTitle, setNewSessionTitle] = useState('');
  const [newSessionLanguage, setNewSessionLanguage] = useState('python');

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const [myData, activeData] = await Promise.all([
        listSessions(true),
        listActiveSessions(10),
      ]);
      setMySessions(myData.sessions);
      setActiveSessions(activeData.sessions);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async () => {
    if (!newSessionTitle.trim()) return;

    try {
      setCreating(true);
      const session = await createSession({
        title: newSessionTitle,
        language: newSessionLanguage,
      });
      navigate(`/collaboration/${session.id}`);
    } catch (error) {
      console.error('Failed to create session:', error);
    } finally {
      setCreating(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const colors: Record<string, string> = {
      waiting: 'bg-yellow-100 text-yellow-800',
      active: 'bg-green-100 text-green-800',
      closed: 'bg-gray-100 text-gray-800',
    };
    return colors[status] || colors.closed;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Collaboration Sessions</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
        >
          Create Session
        </button>
      </div>

      {/* My Sessions */}
      <section className="mb-12">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">My Sessions</h2>
        {mySessions.length === 0 ? (
          <p className="text-gray-500">No active sessions. Create one to start collaborating!</p>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {mySessions.map((session) => (
              <div
                key={session.id}
                onClick={() => navigate(`/collaboration/${session.id}`)}
                className="p-4 bg-white rounded-lg shadow hover:shadow-md transition-shadow cursor-pointer border border-gray-200"
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-medium text-gray-900 truncate">{session.title}</h3>
                  <span
                    className={`px-2 py-0.5 text-xs rounded-full ${getStatusBadge(session.status)}`}
                  >
                    {session.status}
                  </span>
                </div>
                <div className="text-sm text-gray-500 space-y-1">
                  <p>Language: {session.language}</p>
                  <p>
                    Participants: {session.participant_count} / {session.max_participants}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Active Public Sessions */}
      <section>
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Active Sessions</h2>
        {activeSessions.length === 0 ? (
          <p className="text-gray-500">No active public sessions.</p>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {activeSessions.map((session) => (
              <div
                key={session.id}
                onClick={() => navigate(`/collaboration/${session.id}`)}
                className="p-4 bg-white rounded-lg shadow hover:shadow-md transition-shadow cursor-pointer border border-gray-200"
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-medium text-gray-900 truncate">{session.title}</h3>
                  <span className="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-800">
                    active
                  </span>
                </div>
                <div className="text-sm text-gray-500 space-y-1">
                  <p>Language: {session.language}</p>
                  <p>
                    Participants: {session.participant_count} / {session.max_participants}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Create New Session</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Session Title
                </label>
                <input
                  type="text"
                  value={newSessionTitle}
                  onChange={(e) => setNewSessionTitle(e.target.value)}
                  placeholder="Enter session title"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Language
                </label>
                <select
                  value={newSessionLanguage}
                  onChange={(e) => setNewSessionLanguage(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="python">Python</option>
                  <option value="javascript">JavaScript</option>
                  <option value="typescript">TypeScript</option>
                  <option value="java">Java</option>
                  <option value="cpp">C++</option>
                </select>
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateSession}
                disabled={creating || !newSessionTitle.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {creating ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
