/**
 * Collaboration Sessions List Page - Enhanced with modern design
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Users,
  Plus,
  Loader2,
  Code2,
  Sparkles,
  Clock,
  CheckCircle,
  XCircle,
  Globe,
  User,
  MessageSquare,
  Zap,
  X,
} from 'lucide-react';
import {
  createSession,
  listActiveSessions,
  listSessions,
} from '../../api/collaboration';
import type { SessionResponse } from '../../api/collaboration';

const LANGUAGE_CONFIG: Record<string, { label: string; color: string }> = {
  python: { label: 'Python', color: 'bg-blue-500/20 text-blue-400' },
  javascript: { label: 'JavaScript', color: 'bg-yellow-500/20 text-yellow-400' },
  typescript: { label: 'TypeScript', color: 'bg-blue-400/20 text-blue-300' },
  java: { label: 'Java', color: 'bg-orange-500/20 text-orange-400' },
  cpp: { label: 'C++', color: 'bg-purple-500/20 text-purple-400' },
};

const STATUS_CONFIG: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  waiting: {
    icon: <Clock className="w-3.5 h-3.5" />,
    color: 'bg-amber-500/20 text-amber-400',
    label: '대기중',
  },
  active: {
    icon: <CheckCircle className="w-3.5 h-3.5" />,
    color: 'bg-emerald-500/20 text-emerald-400',
    label: '활성',
  },
  closed: {
    icon: <XCircle className="w-3.5 h-3.5" />,
    color: 'bg-slate-500/20 text-slate-400',
    label: '종료',
  },
};

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

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-violet-500 to-purple-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-slate-400">세션 불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl" />
          <Users className="absolute top-10 right-[10%] w-16 h-16 text-white/10" />
          <MessageSquare className="absolute bottom-8 left-[15%] w-12 h-12 text-white/10" />
          <Sparkles className="absolute top-16 left-[25%] w-8 h-8 text-white/10" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Zap className="w-4 h-4" />
                실시간 협업
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Users className="w-10 h-10 text-violet-200" />
                협업 세션
              </h1>
              <p className="text-violet-100 text-lg max-w-md">
                실시간으로 코드를 공유하고 함께 문제를 해결하세요
              </p>
            </div>

            {/* Stats & Create Button */}
            <div className="flex flex-col sm:flex-row gap-4 items-center">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                  <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-violet-500/20 flex items-center justify-center">
                    <User className="w-5 h-5 text-violet-300" />
                  </div>
                  <div className="text-2xl font-bold text-white">{mySessions.length}</div>
                  <div className="text-xs text-violet-200">내 세션</div>
                </div>
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                  <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                    <Globe className="w-5 h-5 text-emerald-300" />
                  </div>
                  <div className="text-2xl font-bold text-white">{activeSessions.length}</div>
                  <div className="text-xs text-violet-200">활성 세션</div>
                </div>
              </div>

              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 px-6 py-3 bg-white/20 hover:bg-white/30 backdrop-blur-sm text-white rounded-xl font-medium transition-all border border-white/30"
              >
                <Plus className="w-5 h-5" />
                새 세션 만들기
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8 -mt-4">
        {/* My Sessions */}
        <section className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center">
              <User className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">내 세션</h2>
              <p className="text-sm text-slate-400">내가 생성한 협업 세션들</p>
            </div>
          </div>

          {mySessions.length === 0 ? (
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-700/50 flex items-center justify-center">
                <Users className="w-8 h-8 text-slate-500" />
              </div>
              <p className="text-slate-400 mb-4">아직 생성한 세션이 없습니다.</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white rounded-xl font-medium transition-all shadow-lg shadow-violet-500/25"
              >
                <Plus className="w-4 h-4" />
                첫 세션 만들기
              </button>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {mySessions.map((session) => {
                const statusInfo = STATUS_CONFIG[session.status] || STATUS_CONFIG.closed;
                const langInfo = LANGUAGE_CONFIG[session.language] || { label: session.language, color: 'bg-slate-500/20 text-slate-400' };

                return (
                  <div
                    key={session.id}
                    onClick={() => navigate(`/collaboration/${session.id}`)}
                    className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-5 hover:border-violet-500/50 hover:bg-slate-800/70 transition-all cursor-pointer group"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="font-semibold text-white group-hover:text-violet-400 transition-colors truncate flex-1 mr-2">
                        {session.title}
                      </h3>
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-lg ${statusInfo.color}`}>
                        {statusInfo.icon}
                        {statusInfo.label}
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Code2 className="w-4 h-4 text-slate-500" />
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-lg ${langInfo.color}`}>
                          {langInfo.label}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-400">
                        <Users className="w-4 h-4 text-slate-500" />
                        <span>{session.participant_count} / {session.max_participants} 참가자</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* Active Public Sessions */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
              <Globe className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">활성 세션</h2>
              <p className="text-sm text-slate-400">참여 가능한 공개 세션들</p>
            </div>
          </div>

          {activeSessions.length === 0 ? (
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-700/50 flex items-center justify-center">
                <Globe className="w-8 h-8 text-slate-500" />
              </div>
              <p className="text-slate-400">현재 활성화된 공개 세션이 없습니다.</p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {activeSessions.map((session) => {
                const langInfo = LANGUAGE_CONFIG[session.language] || { label: session.language, color: 'bg-slate-500/20 text-slate-400' };

                return (
                  <div
                    key={session.id}
                    onClick={() => navigate(`/collaboration/${session.id}`)}
                    className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-5 hover:border-emerald-500/50 hover:bg-slate-800/70 transition-all cursor-pointer group"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="font-semibold text-white group-hover:text-emerald-400 transition-colors truncate flex-1 mr-2">
                        {session.title}
                      </h3>
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-lg bg-emerald-500/20 text-emerald-400">
                        <CheckCircle className="w-3.5 h-3.5" />
                        활성
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Code2 className="w-4 h-4 text-slate-500" />
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-lg ${langInfo.color}`}>
                          {langInfo.label}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-400">
                        <Users className="w-4 h-4 text-slate-500" />
                        <span>{session.participant_count} / {session.max_participants} 참가자</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 rounded-2xl p-6 w-full max-w-md border border-slate-700/50 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center">
                  <Plus className="w-5 h-5 text-violet-400" />
                </div>
                새 세션 만들기
              </h2>
              <button
                onClick={() => setShowCreateModal(false)}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  세션 제목
                </label>
                <input
                  type="text"
                  value={newSessionTitle}
                  onChange={(e) => setNewSessionTitle(e.target.value)}
                  placeholder="예: Two Sum 문제 풀기"
                  className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  프로그래밍 언어
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(LANGUAGE_CONFIG).map(([key, config]) => (
                    <button
                      key={key}
                      onClick={() => setNewSessionLanguage(key)}
                      className={`flex items-center gap-2 px-4 py-3 rounded-xl font-medium transition-all ${
                        newSessionLanguage === key
                          ? 'bg-violet-600 text-white'
                          : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-600'
                      }`}
                    >
                      <Code2 className="w-4 h-4" />
                      {config.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-8">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-3 text-slate-300 hover:text-white hover:bg-slate-700 rounded-xl font-medium transition-colors border border-slate-600"
              >
                취소
              </button>
              <button
                onClick={handleCreateSession}
                disabled={creating || !newSessionTitle.trim()}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-violet-500/25"
              >
                {creating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    생성 중...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    생성하기
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
