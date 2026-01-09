/**
 * Leaderboard Component - Enhanced with podium, animations, and modern UI
 */

import { useState, useMemo } from 'react';
import { Trophy, Medal, Award, Flame, Target, Crown, Sparkles, TrendingUp, Search, Star } from 'lucide-react';
import type { LeaderboardEntry, LeaderboardResponse } from '../../api/gamification';

interface LeaderboardProps {
  data: LeaderboardResponse;
  currentUserId?: string;
  onPeriodChange?: (period: 'all' | 'weekly' | 'monthly') => void;
}

// Avatar component with gradient background
function UserAvatar({ username, rank, size = 'md' }: { username: string; rank: number; size?: 'sm' | 'md' | 'lg' | 'xl' }) {
  const initial = username.charAt(0).toUpperCase();

  const gradients = [
    'from-yellow-400 to-amber-500', // 1st
    'from-slate-300 to-slate-400', // 2nd
    'from-amber-500 to-orange-600', // 3rd
    'from-indigo-400 to-purple-500',
    'from-blue-400 to-cyan-500',
    'from-green-400 to-emerald-500',
    'from-pink-400 to-rose-500',
    'from-violet-400 to-purple-500',
  ];

  const gradient = rank <= 3 ? gradients[rank - 1] : gradients[(rank % 5) + 3];

  const sizes = {
    sm: 'w-8 h-8 text-sm',
    md: 'w-10 h-10 text-base',
    lg: 'w-14 h-14 text-xl',
    xl: 'w-20 h-20 text-3xl',
  };

  return (
    <div className={`${sizes[size]} rounded-full bg-gradient-to-br ${gradient} flex items-center justify-center text-white font-bold shadow-lg`}>
      {initial}
    </div>
  );
}

// Podium component for top 3
function Podium({ entries, currentUserId }: { entries: LeaderboardEntry[]; currentUserId?: string }) {
  const top3 = entries.slice(0, 3);

  if (top3.length < 3) return null;

  // Reorder for podium: 2nd, 1st, 3rd
  const podiumOrder = [top3[1], top3[0], top3[2]];
  const heights = ['h-24', 'h-32', 'h-20'];
  const positions = ['2nd', '1st', '3rd'];
  const crownColors = ['text-slate-400', 'text-yellow-400', 'text-amber-600'];
  const bgColors = [
    'from-slate-100 to-slate-200 dark:from-slate-700 dark:to-slate-600',
    'from-yellow-100 to-amber-200 dark:from-yellow-900/30 dark:to-amber-900/30',
    'from-amber-100 to-orange-200 dark:from-amber-900/20 dark:to-orange-900/20',
  ];

  return (
    <div className="flex items-end justify-center gap-4 mb-8 pt-12">
      {podiumOrder.map((entry, idx) => {
        if (!entry) return null;
        const isCurrentUser = entry.user_id === currentUserId;
        const actualRank = idx === 1 ? 1 : idx === 0 ? 2 : 3;

        return (
          <div key={entry.user_id} className="flex flex-col items-center animate-fade-in-up" style={{ animationDelay: `${idx * 100}ms` }}>
            {/* Crown for 1st place */}
            {actualRank === 1 && (
              <div className="mb-2 animate-bounce-slow">
                <Crown className="w-8 h-8 text-yellow-400 drop-shadow-lg" fill="currentColor" />
              </div>
            )}

            {/* Avatar */}
            <div className={`relative ${isCurrentUser ? 'ring-4 ring-indigo-400 ring-offset-2 rounded-full' : ''}`}>
              <UserAvatar username={entry.username} rank={actualRank} size={actualRank === 1 ? 'xl' : 'lg'} />
              {actualRank <= 3 && (
                <div className={`absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-white dark:bg-slate-800 flex items-center justify-center shadow-md ${crownColors[actualRank - 1]}`}>
                  {actualRank === 1 && <Trophy className="w-4 h-4" />}
                  {actualRank === 2 && <Medal className="w-4 h-4" />}
                  {actualRank === 3 && <Award className="w-4 h-4" />}
                </div>
              )}
            </div>

            {/* Username */}
            <span className={`mt-2 font-semibold text-sm ${isCurrentUser ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-700 dark:text-slate-200'}`}>
              {entry.username}
              {isCurrentUser && <span className="ml-1 text-xs">(나)</span>}
            </span>

            {/* XP */}
            <span className="text-xs text-slate-500 dark:text-slate-400 mb-2">
              {entry.total_xp.toLocaleString()} XP
            </span>

            {/* Podium Stand */}
            <div className={`${heights[idx]} w-24 bg-gradient-to-t ${bgColors[idx]} rounded-t-lg flex items-center justify-center shadow-inner relative overflow-hidden`}>
              <span className="text-3xl font-black text-slate-400/50 dark:text-slate-500/50">
                {positions[idx].charAt(0)}
              </span>
              {/* Shine effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full hover:translate-x-full transition-transform duration-1000" />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Stats summary cards
function StatsSummary({ data, currentUserId }: { data: LeaderboardResponse; currentUserId?: string }) {
  const totalUsers = data.total ?? data.entries.length;
  const userRank = data.user_rank;
  const percentile = userRank && totalUsers ? Math.round((1 - userRank / totalUsers) * 100) : null;

  const topUser = data.entries[0];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-700">
        <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
          <Target className="w-4 h-4" />
          총 참가자
        </div>
        <div className="text-2xl font-bold text-slate-800 dark:text-white">{(totalUsers ?? 0).toLocaleString()}</div>
      </div>

      {userRank && (
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl p-4 shadow-sm text-white">
          <div className="flex items-center gap-2 text-indigo-100 text-sm mb-1">
            <TrendingUp className="w-4 h-4" />
            내 순위
          </div>
          <div className="text-2xl font-bold">{userRank}위</div>
        </div>
      )}

      {percentile !== null && (
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-sm border border-slate-100 dark:border-slate-700">
          <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm mb-1">
            <Star className="w-4 h-4" />
            상위
          </div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">{percentile}%</div>
        </div>
      )}

      {topUser && (
        <div className="bg-gradient-to-br from-amber-400 to-orange-500 rounded-xl p-4 shadow-sm text-white">
          <div className="flex items-center gap-2 text-amber-100 text-sm mb-1">
            <Crown className="w-4 h-4" />
            1위
          </div>
          <div className="text-lg font-bold truncate">{topUser.username}</div>
        </div>
      )}
    </div>
  );
}

export default function Leaderboard({ data, currentUserId, onPeriodChange }: LeaderboardProps) {
  const [period, setPeriod] = useState<'all' | 'weekly' | 'monthly'>(
    data.period as 'all' | 'weekly' | 'monthly'
  );
  const [searchQuery, setSearchQuery] = useState('');

  const handlePeriodChange = (newPeriod: 'all' | 'weekly' | 'monthly') => {
    setPeriod(newPeriod);
    onPeriodChange?.(newPeriod);
  };

  // Filter entries by search
  const filteredEntries = useMemo(() => {
    if (!searchQuery.trim()) return data.entries;
    return data.entries.filter(entry =>
      entry.username.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [data.entries, searchQuery]);

  // Entries excluding top 3 (they're shown in podium)
  const remainingEntries = useMemo(() => {
    if (searchQuery.trim()) return filteredEntries;
    return filteredEntries.slice(3);
  }, [filteredEntries, searchQuery]);

  const getRankBgColor = (rank: number, isCurrentUser: boolean) => {
    if (isCurrentUser) return 'bg-indigo-50 dark:bg-indigo-900/20 border-l-indigo-500';
    switch (rank) {
      case 1: return 'bg-gradient-to-r from-yellow-50 to-amber-50 dark:from-yellow-900/10 dark:to-amber-900/10 border-l-yellow-400';
      case 2: return 'bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-800 dark:to-gray-800 border-l-slate-400';
      case 3: return 'bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/10 dark:to-orange-900/10 border-l-amber-500';
      default: return 'bg-white dark:bg-slate-800 border-l-transparent hover:border-l-indigo-300';
    }
  };

  return (
    <div className="space-y-6">
      {/* Period Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex gap-1 bg-slate-100 dark:bg-slate-800 rounded-xl p-1.5">
          {(['all', 'weekly', 'monthly'] as const).map((p) => (
            <button
              key={p}
              onClick={() => handlePeriodChange(p)}
              className={`px-5 py-2 rounded-lg text-sm font-medium transition-all ${
                period === p
                  ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              {p === 'all' ? '전체' : p === 'weekly' ? '주간' : '월간'}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="사용자 검색..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-4 py-2 border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 focus:ring-2 focus:ring-indigo-500 focus:border-transparent w-full sm:w-64"
          />
        </div>
      </div>

      {/* Stats Summary */}
      <StatsSummary data={data} currentUserId={currentUserId} />

      {/* Podium for Top 3 */}
      {!searchQuery && data.entries.length >= 3 && (
        <Podium entries={data.entries} currentUserId={currentUserId} />
      )}

      {/* Leaderboard List */}
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden border border-slate-100 dark:border-slate-700">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
          <div className="flex items-center gap-3">
            <Trophy className="w-6 h-6 text-white" />
            <h2 className="text-lg font-bold text-white">
              {searchQuery ? `검색 결과 (${filteredEntries.length}명)` : '순위표'}
            </h2>
          </div>
        </div>

        {/* Table Header */}
        <div className="grid grid-cols-12 gap-4 px-6 py-3 bg-slate-50 dark:bg-slate-900/50 text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">
          <div className="col-span-1">순위</div>
          <div className="col-span-5">사용자</div>
          <div className="col-span-2 text-center">레벨</div>
          <div className="col-span-2 text-center">문제</div>
          <div className="col-span-2 text-right">XP</div>
        </div>

        {/* List */}
        <div className="divide-y divide-slate-100 dark:divide-slate-700">
          {remainingEntries.length === 0 ? (
            <div className="p-8 text-center text-slate-500 dark:text-slate-400">
              {searchQuery ? '검색 결과가 없습니다.' : '아직 리더보드 데이터가 없습니다.'}
            </div>
          ) : (
            remainingEntries.map((entry, idx) => {
              const isCurrentUser = entry.user_id === currentUserId;
              return (
                <div
                  key={entry.user_id}
                  className={`grid grid-cols-12 gap-4 px-6 py-4 items-center border-l-4 transition-colors ${getRankBgColor(entry.rank, isCurrentUser)} animate-fade-in`}
                  style={{ animationDelay: `${idx * 30}ms` }}
                >
                  {/* Rank */}
                  <div className="col-span-1">
                    <span className={`font-bold ${entry.rank <= 10 ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-500 dark:text-slate-400'}`}>
                      {entry.rank}
                    </span>
                  </div>

                  {/* User */}
                  <div className="col-span-5 flex items-center gap-3">
                    <UserAvatar username={entry.username} rank={entry.rank} size="sm" />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`font-medium truncate ${isCurrentUser ? 'text-indigo-700 dark:text-indigo-300' : 'text-slate-900 dark:text-white'}`}>
                          {entry.username}
                        </span>
                        {isCurrentUser && (
                          <span className="flex-shrink-0 px-1.5 py-0.5 bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 rounded text-xs font-medium">
                            나
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
                        <Flame className="w-3 h-3 text-orange-400" />
                        <span>{entry.current_streak}일 스트릭</span>
                      </div>
                    </div>
                  </div>

                  {/* Level */}
                  <div className="col-span-2 text-center">
                    <span className="inline-flex items-center gap-1 px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded-full text-sm">
                      <Sparkles className="w-3 h-3 text-purple-500" />
                      <span className="font-medium text-slate-700 dark:text-slate-200">Lv.{entry.level}</span>
                    </span>
                  </div>

                  {/* Problems */}
                  <div className="col-span-2 text-center">
                    <span className="inline-flex items-center gap-1 text-sm text-slate-600 dark:text-slate-300">
                      <Target className="w-4 h-4 text-green-500" />
                      {entry.problems_solved}
                    </span>
                  </div>

                  {/* XP */}
                  <div className="col-span-2 text-right">
                    <span className="font-bold text-indigo-600 dark:text-indigo-400">
                      {entry.total_xp.toLocaleString()}
                    </span>
                    <span className="text-slate-400 dark:text-slate-500 ml-1 text-sm">XP</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Styles */}
      <style>{`
        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes bounce-slow {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-5px); }
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.5s ease-out forwards;
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-out forwards;
        }
        .animate-bounce-slow {
          animation: bounce-slow 2s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}

// Compact leaderboard for dashboard
interface CompactLeaderboardProps {
  entries: LeaderboardEntry[];
  currentUserId?: string;
  limit?: number;
}

export function CompactLeaderboard({ entries, currentUserId, limit = 5 }: CompactLeaderboardProps) {
  const displayEntries = entries.slice(0, limit);

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1: return <Trophy className="w-4 h-4 text-yellow-500" />;
      case 2: return <Medal className="w-4 h-4 text-slate-400" />;
      case 3: return <Award className="w-4 h-4 text-amber-600" />;
      default: return <span className="w-4 h-4 flex items-center justify-center text-xs font-bold text-slate-400">{rank}</span>;
    }
  };

  return (
    <div className="space-y-2">
      {displayEntries.map((entry) => {
        const isCurrentUser = entry.user_id === currentUserId;
        return (
          <div
            key={entry.user_id}
            className={`flex items-center gap-3 p-2.5 rounded-lg transition-colors ${
              isCurrentUser
                ? 'bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800'
                : 'hover:bg-slate-50 dark:hover:bg-slate-800'
            }`}
          >
            <div className="w-6 flex justify-center">{getRankIcon(entry.rank)}</div>
            <UserAvatar username={entry.username} rank={entry.rank} size="sm" />
            <div className="flex-1 min-w-0">
              <span className={`font-medium truncate block ${isCurrentUser ? 'text-indigo-700 dark:text-indigo-300' : 'text-slate-900 dark:text-white'}`}>
                {entry.username}
              </span>
            </div>
            <span className="font-bold text-indigo-600 dark:text-indigo-400 text-sm">
              {entry.total_xp.toLocaleString()}
            </span>
          </div>
        );
      })}
    </div>
  );
}
