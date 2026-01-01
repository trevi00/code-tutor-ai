/**
 * Leaderboard Component
 */

import { useState } from 'react';
import { Trophy, Medal, Award, Flame, Target } from 'lucide-react';
import type { LeaderboardEntry, LeaderboardResponse } from '../../api/gamification';

interface LeaderboardProps {
  data: LeaderboardResponse;
  currentUserId?: string;
  onPeriodChange?: (period: 'all' | 'weekly' | 'monthly') => void;
}

export default function Leaderboard({ data, currentUserId, onPeriodChange }: LeaderboardProps) {
  const [period, setPeriod] = useState<'all' | 'weekly' | 'monthly'>(
    data.period as 'all' | 'weekly' | 'monthly'
  );

  const handlePeriodChange = (newPeriod: 'all' | 'weekly' | 'monthly') => {
    setPeriod(newPeriod);
    onPeriodChange?.(newPeriod);
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Trophy className="w-6 h-6 text-yellow-500" />;
      case 2:
        return <Medal className="w-6 h-6 text-gray-400" />;
      case 3:
        return <Award className="w-6 h-6 text-amber-600" />;
      default:
        return <span className="w-6 h-6 flex items-center justify-center text-gray-500 font-bold">{rank}</span>;
    }
  };

  const getRankBgColor = (rank: number) => {
    switch (rank) {
      case 1:
        return 'bg-gradient-to-r from-yellow-50 to-amber-50 border-yellow-200';
      case 2:
        return 'bg-gradient-to-r from-gray-50 to-slate-50 border-gray-200';
      case 3:
        return 'bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200';
      default:
        return 'bg-white border-gray-100';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Trophy className="w-6 h-6" />
            리더보드
          </h2>

          {/* Period Tabs */}
          <div className="flex gap-1 bg-white/20 rounded-lg p-1">
            {(['all', 'weekly', 'monthly'] as const).map((p) => (
              <button
                key={p}
                onClick={() => handlePeriodChange(p)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  period === p
                    ? 'bg-white text-indigo-600'
                    : 'text-white/80 hover:text-white hover:bg-white/10'
                }`}
              >
                {p === 'all' ? '전체' : p === 'weekly' ? '주간' : '월간'}
              </button>
            ))}
          </div>
        </div>

        {/* User's rank */}
        {data.user_rank && (
          <div className="mt-3 text-white/90 text-sm">
            나의 순위: <span className="font-bold text-white">{data.user_rank}위</span>
          </div>
        )}
      </div>

      {/* Leaderboard List */}
      <div className="divide-y divide-gray-100">
        {data.entries.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            아직 리더보드 데이터가 없습니다.
          </div>
        ) : (
          data.entries.map((entry) => (
            <LeaderboardRow
              key={entry.user_id}
              entry={entry}
              isCurrentUser={entry.user_id === currentUserId}
              rankIcon={getRankIcon(entry.rank)}
              bgColor={getRankBgColor(entry.rank)}
            />
          ))
        )}
      </div>
    </div>
  );
}

interface LeaderboardRowProps {
  entry: LeaderboardEntry;
  isCurrentUser: boolean;
  rankIcon: React.ReactNode;
  bgColor: string;
}

function LeaderboardRow({ entry, isCurrentUser, rankIcon, bgColor }: LeaderboardRowProps) {
  return (
    <div
      className={`flex items-center px-6 py-4 ${bgColor} border-l-4 ${
        isCurrentUser ? 'border-l-indigo-500 bg-indigo-50' : 'border-l-transparent'
      }`}
    >
      {/* Rank */}
      <div className="w-12 flex-shrink-0">{rankIcon}</div>

      {/* User Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`font-medium ${isCurrentUser ? 'text-indigo-700' : 'text-gray-900'}`}>
            {entry.username}
          </span>
          {isCurrentUser && (
            <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
              나
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 mt-1 text-sm text-gray-500">
          <span className="inline-flex items-center gap-1">
            <span className="font-medium text-indigo-600">Lv.{entry.level}</span>
            <span>{entry.level_title}</span>
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-1 text-gray-500">
          <Target className="w-4 h-4" />
          <span>{entry.problems_solved}</span>
        </div>
        <div className="flex items-center gap-1 text-orange-500">
          <Flame className="w-4 h-4" />
          <span>{entry.current_streak}</span>
        </div>
        <div className="w-24 text-right">
          <span className="font-bold text-indigo-600">{entry.total_xp.toLocaleString()}</span>
          <span className="text-gray-400 ml-1">XP</span>
        </div>
      </div>
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

  return (
    <div className="space-y-2">
      {displayEntries.map((entry) => (
        <div
          key={entry.user_id}
          className={`flex items-center gap-3 p-2 rounded-lg ${
            entry.user_id === currentUserId ? 'bg-indigo-50' : 'hover:bg-gray-50'
          }`}
        >
          <span className="w-6 text-center font-bold text-gray-400">{entry.rank}</span>
          <div className="flex-1 min-w-0">
            <span className="font-medium text-gray-900 truncate">{entry.username}</span>
          </div>
          <span className="font-medium text-indigo-600">{entry.total_xp.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}
