/**
 * Gamification Widget for Dashboard
 */

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Trophy, Award, Target, ChevronRight, Loader2, Flame } from 'lucide-react';
import type { GamificationOverview } from '../../api/gamification';
import { getGamificationOverview } from '../../api/gamification';
import XPBar from './XPBar';
import BadgeCard from './BadgeCard';

interface GamificationWidgetProps {
  compact?: boolean;
}

export default function GamificationWidget({ compact = false }: GamificationWidgetProps) {
  const [overview, setOverview] = useState<GamificationOverview | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadOverview();
  }, []);

  const loadOverview = async () => {
    try {
      setLoading(true);
      const data = await getGamificationOverview();
      setOverview(data);
    } catch (error) {
      console.error('Failed to load gamification overview:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow dark:shadow-none dark:border dark:border-slate-700 p-6 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-indigo-600 dark:text-indigo-400" />
      </div>
    );
  }

  if (!overview) {
    return null;
  }

  const { stats, recent_badges, active_challenges, leaderboard_rank, next_badge_progress } = overview;

  if (compact) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow dark:shadow-none dark:border dark:border-slate-700 p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-neutral-900 dark:text-white">내 레벨</h3>
          <Link to="/badges" className="text-indigo-600 dark:text-indigo-400 text-sm hover:underline">
            배지 보기
          </Link>
        </div>
        <XPBar stats={stats} size="sm" />
        <div className="flex items-center justify-between mt-3 text-sm text-neutral-600 dark:text-slate-400">
          <div className="flex items-center gap-1">
            <Flame className="w-4 h-4 text-orange-500" />
            <span>{stats.current_streak}일 연속</span>
          </div>
          <div className="flex items-center gap-1">
            <Trophy className="w-4 h-4 text-yellow-500" />
            <span>#{leaderboard_rank || '-'}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow dark:shadow-none dark:border dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white">게이미피케이션</h2>
          <Link
            to="/leaderboard"
            className="text-white/80 hover:text-white text-sm flex items-center gap-1"
          >
            리더보드
            <ChevronRight className="w-4 h-4" />
          </Link>
        </div>
      </div>

      <div className="p-6">
        {/* Level & XP */}
        <div className="mb-6">
          <XPBar stats={stats} size="md" />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center p-3 bg-neutral-50 dark:bg-slate-700/50 rounded-lg transition-all hover:scale-105">
            <div className="flex items-center justify-center gap-1 text-orange-500 mb-1">
              <Flame className="w-5 h-5 animate-pulse" />
            </div>
            <div className="text-2xl font-bold text-neutral-900 dark:text-white">{stats.current_streak}</div>
            <div className="text-xs text-neutral-500 dark:text-slate-400">연속 일</div>
          </div>
          <div className="text-center p-3 bg-neutral-50 dark:bg-slate-700/50 rounded-lg transition-all hover:scale-105">
            <div className="flex items-center justify-center gap-1 text-yellow-500 mb-1">
              <Trophy className="w-5 h-5" />
            </div>
            <div className="text-2xl font-bold text-neutral-900 dark:text-white">#{leaderboard_rank || '-'}</div>
            <div className="text-xs text-neutral-500 dark:text-slate-400">순위</div>
          </div>
          <div className="text-center p-3 bg-neutral-50 dark:bg-slate-700/50 rounded-lg transition-all hover:scale-105">
            <div className="flex items-center justify-center gap-1 text-green-500 mb-1">
              <Target className="w-5 h-5" />
            </div>
            <div className="text-2xl font-bold text-neutral-900 dark:text-white">{stats.problems_solved}</div>
            <div className="text-xs text-neutral-500 dark:text-slate-400">해결</div>
          </div>
        </div>

        {/* Recent Badges */}
        {recent_badges.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-neutral-900 dark:text-white flex items-center gap-2">
                <Award className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                최근 획득 배지
              </h3>
              <Link to="/badges" className="text-indigo-600 dark:text-indigo-400 text-sm hover:underline">
                전체 보기
              </Link>
            </div>
            <div className="flex gap-4 overflow-x-auto pb-2">
              {recent_badges.slice(0, 3).map((ub) => (
                <BadgeCard
                  key={ub.id}
                  badge={ub.badge}
                  earned
                  size="sm"
                  showDetails={false}
                />
              ))}
            </div>
          </div>
        )}

        {/* Next Badge Progress */}
        {next_badge_progress && (
          <div className="mb-6">
            <h3 className="font-semibold text-neutral-900 dark:text-white mb-3">다음 배지까지</h3>
            <div className="bg-neutral-50 dark:bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-neutral-700 dark:text-slate-300">
                  {next_badge_progress.badge}
                </span>
                <span className="text-sm text-neutral-500 dark:text-slate-400">
                  {next_badge_progress.current} / {next_badge_progress.required}
                </span>
              </div>
              <div className="w-full bg-neutral-200 dark:bg-slate-600 rounded-full h-2.5 overflow-hidden">
                <div
                  className="h-2.5 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-700 ease-out animate-pulse"
                  style={{ width: `${next_badge_progress.percentage}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Active Challenges */}
        {active_challenges.length > 0 && (
          <div>
            <h3 className="font-semibold text-neutral-900 dark:text-white mb-3">진행 중인 도전</h3>
            <div className="space-y-2">
              {active_challenges.slice(0, 2).map((uc) => (
                <div
                  key={uc.id}
                  className="flex items-center gap-3 p-3 bg-neutral-50 dark:bg-slate-700/50 rounded-lg hover:bg-neutral-100 dark:hover:bg-slate-700 transition-colors"
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium text-neutral-900 dark:text-white">
                      {uc.challenge.name}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-slate-400">
                      {uc.current_progress} / {uc.challenge.target_value}
                    </div>
                  </div>
                  <div className="w-16">
                    <div className="w-full bg-neutral-200 dark:bg-slate-600 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-1.5 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                        style={{ width: `${uc.progress_percentage}%` }}
                      />
                    </div>
                  </div>
                  <span className="text-xs font-medium text-indigo-600 dark:text-indigo-400">
                    +{uc.challenge.xp_reward} XP
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
