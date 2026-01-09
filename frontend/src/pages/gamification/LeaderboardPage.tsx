/**
 * Leaderboard Page - Enhanced with modern design
 */

import { useEffect, useState, useCallback } from 'react';
import { Trophy, Loader2, Users, Flame, Award, Crown } from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import type { LeaderboardResponse } from '../../api/gamification';
import { getLeaderboard } from '../../api/gamification';
import { Leaderboard } from '../../components/gamification';

export default function LeaderboardPage() {
  const { user } = useAuthStore();
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState<'all' | 'weekly' | 'monthly'>('all');

  const loadLeaderboard = useCallback(async (selectedPeriod: 'all' | 'weekly' | 'monthly') => {
    try {
      setLoading(true);
      const data = await getLeaderboard(selectedPeriod, 100, 0);
      setLeaderboard(data);
    } catch (error) {
      console.error('Failed to load leaderboard:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadLeaderboard(period);
  }, [period, loadLeaderboard]);

  const handlePeriodChange = (newPeriod: 'all' | 'weekly' | 'monthly') => {
    setPeriod(newPeriod);
  };

  if (loading && !leaderboard) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">리더보드 불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Trophy className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Crown className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Award className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-5xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Users className="w-4 h-4" />
                실시간 순위
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Trophy className="w-10 h-10 text-yellow-300" />
                리더보드
              </h1>
              <p className="text-indigo-100 text-lg max-w-md">
                Code Tutor AI의 최고 학습자들과 경쟁하고, 함께 성장하세요!
              </p>
            </div>

            {/* Quick Stats */}
            <div className="flex gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Flame className="w-6 h-6 text-orange-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{leaderboard?.total || 0}</div>
                <div className="text-xs text-indigo-200">참가자</div>
              </div>
              {leaderboard?.user_rank && (
                <div className="bg-white/20 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px] border border-white/30">
                  <Award className="w-6 h-6 text-yellow-300 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-white">{leaderboard.user_rank}위</div>
                  <div className="text-xs text-indigo-200">내 순위</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto px-6 py-8 -mt-6">
        {leaderboard && (
          <Leaderboard
            data={leaderboard}
            currentUserId={user?.id}
            onPeriodChange={handlePeriodChange}
          />
        )}
      </div>

      {/* Styles */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(-5deg); }
        }
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float-delayed 5s ease-in-out infinite;
          animation-delay: 1s;
        }
      `}</style>
    </div>
  );
}
