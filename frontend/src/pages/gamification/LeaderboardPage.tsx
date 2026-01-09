/**
 * Leaderboard Page
 */

import { useEffect, useState, useCallback } from 'react';
import { Trophy, Loader2 } from 'lucide-react';
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
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
          <Trophy className="w-8 h-8 text-yellow-500" />
          리더보드
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          코드 튜터 AI의 최고 학습자들을 만나보세요
        </p>
      </div>

      {/* Leaderboard */}
      {leaderboard && (
        <Leaderboard
          data={leaderboard}
          currentUserId={user?.id}
          onPeriodChange={handlePeriodChange}
        />
      )}
    </div>
  );
}
