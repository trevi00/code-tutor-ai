/**
 * Badges Page - Enhanced with modern design
 */

import { useEffect, useState } from 'react';
import {
  Award,
  Loader2,
  Trophy,
  Sparkles,
  Lock,
  Target,
  Flame,
  Star,
  Users,
  Zap,
} from 'lucide-react';
import type { UserBadgesResponse, BadgeCategory } from '../../api/gamification';
import { getMyBadges } from '../../api/gamification';
import { BadgeCard } from '../../components/gamification';

const CATEGORY_LABELS: Record<BadgeCategory, string> = {
  problem_solving: '문제 해결',
  streak: '연속 학습',
  mastery: '마스터리',
  social: '소셜',
  special: '특별',
};

const CATEGORY_ICONS: Record<BadgeCategory, React.ReactNode> = {
  problem_solving: <Target className="w-4 h-4" />,
  streak: <Flame className="w-4 h-4" />,
  mastery: <Star className="w-4 h-4" />,
  social: <Users className="w-4 h-4" />,
  special: <Zap className="w-4 h-4" />,
};

export default function BadgesPage() {
  const [badges, setBadges] = useState<UserBadgesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<BadgeCategory | 'all'>('all');

  useEffect(() => {
    loadBadges();
  }, []);

  const loadBadges = async () => {
    try {
      setLoading(true);
      const data = await getMyBadges();
      setBadges(data);
    } catch (error) {
      console.error('Failed to load badges:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-neutral-50 dark:from-slate-900 to-neutral-100 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-amber-500 to-orange-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-neutral-900 dark:text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-neutral-500 dark:text-slate-400">배지 컬렉션 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (!badges) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-neutral-50 dark:from-slate-900 to-neutral-100 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
            <Award className="w-10 h-10 text-red-400" />
          </div>
          <p className="text-neutral-500 dark:text-slate-400">배지 정보를 불러올 수 없습니다.</p>
        </div>
      </div>
    );
  }

  // Filter badges by category
  const filterByCategory = (badge: { category: BadgeCategory }) => {
    return selectedCategory === 'all' || badge.category === selectedCategory;
  };

  const earnedFiltered = badges.earned.filter((ub) => filterByCategory(ub.badge));
  const availableFiltered = badges.available.filter(filterByCategory);
  const collectionRate = ((badges.total_earned / (badges.total_earned + badges.total_available)) * 100);

  return (
    <div className="min-h-screen bg-gradient-to-b from-neutral-50 dark:from-slate-900 to-neutral-100 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-amber-600 via-orange-600 to-yellow-600 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-orange-500/20 rounded-full blur-3xl" />
          <Award className="absolute top-10 right-[10%] w-16 h-16 text-white/10" />
          <Trophy className="absolute bottom-8 left-[15%] w-12 h-12 text-white/10" />
          <Sparkles className="absolute top-16 left-[25%] w-8 h-8 text-white/10" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Sparkles className="w-4 h-4" />
                배지 컬렉션
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Award className="w-10 h-10 text-yellow-200" />
                나의 배지
              </h1>
              <p className="text-amber-100 text-lg max-w-md">
                학습 성과를 배지로 기록하고, 모든 배지를 수집해보세요!
              </p>
            </div>

            {/* Stats Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                  <Trophy className="w-5 h-5 text-emerald-300" />
                </div>
                <div className="text-2xl font-bold text-neutral-900 dark:text-white">{badges.total_earned}</div>
                <div className="text-xs text-amber-200">획득</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-slate-500/20 flex items-center justify-center">
                  <Lock className="w-5 h-5 text-slate-300" />
                </div>
                <div className="text-2xl font-bold text-neutral-900 dark:text-white">{badges.total_available}</div>
                <div className="text-xs text-amber-200">미획득</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Award className="w-5 h-5 text-blue-300" />
                </div>
                <div className="text-2xl font-bold text-neutral-900 dark:text-white">{badges.total_earned + badges.total_available}</div>
                <div className="text-xs text-amber-200">전체</div>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px] border border-white/30">
                <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <Star className="w-5 h-5 text-yellow-300" />
                </div>
                <div className="text-2xl font-bold text-neutral-900 dark:text-white">{collectionRate.toFixed(0)}%</div>
                <div className="text-xs text-amber-200">수집률</div>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-8 max-w-md mx-auto md:mx-0">
            <div className="flex items-center justify-between text-sm text-amber-100 mb-2">
              <span>수집 진행률</span>
              <span>{badges.total_earned} / {badges.total_earned + badges.total_available}</span>
            </div>
            <div className="h-3 bg-white/20 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-yellow-400 to-amber-300 rounded-full transition-all duration-500"
                style={{ width: `${collectionRate}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8 -mt-4">
        {/* Category Filter */}
        <div className="bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-neutral-200/50 dark:border-slate-700/50 p-4 mb-8">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedCategory('all')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium transition-all ${
                selectedCategory === 'all'
                  ? 'bg-gradient-to-r from-amber-600 to-orange-600 text-white shadow-lg shadow-amber-500/25'
                  : 'bg-neutral-200/50 dark:bg-slate-700/50 text-neutral-600 dark:text-slate-300 hover:bg-neutral-300 dark:hover:bg-slate-700 hover:text-neutral-900 dark:hover:text-white'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              전체
            </button>
            {(Object.keys(CATEGORY_LABELS) as BadgeCategory[]).map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium transition-all ${
                  selectedCategory === category
                    ? 'bg-gradient-to-r from-amber-600 to-orange-600 text-white shadow-lg shadow-amber-500/25'
                    : 'bg-neutral-200/50 dark:bg-slate-700/50 text-neutral-600 dark:text-slate-300 hover:bg-neutral-300 dark:hover:bg-slate-700 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                {CATEGORY_ICONS[category]}
                {CATEGORY_LABELS[category]}
              </button>
            ))}
          </div>
        </div>

        {/* Earned Badges */}
        {earnedFiltered.length > 0 && (
          <section className="mb-12">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-neutral-900 dark:text-white">획득한 배지</h2>
                <p className="text-sm text-neutral-500 dark:text-slate-400">{earnedFiltered.length}개의 배지를 획득했습니다</p>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {earnedFiltered.map((userBadge) => (
                <BadgeCard
                  key={userBadge.id}
                  badge={userBadge.badge}
                  earned
                  earnedAt={userBadge.earned_at}
                />
              ))}
            </div>
          </section>
        )}

        {/* Available Badges */}
        {availableFiltered.length > 0 && (
          <section>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-slate-500/20 flex items-center justify-center">
                <Lock className="w-5 h-5 text-slate-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-neutral-900 dark:text-white">미획득 배지</h2>
                <p className="text-sm text-neutral-500 dark:text-slate-400">{availableFiltered.length}개의 배지에 도전하세요</p>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {availableFiltered.map((badge) => (
                <BadgeCard key={badge.id} badge={badge} earned={false} />
              ))}
            </div>
          </section>
        )}

        {/* Empty State */}
        {earnedFiltered.length === 0 && availableFiltered.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-neutral-200/50 dark:bg-slate-700/50 flex items-center justify-center">
              <Award className="w-10 h-10 text-slate-500" />
            </div>
            <h3 className="text-xl font-bold text-neutral-900 dark:text-white mb-2">배지가 없습니다</h3>
            <p className="text-neutral-500 dark:text-slate-400">이 카테고리에는 아직 배지가 없습니다.</p>
          </div>
        )}
      </div>
    </div>
  );
}
