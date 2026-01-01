/**
 * Badges Page
 */

import { useEffect, useState } from 'react';
import { Award, Loader2 } from 'lucide-react';
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
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (!badges) {
    return (
      <div className="text-center py-12 text-gray-500">
        배지 정보를 불러올 수 없습니다.
      </div>
    );
  }

  // Filter badges by category
  const filterByCategory = (badge: { category: BadgeCategory }) => {
    return selectedCategory === 'all' || badge.category === selectedCategory;
  };

  const earnedFiltered = badges.earned.filter((ub) => filterByCategory(ub.badge));
  const availableFiltered = badges.available.filter(filterByCategory);

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
          <Award className="w-8 h-8 text-indigo-600" />
          배지 컬렉션
        </h1>
        <p className="text-gray-600 mt-2">
          획득한 배지와 앞으로 도전할 배지들을 확인하세요
        </p>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-xl shadow p-4 text-center">
          <div className="text-3xl font-bold text-indigo-600">{badges.total_earned}</div>
          <div className="text-sm text-gray-500">획득한 배지</div>
        </div>
        <div className="bg-white rounded-xl shadow p-4 text-center">
          <div className="text-3xl font-bold text-gray-400">{badges.total_available}</div>
          <div className="text-sm text-gray-500">미획득 배지</div>
        </div>
        <div className="bg-white rounded-xl shadow p-4 text-center">
          <div className="text-3xl font-bold text-green-600">
            {badges.total_earned + badges.total_available}
          </div>
          <div className="text-sm text-gray-500">전체 배지</div>
        </div>
        <div className="bg-white rounded-xl shadow p-4 text-center">
          <div className="text-3xl font-bold text-yellow-600">
            {((badges.total_earned / (badges.total_earned + badges.total_available)) * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-500">수집률</div>
        </div>
      </div>

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'all'
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          전체
        </button>
        {(Object.keys(CATEGORY_LABELS) as BadgeCategory[]).map((category) => (
          <button
            key={category}
            onClick={() => setSelectedCategory(category)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedCategory === category
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {CATEGORY_LABELS[category]}
          </button>
        ))}
      </div>

      {/* Earned Badges */}
      {earnedFiltered.length > 0 && (
        <section className="mb-12">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            ✨ 획득한 배지
            <span className="text-sm font-normal text-gray-500">({earnedFiltered.length}개)</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
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
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            🔒 미획득 배지
            <span className="text-sm font-normal text-gray-500">({availableFiltered.length}개)</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {availableFiltered.map((badge) => (
              <BadgeCard key={badge.id} badge={badge} earned={false} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
