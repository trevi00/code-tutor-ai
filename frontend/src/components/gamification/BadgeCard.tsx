/**
 * Badge Card Component
 */

import type { Badge, UserBadge } from '../../api/gamification';
import { RARITY_COLORS, RARITY_LABELS } from '../../api/gamification';

interface BadgeCardProps {
  badge: Badge;
  earned?: boolean;
  earnedAt?: string;
  size?: 'sm' | 'md' | 'lg';
  showDetails?: boolean;
}

export default function BadgeCard({
  badge,
  earned = false,
  earnedAt,
  size = 'md',
  showDetails = true,
}: BadgeCardProps) {
  const colors = RARITY_COLORS[badge.rarity];

  const sizeClasses = {
    sm: 'w-16 h-16',
    md: 'w-24 h-24',
    lg: 'w-32 h-32',
  };

  const iconSizeClasses = {
    sm: 'text-2xl',
    md: 'text-4xl',
    lg: 'text-5xl',
  };

  return (
    <div className={`flex flex-col items-center ${!earned ? 'opacity-50 grayscale' : ''}`}>
      {/* Badge Icon */}
      <div
        className={`${sizeClasses[size]} ${colors.bg} ${colors.border} border-2 rounded-full flex items-center justify-center shadow-md relative`}
      >
        <span className={iconSizeClasses[size]}>{badge.icon}</span>

        {/* Rarity indicator */}
        {earned && (
          <div
            className={`absolute -bottom-1 px-2 py-0.5 rounded-full text-xs font-medium ${colors.bg} ${colors.text} ${colors.border} border`}
          >
            {RARITY_LABELS[badge.rarity]}
          </div>
        )}
      </div>

      {/* Badge Info */}
      {showDetails && (
        <div className="mt-3 text-center">
          <h4 className="font-medium text-gray-900">{badge.name}</h4>
          <p className="text-sm text-gray-500 mt-1 max-w-[150px]">{badge.description}</p>

          {badge.xp_reward > 0 && (
            <div className="mt-1 text-xs text-indigo-600 font-medium">
              +{badge.xp_reward} XP
            </div>
          )}

          {earnedAt && (
            <div className="mt-1 text-xs text-gray-400">
              {new Date(earnedAt).toLocaleDateString('ko-KR')} 획득
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Compact badge display for lists
interface BadgeListItemProps {
  userBadge: UserBadge;
}

export function BadgeListItem({ userBadge }: BadgeListItemProps) {
  const colors = RARITY_COLORS[userBadge.badge.rarity];

  return (
    <div className="flex items-center gap-3 p-3 bg-white rounded-lg shadow-sm border">
      <div
        className={`w-12 h-12 ${colors.bg} ${colors.border} border rounded-full flex items-center justify-center`}
      >
        <span className="text-2xl">{userBadge.badge.icon}</span>
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <h4 className="font-medium text-gray-900">{userBadge.badge.name}</h4>
          <span className={`px-1.5 py-0.5 rounded text-xs ${colors.bg} ${colors.text}`}>
            {RARITY_LABELS[userBadge.badge.rarity]}
          </span>
        </div>
        <p className="text-sm text-gray-500">{userBadge.badge.description}</p>
      </div>
      <div className="text-xs text-gray-400">
        {new Date(userBadge.earned_at).toLocaleDateString('ko-KR')}
      </div>
    </div>
  );
}
