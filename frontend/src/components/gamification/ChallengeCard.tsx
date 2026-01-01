/**
 * Challenge Card Component
 */

import { Clock, Target, CheckCircle, Play } from 'lucide-react';
import type { Challenge, UserChallenge, ChallengeType } from '../../api/gamification';

interface ChallengeCardProps {
  challenge: Challenge;
  userChallenge?: UserChallenge;
  onJoin?: () => void;
}

export default function ChallengeCard({ challenge, userChallenge, onJoin }: ChallengeCardProps) {
  const isJoined = !!userChallenge;
  const isCompleted = userChallenge?.status === 'completed';
  const progress = userChallenge?.progress_percentage || 0;

  const typeColors: Record<ChallengeType, { bg: string; text: string; border: string }> = {
    daily: { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-200' },
    weekly: { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200' },
    monthly: { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-200' },
    special: { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-200' },
  };

  const typeLabels: Record<ChallengeType, string> = {
    daily: '일일',
    weekly: '주간',
    monthly: '월간',
    special: '특별',
  };

  const colors = typeColors[challenge.challenge_type];

  return (
    <div
      className={`bg-white rounded-xl shadow-md overflow-hidden border ${
        isCompleted ? 'border-green-200 bg-green-50' : 'border-gray-100'
      }`}
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-1 rounded text-xs font-medium ${colors.bg} ${colors.text}`}>
            {typeLabels[challenge.challenge_type]}
          </span>
          <h3 className="font-medium text-gray-900">{challenge.name}</h3>
        </div>

        {isCompleted && <CheckCircle className="w-5 h-5 text-green-500" />}
      </div>

      {/* Body */}
      <div className="p-4">
        <p className="text-sm text-gray-600 mb-3">{challenge.description}</p>

        {/* Progress */}
        {isJoined && (
          <div className="mb-3">
            <div className="flex justify-between items-center mb-1 text-sm">
              <span className="text-gray-500">
                {userChallenge?.current_progress || 0} / {challenge.target_value}
              </span>
              <span className="font-medium text-indigo-600">{progress.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  isCompleted ? 'bg-green-500' : 'bg-indigo-500'
                }`}
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <Target className="w-4 h-4" />
              <span>+{challenge.xp_reward} XP</span>
            </div>
            {challenge.time_remaining && (
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                <span>{challenge.time_remaining}</span>
              </div>
            )}
          </div>

          {!isJoined && onJoin && (
            <button
              onClick={onJoin}
              className="flex items-center gap-1 px-3 py-1.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors"
            >
              <Play className="w-4 h-4" />
              참여하기
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Compact challenge for lists
interface ChallengeListItemProps {
  userChallenge: UserChallenge;
}

export function ChallengeListItem({ userChallenge }: ChallengeListItemProps) {
  const challenge = userChallenge.challenge;
  const isCompleted = userChallenge.status === 'completed';

  return (
    <div className={`flex items-center gap-4 p-3 rounded-lg ${isCompleted ? 'bg-green-50' : 'bg-gray-50'}`}>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900">{challenge.name}</span>
          {isCompleted && <CheckCircle className="w-4 h-4 text-green-500" />}
        </div>
        <div className="text-sm text-gray-500 mt-0.5">
          {userChallenge.current_progress} / {challenge.target_value}
        </div>
      </div>

      {/* Mini progress */}
      <div className="w-20">
        <div className="w-full bg-gray-200 rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-1.5 rounded-full ${isCompleted ? 'bg-green-500' : 'bg-indigo-500'}`}
            style={{ width: `${userChallenge.progress_percentage}%` }}
          />
        </div>
      </div>

      <span className="text-sm font-medium text-indigo-600">+{challenge.xp_reward} XP</span>
    </div>
  );
}
