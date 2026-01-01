import { Link } from 'react-router-dom';
import type {
  QualityProfile,
  QualityRecommendation,
  QualityImprovementSuggestion,
} from '@/types';

interface QualityRecommendationsProps {
  profile: QualityProfile;
  recommendations: QualityRecommendation[];
  suggestions: QualityImprovementSuggestion[];
}

export function QualityRecommendations({
  profile,
  recommendations,
  suggestions,
}: QualityRecommendationsProps) {
  // Dimension labels
  const dimensionLabels: Record<string, string> = {
    correctness: '정확성',
    efficiency: '효율성',
    readability: '가독성',
    best_practices: '모범사례',
  };

  // Difficulty styles
  const difficultyStyles: Record<string, string> = {
    easy: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    hard: 'bg-red-100 text-red-800',
  };

  const difficultyLabels: Record<string, string> = {
    easy: 'Easy',
    medium: 'Medium',
    hard: 'Hard',
  };

  // Trend icons and styles
  const trendConfig: Record<string, { icon: string; label: string; color: string }> = {
    improving: { icon: '📈', label: '향상 중', color: 'text-green-600' },
    stable: { icon: '➡️', label: '유지 중', color: 'text-blue-600' },
    declining: { icon: '📉', label: '하락 중', color: 'text-yellow-600' },
    new_user: { icon: '🌱', label: '시작 단계', color: 'text-purple-600' },
    insufficient_data: { icon: '📊', label: '데이터 부족', color: 'text-gray-500' },
  };

  // Suggestion type icons
  const suggestionIcons: Record<string, string> = {
    dimension: '🎯',
    smell: '👃',
    complexity: '🔄',
    encouragement: '🎉',
    warning: '⚠️',
    start: '🚀',
  };

  if (!profile.has_data) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-800">품질 기반 추천</h2>
          <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">AI Powered</span>
        </div>
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-2 block">🎯</span>
          <p>코드를 제출하면 품질 분석을 기반으로</p>
          <p>맞춤형 문제를 추천해드려요!</p>
        </div>
      </div>
    );
  }

  const trend = trendConfig[profile.improvement_trend] || trendConfig.stable;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">품질 기반 추천</h2>
        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">AI Powered</span>
      </div>

      {/* Quality Profile Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {/* Improvement Trend */}
        <div className="bg-gradient-to-br from-gray-50 to-slate-50 rounded-lg p-3 text-center">
          <span className="text-xl">{trend.icon}</span>
          <p className={`text-sm font-medium ${trend.color}`}>{trend.label}</p>
        </div>

        {/* Weak Areas */}
        <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-lg p-3 text-center">
          <span className="text-xl">🔧</span>
          <p className="text-sm font-medium text-red-600">
            {profile.weak_areas.length > 0
              ? profile.weak_areas.map(a => dimensionLabels[a] || a).join(', ')
              : '없음'}
          </p>
          <p className="text-xs text-gray-500">보완 필요</p>
        </div>

        {/* Strong Areas */}
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-3 text-center">
          <span className="text-xl">💪</span>
          <p className="text-sm font-medium text-green-600">
            {profile.strong_areas.length > 0
              ? profile.strong_areas.map(a => dimensionLabels[a] || a).join(', ')
              : '개발 중'}
          </p>
          <p className="text-xs text-gray-500">강점</p>
        </div>

        {/* Complexity */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-3 text-center">
          <span className="text-xl">🔄</span>
          <p className={`text-sm font-medium ${
            profile.avg_complexity > 10 ? 'text-red-600' :
            profile.avg_complexity > 5 ? 'text-yellow-600' : 'text-green-600'
          }`}>
            {profile.avg_complexity.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500">평균 복잡도</p>
        </div>
      </div>

      {/* Recommended Problems */}
      {recommendations.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">추천 문제</h3>
          <div className="space-y-2">
            {recommendations.slice(0, 3).map((rec) => (
              <Link
                key={rec.id}
                to={`/problems/${rec.id}`}
                className="flex items-center justify-between p-3 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-blue-900 truncate">{rec.title}</p>
                  <p className="text-xs text-blue-700">{rec.reason}</p>
                </div>
                <div className="flex items-center gap-2 ml-2">
                  <span className={`px-2 py-0.5 text-xs font-medium rounded ${difficultyStyles[rec.difficulty]}`}>
                    {difficultyLabels[rec.difficulty]}
                  </span>
                </div>
              </Link>
            ))}
          </div>
          {recommendations.length > 3 && (
            <Link
              to="/problems"
              className="block mt-2 text-center text-sm text-blue-600 hover:text-blue-800"
            >
              더 많은 추천 보기 →
            </Link>
          )}
        </div>
      )}

      {/* Improvement Suggestions */}
      {suggestions.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">개선 제안</h3>
          <div className="space-y-2">
            {suggestions.slice(0, 3).map((sug, idx) => {
              const icon = suggestionIcons[sug.type] || '💡';
              const bgColor = sug.type === 'encouragement' ? 'bg-green-50' :
                             sug.type === 'warning' ? 'bg-yellow-50' : 'bg-gray-50';
              const textColor = sug.type === 'encouragement' ? 'text-green-800' :
                               sug.type === 'warning' ? 'text-yellow-800' : 'text-gray-800';

              return (
                <div
                  key={idx}
                  className={`p-3 rounded-lg ${bgColor}`}
                >
                  <div className="flex items-start">
                    <span className="mr-2">{icon}</span>
                    <div className="flex-1">
                      <p className={`text-sm ${textColor}`}>{sug.message}</p>
                      {sug.tips && sug.tips.length > 0 && (
                        <ul className="mt-2 text-xs text-gray-600 space-y-1">
                          {sug.tips.slice(0, 2).map((tip, tipIdx) => (
                            <li key={tipIdx} className="flex items-start">
                              <span className="mr-1">•</span>
                              <span>{tip}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Common Smells */}
      {profile.common_smells.length > 0 && (
        <div className="mt-4 pt-4 border-t">
          <h4 className="text-xs font-medium text-gray-500 mb-2">자주 발견되는 코드 스멜</h4>
          <div className="flex flex-wrap gap-2">
            {profile.common_smells.slice(0, 4).map(([smell, count]) => (
              <span
                key={smell}
                className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full"
              >
                {smell} ({count})
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
