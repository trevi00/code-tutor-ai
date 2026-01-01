import type { InsightsData } from '@/types';

interface LearningInsightsProps {
  insights: InsightsData;
}

export function LearningInsights({ insights }: LearningInsightsProps) {
  const { velocity, prediction, schedule, skill_gaps, insights: messages } = insights;

  // Velocity status configuration
  const velocityConfig: Record<string, { label: string; color: string; icon: string }> = {
    improving: { label: '성장 중', color: 'text-green-600', icon: '📈' },
    steady: { label: '안정적', color: 'text-blue-600', icon: '➡️' },
    declining: { label: '주의 필요', color: 'text-yellow-600', icon: '📉' },
    new_user: { label: '시작 단계', color: 'text-purple-600', icon: '🌱' },
  };

  // Trend configuration
  const trendConfig: Record<string, { label: string; color: string; arrow: string }> = {
    improving: { label: '상승', color: 'text-green-600', arrow: '↑' },
    stable: { label: '유지', color: 'text-blue-600', arrow: '→' },
    declining: { label: '하락', color: 'text-red-600', arrow: '↓' },
  };

  // Category labels
  const categoryLabels: Record<string, string> = {
    array: '배열',
    string: '문자열',
    linked_list: '연결 리스트',
    stack: '스택',
    queue: '큐',
    hash_table: '해시 테이블',
    tree: '트리',
    graph: '그래프',
    dp: 'DP',
    greedy: '그리디',
    binary_search: '이진 탐색',
    sorting: '정렬',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">AI 학습 분석</h2>
        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">LSTM Powered</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Velocity Card */}
        {velocity && (
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">학습 속도</span>
              <span className="text-xl">{velocityConfig[velocity.velocity]?.icon || '📊'}</span>
            </div>
            <p className={`text-lg font-bold ${velocityConfig[velocity.velocity]?.color || 'text-gray-800'}`}>
              {velocityConfig[velocity.velocity]?.label || velocity.velocity}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              일 평균 {velocity.problems_per_day.toFixed(1)}문제
            </p>
          </div>
        )}

        {/* Prediction Card */}
        {prediction && (
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">성공률 예측</span>
              <span className={`text-lg font-bold ${trendConfig[prediction.trend]?.color || 'text-gray-600'}`}>
                {trendConfig[prediction.trend]?.arrow || '→'}
              </span>
            </div>
            <p className="text-lg font-bold text-gray-800">
              {prediction.current_success_rate.toFixed(0)}% → {prediction.predicted_success_rate.toFixed(0)}%
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {prediction.days_ahead || 7}일 후 예측 (신뢰도 {(prediction.confidence * 100).toFixed(0)}%)
            </p>
          </div>
        )}

        {/* Consistency Card */}
        {velocity && (
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">꾸준함 점수</span>
              <span className="text-xl">🎯</span>
            </div>
            <p className="text-lg font-bold text-gray-800">{velocity.consistency_score}점</p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(velocity.consistency_score, 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* Improvement Rate Card */}
        {velocity && (
          <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">성장률</span>
              <span className="text-xl">🚀</span>
            </div>
            <p className={`text-lg font-bold ${velocity.improvement_rate >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {velocity.improvement_rate >= 0 ? '+' : ''}{velocity.improvement_rate.toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500 mt-1">지난 주 대비</p>
          </div>
        )}
      </div>

      {/* Study Recommendations */}
      {schedule && schedule.recommendations && schedule.recommendations.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">학습 추천</h3>
          <div className="space-y-2">
            {schedule.recommendations.slice(0, 3).map((rec, idx) => (
              <div
                key={idx}
                className="flex items-start p-3 bg-blue-50 rounded-lg"
              >
                <span className="text-blue-500 mr-2">💡</span>
                <p className="text-sm text-blue-800">{rec.message}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Skill Gaps */}
      {skill_gaps && skill_gaps.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">보완이 필요한 영역</h3>
          <div className="flex flex-wrap gap-2">
            {skill_gaps.slice(0, 5).map((gap, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-red-50 text-red-700 text-sm rounded-full border border-red-200"
              >
                {categoryLabels[gap.category] || gap.category}
                <span className="ml-1 text-red-500">
                  ({gap.current_rate.toFixed(0)}% → {gap.target_rate.toFixed(0)}%)
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Insights Messages */}
      {messages && messages.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">AI 인사이트</h3>
          <div className="space-y-2">
            {messages.slice(0, 3).map((msg, idx) => {
              const sentimentStyles: Record<string, string> = {
                positive: 'bg-green-50 text-green-800 border-green-200',
                neutral: 'bg-gray-50 text-gray-800 border-gray-200',
                negative: 'bg-yellow-50 text-yellow-800 border-yellow-200',
              };
              const sentimentIcons: Record<string, string> = {
                positive: '✨',
                neutral: '📌',
                negative: '⚠️',
              };
              const style = sentimentStyles[msg.sentiment || 'neutral'] || sentimentStyles.neutral;
              const icon = sentimentIcons[msg.sentiment || 'neutral'] || sentimentIcons.neutral;

              return (
                <div
                  key={idx}
                  className={`flex items-start p-3 rounded-lg border ${style}`}
                >
                  <span className="mr-2">{icon}</span>
                  <p className="text-sm">{msg.message}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Fallback for no data */}
      {!velocity && !prediction && (!messages || messages.length === 0) && (
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-2 block">📊</span>
          <p>더 많은 문제를 풀면 AI가 학습 패턴을 분석해드려요!</p>
        </div>
      )}
    </div>
  );
}
