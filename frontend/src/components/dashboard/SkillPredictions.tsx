import type { SkillPrediction } from '@/types';

interface SkillPredictionsProps {
  predictions: SkillPrediction[];
}

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
  design: '설계',
  dfs: 'DFS',
  bfs: 'BFS',
  math: '수학',
  bit_manipulation: '비트 연산',
  recursion: '재귀',
};

export function SkillPredictions({ predictions }: SkillPredictionsProps) {
  if (predictions.length === 0) {
    return null;
  }

  // Sort by recommended_focus first, then by current_level
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (a.recommended_focus !== b.recommended_focus) {
      return a.recommended_focus ? -1 : 1;
    }
    return a.current_level - b.current_level;
  });

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">스킬 분석</h2>
        <span className="text-xs text-gray-500">AI 기반 예측</span>
      </div>

      <div className="space-y-4">
        {sortedPredictions.slice(0, 6).map((prediction) => (
          <SkillItem key={prediction.category} prediction={prediction} />
        ))}
      </div>

      {predictions.some(p => p.recommended_focus) && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700">
          <span className="font-medium">💡 추천:</span> 표시된 카테고리에 더 집중하면 실력 향상에 도움이 됩니다.
        </div>
      )}
    </div>
  );
}

function SkillItem({ prediction }: { prediction: SkillPrediction }) {
  const categoryName = categoryLabels[prediction.category] || prediction.category;
  const currentLevel = Math.round(prediction.current_level);
  const predictedLevel = Math.round(prediction.predicted_level);
  const improvement = predictedLevel - currentLevel;

  return (
    <div className={`p-3 rounded-lg ${prediction.recommended_focus ? 'bg-yellow-50 border border-yellow-200' : 'bg-gray-50'}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-800">{categoryName}</span>
          {prediction.recommended_focus && (
            <span className="px-2 py-0.5 text-xs bg-yellow-200 text-yellow-800 rounded-full">
              집중 추천
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-600">{currentLevel}%</span>
          <span className="text-gray-400">→</span>
          <span className="text-green-600 font-medium">{predictedLevel}%</span>
          {improvement > 0 && (
            <span className="text-green-500 text-xs">+{improvement}</span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
        {/* Current level */}
        <div
          className="absolute h-full bg-blue-500 rounded-full transition-all duration-500"
          style={{ width: `${currentLevel}%` }}
        />
        {/* Predicted level indicator */}
        <div
          className="absolute h-full w-1 bg-green-400 rounded-full"
          style={{ left: `${predictedLevel}%` }}
        />
      </div>

      {/* Confidence indicator */}
      <div className="mt-1 flex items-center gap-1 text-xs text-gray-500">
        <span>신뢰도:</span>
        <div className="flex gap-0.5">
          {[0.25, 0.5, 0.75, 1].map((threshold, i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full ${
                prediction.confidence >= threshold ? 'bg-blue-500' : 'bg-gray-300'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
