import type { QualityStats } from '@/types';

interface CodeQualityCardProps {
  stats: QualityStats;
}

export function CodeQualityCard({ stats }: CodeQualityCardProps) {
  const {
    total_analyses,
    avg_overall,
    avg_correctness,
    avg_efficiency,
    avg_readability,
    avg_best_practices,
    avg_cyclomatic,
    total_smells,
    grade_distribution,
  } = stats;

  // Grade color configuration
  const gradeColors: Record<string, string> = {
    A: 'bg-green-500',
    B: 'bg-blue-500',
    C: 'bg-yellow-500',
    D: 'bg-orange-500',
    F: 'bg-red-500',
  };

  // Calculate dominant grade
  const dominantGrade = Object.entries(grade_distribution).reduce(
    (max, [grade, count]) => (count > max.count ? { grade, count } : max),
    { grade: 'C', count: 0 }
  ).grade;

  // Score bar component
  const ScoreBar = ({ label, score, color }: { label: string; score: number; color: string }) => (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium">{score.toFixed(0)}점</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
    </div>
  );

  if (total_analyses === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-800">코드 품질 분석</h2>
          <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">CodeBERT</span>
        </div>
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-2 block">🔍</span>
          <p>코드를 제출하면 품질 분석 결과를 확인할 수 있어요!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">코드 품질 분석</h2>
        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">CodeBERT</span>
      </div>

      {/* Overall Score and Grade */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Overall Score */}
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-4 text-center">
          <span className="text-sm text-gray-600 block mb-2">종합 점수</span>
          <span className="text-3xl font-bold text-indigo-600">{avg_overall.toFixed(0)}</span>
          <span className="text-lg text-gray-500">/100</span>
        </div>

        {/* Dominant Grade */}
        <div className="bg-gradient-to-br from-gray-50 to-slate-50 rounded-lg p-4 text-center">
          <span className="text-sm text-gray-600 block mb-2">주요 등급</span>
          <span className={`inline-block w-12 h-12 rounded-full ${gradeColors[dominantGrade] || gradeColors.C} text-white text-2xl font-bold leading-[3rem]`}>
            {dominantGrade}
          </span>
        </div>

        {/* Total Analyses */}
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4 text-center">
          <span className="text-sm text-gray-600 block mb-2">분석 횟수</span>
          <span className="text-3xl font-bold text-blue-600">{total_analyses}</span>
          <span className="text-sm text-gray-500 block">회</span>
        </div>
      </div>

      {/* Dimension Scores */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-4">품질 차원별 점수</h3>
        <ScoreBar label="정확성" score={avg_correctness} color="bg-green-500" />
        <ScoreBar label="효율성" score={avg_efficiency} color="bg-blue-500" />
        <ScoreBar label="가독성" score={avg_readability} color="bg-purple-500" />
        <ScoreBar label="모범 사례" score={avg_best_practices} color="bg-orange-500" />
      </div>

      {/* Complexity and Smells */}
      <div className="grid grid-cols-2 gap-4">
        {/* Complexity */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">평균 복잡도</span>
            <span className="text-xl">🔄</span>
          </div>
          <p className={`text-lg font-bold ${avg_cyclomatic > 10 ? 'text-red-600' : avg_cyclomatic > 5 ? 'text-yellow-600' : 'text-green-600'}`}>
            {avg_cyclomatic.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500">
            {avg_cyclomatic > 10 ? '높음 - 리팩토링 권장' : avg_cyclomatic > 5 ? '보통' : '낮음 - 좋음'}
          </p>
        </div>

        {/* Code Smells */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">코드 스멜</span>
            <span className="text-xl">🔍</span>
          </div>
          <p className={`text-lg font-bold ${total_smells > 20 ? 'text-red-600' : total_smells > 10 ? 'text-yellow-600' : 'text-green-600'}`}>
            {total_smells}개
          </p>
          <p className="text-xs text-gray-500">
            {total_smells > 20 ? '개선 필요' : total_smells > 10 ? '일부 개선 권장' : '양호'}
          </p>
        </div>
      </div>

      {/* Grade Distribution */}
      {Object.keys(grade_distribution).length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">등급 분포</h3>
          <div className="flex gap-2">
            {['A', 'B', 'C', 'D', 'F'].map((grade) => {
              const count = grade_distribution[grade] || 0;
              const total = Object.values(grade_distribution).reduce((a, b) => a + b, 0);
              const percentage = total > 0 ? (count / total) * 100 : 0;

              return (
                <div key={grade} className="flex-1 text-center">
                  <div className="h-16 flex items-end justify-center mb-1">
                    <div
                      className={`w-8 ${gradeColors[grade]} rounded-t transition-all duration-500`}
                      style={{ height: `${Math.max(percentage, 4)}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium text-gray-600">{grade}</span>
                  <span className="text-xs text-gray-400 block">{count}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
