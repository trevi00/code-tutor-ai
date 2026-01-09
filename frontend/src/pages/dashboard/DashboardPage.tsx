import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { dashboardApi, qualityApi } from '@/api';
import { ActivityHeatmap } from '@/components/dashboard/ActivityHeatmap';
import { CodeQualityCard } from '@/components/dashboard/CodeQualityCard';
import { LearningInsights } from '@/components/dashboard/LearningInsights';
import { QualityRecommendations } from '@/components/dashboard/QualityRecommendations';
import { QualityTrendChart } from '@/components/dashboard/QualityTrendChart';
import { SkillPredictions } from '@/components/dashboard/SkillPredictions';
import { GamificationWidget } from '@/components/gamification';
import type {
  DashboardData,
  InsightsData,
  CategoryProgress,
  RecentSubmission,
  QualityStats,
  QualityTrendPoint,
  QualityProfile,
  QualityRecommendation,
  QualityImprovementSuggestion,
} from '@/types';

// Status badge styles
const statusStyles: Record<string, string> = {
  accepted: 'bg-green-100 text-green-800',
  wrong_answer: 'bg-red-100 text-red-800',
  runtime_error: 'bg-orange-100 text-orange-800',
  time_limit_exceeded: 'bg-yellow-100 text-yellow-800',
  memory_limit_exceeded: 'bg-purple-100 text-purple-800',
  pending: 'bg-gray-100 text-gray-800',
  running: 'bg-blue-100 text-blue-800',
};

const statusLabels: Record<string, string> = {
  accepted: '정답',
  wrong_answer: '오답',
  runtime_error: '런타임 에러',
  time_limit_exceeded: '시간 초과',
  memory_limit_exceeded: '메모리 초과',
  pending: '대기중',
  running: '실행중',
};

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [insights, setInsights] = useState<InsightsData | null>(null);
  const [qualityStats, setQualityStats] = useState<QualityStats | null>(null);
  const [qualityTrends, setQualityTrends] = useState<QualityTrendPoint[]>([]);
  const [qualityProfile, setQualityProfile] = useState<QualityProfile | null>(null);
  const [qualityRecommendations, setQualityRecommendations] = useState<QualityRecommendation[]>([]);
  const [qualitySuggestions, setQualitySuggestions] = useState<QualityImprovementSuggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch dashboard, insights, and quality data in parallel
        const [
          dashboardData,
          insightsData,
          qualityStatsData,
          qualityTrendsData,
          profileData,
          recommendationsData,
          suggestionsData,
        ] = await Promise.all([
          dashboardApi.getDashboard(),
          dashboardApi.getInsights().catch(() => null),
          qualityApi.getQualityStats().catch(() => null),
          qualityApi.getQualityTrends(30).catch(() => ({ trends: [], days: 30 })),
          qualityApi.getQualityProfile().catch(() => null),
          qualityApi.getQualityRecommendations(5).catch(() => ({ recommendations: [], total: 0 })),
          qualityApi.getImprovementSuggestions().catch(() => ({ suggestions: [], total: 0 })),
        ]);
        setDashboard(dashboardData);
        setInsights(insightsData);
        setQualityStats(qualityStatsData);
        setQualityTrends(qualityTrendsData?.trends || []);
        setQualityProfile(profileData);
        setQualityRecommendations(recommendationsData?.recommendations || []);
        setQualitySuggestions(suggestionsData?.suggestions || []);
      } catch (err) {
        setError('대시보드를 불러오는데 실패했습니다.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  if (!dashboard) return null;

  const { stats, category_progress, recent_submissions, heatmap, skill_predictions } = dashboard;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-8">대시보드</h1>

      {/* Activity Heatmap */}
      {heatmap && heatmap.length > 0 && (
        <div className="mb-8">
          <ActivityHeatmap data={heatmap} months={6} />
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="푼 문제"
          value={stats.total_problems_solved}
          subtext={`${stats.total_problems_attempted}문제 시도`}
          icon="check"
        />
        <StatCard
          title="총 제출"
          value={stats.total_submissions}
          subtext={`성공률 ${stats.overall_success_rate.toFixed(1)}%`}
          icon="upload"
        />
        <StatCard
          title="현재 스트릭"
          value={stats.streak.current_streak}
          subtext={`최장 ${stats.streak.longest_streak}일`}
          icon="fire"
        />
        <StatCard
          title="난이도별"
          value={null}
          icon="chart"
        >
          <div className="flex justify-between mt-2 text-sm">
            <span className="text-green-600">Easy: {stats.easy_solved}</span>
            <span className="text-yellow-600">Medium: {stats.medium_solved}</span>
            <span className="text-red-600">Hard: {stats.hard_solved}</span>
          </div>
        </StatCard>
      </div>

      {/* Gamification & AI Learning Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <div className="lg:col-span-2">
          {insights && <LearningInsights insights={insights} />}
        </div>
        <div>
          <GamificationWidget />
        </div>
      </div>

      {/* Code Quality Analysis */}
      {qualityStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <CodeQualityCard stats={qualityStats} />
          <QualityTrendChart trends={qualityTrends} days={30} />
        </div>
      )}

      {/* Quality-Based Recommendations */}
      {qualityProfile && (
        <div className="mb-8">
          <QualityRecommendations
            profile={qualityProfile}
            recommendations={qualityRecommendations}
            suggestions={qualitySuggestions}
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Category Progress */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">카테고리별 진행률</h2>
          <div className="space-y-4">
            {category_progress.length > 0 ? (
              category_progress.slice(0, 8).map((cp) => (
                <CategoryProgressBar key={cp.category} progress={cp} />
              ))
            ) : (
              <p className="text-gray-500 dark:text-gray-400 text-center py-4">
                아직 푼 문제가 없습니다. 문제를 풀어보세요!
              </p>
            )}
          </div>
          {category_progress.length > 8 && (
            <Link
              to="/problems"
              className="block mt-4 text-center text-blue-600 hover:text-blue-800"
            >
              더 보기
            </Link>
          )}
        </div>

        {/* Recent Submissions */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">최근 제출</h2>
          <div className="space-y-3">
            {recent_submissions.length > 0 ? (
              recent_submissions.map((submission) => (
                <RecentSubmissionItem key={submission.id} submission={submission} />
              ))
            ) : (
              <p className="text-gray-500 dark:text-gray-400 text-center py-4">
                아직 제출 기록이 없습니다.
              </p>
            )}
          </div>
          {recent_submissions.length > 0 && (
            <Link
              to="/submissions"
              className="block mt-4 text-center text-blue-600 hover:text-blue-800 font-medium"
            >
              전체 제출 기록 보기 →
            </Link>
          )}
        </div>
      </div>

      {/* Skill Predictions */}
      {skill_predictions && skill_predictions.length > 0 && (
        <div className="mb-8">
          <SkillPredictions predictions={skill_predictions} />
        </div>
      )}

      {/* Quick Actions */}
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-4">빠른 시작</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            to="/problems"
            className="flex items-center p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
          >
            <span className="text-2xl mr-3">📝</span>
            <div>
              <h3 className="font-medium text-blue-900 dark:text-blue-100">문제 풀기</h3>
              <p className="text-sm text-blue-700 dark:text-blue-300">알고리즘 문제 도전</p>
            </div>
          </Link>
          <Link
            to="/chat"
            className="flex items-center p-4 bg-green-50 dark:bg-green-900/30 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
          >
            <span className="text-2xl mr-3">💬</span>
            <div>
              <h3 className="font-medium text-green-900 dark:text-green-100">AI 튜터</h3>
              <p className="text-sm text-green-700 dark:text-green-300">코드 리뷰 받기</p>
            </div>
          </Link>
          <Link
            to="/problems?difficulty=easy"
            className="flex items-center p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
          >
            <span className="text-2xl mr-3">🎯</span>
            <div>
              <h3 className="font-medium text-purple-900 dark:text-purple-100">초급 문제</h3>
              <p className="text-sm text-purple-700 dark:text-purple-300">Easy 문제부터 시작</p>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}

// Stat Card Component
interface StatCardProps {
  title: string;
  value: number | null;
  subtext?: string;
  icon: string;
  children?: React.ReactNode;
}

function StatCard({ title, value, subtext, icon, children }: StatCardProps) {
  const icons: Record<string, string> = {
    check: '✅',
    upload: '📤',
    fire: '🔥',
    chart: '📊',
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</h3>
        <span className="text-2xl">{icons[icon]}</span>
      </div>
      {value !== null ? (
        <>
          <p className="text-3xl font-bold text-gray-800 dark:text-gray-100">{value}</p>
          {subtext && <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{subtext}</p>}
        </>
      ) : (
        children
      )}
    </div>
  );
}

// Category Progress Bar Component
function CategoryProgressBar({ progress }: { progress: CategoryProgress }) {
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
  };

  const percentage = (progress.solved_problems / progress.total_problems) * 100;

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium text-gray-700 dark:text-gray-300">
          {categoryLabels[progress.category] || progress.category}
        </span>
        <span className="text-gray-500 dark:text-gray-400">
          {progress.solved_problems}/{progress.total_problems}
        </span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2">
        <div
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// Recent Submission Item Component
function RecentSubmissionItem({ submission }: { submission: RecentSubmission }) {
  const formattedDate = new Date(submission.submitted_at).toLocaleDateString('ko-KR', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <Link
      to={`/problems/${submission.problem_id}`}
      className="flex items-center justify-between p-3 bg-gray-50 dark:bg-slate-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-700 transition-colors"
    >
      <div className="flex-1 min-w-0">
        <p className="font-medium text-gray-800 dark:text-gray-100 truncate">{submission.problem_title}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400">{formattedDate}</p>
      </div>
      <span
        className={`px-2 py-1 text-xs font-medium rounded ${
          statusStyles[submission.status] || 'bg-gray-100 text-gray-800'
        }`}
      >
        {statusLabels[submission.status] || submission.status}
      </span>
    </Link>
  );
}
