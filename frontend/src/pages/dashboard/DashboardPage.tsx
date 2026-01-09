/**
 * Dashboard Page - Enhanced with modern UI, Hero section, and improved visuals
 */

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  CheckCircle2,
  Upload,
  Flame,
  BarChart3,
  BookOpen,
  MessageSquare,
  Target,
  Trophy,
  Sparkles,
  TrendingUp,
  Clock,
  ArrowRight,
  Zap,
  Calendar,
  Award,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { dashboardApi, qualityApi } from '@/api';
import { useAuthStore } from '@/store/authStore';
import { ActivityHeatmap } from '@/components/dashboard/ActivityHeatmap';
import { CategoryProgressChart } from '@/components/dashboard/CategoryProgressChart';
import { CodeQualityCard } from '@/components/dashboard/CodeQualityCard';
import { LearningInsights } from '@/components/dashboard/LearningInsights';
import { QualityRecommendations } from '@/components/dashboard/QualityRecommendations';
import { QualityTrendChart } from '@/components/dashboard/QualityTrendChart';
import { SkillPredictions } from '@/components/dashboard/SkillPredictions';
import { GamificationWidget } from '@/components/gamification';
import type {
  DashboardData,
  InsightsData,
  QualityStats,
  QualityTrendPoint,
  QualityProfile,
  QualityRecommendation,
  QualityImprovementSuggestion,
  RecentSubmission,
} from '@/types';

// Status badge styles with dark mode support
const statusStyles: Record<string, string> = {
  accepted: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
  wrong_answer: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  runtime_error: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
  time_limit_exceeded: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
  memory_limit_exceeded: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
  pending: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
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

// Get greeting based on time of day
function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 6) return '늦은 밤이에요';
  if (hour < 12) return '좋은 아침이에요';
  if (hour < 18) return '좋은 오후예요';
  return '좋은 저녁이에요';
}

export default function DashboardPage() {
  const { user } = useAuthStore();
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
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">대시보드 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
            <RefreshCw className="w-8 h-8 text-red-500" />
          </div>
          <p className="text-red-500 dark:text-red-400">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl hover:from-blue-600 hover:to-indigo-600 transition-all shadow-lg hover:shadow-xl font-medium"
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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl" />
          <Sparkles className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Trophy className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Zap className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-7xl mx-auto px-6 py-10 relative">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            {/* Welcome Message */}
            <div className="text-white">
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-white/90 text-sm mb-3">
                <Calendar className="w-4 h-4" />
                {new Date().toLocaleDateString('ko-KR', { month: 'long', day: 'numeric', weekday: 'long' })}
              </div>
              <h1 className="text-2xl md:text-3xl font-bold mb-2">
                {getGreeting()}, <span className="text-yellow-300">{user?.username || '학습자'}</span>님!
              </h1>
              <p className="text-blue-100 text-lg">
                오늘도 함께 성장해볼까요?
              </p>
            </div>

            {/* Quick Stats */}
            <div className="flex gap-3 flex-wrap">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px]">
                <CheckCircle2 className="w-5 h-5 text-green-300 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">{stats.total_problems_solved}</div>
                <div className="text-xs text-blue-200">해결</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px]">
                <Flame className="w-5 h-5 text-orange-300 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">{stats.streak.current_streak}</div>
                <div className="text-xs text-blue-200">스트릭</div>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px] border border-white/30">
                <TrendingUp className="w-5 h-5 text-yellow-300 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">{stats.overall_success_rate.toFixed(0)}%</div>
                <div className="text-xs text-blue-200">성공률</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 -mt-6">
        {/* Activity Heatmap */}
        {heatmap && heatmap.length > 0 && (
          <div className="mb-8 animate-fade-in">
            <ActivityHeatmap data={heatmap} months={6} />
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
          <StatCard
            title="푼 문제"
            value={stats.total_problems_solved}
            subtext={`${stats.total_problems_attempted}문제 시도`}
            icon={<CheckCircle2 className="w-6 h-6" />}
            gradient="from-green-500 to-emerald-500"
            delay={0}
          />
          <StatCard
            title="총 제출"
            value={stats.total_submissions}
            subtext={`성공률 ${stats.overall_success_rate.toFixed(1)}%`}
            icon={<Upload className="w-6 h-6" />}
            gradient="from-blue-500 to-cyan-500"
            delay={1}
          />
          <StatCard
            title="현재 스트릭"
            value={stats.streak.current_streak}
            subtext={`최장 ${stats.streak.longest_streak}일`}
            icon={<Flame className="w-6 h-6" />}
            gradient="from-orange-500 to-red-500"
            delay={2}
          />
          <DifficultyCard
            easy={stats.easy_solved}
            medium={stats.medium_solved}
            hard={stats.hard_solved}
            delay={3}
          />
        </div>

        {/* Gamification & AI Learning Insights */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 animate-fade-in" style={{ animationDelay: '200ms' }}>
            {insights && <LearningInsights insights={insights} />}
          </div>
          <div className="animate-fade-in" style={{ animationDelay: '300ms' }}>
            <GamificationWidget />
          </div>
        </div>

        {/* Code Quality Analysis */}
        {qualityStats && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="animate-fade-in" style={{ animationDelay: '400ms' }}>
              <CodeQualityCard stats={qualityStats} />
            </div>
            <div className="animate-fade-in" style={{ animationDelay: '500ms' }}>
              <QualityTrendChart trends={qualityTrends} days={30} />
            </div>
          </div>
        )}

        {/* Quality-Based Recommendations */}
        {qualityProfile && (
          <div className="mb-8 animate-fade-in" style={{ animationDelay: '600ms' }}>
            <QualityRecommendations
              profile={qualityProfile}
              recommendations={qualityRecommendations}
              suggestions={qualitySuggestions}
            />
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Category Progress Chart */}
          <div className="animate-fade-in" style={{ animationDelay: '700ms' }}>
            <CategoryProgressChart data={category_progress} maxItems={8} />
          </div>

          {/* Recent Submissions */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-6 animate-fade-in" style={{ animationDelay: '800ms' }}>
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                  <Clock className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-lg font-bold text-slate-800 dark:text-white">최근 제출</h2>
              </div>
              {recent_submissions.length > 0 && (
                <Link
                  to="/submissions"
                  className="text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 font-medium flex items-center gap-1 transition-colors"
                >
                  전체 보기
                  <ArrowRight className="w-4 h-4" />
                </Link>
              )}
            </div>
            <div className="space-y-3">
              {recent_submissions.length > 0 ? (
                recent_submissions.slice(0, 5).map((submission, idx) => (
                  <RecentSubmissionItem key={submission.id} submission={submission} delay={idx} />
                ))
              ) : (
                <div className="text-center py-8">
                  <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center">
                    <Upload className="w-7 h-7 text-slate-400" />
                  </div>
                  <p className="text-slate-500 dark:text-slate-400 font-medium">아직 제출 기록이 없습니다</p>
                  <p className="text-sm text-slate-400 dark:text-slate-500 mt-1">문제를 풀고 제출해보세요!</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Skill Predictions */}
        {skill_predictions && skill_predictions.length > 0 && (
          <div className="mb-8 animate-fade-in" style={{ animationDelay: '900ms' }}>
            <SkillPredictions predictions={skill_predictions} />
          </div>
        )}

        {/* Quick Actions */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-6 animate-fade-in" style={{ animationDelay: '1000ms' }}>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-lg font-bold text-slate-800 dark:text-white">빠른 시작</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <QuickActionCard
              to="/problems"
              icon={<BookOpen className="w-6 h-6" />}
              title="문제 풀기"
              description="알고리즘 문제 도전"
              gradient="from-blue-500 to-indigo-500"
              bgLight="bg-blue-50"
              bgDark="dark:bg-blue-900/20"
            />
            <QuickActionCard
              to="/chat"
              icon={<MessageSquare className="w-6 h-6" />}
              title="AI 튜터"
              description="코드 리뷰 받기"
              gradient="from-green-500 to-emerald-500"
              bgLight="bg-green-50"
              bgDark="dark:bg-green-900/20"
            />
            <QuickActionCard
              to="/problems?difficulty=easy"
              icon={<Target className="w-6 h-6" />}
              title="초급 문제"
              description="Easy 문제부터 시작"
              gradient="from-purple-500 to-pink-500"
              bgLight="bg-purple-50"
              bgDark="dark:bg-purple-900/20"
            />
          </div>
        </div>
      </div>

      {/* Styles */}
      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(-5deg); }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out forwards;
          opacity: 0;
        }
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float-delayed 5s ease-in-out infinite;
          animation-delay: 1s;
        }
      `}</style>
    </div>
  );
}

// Enhanced Stat Card Component
interface StatCardProps {
  title: string;
  value: number;
  subtext?: string;
  icon: React.ReactNode;
  gradient: string;
  delay: number;
}

function StatCard({ title, value, subtext, icon, gradient, delay }: StatCardProps) {
  return (
    <div
      className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-5 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 animate-fade-in"
      style={{ animationDelay: `${delay * 100}ms` }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-white shadow-lg`}>
          {icon}
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold text-slate-800 dark:text-white">{value.toLocaleString()}</p>
        </div>
      </div>
      <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">{title}</h3>
      {subtext && (
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">{subtext}</p>
      )}
    </div>
  );
}

// Difficulty Card Component
interface DifficultyCardProps {
  easy: number;
  medium: number;
  hard: number;
  delay: number;
}

function DifficultyCard({ easy, medium, hard, delay }: DifficultyCardProps) {
  const total = easy + medium + hard || 1;
  const easyPercent = (easy / total) * 100;
  const mediumPercent = (medium / total) * 100;
  const hardPercent = (hard / total) * 100;

  return (
    <div
      className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-5 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 animate-fade-in"
      style={{ animationDelay: `${delay * 100}ms` }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center text-white shadow-lg">
          <BarChart3 className="w-6 h-6" />
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold text-slate-800 dark:text-white">{easy + medium + hard}</p>
        </div>
      </div>
      <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-3">난이도별</h3>

      {/* Progress bar */}
      <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden flex mb-3">
        <div className="bg-green-500 h-full transition-all" style={{ width: `${easyPercent}%` }} />
        <div className="bg-yellow-500 h-full transition-all" style={{ width: `${mediumPercent}%` }} />
        <div className="bg-red-500 h-full transition-all" style={{ width: `${hardPercent}%` }} />
      </div>

      {/* Legend */}
      <div className="flex justify-between text-xs">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-500"></span>
          <span className="text-slate-500 dark:text-slate-400">Easy</span>
          <span className="font-bold text-green-600 dark:text-green-400">{easy}</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
          <span className="text-slate-500 dark:text-slate-400">Medium</span>
          <span className="font-bold text-yellow-600 dark:text-yellow-400">{medium}</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500"></span>
          <span className="text-slate-500 dark:text-slate-400">Hard</span>
          <span className="font-bold text-red-600 dark:text-red-400">{hard}</span>
        </span>
      </div>
    </div>
  );
}

// Quick Action Card Component
interface QuickActionCardProps {
  to: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  gradient: string;
  bgLight: string;
  bgDark: string;
}

function QuickActionCard({ to, icon, title, description, gradient, bgLight, bgDark }: QuickActionCardProps) {
  return (
    <Link
      to={to}
      className={`group flex items-center gap-4 p-4 ${bgLight} ${bgDark} rounded-xl hover:scale-[1.02] transition-all duration-300 border border-transparent hover:border-slate-200 dark:hover:border-slate-600`}
    >
      <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-white shadow-lg group-hover:scale-110 transition-transform`}>
        {icon}
      </div>
      <div className="flex-1">
        <h3 className="font-semibold text-slate-800 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
          {title}
        </h3>
        <p className="text-sm text-slate-500 dark:text-slate-400">{description}</p>
      </div>
      <ArrowRight className="w-5 h-5 text-slate-300 dark:text-slate-600 group-hover:text-indigo-500 group-hover:translate-x-1 transition-all" />
    </Link>
  );
}

// Recent Submission Item Component
interface RecentSubmissionItemProps {
  submission: RecentSubmission;
  delay: number;
}

function RecentSubmissionItem({ submission, delay }: RecentSubmissionItemProps) {
  const formattedDate = new Date(submission.submitted_at).toLocaleDateString('ko-KR', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  const isAccepted = submission.status === 'accepted';

  return (
    <Link
      to={`/problems/${submission.problem_id}`}
      className="flex items-center gap-3 p-3 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-all group animate-fade-in"
      style={{ animationDelay: `${(delay + 8) * 100}ms` }}
    >
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
        isAccepted
          ? 'bg-green-100 dark:bg-green-900/30'
          : 'bg-slate-100 dark:bg-slate-700'
      }`}>
        {isAccepted ? (
          <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
        ) : (
          <Award className="w-5 h-5 text-slate-400 dark:text-slate-500" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium text-slate-800 dark:text-white truncate group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
          {submission.problem_title}
        </p>
        <p className="text-xs text-slate-400 dark:text-slate-500">{formattedDate}</p>
      </div>
      <span
        className={`px-2.5 py-1 text-xs font-medium rounded-lg ${
          statusStyles[submission.status] || 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
        }`}
      >
        {statusLabels[submission.status] || submission.status}
      </span>
    </Link>
  );
}
