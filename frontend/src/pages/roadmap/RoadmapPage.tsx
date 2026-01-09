/**
 * Learning Roadmap Page - Enhanced with modern design
 * Displays all learning paths and user progress
 */
import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  BookOpen,
  CheckCircle,
  Clock,
  Trophy,
  ChevronRight,
  Zap,
  Star,
  Lock,
  Play,
  Map,
  Target,
  Loader2,
  Sparkles,
  Award,
  TrendingUp,
  ArrowRight,
} from 'lucide-react';
import { roadmapApi } from '../../api/roadmap';
import type { LearningPath, UserProgress } from '../../api/roadmap';

// Stat card component
interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  gradient: string;
  delay?: number;
}

function StatCard({ icon, label, value, gradient, delay = 0 }: StatCardProps) {
  return (
    <div
      className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-xl p-4 shadow-lg border border-white/20 dark:border-slate-700/50 animate-fade-in hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center mb-3 shadow-lg`}>
        {icon}
      </div>
      <p className="text-2xl font-bold text-slate-800 dark:text-white">{value}</p>
      <p className="text-sm text-slate-500 dark:text-slate-400">{label}</p>
    </div>
  );
}

// Learning path card component
interface PathCardProps {
  path: LearningPath;
  index: number;
  isLocked: boolean;
  onStart: (pathId: string) => void;
}

function PathCard({ path, index, isLocked, onStart }: PathCardProps) {
  const getLevelStyles = (level: string) => {
    switch (level) {
      case 'beginner':
        return {
          gradient: 'from-emerald-500 to-green-600',
          bg: 'bg-emerald-50 dark:bg-emerald-900/20',
          text: 'text-emerald-700 dark:text-emerald-400',
          ring: 'ring-emerald-500',
        };
      case 'elementary':
        return {
          gradient: 'from-blue-500 to-cyan-600',
          bg: 'bg-blue-50 dark:bg-blue-900/20',
          text: 'text-blue-700 dark:text-blue-400',
          ring: 'ring-blue-500',
        };
      case 'intermediate':
        return {
          gradient: 'from-purple-500 to-indigo-600',
          bg: 'bg-purple-50 dark:bg-purple-900/20',
          text: 'text-purple-700 dark:text-purple-400',
          ring: 'ring-purple-500',
        };
      case 'advanced':
        return {
          gradient: 'from-orange-500 to-red-600',
          bg: 'bg-orange-50 dark:bg-orange-900/20',
          text: 'text-orange-700 dark:text-orange-400',
          ring: 'ring-orange-500',
        };
      default:
        return {
          gradient: 'from-slate-500 to-slate-600',
          bg: 'bg-slate-50 dark:bg-slate-900/20',
          text: 'text-slate-700 dark:text-slate-400',
          ring: 'ring-slate-500',
        };
    }
  };

  const styles = getLevelStyles(path.level);

  const getStatusBadge = () => {
    switch (path.status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
            <CheckCircle className="w-3.5 h-3.5" />
            완료
          </span>
        );
      case 'in_progress':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
            <Play className="w-3.5 h-3.5" />
            진행 중
          </span>
        );
      default:
        return isLocked ? (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-500 dark:bg-slate-700/50 dark:text-slate-400">
            <Lock className="w-3.5 h-3.5" />
            잠김
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400">
            <Sparkles className="w-3.5 h-3.5" />
            시작 가능
          </span>
        );
    }
  };

  return (
    <div
      className={`relative animate-fade-in ${isLocked ? 'opacity-60' : ''}`}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Timeline connector */}
      {index > 0 && (
        <div className="absolute left-8 -top-6 w-0.5 h-6 bg-gradient-to-b from-slate-300 to-slate-200 dark:from-slate-600 dark:to-slate-700" />
      )}

      <div
        className={`bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden border-2 transition-all duration-300 hover:shadow-xl ${
          path.status === 'completed'
            ? 'border-green-300 dark:border-green-700'
            : path.status === 'in_progress'
            ? 'border-blue-300 dark:border-blue-700'
            : 'border-transparent hover:border-slate-200 dark:hover:border-slate-600'
        }`}
      >
        {/* Card Header */}
        <div className={`bg-gradient-to-r ${styles.gradient} p-6 text-white relative overflow-hidden`}>
          {/* Decorative elements */}
          <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2" />
          <div className="absolute bottom-0 left-0 w-24 h-24 bg-black/10 rounded-full translate-y-1/2 -translate-x-1/2" />

          <div className="relative flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center text-3xl shadow-lg">
                {path.icon}
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2.5 py-0.5 bg-white/20 backdrop-blur-sm rounded-full text-xs font-medium">
                    {path.level_display}
                  </span>
                </div>
                <h2 className="text-xl font-bold">{path.title}</h2>
                <p className="text-sm text-white/80 mt-1 max-w-md">{path.description}</p>
              </div>
            </div>
            <div className="hidden md:block">
              {getStatusBadge()}
            </div>
          </div>
        </div>

        {/* Card Body */}
        <div className="p-6">
          {/* Mobile status badge */}
          <div className="md:hidden mb-4">
            {getStatusBadge()}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
              <div className={`w-8 h-8 rounded-lg ${styles.bg} flex items-center justify-center`}>
                <BookOpen className={`w-4 h-4 ${styles.text}`} />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-800 dark:text-white">{path.module_count}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">모듈</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
              <div className={`w-8 h-8 rounded-lg ${styles.bg} flex items-center justify-center`}>
                <Star className={`w-4 h-4 ${styles.text}`} />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-800 dark:text-white">{path.lesson_count}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">레슨</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
              <div className={`w-8 h-8 rounded-lg ${styles.bg} flex items-center justify-center`}>
                <Clock className={`w-4 h-4 ${styles.text}`} />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-800 dark:text-white">{path.estimated_hours}h</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">예상 시간</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
              <div className="w-8 h-8 rounded-lg bg-amber-50 dark:bg-amber-900/20 flex items-center justify-center">
                <Zap className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-800 dark:text-white">{path.total_xp}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">XP</p>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          {path.status !== 'not_started' && (
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">진행률</span>
                <span className="text-sm font-bold text-slate-800 dark:text-white">
                  {path.completion_rate.toFixed(0)}%
                </span>
              </div>
              <div className="relative w-full h-3 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`absolute top-0 left-0 h-full rounded-full transition-all duration-500 ${
                    path.status === 'completed'
                      ? 'bg-gradient-to-r from-green-400 to-emerald-500'
                      : 'bg-gradient-to-r from-blue-400 to-indigo-500'
                  }`}
                  style={{ width: `${path.completion_rate}%` }}
                />
                {/* Animated shine effect */}
                {path.status === 'in_progress' && (
                  <div className="absolute top-0 left-0 w-full h-full overflow-hidden">
                    <div className="w-20 h-full bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shine" />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Row */}
          <div className="flex items-center justify-between">
            {isLocked ? (
              <p className="text-sm text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                <Lock className="w-4 h-4" />
                이전 경로를 먼저 완료하세요
              </p>
            ) : (
              <div />
            )}

            {path.status === 'completed' ? (
              <Link
                to={`/roadmap/${path.id}`}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-green-50 hover:bg-green-100 dark:bg-green-900/20 dark:hover:bg-green-900/30 text-green-700 dark:text-green-400 rounded-xl font-medium transition-all duration-200"
              >
                복습하기
                <ChevronRight className="w-4 h-4" />
              </Link>
            ) : path.status === 'in_progress' ? (
              <Link
                to={`/roadmap/${path.id}`}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white rounded-xl font-medium shadow-lg shadow-blue-500/30 transition-all duration-200 hover:shadow-xl hover:-translate-y-0.5"
              >
                계속하기
                <ChevronRight className="w-4 h-4" />
              </Link>
            ) : (
              <button
                onClick={() => onStart(path.id)}
                disabled={isLocked}
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white rounded-xl font-medium shadow-lg shadow-indigo-500/30 transition-all duration-200 hover:shadow-xl hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none disabled:hover:translate-y-0"
              >
                시작하기
                <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RoadmapPage() {
  const [paths, setPaths] = useState<LearningPath[]>([]);
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [pathsRes, progressRes] = await Promise.all([
        roadmapApi.listPaths(),
        roadmapApi.getProgress().catch(() => null),
      ]);
      setPaths(pathsRes.items);
      setProgress(progressRes);
    } catch (err) {
      setError('학습 경로를 불러오는데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleStartPath = async (pathId: string) => {
    try {
      await roadmapApi.startPath(pathId);
      navigate(`/roadmap/${pathId}`);
    } catch (err) {
      console.error('Failed to start path:', err);
      navigate(`/roadmap/${pathId}`);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">학습 로드맵 불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Map className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Target className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <BookOpen className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 backdrop-blur-sm rounded-full text-white/90 text-sm mb-4">
                <TrendingUp className="w-4 h-4" />
                체계적인 학습
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Map className="w-10 h-10" />
                학습 로드맵
              </h1>
              <p className="text-indigo-100 text-lg max-w-md">
                초보자부터 고수까지, 단계별로 알고리즘을 마스터하세요
              </p>
            </div>

            {/* Quick Stats in Hero */}
            {progress && (
              <div className="flex gap-3">
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                  <BookOpen className="w-6 h-6 text-indigo-200 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-white">{progress.in_progress_paths}</div>
                  <div className="text-xs text-indigo-200">진행 중</div>
                </div>
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                  <CheckCircle className="w-6 h-6 text-green-300 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-white">{progress.completed_paths}</div>
                  <div className="text-xs text-indigo-200">완료</div>
                </div>
                <div className="bg-white/20 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px] border border-white/30">
                  <Zap className="w-6 h-6 text-yellow-300 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-white">{progress.total_xp_earned}</div>
                  <div className="text-xs text-indigo-200">XP</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Progress Summary Cards */}
        {progress && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 -mt-8 mb-8 relative z-10">
            <StatCard
              icon={<BookOpen className="w-5 h-5 text-white" />}
              label="진행 중인 경로"
              value={progress.in_progress_paths}
              gradient="from-blue-500 to-indigo-600"
              delay={0}
            />
            <StatCard
              icon={<CheckCircle className="w-5 h-5 text-white" />}
              label="완료한 경로"
              value={progress.completed_paths}
              gradient="from-green-500 to-emerald-600"
              delay={100}
            />
            <StatCard
              icon={<Award className="w-5 h-5 text-white" />}
              label="완료 레슨"
              value={`${progress.completed_lessons}/${progress.total_lessons}`}
              gradient="from-purple-500 to-pink-600"
              delay={200}
            />
            <StatCard
              icon={<Zap className="w-5 h-5 text-white" />}
              label="획득 XP"
              value={progress.total_xp_earned.toLocaleString()}
              gradient="from-amber-500 to-orange-600"
              delay={300}
            />
          </div>
        )}

        {/* Next Lesson Recommendation */}
        {progress?.next_lesson && (
          <div className="mb-8 animate-fade-in" style={{ animationDelay: '400ms' }}>
            <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl p-1 shadow-xl shadow-indigo-500/20">
              <div className="bg-white dark:bg-slate-800 rounded-xl p-6">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                      <Sparkles className="w-7 h-7 text-white" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400 mb-1">
                        다음 추천 레슨
                      </p>
                      <h3 className="text-xl font-bold text-slate-800 dark:text-white">
                        {progress.next_lesson.title}
                      </h3>
                      <div className="flex items-center gap-3 mt-1 text-sm text-slate-500 dark:text-slate-400">
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {progress.next_lesson.estimated_minutes}분
                        </span>
                        <span className="flex items-center gap-1">
                          <Zap className="w-4 h-4 text-amber-500" />
                          {progress.next_lesson.xp_reward} XP
                        </span>
                      </div>
                    </div>
                  </div>
                  <Link
                    to={`/roadmap/lesson/${progress.next_lesson.id}`}
                    className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white rounded-xl font-medium shadow-lg shadow-indigo-500/30 transition-all duration-200 hover:shadow-xl hover:-translate-y-0.5 group"
                  >
                    계속하기
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-4 rounded-xl mb-6 flex items-center gap-3">
            <div className="w-10 h-10 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-xl">!</span>
            </div>
            {error}
          </div>
        )}

        {/* Section Title */}
        <div className="mb-6">
          <h2 className="text-xl font-bold text-slate-800 dark:text-white flex items-center gap-2">
            <Target className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            학습 경로
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            순서대로 학습하여 알고리즘 실력을 키워보세요
          </p>
        </div>

        {/* Learning Paths */}
        <div className="space-y-6">
          {paths.map((path, index) => {
            const isLocked = index > 0 && paths[index - 1].status !== 'completed' && path.status === 'not_started';
            return (
              <PathCard
                key={path.id}
                path={path}
                index={index}
                isLocked={isLocked}
                onStart={handleStartPath}
              />
            );
          })}
        </div>

        {/* No Paths */}
        {paths.length === 0 && (
          <div className="text-center py-16 bg-white dark:bg-slate-800 rounded-2xl shadow-lg animate-fade-in">
            <div className="w-20 h-20 mx-auto mb-6 bg-slate-100 dark:bg-slate-700 rounded-2xl flex items-center justify-center">
              <BookOpen className="w-10 h-10 text-slate-400" />
            </div>
            <h3 className="text-lg font-semibold text-slate-800 dark:text-white mb-2">
              학습 경로가 없습니다
            </h3>
            <p className="text-slate-500 dark:text-slate-400">
              곧 새로운 학습 경로가 추가될 예정입니다.
            </p>
          </div>
        )}
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
        @keyframes shine {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
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
        .animate-shine {
          animation: shine 2s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
