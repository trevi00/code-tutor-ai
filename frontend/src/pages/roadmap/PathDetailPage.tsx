/**
 * Learning Path Detail Page - Enhanced with modern design
 * Shows modules and lessons for a specific path
 */
import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  BookOpen,
  CheckCircle,
  Clock,
  ChevronRight,
  ChevronDown,
  Zap,
  ArrowLeft,
  FileText,
  Code,
  Keyboard,
  HelpCircle,
  BookMarked,
  Lock,
  Loader2,
  Star,
  Play,
  Trophy,
  Target,
  Sparkles,
} from 'lucide-react';
import { roadmapApi } from '../../api/roadmap';
import type { LearningPath, LessonType, Module, Lesson } from '../../api/roadmap';

// Lesson type styles
const LESSON_TYPE_STYLES: Record<LessonType, { bg: string; text: string; icon: React.ReactNode; label: string }> = {
  concept: {
    bg: 'bg-blue-100 dark:bg-blue-900/30',
    text: 'text-blue-600 dark:text-blue-400',
    icon: <FileText className="w-4 h-4" />,
    label: '개념',
  },
  problem: {
    bg: 'bg-purple-100 dark:bg-purple-900/30',
    text: 'text-purple-600 dark:text-purple-400',
    icon: <Code className="w-4 h-4" />,
    label: '문제',
  },
  typing: {
    bg: 'bg-green-100 dark:bg-green-900/30',
    text: 'text-green-600 dark:text-green-400',
    icon: <Keyboard className="w-4 h-4" />,
    label: '타이핑',
  },
  pattern: {
    bg: 'bg-amber-100 dark:bg-amber-900/30',
    text: 'text-amber-600 dark:text-amber-400',
    icon: <BookMarked className="w-4 h-4" />,
    label: '패턴',
  },
  quiz: {
    bg: 'bg-pink-100 dark:bg-pink-900/30',
    text: 'text-pink-600 dark:text-pink-400',
    icon: <HelpCircle className="w-4 h-4" />,
    label: '퀴즈',
  },
};

// Level styles
const LEVEL_STYLES: Record<string, { gradient: string; bg: string; text: string }> = {
  beginner: {
    gradient: 'from-emerald-500 to-green-600',
    bg: 'bg-emerald-50 dark:bg-emerald-900/20',
    text: 'text-emerald-600 dark:text-emerald-400',
  },
  elementary: {
    gradient: 'from-blue-500 to-cyan-600',
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    text: 'text-blue-600 dark:text-blue-400',
  },
  intermediate: {
    gradient: 'from-purple-500 to-indigo-600',
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    text: 'text-purple-600 dark:text-purple-400',
  },
  advanced: {
    gradient: 'from-orange-500 to-red-600',
    bg: 'bg-orange-50 dark:bg-orange-900/20',
    text: 'text-orange-600 dark:text-orange-400',
  },
};

// Module card component
interface ModuleCardProps {
  module: Module;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
  levelStyles: { gradient: string; bg: string; text: string };
}

function ModuleCard({ module, index, isExpanded, onToggle, levelStyles }: ModuleCardProps) {
  const isCompleted = module.completion_rate >= 100;
  const isInProgress = module.completion_rate > 0 && module.completion_rate < 100;

  return (
    <div
      className={`bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden border-2 transition-all duration-300 animate-fade-in ${
        isCompleted
          ? 'border-green-300 dark:border-green-700'
          : isInProgress
          ? 'border-blue-300 dark:border-blue-700'
          : 'border-transparent hover:border-slate-200 dark:hover:border-slate-600'
      }`}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Module Header */}
      <button
        onClick={onToggle}
        className="w-full p-5 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
      >
        <div className="flex items-center gap-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center text-lg font-bold shadow-lg ${
              isCompleted
                ? 'bg-gradient-to-br from-green-400 to-emerald-500 text-white'
                : isInProgress
                ? 'bg-gradient-to-br from-blue-400 to-indigo-500 text-white'
                : `bg-gradient-to-br ${levelStyles.gradient} text-white/80`
            }`}
          >
            {isCompleted ? (
              <CheckCircle className="w-6 h-6" />
            ) : (
              <span>{index + 1}</span>
            )}
          </div>
          <div className="text-left">
            <div className="flex items-center gap-2 mb-1">
              {isCompleted && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                  <CheckCircle className="w-3 h-3" />
                  완료
                </span>
              )}
              {isInProgress && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                  <Play className="w-3 h-3" />
                  진행 중
                </span>
              )}
            </div>
            <h3 className="font-bold text-lg text-slate-800 dark:text-white">
              {module.title}
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
              {module.description}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Progress indicator */}
          <div className="hidden sm:flex items-center gap-3">
            <div className="text-right">
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                {module.completed_lessons}/{module.lesson_count}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">레슨</p>
            </div>
            <div className="w-16 h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isCompleted
                    ? 'bg-gradient-to-r from-green-400 to-emerald-500'
                    : 'bg-gradient-to-r from-blue-400 to-indigo-500'
                }`}
                style={{ width: `${module.completion_rate}%` }}
              />
            </div>
          </div>
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center transition-transform duration-200 ${
              isExpanded ? 'rotate-180 bg-slate-100 dark:bg-slate-700' : ''
            }`}
          >
            <ChevronDown className="w-5 h-5 text-slate-400" />
          </div>
        </div>
      </button>

      {/* Lessons */}
      {isExpanded && (
        <div className="border-t border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-900/30">
          {module.lessons.map((lesson, lessonIndex) => (
            <LessonItem
              key={lesson.id}
              lesson={lesson}
              index={lessonIndex}
              isLocked={
                lessonIndex > 0 &&
                module.lessons[lessonIndex - 1].status !== 'completed'
              }
              levelStyles={levelStyles}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Lesson item component
interface LessonItemProps {
  lesson: Lesson;
  index: number;
  isLocked: boolean;
  levelStyles: { gradient: string; bg: string; text: string };
}

function LessonItem({ lesson, index, isLocked, levelStyles }: LessonItemProps) {
  const typeStyle = LESSON_TYPE_STYLES[lesson.lesson_type] || LESSON_TYPE_STYLES.concept;
  const isCompleted = lesson.status === 'completed';
  const isInProgress = lesson.status === 'in_progress';

  return (
    <Link
      to={isLocked ? '#' : `/roadmap/lesson/${lesson.id}`}
      className={`flex items-center gap-4 p-4 border-b border-slate-100 dark:border-slate-700/50 last:border-b-0 transition-all duration-200 ${
        isLocked
          ? 'opacity-50 cursor-not-allowed'
          : 'hover:bg-white dark:hover:bg-slate-800 hover:shadow-sm'
      }`}
      onClick={(e) => {
        if (isLocked) {
          e.preventDefault();
        }
      }}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Status Icon */}
      <div
        className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-all ${
          isCompleted
            ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 shadow-sm'
            : isInProgress
            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 shadow-sm'
            : isLocked
            ? 'bg-slate-100 dark:bg-slate-700 text-slate-400'
            : `${typeStyle.bg} ${typeStyle.text}`
        }`}
      >
        {isCompleted ? (
          <CheckCircle className="w-5 h-5" />
        ) : isLocked ? (
          <Lock className="w-5 h-5" />
        ) : (
          typeStyle.icon
        )}
      </div>

      {/* Lesson Info */}
      <div className="flex-1 min-w-0">
        <h4 className="font-medium text-slate-800 dark:text-white truncate">
          {lesson.title}
        </h4>
        <div className="flex items-center gap-2 mt-1.5 flex-wrap">
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium ${typeStyle.bg} ${typeStyle.text}`}>
            {typeStyle.icon}
            {typeStyle.label}
          </span>
          <span className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
            <Clock className="w-3 h-3" />
            {lesson.estimated_minutes}분
          </span>
          <span className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
            <Zap className="w-3 h-3" />
            {lesson.xp_reward} XP
          </span>
        </div>
      </div>

      {/* Action */}
      {!isLocked && (
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${levelStyles.bg}`}>
          <ChevronRight className={`w-5 h-5 ${levelStyles.text}`} />
        </div>
      )}
    </Link>
  );
}

export default function PathDetailPage() {
  const { pathId } = useParams<{ pathId: string }>();
  const [path, setPath] = useState<LearningPath | null>(null);
  const [expandedModules, setExpandedModules] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (pathId) {
      loadPath();
    }
  }, [pathId]);

  const loadPath = async () => {
    try {
      setLoading(true);
      const pathData = await roadmapApi.getPath(pathId!);
      setPath(pathData);

      // Auto-expand first incomplete module
      const firstIncomplete = pathData.modules.find(
        (m) => m.completion_rate < 100
      );
      if (firstIncomplete) {
        setExpandedModules(new Set([firstIncomplete.id]));
      } else if (pathData.modules.length > 0) {
        setExpandedModules(new Set([pathData.modules[0].id]));
      }
    } catch (err) {
      setError('학습 경로를 불러오는데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleModule = (moduleId: string) => {
    setExpandedModules((prev) => {
      const next = new Set(prev);
      if (next.has(moduleId)) {
        next.delete(moduleId);
      } else {
        next.add(moduleId);
      }
      return next;
    });
  };

  const levelStyles = LEVEL_STYLES[path?.level || 'beginner'] || LEVEL_STYLES.beginner;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">학습 경로 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !path) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-6 rounded-2xl flex items-center gap-4">
            <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-xl flex items-center justify-center flex-shrink-0">
              <span className="text-2xl">!</span>
            </div>
            <div>
              <h3 className="font-semibold mb-1">오류가 발생했습니다</h3>
              <p>{error || '학습 경로를 찾을 수 없습니다.'}</p>
            </div>
          </div>
          <Link
            to="/roadmap"
            className="mt-6 inline-flex items-center gap-2 text-indigo-600 dark:text-indigo-400 hover:underline font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            로드맵으로 돌아가기
          </Link>
        </div>
      </div>
    );
  }

  // Calculate stats
  const completedLessons = path.modules.reduce((sum, m) => sum + m.completed_lessons, 0);
  const totalLessons = path.modules.reduce((sum, m) => sum + m.lesson_count, 0);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className={`bg-gradient-to-r ${levelStyles.gradient} relative overflow-hidden`}>
        {/* Decorative elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-black/10 rounded-full blur-3xl" />
          <Target className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Sparkles className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
        </div>

        <div className="max-w-5xl mx-auto px-6 py-8 relative">
          {/* Back Link */}
          <Link
            to="/roadmap"
            className="inline-flex items-center gap-2 text-white/80 hover:text-white mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            로드맵으로 돌아가기
          </Link>

          {/* Path Header */}
          <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
            <div className="w-20 h-20 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center text-5xl shadow-lg">
              {path.icon}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-3 py-1 bg-white/20 backdrop-blur-sm rounded-full text-sm font-medium text-white">
                  {path.level_display}
                </span>
                {path.completion_rate >= 100 && (
                  <span className="inline-flex items-center gap-1 px-3 py-1 bg-white/30 backdrop-blur-sm rounded-full text-sm font-medium text-white">
                    <Trophy className="w-4 h-4" />
                    완료!
                  </span>
                )}
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">{path.title}</h1>
              <p className="text-white/80 max-w-2xl">{path.description}</p>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 text-white/80 mb-1">
                <BookOpen className="w-4 h-4" />
                <span className="text-sm">모듈</span>
              </div>
              <p className="text-2xl font-bold text-white">{path.module_count}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 text-white/80 mb-1">
                <Star className="w-4 h-4" />
                <span className="text-sm">레슨</span>
              </div>
              <p className="text-2xl font-bold text-white">{completedLessons}/{totalLessons}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 text-white/80 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-sm">예상 시간</span>
              </div>
              <p className="text-2xl font-bold text-white">{path.estimated_hours}h</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 text-white/80 mb-1">
                <Zap className="w-4 h-4" />
                <span className="text-sm">총 XP</span>
              </div>
              <p className="text-2xl font-bold text-white">{path.total_xp}</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-8">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-white/80">전체 진행률</span>
              <span className="text-sm font-bold text-white">{path.completion_rate.toFixed(0)}%</span>
            </div>
            <div className="relative w-full h-4 bg-white/20 rounded-full overflow-hidden">
              <div
                className="absolute top-0 left-0 h-full bg-white rounded-full transition-all duration-500"
                style={{ width: `${path.completion_rate}%` }}
              />
              {/* Animated shine */}
              {path.completion_rate > 0 && path.completion_rate < 100 && (
                <div className="absolute top-0 left-0 w-full h-full overflow-hidden">
                  <div className="w-20 h-full bg-gradient-to-r from-transparent via-white/40 to-transparent animate-shine" />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Section Title */}
        <div className="mb-6">
          <h2 className="text-xl font-bold text-slate-800 dark:text-white flex items-center gap-2">
            <BookOpen className={`w-5 h-5 ${levelStyles.text}`} />
            학습 모듈
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            각 모듈을 순서대로 완료하세요
          </p>
        </div>

        {/* Modules */}
        <div className="space-y-4">
          {path.modules.map((module, index) => (
            <ModuleCard
              key={module.id}
              module={module}
              index={index}
              isExpanded={expandedModules.has(module.id)}
              onToggle={() => toggleModule(module.id)}
              levelStyles={levelStyles}
            />
          ))}
        </div>

        {/* Empty State */}
        {path.modules.length === 0 && (
          <div className="text-center py-16 bg-white dark:bg-slate-800 rounded-2xl shadow-lg">
            <div className="w-20 h-20 mx-auto mb-6 bg-slate-100 dark:bg-slate-700 rounded-2xl flex items-center justify-center">
              <BookOpen className="w-10 h-10 text-slate-400" />
            </div>
            <h3 className="text-lg font-semibold text-slate-800 dark:text-white mb-2">
              모듈이 없습니다
            </h3>
            <p className="text-slate-500 dark:text-slate-400">
              곧 새로운 학습 모듈이 추가될 예정입니다.
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
