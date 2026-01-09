/**
 * Typing Practice List Page - Enhanced with modern design
 */
import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Keyboard,
  Clock,
  Target,
  CheckCircle,
  Trophy,
  Loader2,
  Sparkles,
  Zap,
  BookOpen,
  Code2,
  Brain,
  FileCode,
  ArrowRight,
  Star,
} from 'lucide-react';
import { typingPracticeApi } from '../../api/typingPractice';
import type { TypingExercise, UserTypingStats } from '../../api/typingPractice';

// Category styles
const CATEGORY_STYLES: Record<string, { icon: React.ReactNode; bg: string; text: string; label: string }> = {
  template: {
    icon: <FileCode className="w-4 h-4" />,
    bg: 'bg-blue-100 dark:bg-blue-900/30',
    text: 'text-blue-700 dark:text-blue-400',
    label: '템플릿',
  },
  method: {
    icon: <Code2 className="w-4 h-4" />,
    bg: 'bg-purple-100 dark:bg-purple-900/30',
    text: 'text-purple-700 dark:text-purple-400',
    label: '메서드',
  },
  algorithm: {
    icon: <Brain className="w-4 h-4" />,
    bg: 'bg-amber-100 dark:bg-amber-900/30',
    text: 'text-amber-700 dark:text-amber-400',
    label: '알고리즘',
  },
  pattern: {
    icon: <Sparkles className="w-4 h-4" />,
    bg: 'bg-emerald-100 dark:bg-emerald-900/30',
    text: 'text-emerald-700 dark:text-emerald-400',
    label: '패턴',
  },
};

// Difficulty styles
const DIFFICULTY_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  easy: {
    bg: 'bg-emerald-100 dark:bg-emerald-900/30',
    text: 'text-emerald-700 dark:text-emerald-400',
    label: 'Easy',
  },
  medium: {
    bg: 'bg-amber-100 dark:bg-amber-900/30',
    text: 'text-amber-700 dark:text-amber-400',
    label: 'Medium',
  },
  hard: {
    bg: 'bg-red-100 dark:bg-red-900/30',
    text: 'text-red-700 dark:text-red-400',
    label: 'Hard',
  },
};

function getCategoryStyle(category: string) {
  return CATEGORY_STYLES[category] || {
    icon: <BookOpen className="w-4 h-4" />,
    bg: 'bg-gray-100 dark:bg-gray-700',
    text: 'text-gray-700 dark:text-gray-300',
    label: category,
  };
}

function getDifficultyStyle(difficulty: string) {
  return DIFFICULTY_STYLES[difficulty] || DIFFICULTY_STYLES.easy;
}

interface ExerciseCardProps {
  exercise: TypingExercise;
  isMastered: boolean;
  index: number;
}

function ExerciseCard({ exercise, isMastered, index }: ExerciseCardProps) {
  const categoryStyle = getCategoryStyle(exercise.category);
  const difficultyStyle = getDifficultyStyle(exercise.difficulty);

  return (
    <Link
      to={`/typing-practice/${exercise.id}`}
      className={`group block rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 overflow-hidden border hover:-translate-y-1 ${
        isMastered
          ? 'bg-gradient-to-br from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 border-emerald-300 dark:border-emerald-700'
          : 'bg-white dark:bg-slate-800 border-gray-200 dark:border-slate-700'
      }`}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Card Header */}
      <div className={`h-1.5 ${isMastered ? 'bg-gradient-to-r from-emerald-500 to-green-500' : 'bg-gradient-to-r from-orange-500 to-amber-500'}`} />

      <div className="p-5">
        {/* Title and Mastered Badge */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">
              {exercise.title}
            </h3>
            {isMastered && (
              <div className="flex-shrink-0">
                <CheckCircle className="w-5 h-5 text-emerald-500" />
              </div>
            )}
          </div>
          <span className={`flex-shrink-0 px-2.5 py-1 rounded-full text-xs font-medium ${difficultyStyle.bg} ${difficultyStyle.text}`}>
            {difficultyStyle.label}
          </span>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400 mb-3">
          <span className="flex items-center gap-1">
            <FileCode className="w-3.5 h-3.5" />
            {exercise.line_count}줄
          </span>
          <span className="flex items-center gap-1">
            <Keyboard className="w-3.5 h-3.5" />
            {exercise.char_count}자
          </span>
          <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${categoryStyle.bg} ${categoryStyle.text}`}>
            {categoryStyle.icon}
            {categoryStyle.label}
          </span>
        </div>

        {/* Description */}
        {exercise.description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-4">
            {exercise.description}
          </p>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-100 dark:border-slate-700">
          {isMastered ? (
            <span className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
              <Star className="w-3.5 h-3.5 fill-current" />
              마스터 완료
            </span>
          ) : (
            <span className="text-xs text-gray-400 dark:text-gray-500">
              {exercise.required_completions}회 완료 필요
            </span>
          )}
          <span className="flex items-center gap-1 text-orange-600 dark:text-orange-400 text-sm font-medium group-hover:gap-2 transition-all">
            {isMastered ? '복습하기' : '시작하기'}
            <ArrowRight className="w-4 h-4" />
          </span>
        </div>
      </div>
    </Link>
  );
}

export default function TypingPracticeListPage() {
  const [exercises, setExercises] = useState<TypingExercise[]>([]);
  const [stats, setStats] = useState<UserTypingStats | null>(null);
  const [masteredIds, setMasteredIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const categories = [
    { value: 'all', label: '전체', icon: <BookOpen className="w-4 h-4" /> },
    { value: 'template', label: '템플릿', icon: <FileCode className="w-4 h-4" /> },
    { value: 'method', label: '메서드', icon: <Code2 className="w-4 h-4" /> },
    { value: 'algorithm', label: '알고리즘', icon: <Brain className="w-4 h-4" /> },
  ];

  useEffect(() => {
    loadData();
  }, [selectedCategory]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [exercisesRes, statsRes, masteredRes] = await Promise.all([
        typingPracticeApi.listExercises({
          category: selectedCategory === 'all' ? undefined : selectedCategory,
          page_size: 50,
        }),
        typingPracticeApi.getStats().catch(() => null),
        typingPracticeApi.getMasteredExercises().catch(() => []),
      ]);
      setExercises(exercisesRes.exercises);
      setStats(statsRes);
      setMasteredIds(new Set(masteredRes));
    } catch (err) {
      setError('연습 목록을 불러오는데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Calculate stats summary
  const statsSummary = useMemo(() => {
    return {
      total: exercises.length,
      mastered: masteredIds.size,
      masteredPercent: exercises.length > 0 ? Math.round((masteredIds.size / exercises.length) * 100) : 0,
    };
  }, [exercises, masteredIds]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">연습 목록 불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-orange-600/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Keyboard className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Zap className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Target className="absolute top-20 left-[25%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-7xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Sparkles className="w-4 h-4" />
                코드 암기 연습
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Keyboard className="w-10 h-10" />
                받아쓰기 연습
              </h1>
              <p className="text-orange-100 text-lg max-w-md">
                코드를 반복 타이핑하여 알고리즘 템플릿을 손에 익히세요!
              </p>
            </div>

            {/* Stats Cards */}
            <div className="flex gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Target className="w-6 h-6 text-orange-200 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{stats?.total_exercises_attempted || 0}</div>
                <div className="text-xs text-orange-200">시도</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Trophy className="w-6 h-6 text-yellow-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{stats?.total_exercises_mastered || 0}</div>
                <div className="text-xs text-orange-200">마스터</div>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px] border border-white/30">
                <Zap className="w-6 h-6 text-yellow-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{stats?.best_wpm.toFixed(0) || 0}</div>
                <div className="text-xs text-orange-200">최고 WPM</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8 -mt-6">
        {/* Filter and Progress Card */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-5 mb-6 border border-gray-100 dark:border-slate-700">
          <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
            {/* Category Filter */}
            <div className="flex gap-2 flex-wrap">
              {categories.map((cat) => (
                <button
                  key={cat.value}
                  onClick={() => setSelectedCategory(cat.value)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                    selectedCategory === cat.value
                      ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-slate-700 dark:text-gray-300 dark:hover:bg-slate-600'
                  }`}
                >
                  {cat.icon}
                  {cat.label}
                </button>
              ))}
            </div>

            {/* Progress Summary */}
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-500 dark:text-gray-400">
                <span className="font-semibold text-emerald-600 dark:text-emerald-400">{statsSummary.mastered}</span>
                <span className="mx-1">/</span>
                <span>{statsSummary.total}</span>
                <span className="ml-1">마스터</span>
              </div>
              <div className="w-32 h-2 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-green-500 rounded-full transition-all duration-500"
                  style={{ width: `${statsSummary.masteredPercent}%` }}
                />
              </div>
              <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                {statsSummary.masteredPercent}%
              </span>
            </div>
          </div>

          {/* Detailed Stats Row */}
          {stats && (
            <div className="mt-4 pt-4 border-t border-gray-100 dark:border-slate-700 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                  <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">평균 정확률</div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{stats.average_accuracy.toFixed(1)}%</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">평균 WPM</div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{stats.average_wpm.toFixed(0)}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
                  <Clock className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">총 연습 시간</div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{Math.floor(stats.total_practice_time / 60)}분</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
                  <CheckCircle className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">총 시도</div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{stats.total_attempts}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-xl mb-6 border border-red-200 dark:border-red-800">
            {error}
          </div>
        )}

        {/* Exercise List */}
        {exercises.length === 0 ? (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-12 text-center border border-gray-100 dark:border-slate-700">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
              <Keyboard className="w-10 h-10 text-gray-400 dark:text-gray-500" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              연습 문제가 없습니다
            </h3>
            <p className="text-gray-500 dark:text-gray-400">
              선택한 카테고리에 연습 문제가 없습니다.
            </p>
          </div>
        ) : (
          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
            {exercises.map((exercise, index) => (
              <ExerciseCard
                key={exercise.id}
                exercise={exercise}
                isMastered={masteredIds.has(exercise.id)}
                index={index}
              />
            ))}
          </div>
        )}
      </div>

      {/* Styles */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(-5deg); }
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
