/**
 * Typing Practice List Page
 * Lists all available typing exercises for code memorization
 */
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Keyboard, Clock, Target, CheckCircle, Trophy } from 'lucide-react';
import { typingPracticeApi } from '../../api/typingPractice';
import type { TypingExercise, UserTypingStats } from '../../api/typingPractice';

export default function TypingPracticeListPage() {
  const [exercises, setExercises] = useState<TypingExercise[]>([]);
  const [stats, setStats] = useState<UserTypingStats | null>(null);
  const [masteredIds, setMasteredIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const categories = [
    { value: 'all', label: '전체' },
    { value: 'template', label: '템플릿' },
    { value: 'method', label: '메서드' },
    { value: 'algorithm', label: '알고리즘' },
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

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300';
      case 'hard':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300';
    }
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'template':
        return '템플릿';
      case 'method':
        return '메서드';
      case 'algorithm':
        return '알고리즘';
      case 'pattern':
        return '패턴';
      default:
        return category;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
          <Keyboard className="w-8 h-8 text-indigo-600" />
          받아쓰기 연습
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          코드를 5번 반복 타이핑하여 알고리즘 템플릿을 암기하세요
        </p>
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Target className="w-4 h-4" />
              <span className="text-sm">시도한 연습</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.total_exercises_attempted}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm">마스터</span>
            </div>
            <p className="text-2xl font-bold text-green-600">
              {stats.total_exercises_mastered}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Trophy className="w-4 h-4" />
              <span className="text-sm">최고 속도</span>
            </div>
            <p className="text-2xl font-bold text-indigo-600">
              {stats.best_wpm.toFixed(0)} WPM
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Clock className="w-4 h-4" />
              <span className="text-sm">평균 정확률</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.average_accuracy.toFixed(1)}%
            </p>
          </div>
        </div>
      )}

      {/* Category Filter */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {categories.map((cat) => (
          <button
            key={cat.value}
            onClick={() => setSelectedCategory(cat.value)}
            className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
              selectedCategory === cat.value
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Exercise List */}
      {exercises.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <Keyboard className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            아직 연습 문제가 없습니다.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {exercises.map((exercise) => (
            <Link
              key={exercise.id}
              to={`/typing-practice/${exercise.id}`}
              className={`block rounded-lg shadow hover:shadow-lg transition-shadow p-5 ${
                masteredIds.has(exercise.id)
                  ? 'bg-green-50 dark:bg-green-900/20 border-2 border-green-500'
                  : 'bg-white dark:bg-gray-800'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {exercise.title}
                  </h3>
                  {masteredIds.has(exercise.id) && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                </div>
                <span
                  className={`px-2 py-1 rounded text-xs font-medium ${getDifficultyColor(
                    exercise.difficulty
                  )}`}
                >
                  {exercise.difficulty}
                </span>
              </div>

              <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400 mb-3">
                <span className="flex items-center gap-1">
                  <span className="font-medium">{exercise.line_count}</span> 줄
                </span>
                <span className="flex items-center gap-1">
                  <span className="font-medium">{exercise.char_count}</span> 자
                </span>
                <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                  {getCategoryLabel(exercise.category)}
                </span>
              </div>

              {exercise.description && (
                <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                  {exercise.description}
                </p>
              )}

              <div className="mt-4 flex items-center justify-between">
                {masteredIds.has(exercise.id) ? (
                  <span className="text-xs text-green-600 dark:text-green-400 font-medium flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" />
                    마스터 완료
                  </span>
                ) : (
                  <span className="text-xs text-gray-400">
                    {exercise.required_completions}회 완료 필요
                  </span>
                )}
                <span className="text-indigo-600 dark:text-indigo-400 text-sm font-medium">
                  {masteredIds.has(exercise.id) ? '복습하기 →' : '시작하기 →'}
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
