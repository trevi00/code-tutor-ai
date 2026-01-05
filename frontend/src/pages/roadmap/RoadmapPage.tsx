/**
 * Learning Roadmap Page
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
} from 'lucide-react';
import { roadmapApi } from '../../api/roadmap';
import type { LearningPath, UserProgress } from '../../api/roadmap';

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
      // Navigate anyway, the path detail will handle the start
      navigate(`/roadmap/${pathId}`);
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'beginner':
        return 'from-green-500 to-emerald-600';
      case 'elementary':
        return 'from-blue-500 to-cyan-600';
      case 'intermediate':
        return 'from-purple-500 to-indigo-600';
      case 'advanced':
        return 'from-orange-500 to-red-600';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'in_progress':
        return <Play className="w-6 h-6 text-blue-500" />;
      default:
        return <Lock className="w-6 h-6 text-gray-400" />;
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
          <BookOpen className="w-8 h-8 text-indigo-600" />
          학습 로드맵
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          초보자부터 고수까지, 체계적인 알고리즘 학습 경로
        </p>
      </div>

      {/* Progress Summary */}
      {progress && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <BookOpen className="w-4 h-4" />
              <span className="text-sm">진행 중인 경로</span>
            </div>
            <p className="text-2xl font-bold text-indigo-600">
              {progress.in_progress_paths}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm">완료한 경로</span>
            </div>
            <p className="text-2xl font-bold text-green-600">
              {progress.completed_paths}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Trophy className="w-4 h-4" />
              <span className="text-sm">완료 레슨</span>
            </div>
            <p className="text-2xl font-bold text-purple-600">
              {progress.completed_lessons} / {progress.total_lessons}
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
              <Zap className="w-4 h-4" />
              <span className="text-sm">획득 XP</span>
            </div>
            <p className="text-2xl font-bold text-yellow-600">
              {progress.total_xp_earned}
            </p>
          </div>
        </div>
      )}

      {/* Next Lesson Recommendation */}
      {progress?.next_lesson && (
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg p-6 mb-8 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm opacity-80 mb-1">다음 추천 레슨</p>
              <h3 className="text-xl font-bold">{progress.next_lesson.title}</h3>
              <p className="text-sm opacity-80 mt-1">
                {progress.next_lesson.estimated_minutes}분 · {progress.next_lesson.xp_reward} XP
              </p>
            </div>
            <Link
              to={`/roadmap/lesson/${progress.next_lesson.id}`}
              className="bg-white text-indigo-600 px-4 py-2 rounded-lg font-medium hover:bg-gray-100 transition-colors flex items-center gap-2"
            >
              계속하기
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Learning Paths */}
      <div className="space-y-6">
        {paths.map((path, index) => (
          <div
            key={path.id}
            className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden ${
              path.status === 'completed'
                ? 'ring-2 ring-green-500'
                : path.status === 'in_progress'
                ? 'ring-2 ring-blue-500'
                : ''
            }`}
          >
            {/* Path Header */}
            <div className={`bg-gradient-to-r ${getLevelColor(path.level)} p-6 text-white`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="text-4xl">{path.icon}</span>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-white/20 rounded text-sm">
                        {path.level_display}
                      </span>
                      {getStatusIcon(path.status)}
                    </div>
                    <h2 className="text-2xl font-bold mt-1">{path.title}</h2>
                    <p className="text-sm opacity-90 mt-1">{path.description}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Path Stats */}
            <div className="p-6">
              <div className="flex flex-wrap gap-6 text-sm text-gray-600 dark:text-gray-400 mb-4">
                <span className="flex items-center gap-1">
                  <BookOpen className="w-4 h-4" />
                  {path.module_count}개 모듈
                </span>
                <span className="flex items-center gap-1">
                  <Star className="w-4 h-4" />
                  {path.lesson_count}개 레슨
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  약 {path.estimated_hours}시간
                </span>
                <span className="flex items-center gap-1">
                  <Zap className="w-4 h-4 text-yellow-500" />
                  {path.total_xp} XP
                </span>
              </div>

              {/* Progress Bar */}
              {path.status !== 'not_started' && (
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">진행률</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {path.completion_rate.toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        path.status === 'completed'
                          ? 'bg-green-500'
                          : 'bg-blue-500'
                      }`}
                      style={{ width: `${path.completion_rate}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Action Button */}
              <div className="flex items-center justify-between">
                {/* Prerequisites warning */}
                {index > 0 && paths[index - 1].status !== 'completed' && path.status === 'not_started' && (
                  <p className="text-sm text-amber-600 dark:text-amber-400 flex items-center gap-1">
                    <Lock className="w-4 h-4" />
                    이전 경로를 먼저 완료하세요
                  </p>
                )}
                {(index === 0 || paths[index - 1].status === 'completed' || path.status !== 'not_started') && (
                  <div />
                )}

                {path.status === 'completed' ? (
                  <Link
                    to={`/roadmap/${path.id}`}
                    className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 px-4 py-2 rounded-lg font-medium hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors flex items-center gap-2"
                  >
                    복습하기
                    <ChevronRight className="w-4 h-4" />
                  </Link>
                ) : path.status === 'in_progress' ? (
                  <Link
                    to={`/roadmap/${path.id}`}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center gap-2"
                  >
                    계속하기
                    <ChevronRight className="w-4 h-4" />
                  </Link>
                ) : (
                  <button
                    onClick={() => handleStartPath(path.id)}
                    disabled={index > 0 && paths[index - 1].status !== 'completed'}
                    className="bg-indigo-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-indigo-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    시작하기
                    <ChevronRight className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* No Paths */}
      {paths.length === 0 && (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <BookOpen className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            아직 학습 경로가 없습니다.
          </p>
        </div>
      )}
    </div>
  );
}
