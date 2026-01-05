/**
 * Learning Path Detail Page
 * Shows modules and lessons for a specific path
 */
import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  BookOpen,
  CheckCircle,
  Clock,
  ChevronRight,
  ChevronDown,
  Zap,
  ArrowLeft,
  Play,
  FileText,
  Code,
  Keyboard,
  HelpCircle,
  BookMarked,
  Lock,
} from 'lucide-react';
import { roadmapApi } from '../../api/roadmap';
import type { LearningPath, Module, Lesson, LessonType } from '../../api/roadmap';

export default function PathDetailPage() {
  const { pathId } = useParams<{ pathId: string }>();
  const navigate = useNavigate();
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

  const getLessonIcon = (type: LessonType) => {
    switch (type) {
      case 'concept':
        return <FileText className="w-4 h-4" />;
      case 'problem':
        return <Code className="w-4 h-4" />;
      case 'typing':
        return <Keyboard className="w-4 h-4" />;
      case 'pattern':
        return <BookMarked className="w-4 h-4" />;
      case 'quiz':
        return <HelpCircle className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  const getLessonTypeLabel = (type: LessonType) => {
    switch (type) {
      case 'concept':
        return '개념';
      case 'problem':
        return '문제';
      case 'typing':
        return '타이핑';
      case 'pattern':
        return '패턴';
      case 'quiz':
        return '퀴즈';
      default:
        return type;
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

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error || !path) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-lg">
          {error || '학습 경로를 찾을 수 없습니다.'}
        </div>
        <Link
          to="/roadmap"
          className="mt-4 inline-flex items-center text-indigo-600 hover:underline"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          로드맵으로 돌아가기
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Back Link */}
      <Link
        to="/roadmap"
        className="inline-flex items-center text-gray-600 dark:text-gray-400 hover:text-indigo-600 mb-6"
      >
        <ArrowLeft className="w-4 h-4 mr-1" />
        로드맵으로 돌아가기
      </Link>

      {/* Path Header */}
      <div className={`bg-gradient-to-r ${getLevelColor(path.level)} rounded-xl p-6 text-white mb-8`}>
        <div className="flex items-center gap-4">
          <span className="text-5xl">{path.icon}</span>
          <div>
            <span className="px-2 py-0.5 bg-white/20 rounded text-sm">
              {path.level_display}
            </span>
            <h1 className="text-3xl font-bold mt-2">{path.title}</h1>
            <p className="text-sm opacity-90 mt-1">{path.description}</p>
          </div>
        </div>

        {/* Stats */}
        <div className="flex flex-wrap gap-6 mt-6 text-sm">
          <span className="flex items-center gap-1">
            <BookOpen className="w-4 h-4" />
            {path.module_count}개 모듈
          </span>
          <span className="flex items-center gap-1">
            <CheckCircle className="w-4 h-4" />
            {path.lesson_count}개 레슨
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            약 {path.estimated_hours}시간
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-4 h-4" />
            {path.total_xp} XP
          </span>
        </div>

        {/* Progress */}
        <div className="mt-6">
          <div className="flex justify-between text-sm mb-2">
            <span>진행률</span>
            <span className="font-bold">{path.completion_rate.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-white/20 rounded-full h-3">
            <div
              className="bg-white h-3 rounded-full transition-all"
              style={{ width: `${path.completion_rate}%` }}
            />
          </div>
        </div>
      </div>

      {/* Modules */}
      <div className="space-y-4">
        {path.modules.map((module, moduleIndex) => (
          <div
            key={module.id}
            className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden"
          >
            {/* Module Header */}
            <button
              onClick={() => toggleModule(module.id)}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                    module.completion_rate >= 100
                      ? 'bg-green-500 text-white'
                      : module.completion_rate > 0
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300'
                  }`}
                >
                  {module.completion_rate >= 100 ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    moduleIndex + 1
                  )}
                </div>
                <div className="text-left">
                  <h3 className="font-semibold text-gray-900 dark:text-white">
                    {module.title}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {module.description}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right text-sm text-gray-500 dark:text-gray-400">
                  <span>
                    {module.completed_lessons}/{module.lesson_count} 레슨
                  </span>
                </div>
                {expandedModules.has(module.id) ? (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronRight className="w-5 h-5 text-gray-400" />
                )}
              </div>
            </button>

            {/* Lessons */}
            {expandedModules.has(module.id) && (
              <div className="border-t border-gray-200 dark:border-gray-700">
                {module.lessons.map((lesson, lessonIndex) => {
                  const isLocked =
                    lessonIndex > 0 &&
                    module.lessons[lessonIndex - 1].status !== 'completed';

                  return (
                    <Link
                      key={lesson.id}
                      to={isLocked ? '#' : `/roadmap/lesson/${lesson.id}`}
                      className={`flex items-center gap-3 p-4 border-b border-gray-100 dark:border-gray-700 last:border-b-0 transition-colors ${
                        isLocked
                          ? 'opacity-50 cursor-not-allowed'
                          : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                      onClick={(e) => {
                        if (isLocked) {
                          e.preventDefault();
                        }
                      }}
                    >
                      {/* Status Icon */}
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          lesson.status === 'completed'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-600'
                            : lesson.status === 'in_progress'
                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600'
                            : isLocked
                            ? 'bg-gray-100 dark:bg-gray-700 text-gray-400'
                            : 'bg-gray-100 dark:bg-gray-700 text-gray-500'
                        }`}
                      >
                        {lesson.status === 'completed' ? (
                          <CheckCircle className="w-4 h-4" />
                        ) : isLocked ? (
                          <Lock className="w-4 h-4" />
                        ) : (
                          getLessonIcon(lesson.lesson_type)
                        )}
                      </div>

                      {/* Lesson Info */}
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {lesson.title}
                        </h4>
                        <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400 mt-1">
                          <span className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">
                            {getLessonTypeLabel(lesson.lesson_type)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {lesson.estimated_minutes}분
                          </span>
                          <span className="flex items-center gap-1">
                            <Zap className="w-3 h-3 text-yellow-500" />
                            {lesson.xp_reward} XP
                          </span>
                        </div>
                      </div>

                      {/* Action */}
                      {!isLocked && (
                        <ChevronRight className="w-5 h-5 text-gray-400" />
                      )}
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
