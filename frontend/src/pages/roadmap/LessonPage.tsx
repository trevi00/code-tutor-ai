/**
 * Lesson Page
 * Shows individual lesson content based on lesson type
 */
import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  Clock,
  Zap,
  FileText,
  Code,
  Keyboard,
  BookMarked,
  HelpCircle,
  Loader2,
} from 'lucide-react';
import { roadmapApi } from '../../api/roadmap';
import type { Lesson, LessonType, Module } from '../../api/roadmap';

export default function LessonPage() {
  const { lessonId } = useParams<{ lessonId: string }>();
  const navigate = useNavigate();
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [module, setModule] = useState<Module | null>(null);
  const [loading, setLoading] = useState(true);
  const [completing, setCompleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (lessonId) {
      loadLesson();
    }
  }, [lessonId]);

  const loadLesson = async () => {
    try {
      setLoading(true);
      const lessonData = await roadmapApi.getLesson(lessonId!);
      setLesson(lessonData);

      // Load module for navigation
      const moduleData = await roadmapApi.getModule(lessonData.module_id);
      setModule(moduleData);
    } catch (err) {
      setError('레슨을 불러오는데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async () => {
    if (!lesson) return;

    try {
      setCompleting(true);
      await roadmapApi.completeLesson(lesson.id);

      // Refresh lesson data
      await loadLesson();

      // Navigate to next lesson if available
      const nextLesson = getNextLesson();
      if (nextLesson) {
        navigate(`/roadmap/lesson/${nextLesson.id}`);
      }
    } catch (err) {
      console.error('Failed to complete lesson:', err);
    } finally {
      setCompleting(false);
    }
  };

  const getLessonIcon = (type: LessonType) => {
    switch (type) {
      case 'concept':
        return <FileText className="w-5 h-5" />;
      case 'problem':
        return <Code className="w-5 h-5" />;
      case 'typing':
        return <Keyboard className="w-5 h-5" />;
      case 'pattern':
        return <BookMarked className="w-5 h-5" />;
      case 'quiz':
        return <HelpCircle className="w-5 h-5" />;
      default:
        return <FileText className="w-5 h-5" />;
    }
  };

  const getLessonTypeLabel = (type: LessonType) => {
    switch (type) {
      case 'concept':
        return '개념 학습';
      case 'problem':
        return '문제 풀이';
      case 'typing':
        return '타이핑 연습';
      case 'pattern':
        return '패턴 학습';
      case 'quiz':
        return '퀴즈';
      default:
        return type;
    }
  };

  const getPrevLesson = () => {
    if (!module || !lesson) return null;
    const currentIndex = module.lessons.findIndex((l) => l.id === lesson.id);
    if (currentIndex > 0) {
      return module.lessons[currentIndex - 1];
    }
    return null;
  };

  const getNextLesson = () => {
    if (!module || !lesson) return null;
    const currentIndex = module.lessons.findIndex((l) => l.id === lesson.id);
    if (currentIndex < module.lessons.length - 1) {
      return module.lessons[currentIndex + 1];
    }
    return null;
  };

  const renderLessonContent = () => {
    if (!lesson) return null;

    // For linked content (problem, typing, pattern), show a link to the actual content
    if (lesson.content_id) {
      switch (lesson.lesson_type) {
        case 'problem':
          return (
            <div className="text-center py-12">
              <Code className="w-16 h-16 mx-auto text-indigo-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">문제 풀이</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                {lesson.content}
              </p>
              <Link
                to={`/problems/${lesson.content_id}/solve`}
                className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
              >
                <Code className="w-5 h-5" />
                문제 풀러 가기
              </Link>
            </div>
          );

        case 'typing':
          return (
            <div className="text-center py-12">
              <Keyboard className="w-16 h-16 mx-auto text-green-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">타이핑 연습</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                {lesson.content}
              </p>
              <Link
                to={`/typing-practice/${lesson.content_id}`}
                className="inline-flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                <Keyboard className="w-5 h-5" />
                타이핑 연습하기
              </Link>
            </div>
          );

        case 'pattern':
          return (
            <div className="text-center py-12">
              <BookMarked className="w-16 h-16 mx-auto text-purple-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">패턴 학습</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                {lesson.content}
              </p>
              <Link
                to={`/patterns/${lesson.content_id}`}
                className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                <BookMarked className="w-5 h-5" />
                패턴 학습하기
              </Link>
            </div>
          );

        default:
          break;
      }
    }

    // For concept lessons, render the content directly
    return (
      <div className="prose dark:prose-invert max-w-none">
        <div
          className="whitespace-pre-wrap text-gray-700 dark:text-gray-300 leading-relaxed"
          dangerouslySetInnerHTML={{
            __html: lesson.content
              .replace(/\n/g, '<br/>')
              .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto"><code>$2</code></pre>')
              .replace(/`([^`]+)`/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">$1</code>')
              .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
              .replace(/\*([^*]+)\*/g, '<em>$1</em>'),
          }}
        />
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error || !lesson) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-lg">
          {error || '레슨을 찾을 수 없습니다.'}
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

  const prevLesson = getPrevLesson();
  const nextLesson = getNextLesson();

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Back Link */}
      {module && (
        <Link
          to={`/roadmap/${module.path_id}`}
          className="inline-flex items-center text-gray-600 dark:text-gray-400 hover:text-indigo-600 mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          학습 경로로 돌아가기
        </Link>
      )}

      {/* Lesson Header */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-indigo-100 dark:bg-indigo-900/30 rounded-full flex items-center justify-center text-indigo-600">
            {getLessonIcon(lesson.lesson_type)}
          </div>
          <div>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {module?.title}
            </span>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {lesson.title}
            </h1>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 text-sm text-gray-600 dark:text-gray-400">
          <span className="flex items-center gap-1 px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
            {getLessonIcon(lesson.lesson_type)}
            {getLessonTypeLabel(lesson.lesson_type)}
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            {lesson.estimated_minutes}분
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-4 h-4 text-yellow-500" />
            {lesson.xp_reward} XP
          </span>
          {lesson.status === 'completed' && (
            <span className="flex items-center gap-1 text-green-600">
              <CheckCircle className="w-4 h-4" />
              완료됨
            </span>
          )}
        </div>
      </div>

      {/* Lesson Content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
        {renderLessonContent()}
      </div>

      {/* Navigation & Complete */}
      <div className="flex items-center justify-between">
        {/* Previous Lesson */}
        {prevLesson ? (
          <Link
            to={`/roadmap/lesson/${prevLesson.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-indigo-600 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span className="hidden sm:inline">이전: {prevLesson.title}</span>
            <span className="sm:hidden">이전</span>
          </Link>
        ) : (
          <div />
        )}

        {/* Complete Button */}
        {lesson.status !== 'completed' && (
          <button
            onClick={handleComplete}
            disabled={completing}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
          >
            {completing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                처리 중...
              </>
            ) : (
              <>
                <CheckCircle className="w-5 h-5" />
                완료하기
              </>
            )}
          </button>
        )}

        {/* Next Lesson */}
        {nextLesson ? (
          <Link
            to={`/roadmap/lesson/${nextLesson.id}`}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-indigo-600 transition-colors"
          >
            <span className="hidden sm:inline">다음: {nextLesson.title}</span>
            <span className="sm:hidden">다음</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        ) : lesson.status === 'completed' ? (
          <Link
            to="/roadmap"
            className="flex items-center gap-2 text-green-600 hover:text-green-700 transition-colors"
          >
            <span>로드맵으로</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        ) : (
          <div />
        )}
      </div>
    </div>
  );
}
