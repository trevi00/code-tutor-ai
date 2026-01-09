/**
 * Lesson Page - Enhanced with modern design
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
  Play,
  Sparkles,
  BookOpen,
  Target,
  ChevronRight,
  Award,
} from 'lucide-react';
import { roadmapApi } from '../../api/roadmap';
import type { Lesson, LessonType, Module } from '../../api/roadmap';

// Lesson type styles
const LESSON_TYPE_STYLES: Record<LessonType, {
  gradient: string;
  bgLight: string;
  bgDark: string;
  text: string;
  icon: React.ReactNode;
  label: string;
  buttonBg: string;
  buttonHover: string;
}> = {
  concept: {
    gradient: 'from-blue-500 via-blue-600 to-cyan-600',
    bgLight: 'bg-blue-100',
    bgDark: 'dark:bg-blue-900/30',
    text: 'text-blue-600 dark:text-blue-400',
    icon: <FileText className="w-6 h-6" />,
    label: '개념 학습',
    buttonBg: 'bg-blue-600',
    buttonHover: 'hover:bg-blue-700',
  },
  problem: {
    gradient: 'from-purple-500 via-indigo-600 to-purple-700',
    bgLight: 'bg-purple-100',
    bgDark: 'dark:bg-purple-900/30',
    text: 'text-purple-600 dark:text-purple-400',
    icon: <Code className="w-6 h-6" />,
    label: '문제 풀이',
    buttonBg: 'bg-purple-600',
    buttonHover: 'hover:bg-purple-700',
  },
  typing: {
    gradient: 'from-green-500 via-emerald-600 to-teal-600',
    bgLight: 'bg-green-100',
    bgDark: 'dark:bg-green-900/30',
    text: 'text-green-600 dark:text-green-400',
    icon: <Keyboard className="w-6 h-6" />,
    label: '타이핑 연습',
    buttonBg: 'bg-green-600',
    buttonHover: 'hover:bg-green-700',
  },
  pattern: {
    gradient: 'from-amber-500 via-orange-500 to-yellow-600',
    bgLight: 'bg-amber-100',
    bgDark: 'dark:bg-amber-900/30',
    text: 'text-amber-600 dark:text-amber-400',
    icon: <BookMarked className="w-6 h-6" />,
    label: '패턴 학습',
    buttonBg: 'bg-amber-600',
    buttonHover: 'hover:bg-amber-700',
  },
  quiz: {
    gradient: 'from-pink-500 via-rose-500 to-red-500',
    bgLight: 'bg-pink-100',
    bgDark: 'dark:bg-pink-900/30',
    text: 'text-pink-600 dark:text-pink-400',
    icon: <HelpCircle className="w-6 h-6" />,
    label: '퀴즈',
    buttonBg: 'bg-pink-600',
    buttonHover: 'hover:bg-pink-700',
  },
};

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

  const getCurrentLessonIndex = () => {
    if (!module || !lesson) return 0;
    return module.lessons.findIndex((l) => l.id === lesson.id);
  };

  const renderLessonContent = () => {
    if (!lesson) return null;
    const styles = LESSON_TYPE_STYLES[lesson.lesson_type] || LESSON_TYPE_STYLES.concept;

    // For linked content (problem, typing, pattern), show a link to the actual content
    if (lesson.content_id) {
      const contentConfig = {
        problem: {
          icon: <Code className="w-20 h-20" />,
          title: '문제 풀이',
          link: `/problems/${lesson.content_id}/solve`,
          buttonText: '문제 풀러 가기',
          buttonIcon: <Play className="w-5 h-5" />,
        },
        typing: {
          icon: <Keyboard className="w-20 h-20" />,
          title: '타이핑 연습',
          link: `/typing-practice/${lesson.content_id}`,
          buttonText: '타이핑 연습하기',
          buttonIcon: <Keyboard className="w-5 h-5" />,
        },
        pattern: {
          icon: <BookMarked className="w-20 h-20" />,
          title: '패턴 학습',
          link: `/patterns/${lesson.content_id}`,
          buttonText: '패턴 학습하기',
          buttonIcon: <BookOpen className="w-5 h-5" />,
        },
      };

      const config = contentConfig[lesson.lesson_type as keyof typeof contentConfig];
      if (config) {
        return (
          <div className="text-center py-12 animate-fade-in">
            <div className={`w-32 h-32 mx-auto rounded-3xl ${styles.bgLight} ${styles.bgDark} flex items-center justify-center mb-6 ${styles.text}`}>
              {config.icon}
            </div>
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
              {config.title}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md mx-auto leading-relaxed">
              {lesson.content}
            </p>
            <Link
              to={config.link}
              className={`inline-flex items-center gap-3 px-8 py-4 ${styles.buttonBg} ${styles.buttonHover} text-white rounded-xl font-medium transition-all duration-200 transform hover:scale-105 hover:shadow-lg`}
            >
              {config.buttonIcon}
              {config.buttonText}
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        );
      }
    }

    // For concept lessons, render the content directly
    return (
      <div className="prose dark:prose-invert max-w-none animate-fade-in">
        <div
          className="whitespace-pre-wrap text-gray-700 dark:text-gray-300 leading-relaxed text-lg"
          dangerouslySetInnerHTML={{
            __html: lesson.content
              .replace(/\n/g, '<br/>')
              .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-gray-100 dark:bg-slate-800 p-4 rounded-xl overflow-x-auto my-4 border border-gray-200 dark:border-slate-700"><code class="text-sm">$2</code></pre>')
              .replace(/`([^`]+)`/g, '<code class="bg-gray-100 dark:bg-slate-800 px-2 py-1 rounded-md text-sm font-mono">$1</code>')
              .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-gray-900 dark:text-white">$1</strong>')
              .replace(/\*([^*]+)\*/g, '<em>$1</em>'),
          }}
        />
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">레슨 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !lesson) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 p-6">
        <div className="max-w-2xl mx-auto pt-12">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-6 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <HelpCircle className="w-5 h-5" />
              </div>
              <span className="font-semibold text-lg">오류 발생</span>
            </div>
            <p>{error || '레슨을 찾을 수 없습니다.'}</p>
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

  const prevLesson = getPrevLesson();
  const nextLesson = getNextLesson();
  const currentIndex = getCurrentLessonIndex();
  const totalLessons = module?.lessons.length || 0;
  const styles = LESSON_TYPE_STYLES[lesson.lesson_type] || LESSON_TYPE_STYLES.concept;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className={`bg-gradient-to-r ${styles.gradient} relative overflow-hidden`}>
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-black/10 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Sparkles className="absolute top-8 right-[10%] w-10 h-10 text-white/10 animate-float" />
          <BookOpen className="absolute bottom-8 left-[15%] w-8 h-8 text-white/10 animate-float-delayed" />
          <Target className="absolute top-16 left-[25%] w-6 h-6 text-white/10 animate-float" />
        </div>

        <div className="max-w-4xl mx-auto px-6 py-10 relative">
          {/* Back Link */}
          {module && (
            <Link
              to={`/roadmap/${module.path_id}`}
              className="inline-flex items-center gap-2 text-white/80 hover:text-white mb-6 transition-colors group"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
              <span>{module.title}</span>
            </Link>
          )}

          <div className="flex items-start gap-5">
            {/* Lesson Icon */}
            <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center text-white shrink-0">
              {styles.icon}
            </div>

            <div className="flex-1 min-w-0">
              {/* Type Badge */}
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-white/90 text-sm mb-3">
                {styles.icon}
                <span>{styles.label}</span>
              </div>

              {/* Title */}
              <h1 className="text-2xl md:text-3xl font-bold text-white mb-3">
                {lesson.title}
              </h1>

              {/* Meta Info */}
              <div className="flex flex-wrap items-center gap-4 text-white/80">
                <span className="flex items-center gap-1.5">
                  <Clock className="w-4 h-4" />
                  {lesson.estimated_minutes}분
                </span>
                <span className="flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-yellow-300" />
                  {lesson.xp_reward} XP
                </span>
                {lesson.status === 'completed' && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 bg-green-500/30 rounded-full text-green-100">
                    <CheckCircle className="w-4 h-4" />
                    완료됨
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Progress indicator */}
          {module && (
            <div className="mt-8">
              <div className="flex items-center justify-between text-sm text-white/70 mb-2">
                <span>진행률</span>
                <span>{currentIndex + 1} / {totalLessons} 레슨</span>
              </div>
              <div className="h-2 bg-white/20 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white rounded-full transition-all duration-500 relative overflow-hidden"
                  style={{ width: `${((currentIndex + 1) / totalLessons) * 100}%` }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shine" />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-6 py-8 -mt-4">
        {/* Lesson Content Card */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-gray-100 dark:border-slate-700 p-8 mb-8">
          {renderLessonContent()}
        </div>

        {/* Navigation & Complete */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Previous Lesson */}
          <div className="md:col-span-1">
            {prevLesson ? (
              <Link
                to={`/roadmap/lesson/${prevLesson.id}`}
                className="flex items-center gap-3 p-4 bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 hover:border-indigo-300 dark:hover:border-indigo-600 hover:shadow-md transition-all group"
              >
                <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-slate-700 flex items-center justify-center group-hover:bg-indigo-100 dark:group-hover:bg-indigo-900/30 transition-colors">
                  <ArrowLeft className="w-5 h-5 text-gray-500 dark:text-gray-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 group-hover:-translate-x-0.5 transition-all" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-gray-500 dark:text-gray-400">이전</div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {prevLesson.title}
                  </div>
                </div>
              </Link>
            ) : (
              <div className="p-4 bg-gray-50 dark:bg-slate-800/50 rounded-xl border border-gray-100 dark:border-slate-700/50 opacity-50">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
                    <ArrowLeft className="w-5 h-5 text-gray-400" />
                  </div>
                  <div className="text-sm text-gray-400">이전 레슨 없음</div>
                </div>
              </div>
            )}
          </div>

          {/* Complete Button */}
          <div className="md:col-span-1 flex items-center justify-center">
            {lesson.status !== 'completed' ? (
              <button
                onClick={handleComplete}
                disabled={completing}
                className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-medium hover:from-green-600 hover:to-emerald-700 transition-all duration-200 transform hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:hover:scale-100"
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
            ) : (
              <div className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-xl font-medium">
                <Award className="w-5 h-5" />
                학습 완료!
              </div>
            )}
          </div>

          {/* Next Lesson */}
          <div className="md:col-span-1">
            {nextLesson ? (
              <Link
                to={`/roadmap/lesson/${nextLesson.id}`}
                className="flex items-center gap-3 p-4 bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 hover:border-indigo-300 dark:hover:border-indigo-600 hover:shadow-md transition-all group"
              >
                <div className="flex-1 min-w-0 text-right">
                  <div className="text-xs text-gray-500 dark:text-gray-400">다음</div>
                  <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {nextLesson.title}
                  </div>
                </div>
                <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-slate-700 flex items-center justify-center group-hover:bg-indigo-100 dark:group-hover:bg-indigo-900/30 transition-colors">
                  <ArrowRight className="w-5 h-5 text-gray-500 dark:text-gray-400 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 group-hover:translate-x-0.5 transition-all" />
                </div>
              </Link>
            ) : lesson.status === 'completed' ? (
              <Link
                to="/roadmap"
                className="flex items-center gap-3 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800 hover:shadow-md transition-all group"
              >
                <div className="flex-1 min-w-0 text-right">
                  <div className="text-xs text-indigo-500 dark:text-indigo-400">모듈 완료</div>
                  <div className="text-sm font-medium text-indigo-600 dark:text-indigo-300">
                    로드맵으로 이동
                  </div>
                </div>
                <div className="w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                  <ArrowRight className="w-5 h-5 text-indigo-600 dark:text-indigo-400 group-hover:translate-x-0.5 transition-transform" />
                </div>
              </Link>
            ) : (
              <div className="p-4 bg-gray-50 dark:bg-slate-800/50 rounded-xl border border-gray-100 dark:border-slate-700/50 opacity-50">
                <div className="flex items-center gap-3 justify-end">
                  <div className="text-sm text-gray-400 text-right">다음 레슨 없음</div>
                  <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
                    <ArrowRight className="w-5 h-5 text-gray-400" />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Module lessons overview */}
        {module && module.lessons.length > 1 && (
          <div className="mt-8 bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-gray-100 dark:border-slate-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-500" />
              이 모듈의 레슨
            </h3>
            <div className="space-y-2">
              {module.lessons.map((l, idx) => {
                const lStyles = LESSON_TYPE_STYLES[l.lesson_type] || LESSON_TYPE_STYLES.concept;
                const isCurrent = l.id === lesson.id;
                const isCompleted = l.status === 'completed';

                return (
                  <Link
                    key={l.id}
                    to={`/roadmap/lesson/${l.id}`}
                    className={`flex items-center gap-3 p-3 rounded-xl transition-all ${
                      isCurrent
                        ? `${lStyles.bgLight} ${lStyles.bgDark} border-2 border-current ${lStyles.text}`
                        : 'hover:bg-gray-50 dark:hover:bg-slate-700/50'
                    }`}
                  >
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-medium ${
                      isCompleted
                        ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                        : isCurrent
                          ? `${lStyles.bgLight} ${lStyles.bgDark} ${lStyles.text}`
                          : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-gray-400'
                    }`}>
                      {isCompleted ? <CheckCircle className="w-4 h-4" /> : idx + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`text-sm font-medium truncate ${
                        isCurrent
                          ? lStyles.text
                          : 'text-gray-900 dark:text-white'
                      }`}>
                        {l.title}
                      </div>
                    </div>
                    <div className={`text-xs px-2 py-1 rounded ${lStyles.bgLight} ${lStyles.bgDark} ${lStyles.text}`}>
                      {lStyles.label}
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Styles */}
      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
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
