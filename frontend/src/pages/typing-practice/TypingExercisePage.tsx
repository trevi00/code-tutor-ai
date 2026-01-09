/**
 * Typing Exercise Page - Enhanced with modern design
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Editor from '@monaco-editor/react';
import {
  ArrowLeft,
  RotateCcw,
  Clock,
  Target,
  Zap,
  Loader2,
  Trophy,
  Keyboard,
  Play,
  FileCode,
  AlertCircle,
  Star,
  TrendingUp,
} from 'lucide-react';
import { typingPracticeApi } from '../../api/typingPractice';
import type {
  TypingExercise,
  UserProgress,
  TypingAttempt,
} from '../../api/typingPractice';

interface CompareResult {
  accuracy: number;
  correctChars: number;
  totalTyped: number;
  errorPositions: number[];
}

export default function TypingExercisePage() {
  const { exerciseId } = useParams<{ exerciseId: string }>();
  const navigate = useNavigate();

  const [exercise, setExercise] = useState<TypingExercise | null>(null);
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [currentAttempt, setCurrentAttempt] = useState<TypingAttempt | null>(null);
  const [userCode, setUserCode] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCompleting, setIsCompleting] = useState(false);

  // Stats
  const [accuracy, setAccuracy] = useState(100);
  const [wpm, setWpm] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (exerciseId) {
      loadExercise();
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [exerciseId]);

  // Timer effect
  useEffect(() => {
    if (startTime && !isCompleting) {
      timerRef.current = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [startTime, isCompleting]);

  const loadExercise = async () => {
    try {
      setLoading(true);
      const [exerciseData, progressData] = await Promise.all([
        typingPracticeApi.getExercise(exerciseId!),
        typingPracticeApi.getProgress(exerciseId!).catch(() => null),
      ]);
      setExercise(exerciseData);
      setProgress(progressData);
    } catch (err) {
      setError('연습을 불러오는데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const startAttempt = async () => {
    if (!exerciseId) return;

    try {
      const attempt = await typingPracticeApi.startAttempt({
        exercise_id: exerciseId,
      });
      setCurrentAttempt(attempt);
      setUserCode('');
      setStartTime(Date.now());
      setElapsedTime(0);
      setAccuracy(100);
      setWpm(0);
    } catch (err) {
      setError('연습을 시작하는데 실패했습니다.');
      console.error(err);
    }
  };

  const compareCode = useCallback(
    (source: string, typed: string): CompareResult => {
      let correct = 0;
      const errorPositions: number[] = [];

      for (let i = 0; i < typed.length; i++) {
        if (i < source.length && typed[i] === source[i]) {
          correct++;
        } else {
          errorPositions.push(i);
        }
      }

      const accuracy = typed.length > 0 ? (correct / typed.length) * 100 : 100;

      return {
        accuracy,
        correctChars: correct,
        totalTyped: typed.length,
        errorPositions,
      };
    },
    []
  );

  const calculateWPM = useCallback((charCount: number, seconds: number): number => {
    if (seconds === 0) return 0;
    const words = charCount / 5;
    const minutes = seconds / 60;
    return Math.round(words / minutes);
  }, []);

  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      if (!exercise || !currentAttempt) return;

      const newCode = value || '';
      setUserCode(newCode);

      const result = compareCode(exercise.source_code, newCode);
      setAccuracy(result.accuracy);

      if (startTime) {
        const seconds = (Date.now() - startTime) / 1000;
        setWpm(calculateWPM(result.correctChars, seconds));
      }

      if (newCode === exercise.source_code) {
        completeAttempt(newCode, result.accuracy);
      }
    },
    [exercise, currentAttempt, startTime, compareCode, calculateWPM]
  );

  const completeAttempt = async (finalCode: string, finalAccuracy: number) => {
    if (!currentAttempt || isCompleting) return;

    setIsCompleting(true);
    const timeSeconds = startTime ? (Date.now() - startTime) / 1000 : 0;
    const finalWpm = calculateWPM(finalCode.length, timeSeconds);

    try {
      await typingPracticeApi.completeAttempt(currentAttempt.id, {
        user_code: finalCode,
        accuracy: finalAccuracy,
        wpm: finalWpm,
        time_seconds: timeSeconds,
      });

      const newProgress = await typingPracticeApi.getProgress(exerciseId!);
      setProgress(newProgress);
      setCurrentAttempt(null);
    } catch (err) {
      setError('결과를 저장하는데 실패했습니다.');
      console.error(err);
    } finally {
      setIsCompleting(false);
    }
  };

  const resetAttempt = () => {
    setUserCode('');
    setStartTime(Date.now());
    setElapsedTime(0);
    setAccuracy(100);
    setWpm(0);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getAccuracyColor = (acc: number) => {
    if (acc >= 95) return 'text-emerald-500';
    if (acc >= 80) return 'text-amber-500';
    return 'text-red-500';
  };

  const getAccuracyBg = (acc: number) => {
    if (acc >= 95) return 'bg-emerald-500';
    if (acc >= 80) return 'bg-amber-500';
    return 'bg-red-500';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-slate-400">연습 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !exercise) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto px-4">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-400" />
          </div>
          <p className="text-red-400 mb-4">{error || '연습을 찾을 수 없습니다.'}</p>
          <button
            onClick={() => navigate('/typing-practice')}
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            목록으로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  const completedCount = progress?.completed_attempts || 0;
  const requiredCount = exercise.required_completions;
  const isMastered = progress?.is_mastered || false;
  const progressPercent = exercise.source_code.length > 0
    ? (userCode.length / exercise.source_code.length) * 100
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      {/* Header */}
      <div className="bg-slate-800/80 backdrop-blur-sm border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/typing-practice')}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 flex items-center justify-center">
                  <Keyboard className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="font-semibold text-white">{exercise.title}</h1>
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="flex items-center gap-1">
                      <FileCode className="w-3 h-3" />
                      {exercise.line_count}줄
                    </span>
                    <span>•</span>
                    <span>{exercise.char_count}자</span>
                    <span>•</span>
                    <span className={`px-1.5 py-0.5 rounded text-xs ${
                      exercise.difficulty === 'easy' ? 'bg-emerald-500/20 text-emerald-400' :
                      exercise.difficulty === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {exercise.difficulty}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Progress Dots */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                {Array.from({ length: requiredCount }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 rounded-full transition-all ${
                      i < completedCount
                        ? 'bg-emerald-500 shadow-lg shadow-emerald-500/50'
                        : currentAttempt && i === completedCount
                        ? 'bg-orange-500 animate-pulse shadow-lg shadow-orange-500/50'
                        : 'bg-slate-600'
                    }`}
                  />
                ))}
              </div>
              <span className="text-sm text-slate-400 font-medium">
                {completedCount}/{requiredCount}
              </span>
              {isMastered && (
                <div className="flex items-center gap-1 px-2 py-1 bg-emerald-500/20 rounded-full">
                  <Star className="w-4 h-4 text-emerald-400 fill-current" />
                  <span className="text-xs text-emerald-400 font-medium">마스터</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Source Code (Read-only) */}
          <div className="bg-slate-800 rounded-xl shadow-xl overflow-hidden border border-slate-700">
            <div className="px-4 py-3 bg-slate-700/50 border-b border-slate-600 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="ml-2 text-sm font-medium text-slate-300">원본 코드</span>
              </div>
              <span className="text-xs text-slate-500">{exercise.language}</span>
            </div>
            <div className="h-[400px]">
              <Editor
                height="100%"
                defaultLanguage={exercise.language}
                value={exercise.source_code}
                theme="vs-dark"
                options={{
                  readOnly: true,
                  fontSize: 14,
                  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                  minimap: { enabled: false },
                  scrollBeyondLastLine: false,
                  lineNumbers: 'on',
                  wordWrap: 'on',
                  padding: { top: 16 },
                }}
              />
            </div>
          </div>

          {/* Typing Area */}
          <div className="bg-slate-800 rounded-xl shadow-xl overflow-hidden border border-slate-700">
            <div className="px-4 py-3 bg-slate-700/50 border-b border-slate-600 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-orange-500 animate-pulse" />
                <span className="text-sm font-medium text-slate-300">타이핑 영역</span>
              </div>
              {currentAttempt && (
                <button
                  onClick={resetAttempt}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white hover:bg-slate-600 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  다시 시작
                </button>
              )}
            </div>
            <div className="h-[400px]">
              {currentAttempt ? (
                <Editor
                  height="100%"
                  defaultLanguage={exercise.language}
                  value={userCode}
                  onChange={handleEditorChange}
                  theme="vs-dark"
                  options={{
                    fontSize: 14,
                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                    minimap: { enabled: false },
                    scrollBeyondLastLine: false,
                    lineNumbers: 'on',
                    wordWrap: 'on',
                    automaticLayout: true,
                    padding: { top: 16 },
                  }}
                />
              ) : (
                <div className="h-full flex flex-col items-center justify-center p-8">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 flex items-center justify-center mb-6 shadow-lg shadow-orange-500/30">
                    <Play className="w-10 h-10 text-white ml-1" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">
                    {completedCount > 0 ? '다음 시도 준비 완료!' : '연습을 시작하세요'}
                  </h3>
                  <p className="text-slate-400 text-center mb-6 max-w-sm">
                    왼쪽의 코드를 보고 똑같이 타이핑하세요. {requiredCount}번 완료하면 마스터입니다!
                  </p>
                  <button
                    onClick={startAttempt}
                    className="px-8 py-3 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-xl hover:from-orange-600 hover:to-amber-600 transition-all font-medium shadow-lg shadow-orange-500/30 flex items-center gap-2"
                  >
                    <Keyboard className="w-5 h-5" />
                    {completedCount > 0 ? '다음 시도 시작' : '연습 시작하기'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Stats Bar */}
        {currentAttempt && (
          <div className="mt-6 bg-slate-800 rounded-xl shadow-xl p-5 border border-slate-700">
            <div className="grid grid-cols-4 gap-6">
              {/* Accuracy */}
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-slate-400 mb-2">
                  <Target className="w-5 h-5" />
                  <span className="text-sm font-medium">정확률</span>
                </div>
                <p className={`text-3xl font-bold ${getAccuracyColor(accuracy)}`}>
                  {accuracy.toFixed(1)}%
                </p>
              </div>

              {/* WPM */}
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-slate-400 mb-2">
                  <Zap className="w-5 h-5" />
                  <span className="text-sm font-medium">속도</span>
                </div>
                <p className="text-3xl font-bold text-orange-400">{wpm} WPM</p>
              </div>

              {/* Time */}
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-slate-400 mb-2">
                  <Clock className="w-5 h-5" />
                  <span className="text-sm font-medium">시간</span>
                </div>
                <p className="text-3xl font-bold text-white">{formatTime(elapsedTime)}</p>
              </div>

              {/* Characters */}
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 text-slate-400 mb-2">
                  <FileCode className="w-5 h-5" />
                  <span className="text-sm font-medium">진행</span>
                </div>
                <p className="text-3xl font-bold text-white">
                  {userCode.length}<span className="text-lg text-slate-500">/{exercise.source_code.length}</span>
                </p>
              </div>
            </div>

            {/* Progress bar */}
            <div className="mt-5">
              <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full ${getAccuracyBg(accuracy)} transition-all duration-300 relative`}
                  style={{ width: `${progressPercent}%` }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shine" />
                </div>
              </div>
              <div className="flex justify-between mt-2 text-xs text-slate-500">
                <span>0%</span>
                <span className="font-medium text-slate-400">{progressPercent.toFixed(1)}% 완료</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        )}

        {/* Previous Attempts */}
        {progress && progress.attempts.length > 0 && (
          <div className="mt-6 bg-slate-800 rounded-xl shadow-xl p-5 border border-slate-700">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-slate-400" />
              <h3 className="font-medium text-white">이전 시도 기록</h3>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {progress.attempts
                .filter((a) => a.status === 'completed')
                .map((attempt, index) => (
                  <div
                    key={attempt.id}
                    className="bg-slate-700/50 rounded-xl p-4 text-center border border-slate-600 hover:border-slate-500 transition-colors"
                  >
                    <div className="text-xs text-slate-500 mb-2">#{index + 1}</div>
                    <div className={`text-lg font-bold ${getAccuracyColor(attempt.accuracy)}`}>
                      {attempt.accuracy.toFixed(1)}%
                    </div>
                    <div className="text-sm text-orange-400 font-medium">{attempt.wpm.toFixed(0)} WPM</div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Mastered Message */}
        {isMastered && (
          <div className="mt-6 bg-gradient-to-r from-emerald-900/50 to-green-900/50 rounded-xl p-8 text-center border border-emerald-700">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
              <Trophy className="w-10 h-10 text-emerald-400" />
            </div>
            <h3 className="text-2xl font-bold text-emerald-400 mb-2">
              마스터 완료!
            </h3>
            <p className="text-emerald-300/80 mb-4">
              이 코드를 {requiredCount}번 성공적으로 완료했습니다!
            </p>
            <div className="flex items-center justify-center gap-6 text-sm">
              <div className="flex items-center gap-2 text-emerald-400">
                <Zap className="w-4 h-4" />
                최고 속도: {progress?.best_wpm.toFixed(0)} WPM
              </div>
              <div className="flex items-center gap-2 text-emerald-400">
                <Target className="w-4 h-4" />
                최고 정확률: {progress?.best_accuracy.toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Styles */}
      <style>{`
        @keyframes shine {
          from { transform: translateX(-100%); }
          to { transform: translateX(100%); }
        }
        .animate-shine {
          animation: shine 1.5s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
