/**
 * Typing Exercise Page
 * Main page for typing practice with real-time accuracy tracking
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Editor from '@monaco-editor/react';
import { ArrowLeft, RotateCcw, CheckCircle, Clock, Target, Zap } from 'lucide-react';
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
    // Average word length is 5 characters
    const words = charCount / 5;
    const minutes = seconds / 60;
    return Math.round(words / minutes);
  }, []);

  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      if (!exercise || !currentAttempt) return;

      const newCode = value || '';
      setUserCode(newCode);

      // Calculate accuracy
      const result = compareCode(exercise.source_code, newCode);
      setAccuracy(result.accuracy);

      // Calculate WPM
      if (startTime) {
        const seconds = (Date.now() - startTime) / 1000;
        setWpm(calculateWPM(result.correctChars, seconds));
      }

      // Check if completed
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

      // Reload progress
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

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error || !exercise) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-lg">
          {error || '연습을 찾을 수 없습니다.'}
        </div>
        <button
          onClick={() => navigate('/typing-practice')}
          className="mt-4 text-indigo-600 hover:underline"
        >
          ← 목록으로 돌아가기
        </button>
      </div>
    );
  }

  const completedCount = progress?.completed_attempts || 0;
  const requiredCount = exercise.required_completions;
  const isMastered = progress?.is_mastered || false;

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/typing-practice')}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {exercise.title}
            </h1>
            <p className="text-gray-500 dark:text-gray-400">
              {exercise.line_count}줄 · {exercise.char_count}자
            </p>
          </div>
        </div>

        {/* Progress Dots */}
        <div className="flex items-center gap-2">
          {Array.from({ length: requiredCount }).map((_, i) => (
            <div
              key={i}
              className={`w-4 h-4 rounded-full ${
                i < completedCount
                  ? 'bg-green-500'
                  : currentAttempt && i === completedCount
                  ? 'bg-indigo-500 animate-pulse'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`}
            />
          ))}
          <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
            {completedCount}/{requiredCount}
          </span>
          {isMastered && (
            <CheckCircle className="w-5 h-5 text-green-500 ml-2" />
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Source Code (Read-only) */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600">
            <h2 className="font-medium text-gray-700 dark:text-gray-300">
              원본 코드 (참고용)
            </h2>
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
              }}
            />
          </div>
        </div>

        {/* Typing Area */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-b dark:border-gray-600 flex items-center justify-between">
            <h2 className="font-medium text-gray-700 dark:text-gray-300">
              타이핑 영역
            </h2>
            {currentAttempt && (
              <button
                onClick={resetAttempt}
                className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <RotateCcw className="w-4 h-4" />
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
                }}
              />
            ) : (
              <div className="h-full flex items-center justify-center">
                <button
                  onClick={startAttempt}
                  className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium"
                >
                  {completedCount > 0 ? '다음 시도 시작' : '연습 시작하기'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      {currentAttempt && (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
                <Target className="w-4 h-4" />
                <span className="text-sm">정확률</span>
              </div>
              <p
                className={`text-2xl font-bold ${
                  accuracy >= 95
                    ? 'text-green-600'
                    : accuracy >= 80
                    ? 'text-yellow-600'
                    : 'text-red-600'
                }`}
              >
                {accuracy.toFixed(1)}%
              </p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
                <Zap className="w-4 h-4" />
                <span className="text-sm">속도</span>
              </div>
              <p className="text-2xl font-bold text-indigo-600">{wpm} WPM</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-sm">시간</span>
              </div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatTime(elapsedTime)}
              </p>
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mb-1">
              <span>진행률</span>
              <span>
                {userCode.length}/{exercise.source_code.length} 자
              </span>
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-600 transition-all duration-300"
                style={{
                  width: `${(userCode.length / exercise.source_code.length) * 100}%`,
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Previous Attempts */}
      {progress && progress.attempts.length > 0 && (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <h3 className="font-medium text-gray-700 dark:text-gray-300 mb-4">
            이전 시도 기록
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {progress.attempts
              .filter((a) => a.status === 'completed')
              .map((attempt, index) => (
                <div
                  key={attempt.id}
                  className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center"
                >
                  <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                    #{index + 1}
                  </div>
                  <div className="font-medium text-gray-900 dark:text-white">
                    {attempt.accuracy.toFixed(1)}%
                  </div>
                  <div className="text-sm text-indigo-600">{attempt.wpm.toFixed(0)} WPM</div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Mastered Message */}
      {isMastered && (
        <div className="mt-6 bg-green-50 dark:bg-green-900/20 rounded-lg p-6 text-center">
          <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
          <h3 className="text-xl font-bold text-green-700 dark:text-green-400 mb-2">
            마스터 완료!
          </h3>
          <p className="text-green-600 dark:text-green-300">
            이 코드를 {requiredCount}번 완료했습니다. 최고 속도: {progress?.best_wpm.toFixed(0)} WPM
          </p>
        </div>
      )}
    </div>
  );
}
