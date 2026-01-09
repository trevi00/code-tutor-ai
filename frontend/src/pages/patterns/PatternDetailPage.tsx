/**
 * Pattern Detail Page - Enhanced with modern design
 * Shows detailed information about a specific algorithm pattern
 */
import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Clock,
  Box,
  Copy,
  Check,
  BookOpen,
  Loader2,
  Sparkles,
  Target,
  Lightbulb,
  Code,
  Zap,
  ChevronRight,
  BookMarked,
  Play,
  Tag,
} from 'lucide-react';
import { patternsApi, type Pattern } from '@/api/patterns';

// Complexity color styles
const COMPLEXITY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'O(1)': { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-700 dark:text-emerald-400', border: 'border-emerald-200 dark:border-emerald-800' },
  'O(log n)': { bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-400', border: 'border-green-200 dark:border-green-800' },
  'O(n)': { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400', border: 'border-blue-200 dark:border-blue-800' },
  'O(n log n)': { bg: 'bg-yellow-100 dark:bg-yellow-900/30', text: 'text-yellow-700 dark:text-yellow-400', border: 'border-yellow-200 dark:border-yellow-800' },
  'O(n^2)': { bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-400', border: 'border-orange-200 dark:border-orange-800' },
  'O(2^n)': { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400', border: 'border-red-200 dark:border-red-800' },
};

function getComplexityColor(complexity: string) {
  for (const [key, value] of Object.entries(COMPLEXITY_COLORS)) {
    if (complexity.includes(key)) {
      return value;
    }
  }
  return { bg: 'bg-gray-100 dark:bg-gray-800', text: 'text-gray-700 dark:text-gray-400', border: 'border-gray-200 dark:border-gray-700' };
}

export function PatternDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [pattern, setPattern] = useState<Pattern | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const fetchPattern = async () => {
      if (!id) return;

      try {
        const data = await patternsApi.get(id);
        setPattern(data);
      } catch (err) {
        console.error('Error fetching pattern:', err);
        setError('패턴을 불러오는데 실패했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchPattern();
  }, [id]);

  const handleCopyCode = async () => {
    if (!pattern?.example_code) return;

    try {
      await navigator.clipboard.writeText(pattern.example_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">패턴 정보 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error || !pattern) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 p-6">
        <div className="max-w-2xl mx-auto pt-12">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-6 rounded-xl text-center">
            <p>{error || '패턴을 찾을 수 없습니다.'}</p>
          </div>
          <Link
            to="/patterns"
            className="mt-6 inline-flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:underline font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            패턴 목록으로 돌아가기
          </Link>
        </div>
      </div>
    );
  }

  const timeColor = getComplexityColor(pattern.time_complexity);
  const spaceColor = getComplexityColor(pattern.space_complexity);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-600 via-violet-600 to-indigo-600 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Sparkles className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <BookMarked className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Code className="absolute top-20 left-[25%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-5xl mx-auto px-6 py-10 relative">
          {/* Back Link */}
          <Link
            to="/patterns"
            className="inline-flex items-center gap-2 text-white/80 hover:text-white mb-6 transition-colors group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            패턴 목록으로
          </Link>

          <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
            <div className="flex items-start gap-4">
              {/* Pattern Icon */}
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center text-white shrink-0">
                <BookMarked className="w-8 h-8" />
              </div>

              <div>
                {/* Badge */}
                <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-white/90 text-sm mb-3">
                  <Sparkles className="w-4 h-4" />
                  알고리즘 패턴
                </div>

                {/* Title */}
                <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">
                  {pattern.name_ko}
                </h1>
                <p className="text-lg text-purple-200">{pattern.name}</p>
              </div>
            </div>

            {/* Complexity Cards */}
            <div className="flex gap-3">
              <div className="bg-white/15 backdrop-blur-sm rounded-xl p-4 text-center min-w-[110px]">
                <Clock className="w-5 h-5 text-white/80 mx-auto mb-1" />
                <div className="text-xs text-purple-200 mb-1">시간 복잡도</div>
                <div className="text-sm font-bold text-white font-mono">{pattern.time_complexity}</div>
              </div>
              <div className="bg-white/15 backdrop-blur-sm rounded-xl p-4 text-center min-w-[110px]">
                <Box className="w-5 h-5 text-white/80 mx-auto mb-1" />
                <div className="text-xs text-purple-200 mb-1">공간 복잡도</div>
                <div className="text-sm font-bold text-white font-mono">{pattern.space_complexity}</div>
              </div>
            </div>
          </div>

          {/* Description */}
          <p className="mt-6 text-purple-100 text-lg leading-relaxed max-w-3xl">
            {pattern.description_ko}
          </p>

          {/* Keywords */}
          <div className="mt-6 flex flex-wrap gap-2">
            {pattern.keywords.map((keyword) => (
              <span
                key={keyword}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/20 backdrop-blur-sm text-white rounded-lg text-sm"
              >
                <Tag className="w-3 h-3" />
                {keyword}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto px-6 py-8 -mt-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Use Cases */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-gray-100 dark:border-slate-700 p-6 animate-fade-in">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-5 flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <Target className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              언제 사용하나요?
            </h2>
            <ul className="space-y-3">
              {pattern.use_cases.map((useCase, index) => (
                <li
                  key={index}
                  className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-xl"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <span className="flex-shrink-0 w-7 h-7 bg-green-500 text-white rounded-lg flex items-center justify-center text-sm font-bold">
                    {index + 1}
                  </span>
                  <span className="text-gray-700 dark:text-gray-300 leading-relaxed">{useCase}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Key Points */}
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-gray-100 dark:border-slate-700 p-6 animate-fade-in" style={{ animationDelay: '100ms' }}>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-5 flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
                <Lightbulb className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              </div>
              핵심 포인트
            </h2>
            <div className="space-y-4">
              {/* Time Complexity */}
              <div className={`p-4 rounded-xl border ${timeColor.bg} ${timeColor.border}`}>
                <div className="flex items-center gap-2 mb-2">
                  <Clock className={`w-4 h-4 ${timeColor.text}`} />
                  <h3 className={`font-semibold ${timeColor.text}`}>시간 복잡도</h3>
                </div>
                <p className={`text-sm ${timeColor.text}`}>
                  이 패턴의 시간 복잡도는 <code className="px-1.5 py-0.5 bg-white/50 dark:bg-black/20 rounded font-mono font-bold">{pattern.time_complexity}</code>입니다.
                </p>
              </div>

              {/* Space Complexity */}
              <div className={`p-4 rounded-xl border ${spaceColor.bg} ${spaceColor.border}`}>
                <div className="flex items-center gap-2 mb-2">
                  <Box className={`w-4 h-4 ${spaceColor.text}`} />
                  <h3 className={`font-semibold ${spaceColor.text}`}>공간 복잡도</h3>
                </div>
                <p className={`text-sm ${spaceColor.text}`}>
                  추가 공간은 <code className="px-1.5 py-0.5 bg-white/50 dark:bg-black/20 rounded font-mono font-bold">{pattern.space_complexity}</code> 필요합니다.
                </p>
              </div>

              {/* Keywords Hint */}
              <div className="p-4 rounded-xl bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <h3 className="font-semibold text-purple-700 dark:text-purple-400">키워드 힌트</h3>
                </div>
                <p className="text-sm text-purple-700 dark:text-purple-400">
                  문제에서 <span className="font-semibold">{pattern.keywords.slice(0, 3).join(', ')}</span> 키워드가 나오면 이 패턴을 고려해보세요.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Example Code */}
        {pattern.example_code && (
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-gray-100 dark:border-slate-700 overflow-hidden mb-8 animate-fade-in" style={{ animationDelay: '200ms' }}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-slate-700 bg-gray-50 dark:bg-slate-900/50">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-purple-500" />
                템플릿 코드
              </h2>
              <button
                onClick={handleCopyCode}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                  copied
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                    : 'bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600'
                }`}
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4" />
                    <span>복사됨!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    <span>코드 복사</span>
                  </>
                )}
              </button>
            </div>
            <div className="p-6 bg-slate-900 dark:bg-slate-950">
              <pre className="overflow-x-auto">
                <code className="text-sm font-mono text-gray-100 leading-relaxed">{pattern.example_code}</code>
              </pre>
            </div>
          </div>
        )}

        {/* CTA - Practice Problems */}
        <div className="bg-gradient-to-r from-purple-600 via-violet-600 to-indigo-600 rounded-2xl shadow-xl shadow-purple-500/20 p-8 text-white relative overflow-hidden animate-fade-in" style={{ animationDelay: '300ms' }}>
          {/* Background decoration */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute -top-12 -right-12 w-48 h-48 bg-white/10 rounded-full blur-2xl" />
            <div className="absolute -bottom-12 -left-12 w-48 h-48 bg-pink-500/20 rounded-full blur-2xl" />
          </div>

          <div className="relative flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-sm mb-3">
                <Play className="w-4 h-4" />
                실전 연습
              </div>
              <h2 className="text-2xl font-bold mb-2">이 패턴으로 문제를 풀어보세요!</h2>
              <p className="text-purple-100">
                {pattern.name_ko} 패턴을 활용하는 연습 문제로 실력을 향상시켜보세요.
              </p>
            </div>

            <Link
              to={`/problems?pattern=${pattern.id}`}
              className="inline-flex items-center gap-3 px-6 py-4 bg-white text-purple-600 rounded-xl font-bold hover:bg-purple-50 transition-all duration-200 transform hover:scale-105 hover:shadow-lg shrink-0"
            >
              <BookOpen className="w-5 h-5" />
              문제 풀러 가기
              <ChevronRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
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
      `}</style>
    </div>
  );
}
