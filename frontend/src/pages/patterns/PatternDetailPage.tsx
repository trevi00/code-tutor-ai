import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Clock,
  Box,
  CheckCircle,
  Copy,
  Check,
  BookOpen,
  Loader2
} from 'lucide-react';
import { patternsApi, type Pattern } from '@/api/patterns';

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
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-2 text-neutral-600">패턴 정보를 불러오는 중...</span>
        </div>
      </div>
    );
  }

  if (error || !pattern) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-20">
          <p className="text-red-500">{error || '패턴을 찾을 수 없습니다.'}</p>
          <Link
            to="/patterns"
            className="mt-4 inline-flex items-center gap-2 text-blue-600 hover:text-blue-700"
          >
            <ArrowLeft className="h-4 w-4" />
            패턴 목록으로 돌아가기
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Back Button */}
      <Link
        to="/patterns"
        className="inline-flex items-center gap-2 text-neutral-600 hover:text-purple-600 mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        패턴 목록으로
      </Link>

      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-8 mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-neutral-900 mb-2">
              {pattern.name_ko}
            </h1>
            <p className="text-lg text-neutral-500">{pattern.name}</p>
          </div>
          <div className="flex gap-4">
            <div className="text-center px-4 py-2 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-1 text-blue-600">
                <Clock className="h-4 w-4" />
                <span className="font-semibold">시간</span>
              </div>
              <p className="text-sm font-mono">{pattern.time_complexity}</p>
            </div>
            <div className="text-center px-4 py-2 bg-green-50 rounded-lg">
              <div className="flex items-center gap-1 text-green-600">
                <Box className="h-4 w-4" />
                <span className="font-semibold">공간</span>
              </div>
              <p className="text-sm font-mono">{pattern.space_complexity}</p>
            </div>
          </div>
        </div>

        <p className="mt-6 text-neutral-600 text-lg leading-relaxed">
          {pattern.description_ko}
        </p>

        <div className="mt-4 flex flex-wrap gap-2">
          {pattern.keywords.map((keyword) => (
            <span
              key={keyword}
              className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm"
            >
              {keyword}
            </span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Use Cases */}
        <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-6">
          <h2 className="text-xl font-bold text-neutral-900 mb-4 flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            언제 사용하나요?
          </h2>
          <ul className="space-y-3">
            {pattern.use_cases.map((useCase, index) => (
              <li key={index} className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-green-100 text-green-700 rounded-full flex items-center justify-center text-sm font-medium">
                  {index + 1}
                </span>
                <span className="text-neutral-600">{useCase}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Pattern Tips */}
        <div className="bg-white rounded-xl shadow-sm border border-neutral-200 p-6">
          <h2 className="text-xl font-bold text-neutral-900 mb-4 flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-blue-600" />
            핵심 포인트
          </h2>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">시간 복잡도</h3>
              <p className="text-blue-700 text-sm">
                이 패턴의 시간 복잡도는 <code className="bg-blue-100 px-1 rounded">{pattern.time_complexity}</code>입니다.
              </p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold text-green-900 mb-2">공간 복잡도</h3>
              <p className="text-green-700 text-sm">
                추가 공간은 <code className="bg-green-100 px-1 rounded">{pattern.space_complexity}</code> 필요합니다.
              </p>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <h3 className="font-semibold text-purple-900 mb-2">키워드</h3>
              <p className="text-purple-700 text-sm">
                문제에서 다음 키워드가 나오면 이 패턴을 고려해보세요: {pattern.keywords.slice(0, 3).join(', ')}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Example Code */}
      {pattern.example_code && (
        <div className="mt-6 bg-white rounded-xl shadow-sm border border-neutral-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-neutral-900">템플릿 코드</h2>
            <button
              onClick={handleCopyCode}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-neutral-100 hover:bg-neutral-200 rounded-lg transition-colors"
            >
              {copied ? (
                <>
                  <Check className="h-4 w-4 text-green-600" />
                  <span className="text-green-600">복사됨!</span>
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4" />
                  <span>코드 복사</span>
                </>
              )}
            </button>
          </div>
          <pre className="bg-neutral-900 text-neutral-100 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm font-mono">{pattern.example_code}</code>
          </pre>
        </div>
      )}

      {/* Related Problems CTA */}
      <div className="mt-6 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl p-6 text-white">
        <h2 className="text-xl font-bold mb-2">이 패턴으로 문제를 풀어보세요!</h2>
        <p className="text-purple-100 mb-4">
          {pattern.name_ko} 패턴을 활용하는 연습 문제를 풀어보세요.
        </p>
        <Link
          to={`/problems?pattern=${pattern.id}`}
          className="inline-flex items-center gap-2 px-4 py-2 bg-white text-purple-600 rounded-lg font-medium hover:bg-purple-50 transition-colors"
        >
          <BookOpen className="h-5 w-5" />
          문제 풀러 가기
        </Link>
      </div>
    </div>
  );
}
