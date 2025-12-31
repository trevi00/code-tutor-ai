import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, Clock, Box, Search, ChevronRight, Loader2 } from 'lucide-react';
import { patternsApi, type Pattern } from '@/api/patterns';

const DIFFICULTY_MAP: Record<string, { label: string; color: string }> = {
  'O(1)': { label: '상수', color: 'bg-green-100 text-green-700' },
  'O(log n)': { label: '로그', color: 'bg-green-100 text-green-700' },
  'O(n)': { label: '선형', color: 'bg-blue-100 text-blue-700' },
  'O(n log n)': { label: '선형로그', color: 'bg-yellow-100 text-yellow-700' },
  'O(n^2)': { label: '이차', color: 'bg-orange-100 text-orange-700' },
  'O(2^n)': { label: '지수', color: 'bg-red-100 text-red-700' },
};

function getComplexityInfo(complexity: string) {
  for (const [key, value] of Object.entries(DIFFICULTY_MAP)) {
    if (complexity.includes(key)) {
      return value;
    }
  }
  return { label: complexity, color: 'bg-neutral-100 text-neutral-700' };
}

export function PatternsPage() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<'all' | 'basic' | 'advanced'>('all');

  useEffect(() => {
    const fetchPatterns = async () => {
      try {
        const response = await patternsApi.list();
        setPatterns(response.patterns);
      } catch (err) {
        console.error('Error fetching patterns:', err);
        setError('패턴을 불러오는데 실패했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchPatterns();
  }, []);

  const basicPatterns = [
    'two-pointers', 'sliding-window', 'binary-search', 'bfs', 'dfs',
    'stack', 'queue', 'hash-table', 'linked-list-reversal', 'greedy'
  ];

  const filteredPatterns = patterns.filter((pattern) => {
    const matchesSearch =
      pattern.name_ko.toLowerCase().includes(search.toLowerCase()) ||
      pattern.name.toLowerCase().includes(search.toLowerCase()) ||
      pattern.keywords.some((k) => k.toLowerCase().includes(search.toLowerCase()));

    if (filter === 'basic') {
      return matchesSearch && basicPatterns.includes(pattern.id);
    }
    if (filter === 'advanced') {
      return matchesSearch && !basicPatterns.includes(pattern.id);
    }
    return matchesSearch;
  });

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-2 text-neutral-600">패턴 목록을 불러오는 중...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-20">
          <p className="text-red-500">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Sparkles className="h-8 w-8 text-purple-600" />
          알고리즘 패턴 학습
        </h1>
        <p className="text-neutral-600">
          25개의 핵심 알고리즘 패턴을 마스터하고 코딩 테스트를 정복하세요
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400" />
          <input
            type="text"
            placeholder="패턴 검색..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === 'all'
                ? 'bg-purple-600 text-white'
                : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
            }`}
          >
            전체 ({patterns.length})
          </button>
          <button
            onClick={() => setFilter('basic')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === 'basic'
                ? 'bg-green-600 text-white'
                : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
            }`}
          >
            기본 패턴
          </button>
          <button
            onClick={() => setFilter('advanced')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === 'advanced'
                ? 'bg-orange-600 text-white'
                : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
            }`}
          >
            고급 패턴
          </button>
        </div>
      </div>

      {/* Pattern Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPatterns.map((pattern) => {
          const timeInfo = getComplexityInfo(pattern.time_complexity);
          const isBasic = basicPatterns.includes(pattern.id);

          return (
            <Link
              key={pattern.id}
              to={`/patterns/${pattern.id}`}
              className="card-hover bg-white rounded-xl shadow-soft border border-neutral-100 p-6 hover:border-purple-300 group"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-bold text-neutral-900 group-hover:text-purple-600 transition-colors">
                    {pattern.name_ko}
                  </h3>
                  <p className="text-sm text-neutral-500">{pattern.name}</p>
                </div>
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                    isBasic ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'
                  }`}
                >
                  {isBasic ? '기본' : '고급'}
                </span>
              </div>

              <p className="text-neutral-600 text-sm mb-4 line-clamp-2">
                {pattern.description_ko}
              </p>

              <div className="flex items-center gap-4 text-sm text-neutral-500 mb-4">
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  <span className={`px-2 py-0.5 rounded ${timeInfo.color}`}>
                    {pattern.time_complexity}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <Box className="h-4 w-4" />
                  <span>{pattern.space_complexity}</span>
                </div>
              </div>

              <div className="flex flex-wrap gap-1">
                {pattern.keywords.slice(0, 4).map((keyword) => (
                  <span
                    key={keyword}
                    className="px-2 py-1 bg-neutral-100 text-neutral-600 text-xs rounded-full"
                  >
                    {keyword}
                  </span>
                ))}
                {pattern.keywords.length > 4 && (
                  <span className="px-2 py-1 bg-neutral-100 text-neutral-400 text-xs rounded-full">
                    +{pattern.keywords.length - 4}
                  </span>
                )}
              </div>

              <div className="mt-4 pt-4 border-t border-neutral-100 flex items-center justify-between text-sm">
                <span className="text-neutral-500">
                  {pattern.use_cases.length}개의 활용 사례
                </span>
                <ChevronRight className="h-5 w-5 text-neutral-400 group-hover:text-purple-600 transition-colors" />
              </div>
            </Link>
          );
        })}
      </div>

      {filteredPatterns.length === 0 && (
        <div className="text-center py-12">
          <p className="text-neutral-500">검색 결과가 없습니다.</p>
        </div>
      )}
    </div>
  );
}
