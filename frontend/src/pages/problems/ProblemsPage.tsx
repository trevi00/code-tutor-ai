import { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Search, Filter, ChevronRight, Loader2, RefreshCw, X } from 'lucide-react';
import type { Difficulty, Category, ProblemSummary } from '@/types';
import { problemsApi } from '@/api/problems';
import { patternsApi, type Pattern } from '@/api/patterns';


const DIFFICULTY_COLORS: Record<Difficulty, string> = {
  easy: 'bg-green-100 text-green-700',
  medium: 'bg-yellow-100 text-yellow-700',
  hard: 'bg-red-100 text-red-700',
};

const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  easy: '쉬움',
  medium: '보통',
  hard: '어려움',
};

const CATEGORY_LABELS: Partial<Record<Category, string>> = {
  array: '배열',
  string: '문자열',
  hash_table: '해시 테이블',
  linked_list: '연결 리스트',
  stack: '스택',
  queue: '큐',
  heap: '힙',
  tree: '트리',
  graph: '그래프',
  dp: '동적 프로그래밍',
  greedy: '그리디',
  binary_search: '이진 탐색',
  sorting: '정렬',
  backtracking: '백트래킹',
};

export function ProblemsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [difficultyFilter, setDifficultyFilter] = useState<Difficulty | ''>('');
  const [categoryFilter, setCategoryFilter] = useState<Category | ''>('');
  const [patternFilter, setPatternFilter] = useState<string>('');
  const [patternInfo, setPatternInfo] = useState<Pattern | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  // Read pattern from URL on mount
  useEffect(() => {
    const patternParam = searchParams.get('pattern');
    if (patternParam) {
      setPatternFilter(patternParam);
      // Fetch pattern info for display
      patternsApi.get(patternParam).then(setPatternInfo).catch(console.error);
    }
  }, [searchParams]);

  const clearPatternFilter = () => {
    setPatternFilter('');
    setPatternInfo(null);
    setSearchParams({});
    setPage(1);
  };

  const fetchProblems = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await problemsApi.list({
        page,
        size: 20,
        difficulty: difficultyFilter || undefined,
        category: categoryFilter || undefined,
        pattern: patternFilter || undefined,
        search: search || undefined,
      });
      setProblems(response.items);
      setTotalPages(response.pages);
    } catch (err) {
      setError('문제를 불러오는데 실패했습니다. 다시 시도해주세요.');
      console.error('Error fetching problems:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProblems();
  }, [page, difficultyFilter, categoryFilter, patternFilter]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setPage(1);
      fetchProblems();
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">문제 목록</h1>
        <p className="text-neutral-600">AI 힌트와 함께 알고리즘 문제를 풀어보세요</p>
      </div>

      {/* Active Pattern Filter Banner */}
      {patternInfo && (
        <div className="mb-6 bg-purple-50 border border-purple-200 rounded-lg p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-purple-600 font-medium">패턴 필터:</span>
            <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">
              {patternInfo.name_ko}
            </span>
            <span className="text-neutral-500 text-sm">
              이 패턴과 관련된 문제만 표시됩니다
            </span>
          </div>
          <button
            onClick={clearPatternFilter}
            className="flex items-center gap-1 px-3 py-1 text-sm text-purple-600 hover:text-purple-700 hover:bg-purple-100 rounded-lg transition-colors"
          >
            <X className="h-4 w-4" />
            필터 해제
          </button>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400" />
          <input
            type="text"
            placeholder="문제 검색..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div className="flex gap-4">
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400" />
            <select
              value={difficultyFilter}
              onChange={(e) => setDifficultyFilter(e.target.value as Difficulty | '')}
              className="pl-10 pr-8 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-blue-500 appearance-none bg-white"
            >
              <option value="">전체 난이도</option>
              <option value="easy">쉬움</option>
              <option value="medium">보통</option>
              <option value="hard">어려움</option>
            </select>
          </div>

          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value as Category | '')}
            className="px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-blue-500 appearance-none bg-white"
          >
            <option value="">전체 카테고리</option>
            {Object.entries(CATEGORY_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Problem List */}
      <div className="bg-white rounded-xl shadow-sm border border-neutral-200 overflow-hidden">
        <table className="w-full">
          <thead className="bg-neutral-50 border-b border-neutral-200">
            <tr>
              <th className="text-left py-4 px-6 font-medium text-neutral-600">제목</th>
              <th className="text-left py-4 px-4 font-medium text-neutral-600">난이도</th>
              <th className="text-left py-4 px-4 font-medium text-neutral-600">카테고리</th>
              <th className="w-12"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-200">
            {problems.map((problem) => (
              <tr key={problem.id} className="hover:bg-neutral-50 transition-colors">
                <td className="py-4 px-6">
                  <Link
                    to={`/problems/${problem.id}/solve`}
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    {problem.title}
                  </Link>
                </td>
                <td className="py-4 px-4">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      DIFFICULTY_COLORS[problem.difficulty]
                    }`}
                  >
                    {DIFFICULTY_LABELS[problem.difficulty]}
                  </span>
                </td>
                <td className="py-4 px-4 text-neutral-600">
                  {CATEGORY_LABELS[problem.category] || problem.category}
                </td>
                <td className="py-4 px-4">
                  <Link
                    to={`/problems/${problem.id}/solve`}
                    className="text-neutral-400 hover:text-blue-600"
                  >
                    <ChevronRight className="h-5 w-5" />
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {loading && (
          <div className="py-12 text-center">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto" />
            <p className="mt-2 text-neutral-500">문제를 불러오는 중...</p>
          </div>
        )}

        {error && !loading && (
          <div className="py-12 text-center">
            <p className="text-red-500 mb-4">{error}</p>
            <button
              onClick={fetchProblems}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              <RefreshCw className="h-4 w-4" />
              다시 시도
            </button>
          </div>
        )}

        {!loading && !error && problems.length === 0 && (
          <div className="py-12 text-center text-neutral-500">
            필터에 맞는 문제가 없습니다.
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center gap-2 mt-6">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-4 py-2 border border-neutral-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-neutral-50"
          >
            이전
          </button>
          <span className="px-4 py-2 text-neutral-600">
            {page} / {totalPages} 페이지
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-4 py-2 border border-neutral-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-neutral-50"
          >
            다음
          </button>
        </div>
      )}
    </div>
  );
}
