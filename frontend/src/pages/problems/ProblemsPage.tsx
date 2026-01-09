/**
 * Problems Page - Enhanced with modern card-based UI
 */

import { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  Search,
  ChevronRight,
  ChevronLeft,
  Loader2,
  RefreshCw,
  X,
  BookOpen,
  Code2,
  Sparkles,
  Target,
  Flame,
  ArrowRight,
  Filter,
  LayoutGrid,
  List,
  Zap,
} from 'lucide-react';
import type { Difficulty, Category, ProblemSummary } from '@/types';
import { problemsApi } from '@/api/problems';
import { patternsApi, type Pattern } from '@/api/patterns';

const DIFFICULTY_STYLES: Record<Difficulty, { bg: string; text: string; border: string; icon: string }> = {
  easy: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    text: 'text-green-700 dark:text-green-400',
    border: 'border-green-200 dark:border-green-800',
    icon: 'text-green-500',
  },
  medium: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    text: 'text-yellow-700 dark:text-yellow-400',
    border: 'border-yellow-200 dark:border-yellow-800',
    icon: 'text-yellow-500',
  },
  hard: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    text: 'text-red-700 dark:text-red-400',
    border: 'border-red-200 dark:border-red-800',
    icon: 'text-red-500',
  },
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
  tree: '트리',
  graph: '그래프',
  dp: '동적 프로그래밍',
  greedy: '그리디',
  binary_search: '이진 탐색',
  sorting: '정렬',
};

const CATEGORY_ICONS: Partial<Record<Category | string, string>> = {
  array: '📊',
  string: '📝',
  hash_table: '🗂️',
  linked_list: '🔗',
  stack: '📚',
  queue: '📋',
  tree: '🌳',
  graph: '🕸️',
  dp: '🧮',
  greedy: '💰',
  binary_search: '🔍',
  sorting: '📈',
  math: '➗',
  simulation: '🎮',
  backtracking: '🔙',
};

export function ProblemsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialPattern = searchParams.get('pattern') || '';
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [difficultyFilter, setDifficultyFilter] = useState<Difficulty | ''>('');
  const [categoryFilter, setCategoryFilter] = useState<Category | ''>('');
  const [patternFilter, setPatternFilter] = useState<string>(initialPattern);
  const [patternInfo, setPatternInfo] = useState<Pattern | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalProblems, setTotalProblems] = useState(0);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  useEffect(() => {
    const patternParam = searchParams.get('pattern') || '';
    setPatternFilter(patternParam);
    if (patternParam) {
      patternsApi.get(patternParam).then(setPatternInfo).catch(console.error);
    } else {
      setPatternInfo(null);
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
      setTotalProblems(response.total);
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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl" />
          <Code2 className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <BookOpen className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Sparkles className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-7xl mx-auto px-6 py-10 relative">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div className="text-white">
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-white/90 text-sm mb-3">
                <BookOpen className="w-4 h-4" />
                알고리즘 학습
              </div>
              <h1 className="text-2xl md:text-3xl font-bold mb-2 flex items-center gap-3">
                <Code2 className="w-8 h-8" />
                문제 목록
              </h1>
              <p className="text-teal-100 text-lg">
                AI 힌트와 함께 알고리즘 문제를 풀어보세요
              </p>
            </div>

            {/* Stats */}
            <div className="flex gap-3 flex-wrap">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px]">
                <BookOpen className="w-5 h-5 text-teal-200 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">{totalProblems}</div>
                <div className="text-xs text-teal-200">전체 문제</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px]">
                <Target className="w-5 h-5 text-green-300 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">
                  {problems.filter(p => p.difficulty === 'easy').length}
                </div>
                <div className="text-xs text-teal-200">Easy</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px]">
                <Flame className="w-5 h-5 text-yellow-300 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">
                  {problems.filter(p => p.difficulty === 'medium').length}
                </div>
                <div className="text-xs text-teal-200">Medium</div>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-xl px-4 py-3 text-center min-w-[90px] border border-white/30">
                <Zap className="w-5 h-5 text-red-300 mx-auto mb-1" />
                <div className="text-xl font-bold text-white">
                  {problems.filter(p => p.difficulty === 'hard').length}
                </div>
                <div className="text-xs text-teal-200">Hard</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 -mt-6">
        {/* Active Pattern Filter Banner */}
        {patternInfo && (
          <div className="mb-6 bg-purple-50 dark:bg-purple-900/30 border border-purple-200 dark:border-purple-800 rounded-2xl shadow-lg p-4 flex items-center justify-between animate-fade-in">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-purple-100 dark:bg-purple-800/50 flex items-center justify-center">
                <Filter className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <span className="text-purple-600 dark:text-purple-400 font-medium">패턴 필터:</span>
                <span className="ml-2 px-3 py-1 bg-purple-100 dark:bg-purple-800/50 text-purple-700 dark:text-purple-300 rounded-full text-sm font-medium">
                  {patternInfo.name_ko}
                </span>
              </div>
              <span className="text-slate-500 dark:text-slate-400 text-sm hidden md:inline">
                이 패턴과 관련된 문제만 표시됩니다
              </span>
            </div>
            <button
              onClick={clearPatternFilter}
              className="flex items-center gap-1 px-3 py-2 text-sm text-purple-600 dark:text-purple-400 hover:text-purple-700 hover:bg-purple-100 dark:hover:bg-purple-800/50 rounded-lg transition-colors"
            >
              <X className="h-4 w-4" />
              필터 해제
            </button>
          </div>
        )}

        {/* Filters */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-4 mb-6 animate-fade-in">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
              <input
                type="text"
                placeholder="문제 검색..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-slate-200 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent bg-slate-50 dark:bg-slate-700 dark:text-white dark:placeholder-slate-400 transition-all"
              />
            </div>

            {/* Filters */}
            <div className="flex flex-wrap gap-3">
              <select
                value={difficultyFilter}
                onChange={(e) => setDifficultyFilter(e.target.value as Difficulty | '')}
                className="px-4 py-3 border border-slate-200 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-teal-500 appearance-none bg-slate-50 dark:bg-slate-700 dark:text-white cursor-pointer min-w-[140px]"
              >
                <option value="">전체 난이도</option>
                <option value="easy">쉬움</option>
                <option value="medium">보통</option>
                <option value="hard">어려움</option>
              </select>

              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value as Category | '')}
                className="px-4 py-3 border border-slate-200 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-teal-500 appearance-none bg-slate-50 dark:bg-slate-700 dark:text-white cursor-pointer min-w-[160px]"
              >
                <option value="">전체 카테고리</option>
                {Object.entries(CATEGORY_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>

              {/* View Mode Toggle */}
              <div className="flex bg-slate-100 dark:bg-slate-700 rounded-xl p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-lg transition-all ${
                    viewMode === 'grid'
                      ? 'bg-white dark:bg-slate-600 shadow-sm text-teal-600 dark:text-teal-400'
                      : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'
                  }`}
                >
                  <LayoutGrid className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-lg transition-all ${
                    viewMode === 'list'
                      ? 'bg-white dark:bg-slate-600 shadow-sm text-teal-600 dark:text-teal-400'
                      : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'
                  }`}
                >
                  <List className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="py-16 text-center">
            <div className="relative inline-block">
              <div className="w-16 h-16 rounded-full bg-gradient-to-r from-teal-500 to-cyan-500 animate-pulse" />
              <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
            </div>
            <p className="mt-4 text-slate-500 dark:text-slate-400">문제를 불러오는 중...</p>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="py-16 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <RefreshCw className="w-8 h-8 text-red-500" />
            </div>
            <p className="text-red-500 dark:text-red-400 mb-4">{error}</p>
            <button
              onClick={fetchProblems}
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-teal-500 to-cyan-500 text-white rounded-xl hover:from-teal-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-xl font-medium"
            >
              <RefreshCw className="h-4 w-4" />
              다시 시도
            </button>
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && problems.length === 0 && (
          <div className="py-16 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center">
              <Search className="w-8 h-8 text-slate-400" />
            </div>
            <p className="text-slate-500 dark:text-slate-400 font-medium">필터에 맞는 문제가 없습니다</p>
            <p className="text-sm text-slate-400 dark:text-slate-500 mt-1">다른 필터를 시도해보세요</p>
          </div>
        )}

        {/* Problem Grid */}
        {!loading && !error && problems.length > 0 && viewMode === 'grid' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-8">
            {problems.map((problem, idx) => (
              <ProblemCard key={problem.id} problem={problem} delay={idx} />
            ))}
          </div>
        )}

        {/* Problem List */}
        {!loading && !error && problems.length > 0 && viewMode === 'list' && (
          <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 overflow-hidden mb-8 animate-fade-in">
            <div className="divide-y divide-slate-100 dark:divide-slate-700">
              {problems.map((problem, idx) => (
                <ProblemListItem key={problem.id} problem={problem} delay={idx} />
              ))}
            </div>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && !loading && (
          <div className="flex justify-center items-center gap-3 animate-fade-in">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="flex items-center gap-2 px-5 py-2.5 border border-slate-200 dark:border-slate-600 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 dark:hover:bg-slate-700 bg-white dark:bg-slate-800 dark:text-white transition-all font-medium"
            >
              <ChevronLeft className="w-4 h-4" />
              이전
            </button>

            <div className="flex items-center gap-2">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (page <= 3) {
                  pageNum = i + 1;
                } else if (page >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = page - 2 + i;
                }
                return (
                  <button
                    key={pageNum}
                    onClick={() => setPage(pageNum)}
                    className={`w-10 h-10 rounded-xl font-medium transition-all ${
                      page === pageNum
                        ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg'
                        : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700'
                    }`}
                  >
                    {pageNum}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="flex items-center gap-2 px-5 py-2.5 border border-slate-200 dark:border-slate-600 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 dark:hover:bg-slate-700 bg-white dark:bg-slate-800 dark:text-white transition-all font-medium"
            >
              다음
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Styles */}
      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
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

// Problem Card Component
interface ProblemCardProps {
  problem: ProblemSummary;
  delay: number;
}

function ProblemCard({ problem, delay }: ProblemCardProps) {
  const styles = DIFFICULTY_STYLES[problem.difficulty];
  const categoryIcon = CATEGORY_ICONS[problem.category] || '📌';

  return (
    <Link
      to={`/problems/${problem.id}/solve`}
      className="group bg-white dark:bg-slate-800 rounded-2xl shadow-lg border border-slate-100 dark:border-slate-700 p-5 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 animate-fade-in block"
      style={{ animationDelay: `${delay * 50}ms` }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`px-3 py-1 rounded-lg text-sm font-medium flex items-center gap-1.5 ${styles.bg} ${styles.text}`}>
          <span className={styles.icon}>
            {problem.difficulty === 'easy' && <Target className="w-3.5 h-3.5" />}
            {problem.difficulty === 'medium' && <Flame className="w-3.5 h-3.5" />}
            {problem.difficulty === 'hard' && <Zap className="w-3.5 h-3.5" />}
          </span>
          {DIFFICULTY_LABELS[problem.difficulty]}
        </div>
        <ArrowRight className="w-5 h-5 text-slate-300 dark:text-slate-600 group-hover:text-teal-500 group-hover:translate-x-1 transition-all" />
      </div>

      <h3 className="font-semibold text-slate-800 dark:text-white mb-2 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors line-clamp-2">
        {problem.title}
      </h3>

      <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
        <span>{categoryIcon}</span>
        <span>{CATEGORY_LABELS[problem.category] || problem.category}</span>
      </div>
    </Link>
  );
}

// Problem List Item Component
interface ProblemListItemProps {
  problem: ProblemSummary;
  delay: number;
}

function ProblemListItem({ problem, delay }: ProblemListItemProps) {
  const styles = DIFFICULTY_STYLES[problem.difficulty];
  const categoryIcon = CATEGORY_ICONS[problem.category] || '📌';

  return (
    <Link
      to={`/problems/${problem.id}/solve`}
      className="group flex items-center gap-4 p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-all animate-fade-in"
      style={{ animationDelay: `${delay * 30}ms` }}
    >
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-2xl ${styles.bg}`}>
        {categoryIcon}
      </div>

      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-slate-800 dark:text-white group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors truncate">
          {problem.title}
        </h3>
        <div className="flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400">
          <span>{CATEGORY_LABELS[problem.category] || problem.category}</span>
        </div>
      </div>

      <div className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 ${styles.bg} ${styles.text}`}>
        {problem.difficulty === 'easy' && <Target className="w-3.5 h-3.5" />}
        {problem.difficulty === 'medium' && <Flame className="w-3.5 h-3.5" />}
        {problem.difficulty === 'hard' && <Zap className="w-3.5 h-3.5" />}
        {DIFFICULTY_LABELS[problem.difficulty]}
      </div>

      <ChevronRight className="w-5 h-5 text-slate-300 dark:text-slate-600 group-hover:text-teal-500 group-hover:translate-x-1 transition-all flex-shrink-0" />
    </Link>
  );
}
