/**
 * Patterns Page - Enhanced with modern design
 * Shows algorithm patterns with filtering and search
 */
import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  Clock,
  Box,
  Search,
  ChevronRight,
  Loader2,
  BookMarked,
  Zap,
  Target,
  Filter,
  Grid3X3,
  List,
  TrendingUp,
  Award,
  Lightbulb,
} from 'lucide-react';
import { patternsApi, type Pattern } from '@/api/patterns';

// Complexity color mapping
const COMPLEXITY_STYLES: Record<string, { label: string; bg: string; text: string }> = {
  'O(1)': { label: '상수', bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-700 dark:text-emerald-400' },
  'O(log n)': { label: '로그', bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-400' },
  'O(n)': { label: '선형', bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400' },
  'O(n log n)': { label: '선형로그', bg: 'bg-yellow-100 dark:bg-yellow-900/30', text: 'text-yellow-700 dark:text-yellow-400' },
  'O(n^2)': { label: '이차', bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-400' },
  'O(2^n)': { label: '지수', bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400' },
};

function getComplexityInfo(complexity: string) {
  for (const [key, value] of Object.entries(COMPLEXITY_STYLES)) {
    if (complexity.includes(key)) {
      return value;
    }
  }
  return { label: complexity, bg: 'bg-gray-100 dark:bg-gray-800', text: 'text-gray-700 dark:text-gray-400' };
}

// Basic pattern IDs
const BASIC_PATTERNS = [
  'two-pointers', 'sliding-window', 'binary-search', 'bfs', 'dfs',
  'stack', 'queue', 'hash-table', 'linked-list-reversal', 'greedy'
];

// Pattern card component
interface PatternCardProps {
  pattern: Pattern;
  isBasic: boolean;
  index: number;
  viewMode: 'grid' | 'list';
}

function PatternCard({ pattern, isBasic, index, viewMode }: PatternCardProps) {
  const timeInfo = getComplexityInfo(pattern.time_complexity);

  if (viewMode === 'list') {
    return (
      <Link
        to={`/patterns/${pattern.id}`}
        className="flex items-center gap-4 p-4 bg-white dark:bg-slate-800 rounded-xl border border-gray-200 dark:border-slate-700 hover:border-purple-300 dark:hover:border-purple-600 hover:shadow-lg transition-all duration-300 group animate-fade-in"
        style={{ animationDelay: `${index * 30}ms` }}
      >
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
          isBasic
            ? 'bg-gradient-to-br from-green-400 to-emerald-500'
            : 'bg-gradient-to-br from-orange-400 to-red-500'
        }`}>
          <BookMarked className="w-6 h-6 text-white" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-bold text-gray-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
              {pattern.name_ko}
            </h3>
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
              isBasic
                ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400'
                : 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-400'
            }`}>
              {isBasic ? '기본' : '고급'}
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{pattern.name}</p>
        </div>

        <div className="hidden md:flex items-center gap-4 text-sm">
          <span className={`px-2 py-1 rounded-lg ${timeInfo.bg} ${timeInfo.text}`}>
            {pattern.time_complexity}
          </span>
          <span className="text-gray-500 dark:text-gray-400">
            {pattern.use_cases.length}개 활용사례
          </span>
        </div>

        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-purple-500 group-hover:translate-x-1 transition-all" />
      </Link>
    );
  }

  return (
    <Link
      to={`/patterns/${pattern.id}`}
      className="bg-white dark:bg-slate-800 rounded-2xl border border-gray-200 dark:border-slate-700 p-6 hover:border-purple-300 dark:hover:border-purple-600 hover:shadow-xl transition-all duration-300 group animate-fade-in"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
          isBasic
            ? 'bg-gradient-to-br from-green-400 to-emerald-500'
            : 'bg-gradient-to-br from-orange-400 to-red-500'
        }`}>
          <BookMarked className="w-6 h-6 text-white" />
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
          isBasic
            ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400'
            : 'bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-400'
        }`}>
          {isBasic ? '기본' : '고급'}
        </span>
      </div>

      {/* Title */}
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
        {pattern.name_ko}
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">{pattern.name}</p>

      {/* Description */}
      <p className="text-gray-600 dark:text-gray-400 text-sm mb-4 line-clamp-2 leading-relaxed">
        {pattern.description_ko}
      </p>

      {/* Complexity */}
      <div className="flex items-center gap-3 text-sm mb-4">
        <div className="flex items-center gap-1.5">
          <Clock className="w-4 h-4 text-gray-400" />
          <span className={`px-2 py-0.5 rounded-lg ${timeInfo.bg} ${timeInfo.text} text-xs font-medium`}>
            {pattern.time_complexity}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
          <Box className="w-4 h-4" />
          <span>{pattern.space_complexity}</span>
        </div>
      </div>

      {/* Keywords */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {pattern.keywords.slice(0, 4).map((keyword) => (
          <span
            key={keyword}
            className="px-2 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 text-xs rounded-lg"
          >
            {keyword}
          </span>
        ))}
        {pattern.keywords.length > 4 && (
          <span className="px-2 py-1 bg-gray-100 dark:bg-slate-700 text-gray-400 text-xs rounded-lg">
            +{pattern.keywords.length - 4}
          </span>
        )}
      </div>

      {/* Footer */}
      <div className="pt-4 border-t border-gray-100 dark:border-slate-700 flex items-center justify-between">
        <span className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-1.5">
          <Lightbulb className="w-4 h-4" />
          {pattern.use_cases.length}개 활용사례
        </span>
        <div className="flex items-center gap-1 text-purple-600 dark:text-purple-400 text-sm font-medium group-hover:gap-2 transition-all">
          학습하기
          <ChevronRight className="w-4 h-4" />
        </div>
      </div>
    </Link>
  );
}

export function PatternsPage() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<'all' | 'basic' | 'advanced'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

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

  const filteredPatterns = useMemo(() => {
    return patterns.filter((pattern) => {
      const matchesSearch =
        pattern.name_ko.toLowerCase().includes(search.toLowerCase()) ||
        pattern.name.toLowerCase().includes(search.toLowerCase()) ||
        pattern.keywords.some((k) => k.toLowerCase().includes(search.toLowerCase()));

      if (filter === 'basic') {
        return matchesSearch && BASIC_PATTERNS.includes(pattern.id);
      }
      if (filter === 'advanced') {
        return matchesSearch && !BASIC_PATTERNS.includes(pattern.id);
      }
      return matchesSearch;
    });
  }, [patterns, search, filter]);

  const basicCount = patterns.filter(p => BASIC_PATTERNS.includes(p.id)).length;
  const advancedCount = patterns.length - basicCount;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="text-slate-500 dark:text-slate-400 animate-pulse">패턴 목록 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800 p-6">
        <div className="max-w-2xl mx-auto pt-12">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-6 rounded-xl text-center">
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

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
          <Target className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <TrendingUp className="w-4 h-4" />
                핵심 알고리즘 학습
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Sparkles className="w-10 h-10 text-yellow-300" />
                알고리즘 패턴
              </h1>
              <p className="text-purple-100 text-lg max-w-md">
                {patterns.length}개의 핵심 패턴을 마스터하고 코딩 테스트를 정복하세요!
              </p>
            </div>

            {/* Stats */}
            <div className="flex gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Award className="w-6 h-6 text-yellow-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{patterns.length}</div>
                <div className="text-xs text-purple-200">전체 패턴</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Zap className="w-6 h-6 text-green-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{basicCount}</div>
                <div className="text-xs text-purple-200">기본 패턴</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Target className="w-6 h-6 text-orange-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{advancedCount}</div>
                <div className="text-xs text-purple-200">고급 패턴</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8 -mt-6">
        {/* Search & Filters Card */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-xl border border-gray-100 dark:border-slate-700 p-6 mb-8">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="패턴 이름, 키워드로 검색..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-gray-200 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-gray-50 dark:bg-slate-700 dark:text-white placeholder-gray-400 transition-all"
              />
            </div>

            {/* Filter buttons */}
            <div className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-gray-400 hidden sm:block" />
              <div className="flex gap-2">
                <button
                  onClick={() => setFilter('all')}
                  className={`px-4 py-2.5 rounded-xl font-medium transition-all ${
                    filter === 'all'
                      ? 'bg-gradient-to-r from-purple-500 to-violet-500 text-white shadow-lg shadow-purple-500/25'
                      : 'bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600'
                  }`}
                >
                  전체
                </button>
                <button
                  onClick={() => setFilter('basic')}
                  className={`px-4 py-2.5 rounded-xl font-medium transition-all ${
                    filter === 'basic'
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg shadow-green-500/25'
                      : 'bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600'
                  }`}
                >
                  기본
                </button>
                <button
                  onClick={() => setFilter('advanced')}
                  className={`px-4 py-2.5 rounded-xl font-medium transition-all ${
                    filter === 'advanced'
                      ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow-lg shadow-orange-500/25'
                      : 'bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600'
                  }`}
                >
                  고급
                </button>
              </div>
            </div>

            {/* View toggle */}
            <div className="flex items-center gap-1 bg-gray-100 dark:bg-slate-700 rounded-xl p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-lg transition-all ${
                  viewMode === 'grid'
                    ? 'bg-white dark:bg-slate-600 shadow text-purple-600 dark:text-purple-400'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
                }`}
              >
                <Grid3X3 className="w-5 h-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-lg transition-all ${
                  viewMode === 'list'
                    ? 'bg-white dark:bg-slate-600 shadow text-purple-600 dark:text-purple-400'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
                }`}
              >
                <List className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 pt-4 border-t border-gray-100 dark:border-slate-700 flex items-center justify-between">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              <span className="font-semibold text-gray-900 dark:text-white">{filteredPatterns.length}</span>개의 패턴
              {search && <span> - "{search}" 검색 결과</span>}
            </p>
          </div>
        </div>

        {/* Pattern Cards */}
        {filteredPatterns.length > 0 ? (
          <div className={viewMode === 'grid'
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            : "flex flex-col gap-3"
          }>
            {filteredPatterns.map((pattern, index) => (
              <PatternCard
                key={pattern.id}
                pattern={pattern}
                isBasic={BASIC_PATTERNS.includes(pattern.id)}
                index={index}
                viewMode={viewMode}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gray-100 dark:bg-slate-800 flex items-center justify-center">
              <Search className="w-10 h-10 text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              검색 결과가 없습니다
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-6">
              다른 키워드로 검색하거나 필터를 변경해보세요.
            </p>
            <button
              onClick={() => { setSearch(''); setFilter('all'); }}
              className="px-6 py-2 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-lg font-medium hover:bg-purple-200 dark:hover:bg-purple-900/50 transition-colors"
            >
              필터 초기화
            </button>
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
