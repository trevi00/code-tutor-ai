/**
 * Playground List Page - Enhanced with modern design
 */

import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Code2,
  Plus,
  Search,
  Filter,
  Loader2,
  Play,
  GitFork,
  Globe,
  Lock,
  Flame,
  Sparkles,
  Terminal,
  FileCode,
  X,
} from 'lucide-react';
import {
  createPlayground,
  getLanguages,
  listMyPlaygrounds,
  listPopularPlaygrounds,
  listPublicPlaygrounds,
  searchPlaygrounds,
} from '../../api/playground';
import type {
  LanguageInfo,
  PlaygroundResponse,
} from '../../api/playground';
import { useAuthStore } from '../../store/authStore';

type TabType = 'my' | 'public' | 'popular';

// Language icons and colors
const LANGUAGE_STYLES: Record<string, { icon: string; bg: string; text: string }> = {
  python: { icon: '🐍', bg: 'bg-yellow-100 dark:bg-yellow-900/30', text: 'text-yellow-700 dark:text-yellow-400' },
  javascript: { icon: '⚡', bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400' },
  typescript: { icon: '📘', bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400' },
  java: { icon: '☕', bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-400' },
  cpp: { icon: '⚙️', bg: 'bg-purple-100 dark:bg-purple-900/30', text: 'text-purple-700 dark:text-purple-400' },
  c: { icon: '🔧', bg: 'bg-slate-100 dark:bg-slate-700/50', text: 'text-slate-700 dark:text-slate-300' },
  go: { icon: '🐹', bg: 'bg-cyan-100 dark:bg-cyan-900/30', text: 'text-cyan-700 dark:text-cyan-400' },
  rust: { icon: '🦀', bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-400' },
  ruby: { icon: '💎', bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400' },
};

function getLanguageStyle(langId: string) {
  return LANGUAGE_STYLES[langId] || {
    icon: '📄',
    bg: 'bg-gray-100 dark:bg-gray-700/50',
    text: 'text-gray-700 dark:text-gray-300'
  };
}

interface PlaygroundCardProps {
  playground: PlaygroundResponse;
  languageDisplayName: string;
  onClick: () => void;
  index: number;
}

function PlaygroundCard({ playground, languageDisplayName, onClick, index }: PlaygroundCardProps) {
  const langStyle = getLanguageStyle(playground.language);

  return (
    <div
      onClick={onClick}
      className="group bg-white dark:bg-slate-800 rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer border border-gray-200 dark:border-slate-700 overflow-hidden hover:-translate-y-1"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Card Header with gradient */}
      <div className={`h-2 bg-gradient-to-r ${
        playground.is_forked
          ? 'from-purple-500 to-pink-500'
          : 'from-emerald-500 to-teal-500'
      }`} />

      <div className="p-5">
        <div className="flex justify-between items-start mb-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className={`w-10 h-10 rounded-lg ${langStyle.bg} flex items-center justify-center flex-shrink-0`}>
              <span className="text-lg">{langStyle.icon}</span>
            </div>
            <div className="min-w-0">
              <h3 className="font-semibold text-gray-900 dark:text-white truncate group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                {playground.title}
              </h3>
              <span className={`text-xs font-medium ${langStyle.text}`}>
                {languageDisplayName}
              </span>
            </div>
          </div>

          {playground.is_forked && (
            <span className="flex items-center gap-1 px-2 py-0.5 text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded-full">
              <GitFork className="w-3 h-3" />
              Forked
            </span>
          )}
        </div>

        {playground.description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-4">
            {playground.description}
          </p>
        )}

        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <Play className="w-3.5 h-3.5" />
            {playground.run_count} runs
          </span>
          <span className="flex items-center gap-1">
            <GitFork className="w-3.5 h-3.5" />
            {playground.fork_count} forks
          </span>
        </div>
      </div>
    </div>
  );
}

export default function PlaygroundListPage() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();

  const [activeTab, setActiveTab] = useState<TabType>(
    isAuthenticated ? 'my' : 'public'
  );
  const [playgrounds, setPlaygrounds] = useState<PlaygroundResponse[]>([]);
  const [languages, setLanguages] = useState<LanguageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('');

  // Create modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newLanguage, setNewLanguage] = useState('python');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    loadLanguages();
    loadPlaygrounds();
  }, [activeTab, selectedLanguage]);

  const loadLanguages = async () => {
    try {
      const data = await getLanguages();
      setLanguages(data.languages);
    } catch (error) {
      console.error('Failed to load languages:', error);
    }
  };

  const loadPlaygrounds = async () => {
    try {
      setLoading(true);
      let data;

      if (searchQuery) {
        data = await searchPlaygrounds(searchQuery, selectedLanguage || undefined);
      } else if (activeTab === 'my') {
        data = await listMyPlaygrounds();
      } else if (activeTab === 'popular') {
        data = await listPopularPlaygrounds();
      } else {
        data = await listPublicPlaygrounds(selectedLanguage || undefined);
      }

      setPlaygrounds(data.playgrounds);
    } catch (error) {
      console.error('Failed to load playgrounds:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    loadPlaygrounds();
  };

  const handleCreatePlayground = async () => {
    if (!newTitle.trim()) return;

    try {
      setCreating(true);
      const playground = await createPlayground({
        title: newTitle,
        language: newLanguage,
      });
      navigate(`/playground/${playground.id}`);
    } catch (error) {
      console.error('Failed to create playground:', error);
    } finally {
      setCreating(false);
    }
  };

  const getLanguageDisplayName = (langId: string) => {
    const lang = languages.find((l) => l.id === langId);
    return lang?.display_name || langId;
  };

  // Stats
  const stats = useMemo(() => ({
    total: playgrounds.length,
    totalRuns: playgrounds.reduce((sum, p) => sum + p.run_count, 0),
    totalForks: playgrounds.reduce((sum, p) => sum + p.fork_count, 0),
  }), [playgrounds]);

  const tabs = [
    ...(isAuthenticated ? [{ id: 'my' as const, label: '내 플레이그라운드', icon: Lock }] : []),
    { id: 'public' as const, label: '공개', icon: Globe },
    { id: 'popular' as const, label: '인기', icon: Flame },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Terminal className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <Code2 className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <FileCode className="absolute top-20 left-[25%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Sparkles className="w-4 h-4" />
                실시간 코드 실행
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Code2 className="w-10 h-10" />
                Code Playground
              </h1>
              <p className="text-emerald-100 text-lg max-w-md">
                다양한 언어로 코드를 작성하고 실행해보세요. 저장하고 공유할 수 있습니다!
              </p>
            </div>

            {/* Stats Cards */}
            <div className="flex gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <FileCode className="w-6 h-6 text-emerald-200 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{stats.total}</div>
                <div className="text-xs text-emerald-200">플레이그라운드</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Play className="w-6 h-6 text-emerald-200 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{stats.totalRuns}</div>
                <div className="text-xs text-emerald-200">실행 횟수</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8 -mt-6">
        {/* Action Bar */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-4 mb-6 border border-gray-100 dark:border-slate-700">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Tabs */}
            <div className="flex gap-2">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                      activeTab === tab.id
                        ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
                        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-700'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* Search and Filter */}
            <div className="flex-1 flex gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="플레이그라운드 검색..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-200 dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-700 text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all"
                />
              </div>

              <div className="relative">
                <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <select
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  className="pl-10 pr-8 py-2 border border-gray-200 dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-emerald-500 appearance-none cursor-pointer"
                >
                  <option value="">모든 언어</option>
                  {languages.map((lang) => (
                    <option key={lang.id} value={lang.id}>
                      {lang.display_name}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleSearch}
                className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors flex items-center gap-2"
              >
                <Search className="w-4 h-4" />
                검색
              </button>
            </div>

            {/* Create Button */}
            {isAuthenticated && (
              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-lg hover:from-emerald-700 hover:to-teal-700 transition-all shadow-md hover:shadow-lg"
              >
                <Plus className="w-5 h-5" />
                새로 만들기
              </button>
            )}
          </div>
        </div>

        {/* Playground List */}
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="relative">
              <div className="w-16 h-16 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 animate-pulse" />
              <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
            </div>
            <p className="mt-4 text-gray-500 dark:text-gray-400">플레이그라운드 불러오는 중...</p>
          </div>
        ) : playgrounds.length === 0 ? (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-12 text-center border border-gray-100 dark:border-slate-700">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gray-100 dark:bg-slate-700 flex items-center justify-center">
              <Code2 className="w-10 h-10 text-gray-400 dark:text-gray-500" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              플레이그라운드가 없습니다
            </h3>
            <p className="text-gray-500 dark:text-gray-400 mb-6">
              {activeTab === 'my'
                ? '첫 번째 플레이그라운드를 만들어보세요!'
                : '검색 조건에 맞는 플레이그라운드가 없습니다.'}
            </p>
            {isAuthenticated && activeTab === 'my' && (
              <button
                onClick={() => setShowCreateModal(true)}
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-lg hover:from-emerald-700 hover:to-teal-700 transition-all shadow-md"
              >
                <Plus className="w-5 h-5" />
                첫 플레이그라운드 만들기
              </button>
            )}
          </div>
        ) : (
          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
            {playgrounds.map((playground, index) => (
              <PlaygroundCard
                key={playground.id}
                playground={playground}
                languageDisplayName={getLanguageDisplayName(playground.language)}
                onClick={() => navigate(`/playground/${playground.id}`)}
                index={index}
              />
            ))}
          </div>
        )}
      </div>

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 w-full max-w-md shadow-2xl animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 flex items-center justify-center">
                  <Plus className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">새 플레이그라운드</h2>
              </div>
              <button
                onClick={() => setShowCreateModal(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  제목
                </label>
                <input
                  type="text"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="My Awesome Code"
                  className="w-full px-4 py-3 border border-gray-200 dark:border-slate-600 rounded-xl bg-gray-50 dark:bg-slate-700 text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  프로그래밍 언어
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {languages.slice(0, 6).map((lang) => {
                    const style = getLanguageStyle(lang.id);
                    const isSelected = newLanguage === lang.id;
                    return (
                      <button
                        key={lang.id}
                        onClick={() => setNewLanguage(lang.id)}
                        className={`flex flex-col items-center gap-1 p-3 rounded-xl border-2 transition-all ${
                          isSelected
                            ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/30'
                            : 'border-gray-200 dark:border-slate-600 hover:border-gray-300 dark:hover:border-slate-500'
                        }`}
                      >
                        <span className="text-xl">{style.icon}</span>
                        <span className={`text-xs font-medium ${isSelected ? 'text-emerald-700 dark:text-emerald-400' : 'text-gray-600 dark:text-gray-400'}`}>
                          {lang.display_name}
                        </span>
                      </button>
                    );
                  })}
                </div>
                {languages.length > 6 && (
                  <select
                    value={newLanguage}
                    onChange={(e) => setNewLanguage(e.target.value)}
                    className="mt-3 w-full px-4 py-2 border border-gray-200 dark:border-slate-600 rounded-xl bg-gray-50 dark:bg-slate-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-emerald-500"
                  >
                    {languages.map((lang) => (
                      <option key={lang.id} value={lang.id}>
                        {lang.display_name}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>

            <div className="flex gap-3 mt-8">
              <button
                onClick={() => setShowCreateModal(false)}
                className="flex-1 px-4 py-3 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-xl font-medium transition-colors"
              >
                취소
              </button>
              <button
                onClick={handleCreatePlayground}
                disabled={creating || !newTitle.trim()}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-xl font-medium hover:from-emerald-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                {creating ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    생성 중...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    만들기
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Styles */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(5deg); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-15px) rotate(-5deg); }
        }
        @keyframes fade-in {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        .animate-float-delayed {
          animation: float-delayed 5s ease-in-out infinite;
          animation-delay: 1s;
        }
        .animate-fade-in {
          animation: fade-in 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}
