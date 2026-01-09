/**
 * Algorithm Visualization Page - Enhanced with modern design
 */

import { useEffect, useState } from 'react';
import {
  Eye,
  Play,
  Settings,
  Info,
  ArrowRight,
  Loader2,
  BarChart3,
  Search,
  Network,
  Sparkles,
  Zap,
  Clock,
  Database,
} from 'lucide-react';
import {
  getAlgorithms,
  getSortingVisualization,
  getSearchVisualization,
  getGraphVisualization,
} from '../../api/visualization';
import type {
  AlgorithmInfo,
  SortingVisualization,
  GraphVisualization,
} from '../../api/visualization';
import { SortingVisualizer, GraphVisualizer } from '../../components/visualization';

type Category = 'sorting' | 'searching' | 'graph';

const CATEGORIES: { id: Category; name: string; description: string; icon: React.ReactNode; gradient: string }[] = [
  {
    id: 'sorting',
    name: '정렬',
    description: '다양한 정렬 알고리즘의 동작 과정',
    icon: <BarChart3 className="w-5 h-5" />,
    gradient: 'from-cyan-500 to-blue-500',
  },
  {
    id: 'searching',
    name: '탐색',
    description: '효율적인 탐색 알고리즘 시각화',
    icon: <Search className="w-5 h-5" />,
    gradient: 'from-emerald-500 to-teal-500',
  },
  {
    id: 'graph',
    name: '그래프',
    description: '그래프 탐색 알고리즘 시각화',
    icon: <Network className="w-5 h-5" />,
    gradient: 'from-violet-500 to-purple-500',
  },
];

export default function VisualizationPage() {
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<Category>('sorting');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [arraySize, setArraySize] = useState(10);
  const [searchTarget, setSearchTarget] = useState<number | undefined>();
  const [loading, setLoading] = useState(false);

  const [sortingViz, setSortingViz] = useState<SortingVisualization | null>(null);
  const [graphViz, setGraphViz] = useState<GraphVisualization | null>(null);

  useEffect(() => {
    loadAlgorithms();
  }, [selectedCategory]);

  const loadAlgorithms = async () => {
    try {
      const data = await getAlgorithms(selectedCategory);
      setAlgorithms(data.algorithms);

      // Auto-select first algorithm
      if (data.algorithms.length > 0 && !selectedAlgorithm) {
        setSelectedAlgorithm(data.algorithms[0].id);
      }
    } catch (error) {
      console.error('Failed to load algorithms:', error);
    }
  };

  const handleCategoryChange = (category: Category) => {
    setSelectedCategory(category);
    setSelectedAlgorithm(null);
    setSortingViz(null);
    setGraphViz(null);
  };

  const handleVisualize = async () => {
    if (!selectedAlgorithm) return;

    try {
      setLoading(true);
      setSortingViz(null);
      setGraphViz(null);

      if (selectedCategory === 'sorting') {
        const viz = await getSortingVisualization(selectedAlgorithm, arraySize);
        setSortingViz(viz);
      } else if (selectedCategory === 'searching') {
        const viz = await getSearchVisualization(selectedAlgorithm, arraySize, searchTarget);
        setSortingViz(viz);
      } else if (selectedCategory === 'graph') {
        const viz = await getGraphVisualization(selectedAlgorithm);
        setGraphViz(viz);
      }
    } catch (error) {
      console.error('Failed to generate visualization:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredAlgorithms = algorithms.filter(
    (algo) => algo.category === selectedCategory
  );

  const selectedAlgoInfo = algorithms.find((a) => a.id === selectedAlgorithm);
  const selectedCategoryInfo = CATEGORIES.find((c) => c.id === selectedCategory);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-cyan-600 via-teal-600 to-cyan-700 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl" />
          {/* Floating icons */}
          <Eye className="absolute top-10 right-[10%] w-12 h-12 text-white/10 animate-float" />
          <BarChart3 className="absolute bottom-10 left-[15%] w-10 h-10 text-white/10 animate-float-delayed" />
          <Network className="absolute top-20 left-[20%] w-8 h-8 text-white/10 animate-float" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4">
                <Sparkles className="w-4 h-4" />
                인터랙티브 학습
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <Eye className="w-10 h-10 text-cyan-200" />
                알고리즘 시각화
              </h1>
              <p className="text-cyan-100 text-lg max-w-md">
                다양한 알고리즘의 동작 과정을 시각적으로 이해해보세요
              </p>
            </div>

            {/* Quick Stats */}
            <div className="flex gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <BarChart3 className="w-6 h-6 text-cyan-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">3</div>
                <div className="text-xs text-cyan-200">카테고리</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[100px]">
                <Zap className="w-6 h-6 text-yellow-300 mx-auto mb-1" />
                <div className="text-2xl font-bold text-white">{filteredAlgorithms.length}</div>
                <div className="text-xs text-cyan-200">알고리즘</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8 -mt-6">
        {/* Category Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => handleCategoryChange(cat.id)}
              className={`relative p-5 rounded-2xl text-left transition-all duration-300 group overflow-hidden ${
                selectedCategory === cat.id
                  ? 'bg-white dark:bg-slate-800 shadow-xl ring-2 ring-cyan-500'
                  : 'bg-white/80 dark:bg-slate-800/80 shadow-lg hover:shadow-xl hover:-translate-y-1'
              }`}
            >
              {/* Gradient overlay when selected */}
              {selectedCategory === cat.id && (
                <div className={`absolute inset-0 bg-gradient-to-br ${cat.gradient} opacity-10`} />
              )}

              <div className="relative">
                <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl mb-3 bg-gradient-to-br ${cat.gradient} text-white shadow-lg`}>
                  {cat.icon}
                </div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">
                  {cat.name}
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {cat.description}
                </p>
                {selectedCategory === cat.id && (
                  <div className={`mt-3 inline-flex items-center gap-1 text-sm font-medium bg-gradient-to-r ${cat.gradient} bg-clip-text text-transparent`}>
                    선택됨
                    <ArrowRight className="w-4 h-4 text-cyan-500" />
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Algorithm Selection */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-5">
              <div className="flex items-center gap-2 mb-4">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${selectedCategoryInfo?.gradient || 'from-cyan-500 to-blue-500'}`}>
                  {selectedCategoryInfo?.icon || <BarChart3 className="w-4 h-4 text-white" />}
                </div>
                <h3 className="font-bold text-slate-900 dark:text-white">알고리즘 선택</h3>
              </div>
              <div className="space-y-2">
                {filteredAlgorithms.map((algo, index) => (
                  <button
                    key={algo.id}
                    onClick={() => setSelectedAlgorithm(algo.id)}
                    className={`w-full text-left px-4 py-3 rounded-xl transition-all duration-200 ${
                      selectedAlgorithm === algo.id
                        ? 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white shadow-lg'
                        : 'bg-slate-50 dark:bg-slate-700/50 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-900 dark:text-white'
                    }`}
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <div className="font-medium">{algo.name}</div>
                    <div className={`text-xs mt-1 font-mono ${
                      selectedAlgorithm === algo.id ? 'text-cyan-100' : 'text-slate-500 dark:text-slate-400'
                    }`}>
                      {algo.time_complexity}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Algorithm Info */}
            {selectedAlgoInfo && (
              <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-5">
                <div className="flex items-center gap-2 mb-4">
                  <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
                    <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="font-bold text-slate-900 dark:text-white">알고리즘 정보</h3>
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                  {selectedAlgoInfo.description}
                </p>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl">
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4 text-amber-500" />
                      <span className="text-sm text-slate-600 dark:text-slate-400">시간 복잡도</span>
                    </div>
                    <span className="font-mono text-sm font-bold text-slate-900 dark:text-white">
                      {selectedAlgoInfo.time_complexity}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl">
                    <div className="flex items-center gap-2">
                      <Database className="w-4 h-4 text-purple-500" />
                      <span className="text-sm text-slate-600 dark:text-slate-400">공간 복잡도</span>
                    </div>
                    <span className="font-mono text-sm font-bold text-slate-900 dark:text-white">
                      {selectedAlgoInfo.space_complexity}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Settings */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg p-5">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
                  <Settings className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                </div>
                <h3 className="font-bold text-slate-900 dark:text-white">설정</h3>
              </div>
              <div className="space-y-4">
                {(selectedCategory === 'sorting' || selectedCategory === 'searching') && (
                  <div>
                    <label className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-400 mb-2">
                      <span>배열 크기</span>
                      <span className="font-bold text-cyan-600 dark:text-cyan-400">{arraySize}</span>
                    </label>
                    <input
                      type="range"
                      min={5}
                      max={20}
                      value={arraySize}
                      onChange={(e) => setArraySize(Number(e.target.value))}
                      className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                      <span>5</span>
                      <span>20</span>
                    </div>
                  </div>
                )}

                {selectedCategory === 'searching' && (
                  <div>
                    <label className="block text-sm text-slate-600 dark:text-slate-400 mb-2">
                      찾을 값 (비워두면 랜덤)
                    </label>
                    <input
                      type="number"
                      value={searchTarget ?? ''}
                      onChange={(e) =>
                        setSearchTarget(e.target.value ? Number(e.target.value) : undefined)
                      }
                      placeholder="랜덤"
                      className="w-full px-4 py-2.5 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-xl focus:ring-2 focus:ring-cyan-500 focus:border-transparent text-slate-900 dark:text-white"
                    />
                  </div>
                )}

                <button
                  onClick={handleVisualize}
                  disabled={!selectedAlgorithm || loading}
                  className="w-full px-4 py-3 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      생성 중...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      시각화 시작
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right Panel - Visualization */}
          <div className="lg:col-span-3">
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
              {/* Panel Header */}
              <div className="px-6 py-4 bg-gradient-to-r from-slate-100 to-slate-50 dark:from-slate-700 dark:to-slate-800 border-b border-slate-200 dark:border-slate-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-red-500" />
                      <div className="w-3 h-3 rounded-full bg-yellow-500" />
                      <div className="w-3 h-3 rounded-full bg-green-500" />
                    </div>
                    <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                      시각화 패널
                    </span>
                  </div>
                  {selectedAlgoInfo && (
                    <span className="text-sm text-slate-500 dark:text-slate-400">
                      {selectedAlgoInfo.name}
                    </span>
                  )}
                </div>
              </div>

              {/* Visualization Content */}
              <div className="p-6">
                {loading ? (
                  <div className="flex flex-col items-center justify-center h-96">
                    <div className="relative">
                      <div className="w-20 h-20 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 animate-pulse" />
                      <Loader2 className="w-10 h-10 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
                    </div>
                    <p className="text-slate-500 dark:text-slate-400 mt-4 animate-pulse">시각화 생성 중...</p>
                  </div>
                ) : sortingViz ? (
                  <SortingVisualizer visualization={sortingViz} />
                ) : graphViz ? (
                  <GraphVisualizer visualization={graphViz} />
                ) : (
                  <div className="flex flex-col items-center justify-center h-96 text-slate-400 dark:text-slate-500">
                    <div className="w-24 h-24 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center mb-6">
                      <Eye className="w-12 h-12 text-slate-300 dark:text-slate-600" />
                    </div>
                    <p className="text-lg font-medium text-slate-600 dark:text-slate-400 mb-2">
                      알고리즘을 선택하세요
                    </p>
                    <p className="text-sm text-slate-400 dark:text-slate-500">
                      "시각화 시작" 버튼을 클릭하여 알고리즘의 동작을 확인하세요
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

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
