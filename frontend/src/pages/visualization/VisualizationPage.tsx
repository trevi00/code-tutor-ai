/**
 * Algorithm Visualization Page
 */

import { useEffect, useState } from 'react';
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

const CATEGORIES: { id: Category; name: string; description: string }[] = [
  { id: 'sorting', name: '정렬', description: '다양한 정렬 알고리즘 시각화' },
  { id: 'searching', name: '탐색', description: '탐색 알고리즘 시각화' },
  { id: 'graph', name: '그래프', description: '그래프 탐색 알고리즘 시각화' },
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

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">알고리즘 시각화</h1>
        <p className="text-gray-600 mt-1">
          다양한 알고리즘의 동작 과정을 시각적으로 이해해보세요
        </p>
      </div>

      {/* Category Tabs */}
      <div className="flex gap-4 mb-6">
        {CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            onClick={() => handleCategoryChange(cat.id)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedCategory === cat.id
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {cat.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Panel - Controls */}
        <div className="lg:col-span-1 space-y-6">
          {/* Algorithm Selection */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="font-medium text-gray-900 mb-3">알고리즘 선택</h3>
            <div className="space-y-2">
              {filteredAlgorithms.map((algo) => (
                <button
                  key={algo.id}
                  onClick={() => setSelectedAlgorithm(algo.id)}
                  className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                    selectedAlgorithm === algo.id
                      ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  <div className="font-medium">{algo.name}</div>
                  <div className="text-xs text-gray-500">{algo.time_complexity}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Algorithm Info */}
          {selectedAlgoInfo && (
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="font-medium text-gray-900 mb-3">알고리즘 정보</h3>
              <div className="space-y-2 text-sm">
                <p className="text-gray-600">{selectedAlgoInfo.description}</p>
                <div className="flex justify-between">
                  <span className="text-gray-500">시간 복잡도:</span>
                  <span className="font-mono">{selectedAlgoInfo.time_complexity}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">공간 복잡도:</span>
                  <span className="font-mono">{selectedAlgoInfo.space_complexity}</span>
                </div>
              </div>
            </div>
          )}

          {/* Settings */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="font-medium text-gray-900 mb-3">설정</h3>
            <div className="space-y-4">
              {(selectedCategory === 'sorting' || selectedCategory === 'searching') && (
                <div>
                  <label className="block text-sm text-gray-600 mb-1">
                    배열 크기: {arraySize}
                  </label>
                  <input
                    type="range"
                    min={5}
                    max={20}
                    value={arraySize}
                    onChange={(e) => setArraySize(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              )}

              {selectedCategory === 'searching' && (
                <div>
                  <label className="block text-sm text-gray-600 mb-1">
                    찾을 값 (비워두면 랜덤)
                  </label>
                  <input
                    type="number"
                    value={searchTarget ?? ''}
                    onChange={(e) =>
                      setSearchTarget(e.target.value ? Number(e.target.value) : undefined)
                    }
                    placeholder="랜덤"
                    className="w-full px-3 py-2 border rounded-lg"
                  />
                </div>
              )}

              <button
                onClick={handleVisualize}
                disabled={!selectedAlgorithm || loading}
                className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {loading ? '생성 중...' : '시각화 시작'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel - Visualization */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow p-6">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              </div>
            ) : sortingViz ? (
              <SortingVisualizer visualization={sortingViz} />
            ) : graphViz ? (
              <GraphVisualizer visualization={graphViz} />
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <p className="text-lg">알고리즘을 선택하고</p>
                <p className="text-lg">"시각화 시작" 버튼을 클릭하세요</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
