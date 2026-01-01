/**
 * Playground List Page
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Code Playground</h1>
          <p className="text-gray-600 mt-1">
            Experiment with code in multiple languages
          </p>
        </div>
        {isAuthenticated && (
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            New Playground
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b">
        {isAuthenticated && (
          <button
            onClick={() => setActiveTab('my')}
            className={`pb-2 px-1 border-b-2 transition-colors ${
              activeTab === 'my'
                ? 'border-indigo-600 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            My Playgrounds
          </button>
        )}
        <button
          onClick={() => setActiveTab('public')}
          className={`pb-2 px-1 border-b-2 transition-colors ${
            activeTab === 'public'
              ? 'border-indigo-600 text-indigo-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Public
        </button>
        <button
          onClick={() => setActiveTab('popular')}
          className={`pb-2 px-1 border-b-2 transition-colors ${
            activeTab === 'popular'
              ? 'border-indigo-600 text-indigo-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Popular
        </button>
      </div>

      {/* Search and Filter */}
      <div className="flex gap-4 mb-6">
        <div className="flex-1">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search playgrounds..."
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        <select
          value={selectedLanguage}
          onChange={(e) => setSelectedLanguage(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
        >
          <option value="">All Languages</option>
          {languages.map((lang) => (
            <option key={lang.id} value={lang.id}>
              {lang.display_name}
            </option>
          ))}
        </select>
        <button
          onClick={handleSearch}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
        >
          Search
        </button>
      </div>

      {/* Playground List */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : playgrounds.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-500">No playgrounds found</p>
          {isAuthenticated && activeTab === 'my' && (
            <button
              onClick={() => setShowCreateModal(true)}
              className="mt-4 text-indigo-600 hover:text-indigo-700"
            >
              Create your first playground
            </button>
          )}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {playgrounds.map((playground) => (
            <div
              key={playground.id}
              onClick={() => navigate(`/playground/${playground.id}`)}
              className="p-4 bg-white rounded-lg shadow hover:shadow-md transition-shadow cursor-pointer border border-gray-200"
            >
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-medium text-gray-900 truncate">
                  {playground.title}
                </h3>
                <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                  {getLanguageDisplayName(playground.language)}
                </span>
              </div>
              {playground.description && (
                <p className="text-sm text-gray-500 line-clamp-2 mb-2">
                  {playground.description}
                </p>
              )}
              <div className="flex items-center gap-4 text-xs text-gray-400">
                <span>{playground.run_count} runs</span>
                <span>{playground.fork_count} forks</span>
                {playground.is_forked && <span>Forked</span>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">New Playground</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Title
                </label>
                <input
                  type="text"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="My Playground"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Language
                </label>
                <select
                  value={newLanguage}
                  onChange={(e) => setNewLanguage(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                >
                  {languages.map((lang) => (
                    <option key={lang.id} value={lang.id}>
                      {lang.display_name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreatePlayground}
                disabled={creating || !newTitle.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {creating ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
