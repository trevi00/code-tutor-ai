import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, Filter, ChevronRight } from 'lucide-react';
import type { Difficulty, Category } from '@/types';

// Mock data - will be replaced with API calls
const MOCK_PROBLEMS = [
  { id: '1', title: 'Two Sum', difficulty: 'easy' as Difficulty, category: 'array' as Category },
  { id: '2', title: 'Valid Parentheses', difficulty: 'easy' as Difficulty, category: 'stack' as Category },
  { id: '3', title: 'Merge Two Sorted Lists', difficulty: 'easy' as Difficulty, category: 'linked_list' as Category },
  { id: '4', title: 'Best Time to Buy and Sell Stock', difficulty: 'easy' as Difficulty, category: 'dp' as Category },
  { id: '5', title: 'Longest Substring Without Repeating', difficulty: 'medium' as Difficulty, category: 'string' as Category },
  { id: '6', title: 'Container With Most Water', difficulty: 'medium' as Difficulty, category: 'array' as Category },
  { id: '7', title: 'Coin Change', difficulty: 'medium' as Difficulty, category: 'dp' as Category },
  { id: '8', title: 'Binary Tree Level Order', difficulty: 'medium' as Difficulty, category: 'tree' as Category },
  { id: '9', title: 'Word Break', difficulty: 'hard' as Difficulty, category: 'dp' as Category },
  { id: '10', title: 'Merge k Sorted Lists', difficulty: 'hard' as Difficulty, category: 'linked_list' as Category },
];

const DIFFICULTY_COLORS: Record<Difficulty, string> = {
  easy: 'bg-green-100 text-green-700',
  medium: 'bg-yellow-100 text-yellow-700',
  hard: 'bg-red-100 text-red-700',
};

const CATEGORY_LABELS: Partial<Record<Category, string>> = {
  array: 'Array',
  string: 'String',
  hash_table: 'Hash Table',
  linked_list: 'Linked List',
  stack: 'Stack',
  queue: 'Queue',
  tree: 'Tree',
  graph: 'Graph',
  dp: 'Dynamic Programming',
  greedy: 'Greedy',
  binary_search: 'Binary Search',
  sorting: 'Sorting',
};

export function ProblemsPage() {
  const [search, setSearch] = useState('');
  const [difficultyFilter, setDifficultyFilter] = useState<Difficulty | ''>('');
  const [categoryFilter, setCategoryFilter] = useState<Category | ''>('');

  const filteredProblems = MOCK_PROBLEMS.filter((problem) => {
    const matchesSearch = problem.title.toLowerCase().includes(search.toLowerCase());
    const matchesDifficulty = !difficultyFilter || problem.difficulty === difficultyFilter;
    const matchesCategory = !categoryFilter || problem.category === categoryFilter;
    return matchesSearch && matchesDifficulty && matchesCategory;
  });

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Problems</h1>
        <p className="text-neutral-600">Practice algorithm problems with AI-powered hints</p>
      </div>

      {/* Filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-neutral-400" />
          <input
            type="text"
            placeholder="Search problems..."
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
              <option value="">All Difficulties</option>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value as Category | '')}
            className="px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-blue-500 appearance-none bg-white"
          >
            <option value="">All Categories</option>
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
              <th className="text-left py-4 px-6 font-medium text-neutral-600">Title</th>
              <th className="text-left py-4 px-4 font-medium text-neutral-600">Difficulty</th>
              <th className="text-left py-4 px-4 font-medium text-neutral-600">Category</th>
              <th className="w-12"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-200">
            {filteredProblems.map((problem) => (
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
                    className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${
                      DIFFICULTY_COLORS[problem.difficulty]
                    }`}
                  >
                    {problem.difficulty}
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

        {filteredProblems.length === 0 && (
          <div className="py-12 text-center text-neutral-500">
            No problems found matching your filters.
          </div>
        )}
      </div>
    </div>
  );
}
