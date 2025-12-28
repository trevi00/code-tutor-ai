import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { dashboardApi } from '@/api';
import type { SubmissionSummary, SubmissionStatus } from '@/types';

// Status badge styles
const statusStyles: Record<SubmissionStatus, string> = {
  accepted: 'bg-green-100 text-green-800',
  wrong_answer: 'bg-red-100 text-red-800',
  runtime_error: 'bg-orange-100 text-orange-800',
  time_limit_exceeded: 'bg-yellow-100 text-yellow-800',
  memory_limit_exceeded: 'bg-purple-100 text-purple-800',
  pending: 'bg-gray-100 text-gray-800',
  running: 'bg-blue-100 text-blue-800',
};

const statusLabels: Record<SubmissionStatus, string> = {
  accepted: '정답',
  wrong_answer: '오답',
  runtime_error: '런타임 에러',
  time_limit_exceeded: '시간 초과',
  memory_limit_exceeded: '메모리 초과',
  pending: '대기중',
  running: '실행중',
};

export default function SubmissionsPage() {
  const [submissions, setSubmissions] = useState<SubmissionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const pageSize = 20;

  const fetchSubmissions = async (pageNum: number, append = false) => {
    try {
      setLoading(true);
      const data = await dashboardApi.getSubmissions(pageSize, pageNum * pageSize);

      if (append) {
        setSubmissions(prev => [...prev, ...data]);
      } else {
        setSubmissions(data);
      }

      setHasMore(data.length === pageSize);
    } catch (err) {
      setError('제출 기록을 불러오는데 실패했습니다.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSubmissions(0);
  }, []);

  const loadMore = () => {
    const nextPage = page + 1;
    setPage(nextPage);
    fetchSubmissions(nextPage, true);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading && submissions.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error && submissions.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <button
            onClick={() => fetchSubmissions(0)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">제출 기록</h1>
        <Link
          to="/dashboard"
          className="text-blue-600 hover:text-blue-800 flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          대시보드로 돌아가기
        </Link>
      </div>

      {submissions.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-6xl mb-4">📝</div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">아직 제출 기록이 없습니다</h2>
          <p className="text-gray-500 mb-6">문제를 풀고 코드를 제출해보세요!</p>
          <Link
            to="/problems"
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            문제 풀러 가기
          </Link>
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-sm font-medium text-gray-500">총 제출</h3>
              <p className="text-2xl font-bold text-gray-800">{submissions.length}+</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-sm font-medium text-gray-500">정답</h3>
              <p className="text-2xl font-bold text-green-600">
                {submissions.filter(s => s.status === 'accepted').length}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-sm font-medium text-gray-500">오답</h3>
              <p className="text-2xl font-bold text-red-600">
                {submissions.filter(s => s.status === 'wrong_answer').length}
              </p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-sm font-medium text-gray-500">기타</h3>
              <p className="text-2xl font-bold text-gray-600">
                {submissions.filter(s => !['accepted', 'wrong_answer'].includes(s.status)).length}
              </p>
            </div>
          </div>

          {/* Submissions Table */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    문제
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    결과
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    실행 시간
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    메모리
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    제출 시간
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {submissions.map((submission) => (
                  <tr
                    key={submission.id}
                    className="hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => window.location.href = `/problems/${submission.problem_id}`}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <Link
                        to={`/problems/${submission.problem_id}`}
                        className="text-blue-600 hover:text-blue-800 font-medium"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {submission.problem_title}
                      </Link>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded ${
                          statusStyles[submission.status] || 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {statusLabels[submission.status] || submission.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {submission.execution_time_ms > 0 ? `${submission.execution_time_ms}ms` : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {submission.memory_usage_mb > 0 ? `${submission.memory_usage_mb.toFixed(1)}MB` : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(submission.submitted_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Load More Button */}
          {hasMore && (
            <div className="mt-6 text-center">
              <button
                onClick={loadMore}
                disabled={loading}
                className="px-6 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
              >
                {loading ? '로딩 중...' : '더 보기'}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
