/**
 * Submissions Page - Enhanced with modern design
 */

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  FileCode2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Cpu,
  HardDrive,
  ArrowLeft,
  Loader2,
  ChevronDown,
  Sparkles,
  TrendingUp,
  AlertCircle,
  Zap,
} from 'lucide-react';
import { dashboardApi } from '@/api';
import type { SubmissionSummary, SubmissionStatus } from '@/types';

// Status configuration with icons and styles
const statusConfig: Record<SubmissionStatus, {
  bg: string;
  text: string;
  icon: React.ReactNode;
  label: string;
}> = {
  accepted: {
    bg: 'bg-emerald-500/20',
    text: 'text-emerald-400',
    icon: <CheckCircle className="w-3.5 h-3.5" />,
    label: '정답',
  },
  wrong_answer: {
    bg: 'bg-red-500/20',
    text: 'text-red-400',
    icon: <XCircle className="w-3.5 h-3.5" />,
    label: '오답',
  },
  runtime_error: {
    bg: 'bg-orange-500/20',
    text: 'text-orange-400',
    icon: <AlertTriangle className="w-3.5 h-3.5" />,
    label: '런타임 에러',
  },
  time_limit_exceeded: {
    bg: 'bg-amber-500/20',
    text: 'text-amber-400',
    icon: <Clock className="w-3.5 h-3.5" />,
    label: '시간 초과',
  },
  memory_limit_exceeded: {
    bg: 'bg-purple-500/20',
    text: 'text-purple-400',
    icon: <HardDrive className="w-3.5 h-3.5" />,
    label: '메모리 초과',
  },
  pending: {
    bg: 'bg-slate-500/20',
    text: 'text-slate-400',
    icon: <Clock className="w-3.5 h-3.5" />,
    label: '대기중',
  },
  running: {
    bg: 'bg-cyan-500/20',
    text: 'text-cyan-400',
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: '실행중',
  },
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

  // Calculate stats
  const acceptedCount = submissions.filter(s => s.status === 'accepted').length;
  const wrongCount = submissions.filter(s => s.status === 'wrong_answer').length;
  const accuracyRate = submissions.length > 0 ? ((acceptedCount / submissions.length) * 100).toFixed(1) : '0';

  if (loading && submissions.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse" />
            <Loader2 className="w-8 h-8 text-white absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
          </div>
          <p className="mt-4 text-slate-400">제출 기록 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error && submissions.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
            <AlertCircle className="w-10 h-10 text-red-400" />
          </div>
          <p className="text-slate-400 mb-6">{error}</p>
          <button
            onClick={() => fetchSubmissions(0)}
            className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl font-medium transition-all shadow-lg shadow-cyan-500/25"
          >
            다시 시도
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 relative overflow-hidden">
        {/* Background decorations */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl" />
          <FileCode2 className="absolute top-10 right-[10%] w-16 h-16 text-white/10" />
          <TrendingUp className="absolute bottom-8 left-[15%] w-12 h-12 text-white/10" />
          <Sparkles className="absolute top-16 left-[25%] w-8 h-8 text-white/10" />
        </div>

        <div className="max-w-6xl mx-auto px-6 py-12 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-center md:text-left">
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/20 rounded-full text-white/90 text-sm mb-4 hover:bg-white/30 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                대시보드로 돌아가기
              </Link>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-3 flex items-center gap-3 justify-center md:justify-start">
                <FileCode2 className="w-10 h-10 text-cyan-200" />
                제출 기록
              </h1>
              <p className="text-cyan-100 text-lg max-w-md">
                모든 코드 제출 내역을 확인하고 학습 진행 상황을 추적하세요
              </p>
            </div>

            {/* Stats Summary */}
            {submissions.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                  <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-blue-500/20 flex items-center justify-center">
                    <FileCode2 className="w-5 h-5 text-blue-300" />
                  </div>
                  <div className="text-2xl font-bold text-white">{submissions.length}+</div>
                  <div className="text-xs text-cyan-200">총 제출</div>
                </div>
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                  <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                    <CheckCircle className="w-5 h-5 text-emerald-300" />
                  </div>
                  <div className="text-2xl font-bold text-white">{acceptedCount}</div>
                  <div className="text-xs text-cyan-200">정답</div>
                </div>
                <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px]">
                  <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-red-500/20 flex items-center justify-center">
                    <XCircle className="w-5 h-5 text-red-300" />
                  </div>
                  <div className="text-2xl font-bold text-white">{wrongCount}</div>
                  <div className="text-xs text-cyan-200">오답</div>
                </div>
                <div className="bg-white/20 backdrop-blur-sm rounded-xl p-4 text-center min-w-[90px] border border-white/30">
                  <div className="w-10 h-10 mx-auto mb-2 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-yellow-300" />
                  </div>
                  <div className="text-2xl font-bold text-white">{accuracyRate}%</div>
                  <div className="text-xs text-cyan-200">정답률</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8 -mt-4">
        {submissions.length === 0 ? (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-12 text-center">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-slate-700/50 flex items-center justify-center">
              <FileCode2 className="w-10 h-10 text-slate-500" />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">아직 제출 기록이 없습니다</h2>
            <p className="text-slate-400 mb-6">문제를 풀고 코드를 제출해보세요!</p>
            <Link
              to="/problems"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl font-medium transition-all shadow-lg shadow-cyan-500/25"
            >
              <Sparkles className="w-5 h-5" />
              문제 풀러 가기
            </Link>
          </div>
        ) : (
          <>
            {/* Submissions Table */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
              {/* Table Header */}
              <div className="grid grid-cols-12 gap-4 px-6 py-4 bg-slate-700/30 border-b border-slate-700/50">
                <div className="col-span-4 text-xs font-medium text-slate-400 uppercase tracking-wider">문제</div>
                <div className="col-span-2 text-xs font-medium text-slate-400 uppercase tracking-wider">결과</div>
                <div className="col-span-2 text-xs font-medium text-slate-400 uppercase tracking-wider">실행 시간</div>
                <div className="col-span-2 text-xs font-medium text-slate-400 uppercase tracking-wider">메모리</div>
                <div className="col-span-2 text-xs font-medium text-slate-400 uppercase tracking-wider">제출 시간</div>
              </div>

              {/* Table Body */}
              <div className="divide-y divide-slate-700/30">
                {submissions.map((submission) => {
                  const config = statusConfig[submission.status] || {
                    bg: 'bg-slate-500/20',
                    text: 'text-slate-400',
                    icon: <Clock className="w-3.5 h-3.5" />,
                    label: submission.status,
                  };

                  return (
                    <Link
                      key={submission.id}
                      to={`/problems/${submission.problem_id}`}
                      className="grid grid-cols-12 gap-4 px-6 py-4 hover:bg-slate-700/20 transition-colors group"
                    >
                      <div className="col-span-4">
                        <span className="text-white font-medium group-hover:text-cyan-400 transition-colors">
                          {submission.problem_title}
                        </span>
                      </div>
                      <div className="col-span-2">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium ${config.bg} ${config.text}`}>
                          {config.icon}
                          {config.label}
                        </span>
                      </div>
                      <div className="col-span-2 flex items-center gap-2 text-sm text-slate-400">
                        <Cpu className="w-4 h-4 text-slate-500" />
                        {submission.execution_time_ms > 0 ? `${submission.execution_time_ms}ms` : '-'}
                      </div>
                      <div className="col-span-2 flex items-center gap-2 text-sm text-slate-400">
                        <HardDrive className="w-4 h-4 text-slate-500" />
                        {submission.memory_usage_mb > 0 ? `${submission.memory_usage_mb.toFixed(1)}MB` : '-'}
                      </div>
                      <div className="col-span-2 flex items-center gap-2 text-sm text-slate-400">
                        <Clock className="w-4 h-4 text-slate-500" />
                        {formatDate(submission.submitted_at)}
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>

            {/* Load More Button */}
            {hasMore && (
              <div className="mt-8 text-center">
                <button
                  onClick={loadMore}
                  disabled={loading}
                  className="inline-flex items-center gap-2 px-8 py-3 bg-slate-700/50 hover:bg-slate-700 text-white rounded-xl font-medium transition-all border border-slate-600/50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      로딩 중...
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-5 h-5" />
                      더 보기
                    </>
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
