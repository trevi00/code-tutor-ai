/**
 * Performance Issues Display Component - Enhanced with modern design
 */

import { AlertTriangle, AlertCircle, Info, XCircle, Lightbulb, CheckCircle2 } from 'lucide-react';
import type { PerformanceIssue, IssueSeverity } from '../../api/performance';
import { ISSUE_TYPE_LABELS } from '../../api/performance';

interface IssuesDisplayProps {
  issues: PerformanceIssue[];
}

export default function IssuesDisplay({ issues }: IssuesDisplayProps) {
  if (issues.length === 0) {
    return (
      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-2xl p-5 flex items-center gap-4 border border-emerald-200 dark:border-emerald-800">
        <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/50 flex items-center justify-center flex-shrink-0">
          <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
        </div>
        <div>
          <p className="font-bold text-emerald-800 dark:text-emerald-200">성능 문제 없음</p>
          <p className="text-sm text-emerald-600 dark:text-emerald-400 mt-0.5">
            코드에서 성능 문제가 발견되지 않았습니다.
          </p>
        </div>
      </div>
    );
  }

  // Sort by severity
  const sortedIssues = [...issues].sort((a, b) => {
    const severityOrder: Record<IssueSeverity, number> = {
      critical: 0,
      error: 1,
      warning: 2,
      info: 3,
    };
    return severityOrder[a.severity] - severityOrder[b.severity];
  });

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden">
      <div className="px-5 py-4 bg-gradient-to-r from-amber-100 to-yellow-100 dark:from-amber-900/30 dark:to-yellow-900/30 border-b border-amber-200 dark:border-amber-800 flex items-center justify-between">
        <h3 className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400" />
          성능 이슈
        </h3>
        <span className="px-3 py-1 bg-amber-200 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 text-sm rounded-lg font-medium">
          {issues.length}개 발견
        </span>
      </div>

      <div className="divide-y divide-slate-100 dark:divide-slate-700">
        {sortedIssues.map((issue, idx) => (
          <IssueItem key={idx} issue={issue} />
        ))}
      </div>
    </div>
  );
}

interface IssueItemProps {
  issue: PerformanceIssue;
}

function SeverityIconDisplay({ severity }: { severity: IssueSeverity; className: string }) {
  const iconClass = getSeverityStyles(severity).icon;
  switch (severity) {
    case 'critical':
      return <XCircle className={`w-5 h-5 ${iconClass}`} />;
    case 'error':
      return <AlertCircle className={`w-5 h-5 ${iconClass}`} />;
    case 'warning':
      return <AlertTriangle className={`w-5 h-5 ${iconClass}`} />;
    case 'info':
    default:
      return <Info className={`w-5 h-5 ${iconClass}`} />;
  }
}

function IssueItem({ issue }: IssueItemProps) {
  const severityStyles = getSeverityStyles(issue.severity);

  return (
    <div className="p-5 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
      <div className="flex items-start gap-4">
        <div className={`p-2 rounded-xl ${severityStyles.bg}`}>
          <SeverityIconDisplay severity={issue.severity} className={severityStyles.icon} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <span className={`text-sm font-bold ${severityStyles.text}`}>
              {ISSUE_TYPE_LABELS[issue.issue_type] || issue.issue_type}
            </span>
            <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
              줄 {issue.line_number}
            </span>
          </div>

          <p className="text-sm text-slate-700 dark:text-slate-300 mb-3">{issue.message}</p>

          {issue.code_snippet && (
            <pre className="text-xs bg-slate-900 text-slate-100 px-4 py-2 rounded-xl font-mono mb-3 overflow-x-auto">
              {issue.code_snippet}
            </pre>
          )}

          <div className="flex items-start gap-2 text-sm bg-emerald-50 dark:bg-emerald-900/20 px-4 py-3 rounded-xl border border-emerald-200 dark:border-emerald-800">
            <Lightbulb className="w-4 h-4 flex-shrink-0 mt-0.5 text-emerald-600 dark:text-emerald-400" />
            <span className="text-emerald-700 dark:text-emerald-300">{issue.suggestion}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function getSeverityStyles(severity: IssueSeverity) {
  switch (severity) {
    case 'critical':
      return {
        bg: 'bg-red-100 dark:bg-red-900/30',
        icon: 'text-red-600 dark:text-red-400',
        text: 'text-red-700 dark:text-red-400',
      };
    case 'error':
      return {
        bg: 'bg-rose-100 dark:bg-rose-900/30',
        icon: 'text-rose-600 dark:text-rose-400',
        text: 'text-rose-700 dark:text-rose-400',
      };
    case 'warning':
      return {
        bg: 'bg-amber-100 dark:bg-amber-900/30',
        icon: 'text-amber-600 dark:text-amber-400',
        text: 'text-amber-700 dark:text-amber-400',
      };
    case 'info':
    default:
      return {
        bg: 'bg-blue-100 dark:bg-blue-900/30',
        icon: 'text-blue-600 dark:text-blue-400',
        text: 'text-blue-700 dark:text-blue-400',
      };
  }
}

// Summary component for quick overview
interface IssuesSummaryProps {
  issues: PerformanceIssue[];
}

export function IssuesSummary({ issues }: IssuesSummaryProps) {
  const counts = {
    critical: issues.filter((i) => i.severity === 'critical').length,
    error: issues.filter((i) => i.severity === 'error').length,
    warning: issues.filter((i) => i.severity === 'warning').length,
    info: issues.filter((i) => i.severity === 'info').length,
  };

  if (issues.length === 0) {
    return (
      <span className="text-sm text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
        <CheckCircle2 className="w-4 h-4" />
        문제 없음
      </span>
    );
  }

  return (
    <div className="flex items-center gap-3 text-sm">
      {counts.critical > 0 && (
        <span className="flex items-center gap-1 text-red-700 dark:text-red-400">
          <XCircle className="w-4 h-4" />
          {counts.critical}
        </span>
      )}
      {counts.error > 0 && (
        <span className="flex items-center gap-1 text-rose-600 dark:text-rose-400">
          <AlertCircle className="w-4 h-4" />
          {counts.error}
        </span>
      )}
      {counts.warning > 0 && (
        <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400">
          <AlertTriangle className="w-4 h-4" />
          {counts.warning}
        </span>
      )}
      {counts.info > 0 && (
        <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400">
          <Info className="w-4 h-4" />
          {counts.info}
        </span>
      )}
    </div>
  );
}
