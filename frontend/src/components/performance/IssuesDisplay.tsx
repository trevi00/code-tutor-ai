/**
 * Performance Issues Display Component
 */

import { AlertTriangle, AlertCircle, Info, XCircle, Lightbulb } from 'lucide-react';
import type { PerformanceIssue, IssueSeverity } from '../../api/performance';
import { SEVERITY_COLORS, ISSUE_TYPE_LABELS } from '../../api/performance';

interface IssuesDisplayProps {
  issues: PerformanceIssue[];
}

export default function IssuesDisplay({ issues }: IssuesDisplayProps) {
  if (issues.length === 0) {
    return (
      <div className="bg-green-50 rounded-lg p-4 flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
          <Lightbulb className="w-5 h-5 text-green-600" />
        </div>
        <div>
          <p className="font-medium text-green-800">성능 문제 없음</p>
          <p className="text-sm text-green-600">코드에서 성능 문제가 발견되지 않았습니다.</p>
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
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="font-medium text-gray-800 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-yellow-600" />
          성능 이슈
        </h3>
        <span className="text-sm text-gray-500">{issues.length}개 발견</span>
      </div>

      <div className="divide-y divide-gray-100">
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

function IssueItem({ issue }: IssueItemProps) {
  const SeverityIcon = getSeverityIcon(issue.severity);
  const severityColor = SEVERITY_COLORS[issue.severity];

  return (
    <div className="p-4 hover:bg-gray-50">
      <div className="flex items-start gap-3">
        <div className={`p-1.5 rounded ${severityColor}`}>
          <SeverityIcon className="w-4 h-4" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-sm font-medium ${severityColor.split(' ')[0]}`}>
              {ISSUE_TYPE_LABELS[issue.issue_type] || issue.issue_type}
            </span>
            <span className="text-xs text-gray-400">줄 {issue.line_number}</span>
          </div>

          <p className="text-sm text-gray-700 mb-2">{issue.message}</p>

          {issue.code_snippet && (
            <pre className="text-xs bg-gray-100 px-2 py-1 rounded font-mono text-gray-600 mb-2 overflow-x-auto">
              {issue.code_snippet}
            </pre>
          )}

          <div className="flex items-start gap-2 text-sm text-green-700 bg-green-50 px-3 py-2 rounded">
            <Lightbulb className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{issue.suggestion}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function getSeverityIcon(severity: IssueSeverity) {
  switch (severity) {
    case 'critical':
      return XCircle;
    case 'error':
      return AlertCircle;
    case 'warning':
      return AlertTriangle;
    case 'info':
    default:
      return Info;
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
      <span className="text-sm text-green-600 flex items-center gap-1">
        <Lightbulb className="w-4 h-4" />
        문제 없음
      </span>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      {counts.critical > 0 && (
        <span className="flex items-center gap-1 text-red-700">
          <XCircle className="w-4 h-4" />
          {counts.critical}
        </span>
      )}
      {counts.error > 0 && (
        <span className="flex items-center gap-1 text-red-600">
          <AlertCircle className="w-4 h-4" />
          {counts.error}
        </span>
      )}
      {counts.warning > 0 && (
        <span className="flex items-center gap-1 text-yellow-600">
          <AlertTriangle className="w-4 h-4" />
          {counts.warning}
        </span>
      )}
      {counts.info > 0 && (
        <span className="flex items-center gap-1 text-blue-600">
          <Info className="w-4 h-4" />
          {counts.info}
        </span>
      )}
    </div>
  );
}
