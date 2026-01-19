/**
 * Collaboration WebSocket hook
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  MessageType,
} from '../api/collaboration';
import type {
  ChatMessageData,
  CodeUpdateData,
  CursorUpdateData,
  SelectionUpdateData,
  SessionStateData,
  UserJoinedData,
  UserLeftData,
  WebSocketMessage,
} from '../api/collaboration';

interface CollaborationParticipant {
  userId: string;
  username: string;
  color: string;
  cursorLine: number | null;
  cursorColumn: number | null;
  selectionStartLine: number | null;
  selectionStartColumn: number | null;
  selectionEndLine: number | null;
  selectionEndColumn: number | null;
  isActive: boolean;
}

interface ChatMessage {
  userId: string;
  username: string;
  message: string;
  timestamp: Date;
}

interface UseCollaborationOptions {
  sessionId: string;
  onCodeChange?: (content: string, version: number) => void;
  onParticipantsChange?: (participants: CollaborationParticipant[]) => void;
  onChatMessage?: (message: ChatMessage) => void;
  onError?: (error: string) => void;
}

export function useCollaboration({
  sessionId,
  onCodeChange,
  onParticipantsChange,
  onChatMessage,
  onError,
}: UseCollaborationOptions) {
  const wsRef = useRef<WebSocket | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [codeContent, setCodeContent] = useState('');
  const [version, setVersion] = useState(0);
  const [participants, setParticipants] = useState<CollaborationParticipant[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);

  // Handle incoming messages - defined before useEffect that uses it
  const handleMessage = useCallback(
    (message: WebSocketMessage) => {
      switch (message.type) {
        case MessageType.SESSION_STATE: {
          const data = message.data as unknown as SessionStateData;
          setCodeContent(data.code_content);
          setVersion(data.version);
          onCodeChange?.(data.code_content, data.version);

          const participantsList = data.participants.map((p) => ({
            userId: p.user_id,
            username: p.username,
            color: p.color,
            cursorLine: p.cursor?.line ?? null,
            cursorColumn: p.cursor?.column ?? null,
            selectionStartLine: p.selection?.start.line ?? null,
            selectionStartColumn: p.selection?.start.column ?? null,
            selectionEndLine: p.selection?.end.line ?? null,
            selectionEndColumn: p.selection?.end.column ?? null,
            isActive: p.is_active,
          }));
          setParticipants(participantsList);
          onParticipantsChange?.(participantsList);
          break;
        }

        case MessageType.CODE_UPDATE: {
          const data = message.data as unknown as CodeUpdateData;
          // Apply operation to local code
          setCodeContent((prev) => {
            let newContent = prev;
            if (data.operation_type === 'insert') {
              newContent =
                prev.slice(0, data.position) +
                data.content +
                prev.slice(data.position);
            } else if (data.operation_type === 'delete') {
              newContent =
                prev.slice(0, data.position) +
                prev.slice(data.position + data.length);
            } else if (data.operation_type === 'replace') {
              newContent =
                prev.slice(0, data.position) +
                data.content +
                prev.slice(data.position + data.length);
            }
            onCodeChange?.(newContent, data.version);
            return newContent;
          });
          setVersion(data.version);
          break;
        }

        case MessageType.CURSOR_UPDATE: {
          const data = message.data as unknown as CursorUpdateData;
          setParticipants((prev) => {
            const updated = prev.map((p) =>
              p.userId === data.user_id
                ? {
                    ...p,
                    cursorLine: data.line,
                    cursorColumn: data.column,
                  }
                : p
            );
            onParticipantsChange?.(updated);
            return updated;
          });
          break;
        }

        case MessageType.SELECTION_UPDATE: {
          const data = message.data as unknown as SelectionUpdateData;
          setParticipants((prev) => {
            const updated = prev.map((p) =>
              p.userId === data.user_id
                ? {
                    ...p,
                    selectionStartLine: data.start_line,
                    selectionStartColumn: data.start_column,
                    selectionEndLine: data.end_line,
                    selectionEndColumn: data.end_column,
                  }
                : p
            );
            onParticipantsChange?.(updated);
            return updated;
          });
          break;
        }

        case MessageType.USER_JOINED: {
          const data = message.data as unknown as UserJoinedData;
          setParticipants((prev) => {
            const existing = prev.find((p) => p.userId === data.user_id);
            if (existing) {
              return prev.map((p) =>
                p.userId === data.user_id ? { ...p, isActive: true } : p
              );
            }
            const updated = [
              ...prev,
              {
                userId: data.user_id,
                username: data.username,
                color: data.color,
                cursorLine: null,
                cursorColumn: null,
                selectionStartLine: null,
                selectionStartColumn: null,
                selectionEndLine: null,
                selectionEndColumn: null,
                isActive: true,
              },
            ];
            onParticipantsChange?.(updated);
            return updated;
          });
          break;
        }

        case MessageType.USER_LEFT: {
          const data = message.data as unknown as UserLeftData;
          setParticipants((prev) => {
            const updated = prev.map((p) =>
              p.userId === data.user_id ? { ...p, isActive: false } : p
            );
            onParticipantsChange?.(updated);
            return updated;
          });
          break;
        }

        case MessageType.CHAT_MESSAGE: {
          const data = message.data as unknown as ChatMessageData;
          const chatMsg: ChatMessage = {
            userId: data.user_id,
            username: data.username,
            message: data.message,
            timestamp: new Date(data.timestamp),
          };
          setChatMessages((prev) => [...prev, chatMsg]);
          onChatMessage?.(chatMsg);
          break;
        }

        case MessageType.ERROR: {
          const error = message.data.error as string;
          onError?.(error);
          break;
        }
      }
    },
    [onCodeChange, onParticipantsChange, onChatMessage, onError]
  );

  // Connect to WebSocket
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token || !sessionId) return;

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = import.meta.env.VITE_WS_URL || `${wsProtocol}//${window.location.host}`;
    const wsUrl = `${wsHost}/api/v1/collaboration/ws/${sessionId}?token=${token}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = () => {
      onError?.('WebSocket connection error');
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleMessage(message);
      } catch {
        onError?.('Failed to parse WebSocket message');
      }
    };

    return () => {
      ws.close();
    };
  }, [sessionId, handleMessage, onError]);

  // Send code change
  const sendCodeChange = useCallback(
    (operationType: 'insert' | 'delete' | 'replace', position: number, content: string, length: number = 0) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      wsRef.current.send(
        JSON.stringify({
          type: MessageType.CODE_CHANGE,
          data: {
            operation_type: operationType,
            position,
            content,
            length,
          },
        })
      );
    },
    []
  );

  // Send cursor update
  const sendCursorUpdate = useCallback((line: number, column: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(
      JSON.stringify({
        type: MessageType.CURSOR_MOVE,
        data: { line, column },
      })
    );
  }, []);

  // Send selection update
  const sendSelectionUpdate = useCallback(
    (startLine: number, startColumn: number, endLine: number, endColumn: number) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      wsRef.current.send(
        JSON.stringify({
          type: MessageType.SELECTION_CHANGE,
          data: {
            start_line: startLine,
            start_column: startColumn,
            end_line: endLine,
            end_column: endColumn,
          },
        })
      );
    },
    []
  );

  // Send chat message
  const sendChatMessage = useCallback((message: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(
      JSON.stringify({
        type: MessageType.CHAT,
        data: { message },
      })
    );
  }, []);

  return {
    isConnected,
    codeContent,
    version,
    participants,
    chatMessages,
    sendCodeChange,
    sendCursorUpdate,
    sendSelectionUpdate,
    sendChatMessage,
  };
}
