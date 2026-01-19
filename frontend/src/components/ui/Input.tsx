import { forwardRef, useId, useState } from 'react';
import clsx from 'clsx';
import { Eye, EyeOff, AlertCircle, CheckCircle } from 'lucide-react';

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string;
  error?: string;
  success?: string;
  hint?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  inputSize?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
}

const sizeStyles = {
  sm: 'h-8 px-3 text-sm',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-4 text-base',
};

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      type = 'text',
      label,
      error,
      success,
      hint,
      leftIcon,
      rightIcon,
      inputSize = 'md',
      fullWidth = true,
      disabled,
      id,
      ...props
    },
    ref
  ) => {
    const [showPassword, setShowPassword] = useState(false);
    const generatedId = useId();
    const inputId = id || props.name || generatedId;
    const isPassword = type === 'password';
    const currentType = isPassword && showPassword ? 'text' : type;

    const hasValidation = error || success;

    return (
      <div className={clsx(fullWidth && 'w-full')}>
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5"
          >
            {label}
          </label>
        )}
        <div className="relative">
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500">
              {leftIcon}
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            type={currentType}
            className={clsx(
              'w-full rounded-lg border bg-white dark:bg-slate-800',
              'text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500',
              'transition-all duration-200',
              'focus:outline-none focus:ring-2 focus:ring-offset-0',
              sizeStyles[inputSize],
              leftIcon && 'pl-10',
              (rightIcon || isPassword || hasValidation) && 'pr-10',
              error
                ? 'border-red-300 dark:border-red-500/50 focus:border-red-500 focus:ring-red-500/20'
                : success
                ? 'border-green-300 dark:border-green-500/50 focus:border-green-500 focus:ring-green-500/20'
                : 'border-gray-300 dark:border-slate-600 focus:border-blue-500 focus:ring-blue-500/20',
              disabled && 'bg-gray-50 dark:bg-slate-900 cursor-not-allowed opacity-60',
              className
            )}
            disabled={disabled}
            {...props}
          />
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
            {isPassword && (
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
                tabIndex={-1}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            )}
            {error && <AlertCircle className="h-4 w-4 text-red-500" />}
            {success && <CheckCircle className="h-4 w-4 text-green-500" />}
            {!isPassword && !hasValidation && rightIcon}
          </div>
        </div>
        {(error || success || hint) && (
          <p
            className={clsx(
              'mt-1.5 text-sm',
              error
                ? 'text-red-600 dark:text-red-400'
                : success
                ? 'text-green-600 dark:text-green-400'
                : 'text-gray-500 dark:text-gray-400'
            )}
          >
            {error || success || hint}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

// Textarea variant
interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  hint?: string;
  fullWidth?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, hint, fullWidth = true, disabled, id, ...props }, ref) => {
    const generatedId = useId();
    const inputId = id || props.name || generatedId;

    return (
      <div className={clsx(fullWidth && 'w-full')}>
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5"
          >
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={inputId}
          className={clsx(
            'w-full rounded-lg border bg-white dark:bg-slate-800 px-4 py-3',
            'text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500',
            'transition-all duration-200',
            'focus:outline-none focus:ring-2 focus:ring-offset-0',
            'resize-y min-h-[100px]',
            error
              ? 'border-red-300 dark:border-red-500/50 focus:border-red-500 focus:ring-red-500/20'
              : 'border-gray-300 dark:border-slate-600 focus:border-blue-500 focus:ring-blue-500/20',
            disabled && 'bg-gray-50 dark:bg-slate-900 cursor-not-allowed opacity-60',
            className
          )}
          disabled={disabled}
          {...props}
        />
        {(error || hint) && (
          <p
            className={clsx(
              'mt-1.5 text-sm',
              error ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'
            )}
          >
            {error || hint}
          </p>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';
