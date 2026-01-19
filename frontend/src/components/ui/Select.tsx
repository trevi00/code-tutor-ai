import { forwardRef, useId } from 'react';
import { ChevronDown } from 'lucide-react';
import clsx from 'clsx';

interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  options: SelectOption[];
  label?: string;
  error?: string;
  hint?: string;
  placeholder?: string;
  selectSize?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
}

const sizeStyles = {
  sm: 'h-8 px-3 text-sm pr-8',
  md: 'h-10 px-4 text-sm pr-10',
  lg: 'h-12 px-4 text-base pr-12',
};

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      className,
      options,
      label,
      error,
      hint,
      placeholder,
      selectSize = 'md',
      fullWidth = true,
      disabled,
      id,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const selectId = id || props.name || generatedId;

    return (
      <div className={clsx(fullWidth && 'w-full')}>
        {label && (
          <label
            htmlFor={selectId}
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5"
          >
            {label}
          </label>
        )}
        <div className="relative">
          <select
            ref={ref}
            id={selectId}
            className={clsx(
              'w-full rounded-lg border bg-white dark:bg-slate-800 appearance-none',
              'text-gray-900 dark:text-gray-100',
              'transition-all duration-200',
              'focus:outline-none focus:ring-2 focus:ring-offset-0',
              sizeStyles[selectSize],
              error
                ? 'border-red-300 dark:border-red-500/50 focus:border-red-500 focus:ring-red-500/20'
                : 'border-gray-300 dark:border-slate-600 focus:border-blue-500 focus:ring-blue-500/20',
              disabled && 'bg-gray-50 dark:bg-slate-900 cursor-not-allowed opacity-60',
              className
            )}
            disabled={disabled}
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 dark:text-gray-500 pointer-events-none" />
        </div>
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

Select.displayName = 'Select';
