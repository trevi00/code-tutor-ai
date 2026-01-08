import { forwardRef } from 'react';
import clsx from 'clsx';
import { Loader2 } from 'lucide-react';

type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: clsx(
    'bg-gradient-to-r from-blue-600 to-indigo-600 text-white',
    'hover:from-blue-700 hover:to-indigo-700',
    'focus-visible:ring-blue-500',
    'shadow-sm hover:shadow-md hover:shadow-blue-500/25',
    'active:scale-[0.98]'
  ),
  secondary: clsx(
    'bg-gray-100 text-gray-900 dark:bg-slate-700 dark:text-gray-100',
    'hover:bg-gray-200 dark:hover:bg-slate-600',
    'focus-visible:ring-gray-500'
  ),
  outline: clsx(
    'border border-gray-300 dark:border-slate-600 bg-transparent',
    'text-gray-700 dark:text-gray-200',
    'hover:bg-gray-50 dark:hover:bg-slate-800',
    'focus-visible:ring-gray-400'
  ),
  ghost: clsx(
    'bg-transparent text-gray-700 dark:text-gray-200',
    'hover:bg-gray-100 dark:hover:bg-slate-800',
    'focus-visible:ring-gray-400'
  ),
  danger: clsx(
    'bg-red-600 text-white',
    'hover:bg-red-700',
    'focus-visible:ring-red-500',
    'shadow-sm hover:shadow-md hover:shadow-red-500/25'
  ),
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-sm gap-1.5 rounded-md',
  md: 'h-10 px-4 text-sm gap-2 rounded-lg',
  lg: 'h-12 px-6 text-base gap-2.5 rounded-lg',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      isLoading = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={clsx(
          'inline-flex items-center justify-center font-medium',
          'transition-all duration-200',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-slate-900',
          'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
          variantStyles[variant],
          sizeStyles[size],
          fullWidth && 'w-full',
          className
        )}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          leftIcon
        )}
        {children}
        {!isLoading && rightIcon}
      </button>
    );
  }
);

Button.displayName = 'Button';

// Icon Button variant
interface IconButtonProps extends Omit<ButtonProps, 'leftIcon' | 'rightIcon' | 'children'> {
  icon: React.ReactNode;
  'aria-label': string;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, size = 'md', icon, ...props }, ref) => {
    const iconSizeStyles: Record<ButtonSize, string> = {
      sm: 'h-8 w-8',
      md: 'h-10 w-10',
      lg: 'h-12 w-12',
    };

    return (
      <Button
        ref={ref}
        className={clsx(iconSizeStyles[size], '!px-0', className)}
        size={size}
        {...props}
      >
        {icon}
      </Button>
    );
  }
);

IconButton.displayName = 'IconButton';
