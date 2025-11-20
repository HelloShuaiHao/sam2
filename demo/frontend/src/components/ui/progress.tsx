import * as React from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
  showValue?: boolean;
  variant?: "default" | "success" | "warning" | "danger";
}

const variantColors = {
  default: "from-blue-500 to-indigo-600",
  success: "from-green-500 to-emerald-600",
  warning: "from-yellow-500 to-orange-600",
  danger: "from-red-500 to-pink-600",
};

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, showValue = false, variant = "default", ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    return (
      <div ref={ref} className={cn("relative w-full", className)} {...props}>
        <div className="h-3 w-full overflow-hidden rounded-full bg-gradient-to-r from-gray-100 to-gray-200">
          <motion.div
            className={cn(
              "h-full rounded-full bg-gradient-to-r shadow-lg",
              variantColors[variant]
            )}
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
        {showValue && (
          <motion.div
            className="absolute -top-7 text-xs font-semibold text-gray-700"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0, left: `${Math.max(percentage - 5, 0)}%` }}
            transition={{ duration: 0.3 }}
          >
            {percentage.toFixed(1)}%
          </motion.div>
        )}
      </div>
    );
  }
);

Progress.displayName = "Progress";

export { Progress };
