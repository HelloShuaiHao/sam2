import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold transition-all",
  {
    variants: {
      variant: {
        default:
          "bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-800 border border-blue-200",
        secondary:
          "bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800 border border-purple-200",
        success:
          "bg-gradient-to-r from-green-100 to-emerald-100 text-green-800 border border-green-200",
        warning:
          "bg-gradient-to-r from-yellow-100 to-orange-100 text-orange-800 border border-orange-200",
        destructive:
          "bg-gradient-to-r from-red-100 to-pink-100 text-red-800 border border-red-200",
        outline: "border-2 border-gray-300 text-gray-700 bg-white",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
