export function SkeletonLine({ className = '' }: { className?: string }) {
  return (
    <div className={`animate-pulse rounded bg-line/60 ${className}`} />
  )
}

export function SkeletonRow() {
  return (
    <div className="flex items-center gap-4 rounded-lg border border-line bg-elev px-4 py-3">
      <div className="h-2.5 w-2.5 shrink-0 animate-pulse rounded-full bg-line/60" />
      <div className="flex-1 space-y-2">
        <SkeletonLine className="h-4 w-3/4" />
        <SkeletonLine className="h-3 w-1/2" />
      </div>
      <SkeletonLine className="h-5 w-12" />
    </div>
  )
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 border-b border-line/40 py-2">
          <SkeletonLine className="h-3.5 w-16 shrink-0" />
          <SkeletonLine className="h-3.5 w-20 shrink-0" />
          <SkeletonLine className="h-3.5 flex-1" />
          <SkeletonLine className="h-3.5 w-32 shrink-0" />
          <SkeletonLine className="h-3.5 w-8 shrink-0" />
        </div>
      ))}
    </div>
  )
}
