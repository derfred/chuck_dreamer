// A tiny inline SVG sparkline (F5.1/F5.4). Plots a series of values in [0,1]
// (or auto-scaled) as a polyline. No dependency, no axes — a quick-glance trend.
// Shared by the dashboard sensing strip and the /ring activity view.

interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  // Fixed max for the y-axis; when omitted, auto-scales to the data (min 0.01
  // so a flat-zero series doesn't divide by zero).
  max?: number;
  ariaLabel?: string;
}

export function Sparkline({
  values,
  width = 160,
  height = 28,
  color = "#2a7",
  max,
  ariaLabel = "sparkline",
}: SparklineProps) {
  if (values.length === 0) {
    return (
      <svg width={width} height={height} role="img" aria-label={`${ariaLabel} (no data)`}>
        <line x1={0} y1={height - 1} x2={width} y2={height - 1} stroke="#ddd" />
      </svg>
    );
  }
  const hi = max ?? Math.max(0.01, ...values);
  const n = values.length;
  const stepX = n > 1 ? width / (n - 1) : width;
  const points = values
    .map((v, i) => {
      const x = i * stepX;
      const y = height - Math.min(1, Math.max(0, v / hi)) * (height - 2) - 1;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} role="img" aria-label={ariaLabel}>
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}
