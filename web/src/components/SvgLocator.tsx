// C6 — offline SVG locator fallback. Renders a simplified world outline +
// GeoJSON overlay on an equirectangular SVG (zero map libs, air-gap safe).
import { useEffect, useState } from 'react'
import type { GeoJSON } from '../api'

// Simplified country polygons (licence-clean Natural-Earth-style, <=200KB).
// Each country: [name, [polygon0, polygon1, ...]] where each polygon is
// [lon, lat] pairs on -180..180 / -90..90 (equirectangular).
import WORLD from '../geo/world110.json'

interface CountryPoly { n: string; polys: number[][][] }

function project(lon: number, lat: number, w: number, h: number): [number, number] {
  const x = (lon + 180) * (w / 360)
  const y = (90 - lat) * (h / 180)
  return [x, y]
}

export default function SvgLocator({ geo }: { geo: GeoJSON }) {
  const W = 720, H = 360
  const [countries, setCountries] = useState<CountryPoly[]>([])

  useEffect(() => {
    // WORLD: { countries: [{n, polys: [[[lon,lat],...],...] }] }
    const raw = (WORLD as any).countries as CountryPoly[]
    setCountries(raw || [])
  }, [])

  const features = geo?.features || []
  const lons: number[] = [], lats: number[] = []
  for (const f of features) for (const [x, y] of f.geometry.coordinates[0]) {
    if (Math.abs(x) <= 180 && Math.abs(y) <= 90) { lons.push(x); lats.push(y) }
  }

  return (
    <div className="relative h-full w-full" style={{ background: '#0f172a' }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
        {/* ocean */}
        <rect x="0" y="0" width={W} height={H} fill="#0f172a" />
        {/* graticule */}
        {[-120, -60, 0, 60, 120].map(lon => {
          const [x] = project(lon, 0, W, H)
          return <line key={`g${lon}`} x1={x} y1={0} x2={x} y2={H} stroke="#1e293b" strokeWidth="0.5" />
        })}
        {[-60, -30, 0, 30, 60].map(lat => {
          const [, y] = project(0, lat, W, H)
          return <line key={`a${lat}`} x1={0} y1={y} x2={W} y2={y} stroke="#1e293b" strokeWidth="0.5" />
        })}
        {/* country outlines */}
        {countries.map((c, ci) =>
          c.polys.map((poly, pi) => {
            const d = poly.map((p, i) => {
              const [x, y] = project(p[0], p[1], W, H)
              return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
            }).join(' ') + ' Z'
            return <path key={`${ci}-${pi}`} d={d} fill="#1e293b" stroke="#475569" strokeWidth="0.6" />
          })
        )}
        {/* GeoJSON overlay */}
        {features.map((f, fi) => (
          <path key={`f${fi}`}
            d={f.geometry.coordinates[0].map(([x, y], i) => {
              const [px, py] = project(x, y, W, H)
              return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`
            }).join(' ') + ' Z'}
            fill={f.properties.kind === 'change' ? '#F59E0B' : '#4C8DF6'}
            fillOpacity={0.3}
            stroke={f.properties.kind === 'change' ? '#F59E0B' : '#4C8DF6'}
            strokeWidth="1"
          />
        ))}
        {/* feature markers */}
        {features.map((f, fi) => {
          const coords = f.geometry.coordinates[0]
          if (!coords.length) return null
          const cx = coords.reduce((s, c) => s + c[0], 0) / coords.length
          const cy = coords.reduce((s, c) => s + c[1], 0) / coords.length
          const [px, py] = project(cx, cy, W, H)
          return <circle key={`m${fi}`} cx={px} cy={py} r="3" fill="#4C8DF6" />
        })}
      </svg>
      <div className="absolute bottom-1 left-2 font-mono text-[10px] text-slate-500">
        offline locator · no tiles · simplified Natural Earth outline
      </div>
    </div>
  )
}

