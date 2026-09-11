import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, Polygon, Marker } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import type { GeoJSON } from '../api'
import SvgLocator from './SvgLocator'

/** GeoJSON overlay on OSM tiles. Falls back to raw geometry on a neutral
 *  canvas when tiles are unavailable (air-gapped mode). */
export default function MapView({ geo }: { geo: GeoJSON | null }) {
  const [tilesOk, setTilesOk] = useState(true)

  if (!geo || !geo.features.length) {
    return <p className="text-base text-muted">
      No georeferenced overlay for this run (inputs lack CRS bounds or the
      task produced no spatial regions).
    </p>
  }

  const lons: number[] = []
  const lats: number[] = []
  for (const f of geo.features) {
    for (const [x, y] of f.geometry.coordinates[0]) {
      if (Math.abs(x) <= 180 && Math.abs(y) <= 90) { lons.push(x); lats.push(y) }
    }
  }
  const valid = lons.length > 0
  const center: [number, number] = valid
    ? [(Math.min(...lats) + Math.max(...lats)) / 2,
       (Math.min(...lons) + Math.max(...lons)) / 2]
    : [0, 0]

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 font-mono text-[15.5px] text-muted">
        <span>CRS: {geo.crs ?? 'pixel space'}</span>
        <span>·</span>
        <span>{geo.features.length} feature(s)</span>
      </div>
      <div className="overflow-hidden rounded-lg border border-line" style={{ height: 380 }}>
        {tilesOk ? (
          <MapContainer center={center} zoom={valid ? 13 : 5}
            style={{ height: '100%', width: '100%', background: '#0f172a' }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="© OpenStreetMap"
              eventHandlers={{ tileerror: () => setTilesOk(false) }} />
            {geo.features.map((f, i) => (
              <Polygon key={i}
                positions={f.geometry.coordinates[0].map(([x, y]) => [y, x] as [number, number])}
                pathOptions={{
                  color: f.properties.kind === 'change' ? '#F59E0B' : '#4C8DF6',
                  fillColor: f.properties.kind === 'change' ? '#F59E0B' : '#4C8DF6',
                  fillOpacity: 0.25, weight: 2,
                }} />
            ))}
          </MapContainer>
        ) : (
          <SvgLocator geo={geo} />
        )}
      </div>
    </div>
  )
}
