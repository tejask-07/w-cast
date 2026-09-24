import {
  CircleMarker,
  ImageOverlay,
  MapContainer,
  Rectangle,
  TileLayer,
  ZoomControl,
  useMap,
} from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import { useEffect } from 'react'
import 'leaflet/dist/leaflet.css'
import './ForecastMap.css'

import { getApiUrl } from '../../services/api'
import { useForecast } from '../../hooks/useForecast'
import type {
  ForecastAOI,
  ForecastVariable,
  SpatialForecastResponse,
} from '../../types/forecast'

type ForecastMapProps = {
  latitude: number
  longitude: number
  city: string
  leadHours?: number
  variable?: ForecastVariable
  overlay?: SpatialForecastResponse | null
  aoi?: ForecastAOI
  initialZoom?: number
  showChrome?: boolean
  satellite?: boolean
}

function MapRecenter({
  latitude,
  longitude,
}: {
  latitude: number
  longitude: number
}) {
  const map = useMap()

  useEffect(() => {
    map.flyTo([latitude, longitude], 6, {
      duration: 0.8,
    })
  }, [latitude, longitude, map])

  return null
}

function ForecastMap({
  latitude,
  longitude,
  city,
  leadHours,
  variable = 'temperature',
  overlay = null,
  aoi,
  initialZoom = 5,
  showChrome = true,
  satellite = false,
}: ForecastMapProps) {
  const {
    mapData,
    mapLoading,
    mapError,
    fetchForecastMap,
  } = useForecast()

  useEffect(() => {
    if (leadHours === undefined) return

    void fetchForecastMap({
      lat: latitude,
      lon: longitude,
      lead_hours: leadHours,
      variable,
      ...aoi,
    }).catch(() => undefined)
  }, [aoi, fetchForecastMap, latitude, longitude, leadHours, variable])

  const spatialOverlay = leadHours === undefined ? overlay : mapData
  const overlayBounds: LatLngBoundsExpression | undefined = spatialOverlay?.bounds
  const overlayUrl = spatialOverlay ? getApiUrl(spatialOverlay.image_url) : undefined
  const legendVariable = spatialOverlay?.variable ?? variable
  const legend = {
    temperature: { label: 'TEMPERATURE', unit: '°C', low: 'COOL', high: 'WARM' },
    rainfall: { label: 'RAINFALL', unit: 'mm', low: 'LOW', high: 'HIGH' },
    wind_speed: { label: 'WIND SPEED', unit: 'm/s', low: 'LOW', high: 'HIGH' },
  }[legendVariable]
  const aoiBounds: LatLngBoundsExpression | undefined = aoi
    ? [[aoi.south, aoi.west], [aoi.north, aoi.east]]
    : undefined

  return (
    <div className="forecast-map">
      <MapContainer
        center={[latitude, longitude]}
        zoom={initialZoom}
        scrollWheelZoom
        zoomControl={false}
        className="leaflet-map"
      >
        <TileLayer
          attribution={satellite ? 'Tiles &copy; Esri' : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}
          url={satellite
            ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'}
          maxZoom={satellite ? 19 : undefined}
        />

        {spatialOverlay && overlayBounds && overlayUrl && (
          <ImageOverlay
            url={overlayUrl}
            bounds={overlayBounds}
            opacity={0.5}
            interactive={false}
          />
        )}

        {aoiBounds && (
          <Rectangle
            bounds={aoiBounds}
            pathOptions={{
              color: '#f7f6f2',
              weight: 0.5,
              fillColor: '#ffffff',
              fillOpacity: 0.12,
            }}
          />
        )}

        <ZoomControl position="topright" />

        <MapRecenter
          latitude={latitude}
          longitude={longitude}
        />

        <CircleMarker
          center={[latitude, longitude]}
          radius={7}
          pathOptions={{
            color: '#111111',
            fillColor: '#111111',
            fillOpacity: 1,
            weight: 2,
          }}
        />

        <CircleMarker
          center={[latitude, longitude]}
          radius={14}
          pathOptions={{
            color: '#111111',
            fillColor: 'transparent',
            fillOpacity: 0,
            weight: 1,
          }}
        />
      </MapContainer>

      {showChrome && (
        <>
          <div className="map-location-label">
            <span>{city.toUpperCase()}</span>

            <span>
              {latitude.toFixed(4)}° N&nbsp;&nbsp;
              {longitude.toFixed(4)}° E
            </span>
          </div>

          <div className="map-info">
            <span>INDIA</span>
            <span>LEFORECAST</span>
            <span>LOCATION INPUT</span>
          </div>
        </>
      )}

      {mapLoading && (
        <div className="forecast-map-status">LOADING SPATIAL FIELD</div>
      )}

      {mapError && (
        <div className="forecast-map-status">SPATIAL FIELD UNAVAILABLE</div>
      )}

      {spatialOverlay && legend && (
        <div className="forecast-map-legend">
          <strong>{legend.label} ({spatialOverlay.unit})</strong>
          <div className={`legend-scale legend-${legendVariable}`} />
          <div className="legend-labels">
            <span>{spatialOverlay.min_value.toFixed(1)}</span>
            <span>{spatialOverlay.max_value.toFixed(1)}</span>
          </div>
          <div className="legend-qualitative">
            <span>{legend.low}</span>
            <span>{legend.high}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default ForecastMap