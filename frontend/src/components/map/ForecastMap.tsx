import {
  CircleMarker,
  MapContainer,
  TileLayer,
  ZoomControl,
  useMap,
} from 'react-leaflet'
import { useEffect } from 'react'
import 'leaflet/dist/leaflet.css'
import './ForecastMap.css'

type ForecastMapProps = {
  latitude: number
  longitude: number
  city: string
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
}: ForecastMapProps) {
  return (
    <div className="forecast-map">
      <MapContainer
        center={[latitude, longitude]}
        zoom={5}
        scrollWheelZoom
        zoomControl={false}
        className="leaflet-map"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

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
    </div>
  )
}

export default ForecastMap