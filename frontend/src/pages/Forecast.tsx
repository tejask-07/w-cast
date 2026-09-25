import { useEffect, useRef, useState } from 'react'
import {
  MapContainer,
  TileLayer,
  useMap,
} from 'react-leaflet'
import type {
  LeafletEvent,
  Map as LeafletMap,
  Rectangle as LeafletRectangle,
} from 'leaflet'
import '@geoman-io/leaflet-geoman-free'
import 'leaflet/dist/leaflet.css'
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css'

import { useNavigate } from 'react-router-dom'

import { useForecast } from '../hooks/useForecast'
import { searchLocations } from '../services/geocoding'
import type { ForecastLocation, ForecastVariable } from '../types/forecast'

import './Forecast.css'

type AOI = {
  north: number
  south: number
  east: number
  west: number
}

type GeomanLayer = LeafletRectangle & {
  pm: {
    enable: (options: {
      allowSelfIntersection: boolean
      draggable: boolean
    }) => void
  }
}

type GeomanMap = LeafletMap & {
  pm: {
    disableDraw: () => void
    enableDraw: (shape: string, options: Record<string, unknown>) => void
  }
}

const defaultLocation: ForecastLocation = {
  name: 'SELECT A LOCATION',
  lat: 20.5937,
  lon: 78.9629,
}

/* ================================================= */
/* AOI HELPERS */
/* ================================================= */

function getAOI(
  layer: LeafletRectangle,
): AOI {
  const bounds = layer.getBounds()

  return {
    north: bounds.getNorth(),
    south: bounds.getSouth(),
    east: bounds.getEast(),
    west: bounds.getWest(),
  }
}

/* ================================================= */
/* MAP CONTROLLER */
/* ================================================= */

function MapController({
  location,
  startDrawing,
  onAOIChange,
  onDrawingChange,
}: {
  location: ForecastLocation
  startDrawing: number
  onAOIChange: (aoi: AOI | null) => void
  onDrawingChange: (drawing: boolean) => void
}) {
  const map = useMap()

  const mapRef = useRef<LeafletMap>(map)

  const aoiLayerRef =
    useRef<LeafletRectangle | null>(null)

  useEffect(() => {
    mapRef.current = map
  }, [map])

  /* --------------------------------------------- */
  /* LOCATION CHANGE                               */

  useEffect(() => {
    map.flyTo(
      [location.lat, location.lon],
      9,
      {
        duration: 0.7,
      },
    )

    if (aoiLayerRef.current) {
      map.removeLayer(
        aoiLayerRef.current,
      )

      aoiLayerRef.current = null
    }

    onAOIChange(null)
    onDrawingChange(false)
  }, [
    location,
    map,
    onAOIChange,
    onDrawingChange,
  ])

  /* --------------------------------------------- */
  /* GEOMAN EVENTS                                 */
  /* --------------------------------------------- */

  useEffect(() => {
    const leafletMap = mapRef.current

    const handleCreate = (
      event: LeafletEvent & { layer: LeafletRectangle },
    ) => {
      const layer =
        event.layer as LeafletRectangle

      if (aoiLayerRef.current) {
        leafletMap.removeLayer(
          aoiLayerRef.current,
        )
      }

      aoiLayerRef.current = layer

      ;(layer as GeomanLayer).pm.enable({
        allowSelfIntersection: false,
        draggable: true,
      })

      onAOIChange(
        getAOI(layer),
      )

      onDrawingChange(false)

      ;(leafletMap as GeomanMap).pm.disableDraw()
    }

    const handleEdit = (
      event: LeafletEvent & { layer: LeafletRectangle },
    ) => {
      const layer =
        event.layer as LeafletRectangle

      if (
        layer ===
        aoiLayerRef.current
      ) {
        onAOIChange(
          getAOI(layer),
        )
      }
    }

    const handleRemove = (
      event: LeafletEvent & { layer: LeafletRectangle },
    ) => {
      const layer =
        event.layer as LeafletRectangle

      if (
        layer ===
        aoiLayerRef.current
      ) {
        aoiLayerRef.current = null

        onAOIChange(null)
        onDrawingChange(false)
      }
    }

    const handleDrawStart = () => {
      onDrawingChange(true)
    }

    leafletMap.on(
      'pm:create',
      handleCreate,
    )

    leafletMap.on(
      'pm:edit',
      handleEdit,
    )

    leafletMap.on(
      'pm:remove',
      handleRemove,
    )

    leafletMap.on(
      'pm:drawstart',
      handleDrawStart,
    )

    return () => {
      leafletMap.off(
        'pm:create',
        handleCreate,
      )

      leafletMap.off(
        'pm:edit',
        handleEdit,
      )

      leafletMap.off(
        'pm:remove',
        handleRemove,
      )

      leafletMap.off(
        'pm:drawstart',
        handleDrawStart,
      )
    }
  }, [
    onAOIChange,
    onDrawingChange,
  ])

  /* --------------------------------------------- */
  /* START AOI DRAWING                             */
  /* --------------------------------------------- */

  useEffect(() => {
    if (!startDrawing) {
      return
    }

    const leafletMap = mapRef.current

    if (aoiLayerRef.current) {
      leafletMap.removeLayer(
        aoiLayerRef.current,
      )

      aoiLayerRef.current = null
    }

    onAOIChange(null)

    ;(leafletMap as GeomanMap).pm.enableDraw(
      'Rectangle',
      {
        snappable: false,

        allowSelfIntersection: false,

        templineStyle: {
          color: '#111111',
          weight: 1,
          dashArray: '5 5',
        },

        hintlineStyle: {
          color: '#111111',
          weight: 1,
          dashArray: '5 5',
        },

        pathOptions: {
          color: '#f7f6f2',
          weight: 1,
          opacity: 1,
          fillColor: '#f7f6f2',
          fillOpacity: 0.06,
        },
      },
    )

    onDrawingChange(true)
  }, [
    startDrawing,
    onAOIChange,
    onDrawingChange,
  ])

  return null
}

/* ================================================= */
/* FORECAST */
/* ================================================= */

function Forecast() {

  const navigate = useNavigate()

  const [
    selectedLocation,
    setSelectedLocation,
  ] = useState<ForecastLocation | null>(null)

  const [
    searchQuery,
    setSearchQuery,
  ] = useState('')

  const [
    locationSuggestions,
    setLocationSuggestions,
  ] = useState<ForecastLocation[]>([])

  const [
    searchingLocations,
    setSearchingLocations,
  ] = useState(false)

  const [
    locationSearchError,
    setLocationSearchError,
  ] = useState(false)

  const [
    locationSelected,
    setLocationSelected,
  ] = useState(false)

  const [
    variable,
    setVariable,
  ] = useState<ForecastVariable>('temperature')

  const [
    leadTime,
    setLeadTime,
  ] = useState(24)

  const [
    aoi,
    setAoi,
  ] = useState<AOI | null>(null)

  const [
    drawingAOI,
    setDrawingAOI,
  ] = useState(false)

  const [
    drawRequest,
    setDrawRequest,
  ] = useState(0)

  const {
    loading,
    error,
    fetchForecast,
  } = useForecast()

  const searchRequest = useRef(0)

  useEffect(() => {
    const query = searchQuery.trim()

    if (locationSelected || query.length < 2) {
      return
    }

    const requestId = searchRequest.current + 1
    searchRequest.current = requestId
    const controller = new AbortController()
    const timeout = window.setTimeout(() => {
      setSearchingLocations(true)
      setLocationSearchError(false)

      void searchLocations(query, controller.signal)
        .then((results) => {
          if (requestId !== searchRequest.current) return
          setLocationSuggestions(results)
        })
        .catch(() => {
          if (controller.signal.aborted || requestId !== searchRequest.current) return
          setLocationSuggestions([])
          setLocationSearchError(true)
        })
        .finally(() => {
          if (requestId === searchRequest.current) {
            setSearchingLocations(false)
          }
        })
    }, 350)

    return () => {
      window.clearTimeout(timeout)
      controller.abort()
    }
  }, [locationSelected, searchQuery])

  const mapLocation = selectedLocation ?? defaultLocation

  const forecastLocation = aoi
    ? {
      name: 'AOI CENTER',
      lat: (aoi.south + aoi.north) / 2,
      lon: (aoi.west + aoi.east) / 2,
    }
    : selectedLocation

  const selectAOI = () => {
    setAoi(null)

    setDrawRequest(
      (value) => value + 1,
    )
  }

  const handleSearchChange = (
    value: string,
  ) => {
    setSearchQuery(value)
    setLocationSelected(false)
    setSelectedLocation(null)
    setLocationSuggestions([])
    setSearchingLocations(false)
    setAoi(null)
    setDrawingAOI(false)
    setLocationSearchError(false)
  }

  const handleLocationSelect = (
    location: ForecastLocation,
  ) => {
    setSelectedLocation(location)
    setSearchQuery(location.name ?? '')
    setLocationSuggestions([])
    setLocationSelected(true)
    setAoi(null)
    setDrawingAOI(false)
  }

  const generateForecast = async () => {
    if (!forecastLocation) return

    try {
      const forecast = await fetchForecast({
        lat: forecastLocation.lat,
        lon: forecastLocation.lon,
        lead_hours: leadTime,
        variable,
      })

      navigate('/forecast/results', {
        state: {
          location: {
            name: forecastLocation.name,
            lat: forecastLocation.lat,
            lon: forecastLocation.lon,
          },
          aoi,
          variable,
          lead_hours: leadTime,
          forecast,
        },
      })
    } catch {
      // The hook exposes the request error in the existing form surface.
    }
  }

  return (
    <main className="forecast-page">

      {/* ======================================== */}
      {/* HEADER */}
      {/* ======================================== */}

      <header className="forecast-header">

        <div className="brand">
          LEFORECAST
        </div>

        <div className="header-divider" />

        <div className="header-meta">
          INDIA
          <span>/</span>
          WEATHER FORECAST INTELLIGENCE
        </div>

        <div className="header-right">
          REAL DATA. BETTER DECISIONS.
        </div>

      </header>

      {/* ======================================== */}
      {/* WORKSPACE */}
      {/* ======================================== */}

      <section className="forecast-workspace">

        {/* ====================================== */}
        {/* SIDEBAR */}
        {/* ====================================== */}

        <aside className="forecast-sidebar">

          <div className="forecast-heading">

            <h1>
              FORECAST
            </h1>

          </div>

          {/* LOCATION */}

          <section className="parameter">

            <div className="parameter-label">
              LOCATION
            </div>

            <div className="location-search">

              <input
                type="search"
                value={searchQuery}
                placeholder="SEARCH LOCATION"
                aria-label="Search location"
                onChange={(event) =>
                  handleSearchChange(event.target.value)
                }
              />

              {!locationSelected && searchQuery.trim().length >= 2 && (
                <div
                  className="location-suggestions"
                  role="listbox"
                >
                  {searchingLocations && (
                    <div className="location-search-state">
                      SEARCHING LOCATIONS
                    </div>
                  )}

                  {!searchingLocations && locationSearchError && (
                    <div className="location-search-state">
                      LOCATION SEARCH UNAVAILABLE
                    </div>
                  )}

                  {!searchingLocations
                    && !locationSearchError
                    && locationSuggestions.length === 0 && (
                    <div className="location-search-state">
                      NO LOCATIONS FOUND
                    </div>
                  )}

                  {locationSuggestions.map((location) => (
                    <button
                      type="button"
                      className="location-suggestion"
                      role="option"
                      key={`${location.name}-${location.lat}-${location.lon}`}
                      onClick={() => handleLocationSelect(location)}
                    >
                      {location.name}
                    </button>
                  ))}
                </div>
              )}

            </div>

            <div className="location-coordinates">

              <span>
                {selectedLocation
                  ? `${selectedLocation.lat.toFixed(4)}° N`
                  : '—'}
              </span>

              <span>
                {selectedLocation
                  ? `${selectedLocation.lon.toFixed(4)}° E`
                  : '—'}
              </span>

            </div>

          </section>

          {/* AOI */}

          <section className="parameter aoi-parameter">

            <div className="parameter-header">

              <span className="parameter-label">
                AREA OF INTEREST
              </span>

              <span
                className={
                  aoi
                    ? 'aoi-state selected'
                    : 'aoi-state'
                }
              >
                {aoi
                  ? 'SELECTED'
                  : 'REQUIRED'}
              </span>

            </div>

            {!aoi && !drawingAOI && (
              <div className="aoi-sidebar-note">
                SELECT AN AREA USING
                <br />
                THE MAP TOOL.
              </div>
            )}

            {drawingAOI && !aoi && (
              <div className="aoi-instruction">

                <div className="instruction-number">
                  01
                </div>

                <div>
                  <strong>
                    DRAW AREA
                  </strong>

                  <p>
                    Click and drag across
                    <br />
                    the satellite map.
                  </p>
                </div>

              </div>
            )}

            {aoi && (
              <div className="aoi-data">

                <div className="aoi-row">
                  <span>NORTH</span>

                  <strong>
                    {aoi.north.toFixed(4)}°
                  </strong>
                </div>

                <div className="aoi-row">
                  <span>SOUTH</span>

                  <strong>
                    {aoi.south.toFixed(4)}°
                  </strong>
                </div>

                <div className="aoi-row">
                  <span>WEST</span>

                  <strong>
                    {aoi.west.toFixed(4)}°
                  </strong>
                </div>

                <div className="aoi-row">
                  <span>EAST</span>

                  <strong>
                    {aoi.east.toFixed(4)}°
                  </strong>
                </div>

              </div>
            )}

          </section>

          {/* WEATHER VARIABLE */}

          <section className="parameter">

            <div className="parameter-label">
              WEATHER VARIABLE
            </div>

            <select
              value={variable}
              onChange={(event) =>
                setVariable(
                  event.target.value as ForecastVariable,
                )
              }
            >
              <option value="temperature">
                Temperature
              </option>

              <option value="rainfall">
                Rainfall
              </option>

              <option value="wind_speed">
                Wind Speed
              </option>
            </select>

          </section>

          {/* LEAD TIME */}

          <section className="parameter">

            <div className="parameter-label">
              LEAD TIME
            </div>

            <div className="lead-options">

              {[24, 48, 72].map(
                (hours) => (
                  <button
                    key={hours}
                    type="button"
                    className={
                      leadTime === hours
                        ? 'active'
                        : ''
                    }
                    onClick={() =>
                      setLeadTime(hours)
                    }
                  >
                    {hours}H
                  </button>
                ),
              )}

            </div>

          </section>

          {/* MODEL */}

          <section className="parameter">

            <div className="parameter-label">
              MODEL BLEND
            </div>

            <div className="model-blend">
              <span>◆</span>
              W-CAST / ADAPTIVE
            </div>

          </section>

          {/* GENERATE */}

          <button
            type="button"
            className="generate-button"
            disabled={!selectedLocation || loading}
            onClick={generateForecast}
          >
            <span>→</span>
            {loading ? 'LOADING FORECAST' : 'GENERATE FORECAST'}
          </button>

          {error && (
            <div className="forecast-request-error" role="alert">
              {error}
              <br />
              TRY GENERATING AGAIN.
            </div>
          )}

        </aside>

        {/* ====================================== */}
        {/* MAP WORKSPACE */}
        {/* ====================================== */}

        <section className="map-workspace">

          {/* MAP HEADER */}

          <div className="map-header">

            <div>

              <span>
                SPATIAL FORECAST INPUT
              </span>

              <strong>
                {(mapLocation.name ?? 'SELECT A LOCATION').toUpperCase()}
              </strong>

            </div>

          </div>

          {/* ==================================== */}
          {/* MAP TOOLBAR */}
          {/* ==================================== */}

          <div className="map-toolbar">

            <div className="map-toolbar-left">

              <span className="toolbar-index">
                AOI
              </span>

              <span className="toolbar-description">
                AREA OF INTEREST
              </span>

            </div>

            <div className="map-toolbar-right">

              {drawingAOI && (
                <span className="drawing-status">
                  CLICK + DRAG ON MAP
                </span>
              )}

              {aoi && !drawingAOI && (
                <span className="selected-status">
                  AOI SELECTED
                </span>
              )}

              <button
                type="button"
                className={
                  drawingAOI
                    ? 'aoi-toolbar-button active'
                    : 'aoi-toolbar-button'
                }
                onClick={selectAOI}
              >
                <span>
                  {aoi ? '↻' : '+'}
                </span>

                {aoi
                  ? 'REDRAW AOI'
                  : 'SELECT AOI'}
              </button>

            </div>

          </div>

          {/* ==================================== */}
          {/* MAP */}
          {/* ==================================== */}

          <div
            className={
              drawingAOI
                ? 'map-frame drawing'
                : 'map-frame'
            }
          >

            <MapContainer
              center={[
                mapLocation.lat,
                mapLocation.lon,
              ]}
              zoom={9}
              scrollWheelZoom
              zoomControl
              className="forecast-map"
            >

              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles © Esri"
                maxZoom={19}
              />

              <MapController
                location={mapLocation}
                startDrawing={drawRequest}
                onAOIChange={setAoi}
                onDrawingChange={
                  setDrawingAOI
                }
              />

            </MapContainer>

            {/* MAP INFO */}

            <div className="map-info">

              <span>
                INDIA
              </span>

              <span>
                LEFORECAST
              </span>

              <span>
                SPATIAL INPUT
              </span>

            </div>

            {/* LOCATION */}

            <div className="map-location">

              <strong>
                {(mapLocation.name ?? 'SELECT A LOCATION').toUpperCase()}
              </strong>

              <span>
                {mapLocation.lat.toFixed(4)}° N
                &nbsp;&nbsp;
                {mapLocation.lon.toFixed(4)}° E
              </span>

            </div>

            {/* DRAWING STATUS */}

            {drawingAOI && (
              <div className="drawing-overlay">

                <span>
                  AOI SELECTION
                </span>

                <strong>
                  CLICK + DRAG
                </strong>

              </div>
            )}

          </div>

          {/* ==================================== */}
          {/* AOI DATA BAR */}
          {/* ==================================== */}

          <div className="aoi-bar">

            <div className="aoi-bar-title">

              <span>
                CURRENT AOI
              </span>

              <strong>
                {aoi
                  ? 'CUSTOM REGION'
                  : 'NO AREA SELECTED'}
              </strong>

            </div>

            <div>
              <span>N</span>

              <strong>
                {aoi
                  ? `${aoi.north.toFixed(4)}°`
                  : '—'}
              </strong>
            </div>

            <div>
              <span>S</span>

              <strong>
                {aoi
                  ? `${aoi.south.toFixed(4)}°`
                  : '—'}
              </strong>
            </div>

            <div>
              <span>W</span>

              <strong>
                {aoi
                  ? `${aoi.west.toFixed(4)}°`
                  : '—'}
              </strong>
            </div>

            <div>
              <span>E</span>

              <strong>
                {aoi
                  ? `${aoi.east.toFixed(4)}°`
                  : '—'}
              </strong>
            </div>

          </div>

        </section>

      </section>

      {/* ======================================== */}
      {/* FOOTER */}
      {/* ======================================== */}

      <footer className="forecast-footer">

        <span>
          SAME SKY. A STRONGER TOMORROW.
        </span>

        <span>
          INDIA
          <b>/</b>
          DATA
          <b>/</b>
          IMPACT
        </span>

      </footer>

    </main>
  )
}

export default Forecast