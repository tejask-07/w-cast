import { useLocation, useNavigate } from 'react-router-dom'
import {
  MapContainer,
  TileLayer,
  Rectangle,
} from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'

import 'leaflet/dist/leaflet.css'
import './Results.css'

type AOI = {
  north: number
  south: number
  east: number
  west: number
}

type ForecastRequest = {
  location: {
    name: string
    lat: number
    lon: number
  }
  aoi: AOI
  variable: string
  lead_hours: number
  model_blend: string
}

function Results() {
  const navigate = useNavigate()
  const location = useLocation()

  const request =
    location.state as ForecastRequest | null

  /*
   * If somebody opens /forecast/results directly,
   * there is no forecast request yet.
   */
  if (!request) {
    return (
      <main className="results-page">

        <header className="results-header">
          <div className="results-brand">
            LEFORECAST
          </div>

          <div className="results-header-divider" />

          <div className="results-header-meta">
            INDIA
            <span>/</span>
            WEATHER FORECAST INTELLIGENCE
          </div>
        </header>

        <section className="results-empty">

          <span className="results-page-index">
            03 / 03
          </span>

          <h1>
            NO FORECAST
          </h1>

          <p>
            SELECT A LOCATION AND AREA OF INTEREST
            <br />
            BEFORE GENERATING A FORECAST.
          </p>

          <button
            type="button"
            onClick={() => navigate('/forecast')}
          >
            ← RETURN TO FORECAST SETUP
          </button>

        </section>

      </main>
    )
  }

  const bounds: LatLngBoundsExpression = [
    [
      request.aoi.south,
      request.aoi.west,
    ],
    [
      request.aoi.north,
      request.aoi.east,
    ],
  ]

  return (
    <main className="results-page">

      {/* ================================================= */}
      {/* HEADER */}
      {/* ================================================= */}

      <header className="results-header">

        <div className="results-brand">
          LEFORECAST
        </div>

        <div className="results-header-divider" />

        <div className="results-header-meta">
          INDIA
          <span>/</span>
          WEATHER FORECAST INTELLIGENCE
        </div>

        <div className="results-header-actions">

          <button
            type="button"
            onClick={() => navigate('/forecast')}
          >
            ← NEW FORECAST
          </button>

        </div>

      </header>

      {/* ================================================= */}
      {/* MAIN RESULTS */}
      {/* ================================================= */}

      <section className="results-layout">

        {/* =============================================== */}
        {/* LEFT NAVIGATION */}
        {/* =============================================== */}

        <aside className="results-sidebar">

          <div className="results-title">

            <h1>
              FORECAST
              <br />
              RESULTS
            </h1>

            <strong>
              {request.location.name.toUpperCase()}
            </strong>

            {/* <i /> */}

          </div>

          <nav className="results-nav">

            <button
              type="button"
              className="active"
            >
              <span>→</span>
              OVERVIEW
            </button>

            <button type="button">
              <span>│</span>
              VARIABLES
            </button>

            <button type="button">
              <span>│</span>
              MAP LAYERS
            </button>

            <button type="button">
              <span>│</span>
              MODEL INSIGHTS
            </button>

          </nav>

          <div className="results-sidebar-footer">

            <span>
              AREA OF INTEREST
            </span>

            <strong>
              CUSTOM REGION
            </strong>

          </div>

        </aside>

        {/* =============================================== */}
        {/* RESULTS CONTENT */}
        {/* =============================================== */}

        <section className="results-content">

          {/* ============================================= */}
          {/* FORECAST META */}
          {/* ============================================= */}

          <div className="results-meta">

            <div className="results-location">

              <h2>
                {request.location.name}
              </h2>

              <span>
                {request.location.lat.toFixed(4)}° N
                &nbsp;&nbsp;
                {request.location.lon.toFixed(4)}° E
              </span>

              <i />

            </div>

            <div>
              <span>VARIABLE</span>
              <strong>
                {request.variable}
              </strong>
            </div>

            <div>
              <span>LEAD TIME</span>
              <strong>
                {request.lead_hours} Hours
              </strong>
            </div>

            <div>
              <span>MODEL BLEND</span>
              <strong>
                W-CAST (Adaptive)
              </strong>
            </div>

            <div>
              <span>STATUS</span>
              <strong>
                AWAITING FORECAST
              </strong>
            </div>

          </div>

          {/* ============================================= */}
          {/* TOP GRID */}
          {/* ============================================= */}

          <div className="results-top-grid">

            {/* ------------------------------------------- */}
            {/* MAP */}
            {/* ------------------------------------------- */}

            <section className="results-map-card">

              <div className="results-map-header">

                <span>
                  FORECAST AREA
                </span>

                <div>
                  <span className="map-dot" />
                  AOI
                </div>

              </div>

              <div className="results-map">

                <MapContainer
                  center={[
                    request.location.lat,
                    request.location.lon,
                  ]}
                  zoom={9}
                  scrollWheelZoom
                  zoomControl
                  className="results-leaflet"
                >

                  <TileLayer
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    attribution="Tiles © Esri"
                    maxZoom={19}
                  />

                  <Rectangle
                    bounds={bounds}
                    pathOptions={{
                      color: '#ffffff',
                      weight: 2,
                      fillColor: '#ffffff',
                      fillOpacity: 0.12,
                    }}
                  />

                </MapContainer>

                <div className="results-map-label">

                  <strong>
                    {request.location.name.toUpperCase()}
                  </strong>

                  <span>
                    AOI / CUSTOM REGION
                  </span>

                </div>

                <div className="results-map-coordinates">

                  <span>
                    N {request.aoi.north.toFixed(4)}°
                  </span>

                  <span>
                    S {request.aoi.south.toFixed(4)}°
                  </span>

                  <span>
                    W {request.aoi.west.toFixed(4)}°
                  </span>

                  <span>
                    E {request.aoi.east.toFixed(4)}°
                  </span>

                </div>

              </div>

            </section>

            {/* ------------------------------------------- */}
            {/* KEY FORECAST */}
            {/* ------------------------------------------- */}

            <section className="key-forecast">

              <div className="section-heading">

                <strong>
                  KEY FORECAST
                </strong>

                <span>
                  [{request.lead_hours}H]
                </span>

              </div>

              <div className="forecast-unavailable">

                <span className="forecast-unavailable-symbol">
                  —
                </span>

                <strong>
                  FORECAST DATA
                  <br />
                  UNAVAILABLE
                </strong>

                <p>
                  The W-CAST backend has not returned
                  <br />
                  a forecast for this request yet.
                </p>

              </div>

              <div className="key-metrics">

                <div>
                  <span>
                    PRECIPITATION
                  </span>

                  <strong>
                    N/A
                  </strong>
                </div>

                <div>
                  <span>
                    WIND SPEED
                  </span>

                  <strong>
                    N/A
                  </strong>
                </div>

                <div>
                  <span>
                    HUMIDITY
                  </span>

                  <strong>
                    N/A
                  </strong>
                </div>

                <div>
                  <span>
                    PRESSURE
                  </span>

                  <strong>
                    N/A
                  </strong>
                </div>

              </div>

            </section>

          </div>

          {/* ============================================= */}
          {/* HOURLY FORECAST */}
          {/* ============================================= */}

          <section className="chart-section">

            <div className="chart-heading">

              <strong>
                HOURLY FORECAST
                {' '}
                (NEXT {request.lead_hours} HOURS)
              </strong>

              <span>
                [ {request.variable === 'Temperature'
                  ? '°C'
                  : request.variable === 'Rainfall'
                    ? 'mm'
                    : 'km/h'} ]
              </span>

            </div>

            <div className="empty-chart">

              <div className="chart-grid-lines">
                <span />
                <span />
                <span />
                <span />
              </div>

              <div className="empty-chart-message">
                HISTORICAL / FORECAST SERIES
                <strong>
                  AWAITING BACKEND DATA
                </strong>
              </div>

            </div>

          </section>

          {/* ============================================= */}
          {/* SECONDARY METRICS */}
          {/* ============================================= */}

          <div className="metric-grid">

            <section className="metric-panel">

              <div className="metric-panel-heading">
                <strong>
                  PRECIPITATION PROBABILITY
                </strong>

                <span>
                  [ % ]
                </span>
              </div>

              <div className="empty-bars">
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>

              <div className="metric-unavailable">
                DATA UNAVAILABLE
              </div>

            </section>

            <section className="metric-panel">

              <div className="metric-panel-heading">
                <strong>
                  WIND SPEED
                </strong>

                <span>
                  [ km/h ]
                </span>
              </div>

              <div className="empty-bars">
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>

              <div className="metric-unavailable">
                DATA UNAVAILABLE
              </div>

            </section>

            <section className="metric-panel">

              <div className="metric-panel-heading">
                <strong>
                  HUMIDITY
                </strong>

                <span>
                  [ % ]
                </span>
              </div>

              <div className="empty-bars">
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>

              <div className="metric-unavailable">
                DATA UNAVAILABLE
              </div>

            </section>

          </div>

          {/* ============================================= */}
          {/* BOTTOM GRID */}
          {/* ============================================= */}

          <div className="results-bottom-grid">

            {/* MODEL COMPARISON */}

            <section className="model-comparison">

              <div className="bottom-heading">

                <strong>
                  MODEL COMPARISON
                  {' '}
                  ({request.variable.toUpperCase()})
                </strong>

                <span>
                  [ DATA ]
                </span>

              </div>

              <div className="model-row">
                <span>
                  W-CAST (Adaptive)
                </span>

                <div className="model-bar">
                  <i />
                </div>

                <strong>
                  N/A
                </strong>
              </div>

              <div className="model-row">
                <span>
                  GFS
                </span>

                <div className="model-bar">
                  <i />
                </div>

                <strong>
                  N/A
                </strong>
              </div>

              <div className="model-row">
                <span>
                  GEFS
                </span>

                <div className="model-bar">
                  <i />
                </div>

                <strong>
                  N/A
                </strong>
              </div>

              <div className="comparison-note">
                MODEL VERIFICATION DATA
                <br />
                UNAVAILABLE
              </div>

            </section>

            {/* INSIGHTS */}

            <section className="forecast-insights">

              <div className="bottom-heading">

                <strong>
                  FORECAST INSIGHTS
                </strong>

              </div>

              <div className="insight-empty">

                <span>
                  —
                </span>

                <p>
                  Insights will appear here once
                  <br />
                  the backend returns regime,
                  <br />
                  extreme-event and forecast data.
                </p>

              </div>

            </section>

            {/* MODEL DETAILS */}

            <section className="model-details">

              <div className="bottom-heading">

                <strong>
                  MODEL DETAILS
                </strong>

              </div>

              <div className="model-detail-title">
                W-CAST (Adaptive)
              </div>

              <p>
                Adaptive forecast blending using
                <br />
                dynamically determined model
                <br />
                weights.
              </p>

              <i />

              <button type="button">
                VIEW MODEL WEIGHTS →
              </button>

              <button type="button">
                SEE TECHNICAL DETAILS →
              </button>

            </section>

          </div>

        </section>

      </section>

      {/* ================================================= */}
      {/* FOOTER */}
      {/* ================================================= */}

      <footer className="results-footer">

        <span>
          LEFORECAST / FORECAST INTELLIGENCE
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

export default Results