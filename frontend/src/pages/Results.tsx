import { useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import 'leaflet/dist/leaflet.css'
import './Results.css'

import ForecastMap from '../components/map/ForecastMap'
import { useForecast } from '../hooks/useForecast'
import type { ForecastResponse, ForecastVariable } from '../types/forecast'

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
  variable: ForecastVariable
  lead_hours: number
  forecast: ForecastResponse
}

function Results() {
  const navigate = useNavigate()
  const location = useLocation()

  const request =
    location.state as ForecastRequest | null

  const {
    hourlyData,
    hourlyLoading,
    hourlyError,
    fetchHourlyForecast,
  } = useForecast()

  useEffect(() => {
    if (!request?.forecast) return

    void fetchHourlyForecast({
      lat: request.location.lat,
      lon: request.location.lon,
      lead_hours: request.lead_hours,
      variable: request.variable,
    }).catch(() => undefined)
  }, [
    fetchHourlyForecast,
    request?.forecast,
    request?.lead_hours,
    request?.location.lat,
    request?.location.lon,
    request?.variable,
  ])

  /*
   * If somebody opens /forecast/results directly,
   * there is no forecast request yet.
   */
  if (!request || !request.forecast) {
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

  const forecast = request.forecast
  const variableLabel = request.variable === 'wind_speed'
    ? 'WIND SPEED'
    : request.variable.toUpperCase()

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
                {variableLabel}
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
                REAL DATA
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

                <ForecastMap
                  latitude={request.location.lat}
                  longitude={request.location.lon}
                  city={request.location.name}
                  leadHours={request.lead_hours}
                  variable={request.variable}
                  aoi={request.aoi}
                  initialZoom={9}
                  showChrome={false}
                  satellite
                />

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
                  W-CAST FORECAST
                  <br />
                  {forecast.forecast[request.variable].toFixed(2)} {request.variable === 'temperature' ? '°C' : request.variable === 'rainfall' ? 'mm' : 'm/s'}
                </strong>

                <p>
                  {forecast.regime} REGIME / {forecast.extremes.risk_level.toUpperCase()} RISK
                </p>

              </div>

              <div className="key-metrics">

                <div>
                  <span>
                    PRECIPITATION
                  </span>

                  <strong>
                    {forecast.forecast.rainfall.toFixed(2)} mm
                  </strong>
                </div>

                <div>
                  <span>
                    WIND SPEED
                  </span>

                  <strong>
                    {(forecast.forecast.wind_speed * 3.6).toFixed(2)} km/h
                  </strong>
                </div>

                <div>
                  <span>
                    TEMPERATURE
                  </span>

                  <strong>
                    {forecast.forecast.temperature.toFixed(2)} °C
                  </strong>
                </div>

                <div>
                  <span>
                    REGIME
                  </span>

                  <strong>
                    {forecast.regime}
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
                [ {request.variable === 'temperature'
                  ? '°C'
                  : request.variable === 'rainfall'
                    ? 'mm'
                    : 'm/s'} ]
              </span>

            </div>

            <div className="forecast-line-chart">
              {hourlyLoading && (
                <div className="chart-state">LOADING HOURLY FORECAST</div>
              )}

              {hourlyError && (
                <div className="chart-state">HOURLY FORECAST UNAVAILABLE</div>
              )}

              {!hourlyLoading && !hourlyError && hourlyData && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={hourlyData.points} margin={{ top: 12, right: 12, bottom: 6, left: 0 }}>
                    <CartesianGrid stroke="#dddddd" strokeDasharray="2 3" />
                    <XAxis
                      dataKey="hour"
                      tick={{ fontSize: 8, fill: '#777' }}
                      tickLine={false}
                      axisLine={{ stroke: '#999' }}
                      tickFormatter={(hour: number) => `${hour}H`}
                      interval={Math.max(0, Math.ceil(hourlyData.points.length / 8) - 1)}
                    />
                    <YAxis
                      tick={{ fontSize: 8, fill: '#777' }}
                      tickLine={false}
                      axisLine={{ stroke: '#999' }}
                      width={42}
                      tickFormatter={(value: number) => value.toFixed(1)}
                    />
                    <Tooltip
                      contentStyle={{
                        border: '1px solid #111',
                        borderRadius: 0,
                        fontFamily: 'inherit',
                        fontSize: 10,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name={variableLabel}
                      stroke="#111111"
                      strokeWidth={1.5}
                      dot={{ r: 2, fill: '#111111' }}
                      activeDot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>

          </section>

          {/* ============================================= */}
          {/* SECONDARY METRICS */}
          {/* ============================================= */}

          <div className="metric-grid">

            <section className="metric-panel">

              <div className="metric-panel-heading">
                <strong>
                  PRECIPITATION
                </strong>

                <span>
                  [ mm ]
                </span>
              </div>

              <div className="metric-unavailable">
                {forecast.forecast.rainfall.toFixed(2)} mm
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

              <div className="metric-unavailable">
                {(forecast.forecast.wind_speed * 3.6).toFixed(2)} km/h
              </div>

            </section>

            <section className="metric-panel">

              <div className="metric-panel-heading">
                <strong>
                  TEMPERATURE
                </strong>

                <span>
                  [ °C ]
                </span>
              </div>

              <div className="metric-unavailable">
                {forecast.forecast.temperature.toFixed(2)} °C
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
                  MODEL WEIGHTS
                </strong>

                <span>
                  [ DATA ]
                </span>

              </div>

              <div className="model-row">
                <span>
                  GFS
                </span>

                <div className="model-bar">
                  <i style={{ width: `${forecast.weights.gfs * 100}%` }} />
                </div>

                <strong>
                  {(forecast.weights.gfs * 100).toFixed(1)}%
                </strong>
              </div>

              <div className="model-row">
                <span>
                  GEFS
                </span>

                <div className="model-bar">
                  <i style={{ width: `${forecast.weights.gefs * 100}%` }} />
                </div>

                <strong>
                  {(forecast.weights.gefs * 100).toFixed(1)}%
                </strong>
              </div>

              <div className="comparison-note">
                W-CAST IS THE ADAPTIVE BLEND
                <br />
                GFS + GEFS WEIGHTS FROM BACKEND
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
                  REGIME: {forecast.regime}
                  <br />
                  HEAVY RAIN: {forecast.extremes.heavy_rain ? 'YES' : 'NO'}
                  <br />
                  HEAT WAVE: {forecast.extremes.heat_wave ? 'YES' : 'NO'}
                  <br />
                  HIGH WIND: {forecast.extremes.high_wind ? 'YES' : 'NO'}
                  <br />
                  RISK: {forecast.extremes.risk_level.toUpperCase()}
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