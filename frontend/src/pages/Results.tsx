import { useEffect, useState } from 'react'
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
  aoi: AOI | null
  variable: ForecastVariable
  lead_hours: number
  forecast: ForecastResponse
}

type SeriesPoint = { hour: number; value: number | null }

const hourlyMetricConfig: Record<ForecastVariable, {
  label: string
  unit: string
}> = {
  temperature: { label: 'TEMPERATURE', unit: '°C' },
  wind_speed: { label: 'WIND SPEED', unit: 'm/s' },
  rainfall: { label: 'PRECIPITATION', unit: 'mm' },
}

function seriesStats(points: SeriesPoint[]) {
  const values = points
    .map((point) => point.value)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  if (!values.length) return null
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    total: values.reduce((sum, value) => sum + value, 0),
  }
}

function trendFor(points: SeriesPoint[]) {
  const validPoints = points.filter(
    (point): point is { hour: number; value: number } =>
      typeof point.value === 'number' && Number.isFinite(point.value),
  )
  if (validPoints.length < 2) return 'HOURLY SERIES UNAVAILABLE'
  const difference = validPoints[validPoints.length - 1].value - validPoints[0].value
  const tolerance = Math.max(0.05, Math.abs(validPoints[0].value) * 0.01)
  if (difference > tolerance) return 'INCREASING'
  if (difference < -tolerance) return 'DECREASING'
  return 'RELATIVELY STABLE'
}

function Sparkline({ points }: { points: SeriesPoint[] }) {
  const validPoints = points.filter(
    (point): point is { hour: number; value: number } =>
      typeof point.value === 'number' && Number.isFinite(point.value),
  )
  if (validPoints.length < 2) return <div className="sparkline-empty">SELECTED SERIES ONLY</div>
  const values = validPoints.map((point) => point.value)
  const min = Math.min(...values)
  const range = Math.max(...values) - min || 1
  const path = validPoints.map((point, index) => {
    const x = (index / (validPoints.length - 1)) * 100
    const y = 30 - ((point.value - min) / range) * 24
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
  }).join(' ')
  return <svg className="metric-sparkline" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true"><path d={path} fill="none" stroke="currentColor" strokeWidth="1.2" vectorEffect="non-scaling-stroke" /></svg>
}

function Results() {
  const navigate = useNavigate()
  const location = useLocation()

  const request =
    location.state as ForecastRequest | null

  const {
    hourlyData,
    hourlyDataByVariable,
    hourlyLoading,
    hourlyError,
    fetchHourlyForecast,
  } = useForecast()

  const [selectedHourlyMetric, setSelectedHourlyMetric] =
    useState<ForecastVariable>('temperature')

  useEffect(() => {
    if (!request?.forecast) return

    void Promise.all(
      (['temperature', 'rainfall', 'wind_speed'] as ForecastVariable[]).map((variable) =>
        fetchHourlyForecast({
          lat: request.location.lat,
          lon: request.location.lon,
          lead_hours: request.lead_hours,
          variable,
        }),
      ),
    ).catch(() => undefined)
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
  const hourlyPoints = hourlyDataByVariable[request.variable]?.points ?? hourlyData?.points ?? []
  const precipitationPoints = hourlyDataByVariable.rainfall?.points ?? []
  const windPoints = hourlyDataByVariable.wind_speed?.points ?? []
  const temperaturePoints = hourlyDataByVariable.temperature?.points ?? []
  const hourlyStats = seriesStats(hourlyPoints)
  const selectedMetric = hourlyMetricConfig[selectedHourlyMetric]
  const selectedHourlyData = hourlyDataByVariable[selectedHourlyMetric]
    ?? (hourlyData?.variable === selectedHourlyMetric ? hourlyData : undefined)
  const selectedUnit = selectedHourlyData?.unit ?? selectedMetric.unit
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
              {request.aoi ? 'CUSTOM REGION' : 'SELECTED LOCATION'}
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
                {forecast.model_source ?? 'W-CAST (Adaptive)'}
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
                  aoi={request.aoi ?? undefined}
                  initialZoom={9}
                  showChrome={false}
                  satellite
                />

                <div className="results-map-label">

                  <strong>
                    {request.location.name.toUpperCase()}
                  </strong>

                  <span>
                    {request.aoi ? 'AOI / CUSTOM REGION' : 'SELECTED LOCATION'}
                  </span>

                </div>

                {request.aoi && (
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
                )}

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

              <div className="hourly-chart-title">
                <strong>
                  HOURLY FORECAST
                  {' '}
                  (NEXT {request.lead_hours} HOURS)
                </strong>

                <div className="hourly-metric-selector" role="tablist" aria-label="Hourly forecast metric">
                  {(['temperature', 'wind_speed', 'rainfall'] as ForecastVariable[]).map((metric) => (
                    <button
                      key={metric}
                      type="button"
                      className={selectedHourlyMetric === metric ? 'active' : ''}
                      role="tab"
                      aria-selected={selectedHourlyMetric === metric}
                      onClick={() => setSelectedHourlyMetric(metric)}
                    >
                      {hourlyMetricConfig[metric].label}
                    </button>
                  ))}
                </div>
              </div>

              <span>
                [ {selectedMetric.unit} ]
              </span>

            </div>

            <div className="forecast-line-chart">
              {hourlyLoading && (
                <div className="chart-state">LOADING HOURLY FORECAST</div>
              )}

              {hourlyError && (
                <div className="chart-state">HOURLY FORECAST UNAVAILABLE</div>
              )}

              {!hourlyLoading && !hourlyError && selectedHourlyData && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selectedHourlyData.points} margin={{ top: 12, right: 12, bottom: 6, left: 0 }}>
                    <CartesianGrid stroke="#dddddd" strokeDasharray="2 3" />
                    <XAxis
                      dataKey="hour"
                      tick={{ fontSize: 11, fill: '#777' }}
                      tickLine={false}
                      axisLine={{ stroke: '#999' }}
                      tickFormatter={(hour: number) => `${hour}H`}
                      interval={Math.max(0, Math.ceil(selectedHourlyData.points.length / 8) - 1)}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: '#777' }}
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
                        fontSize: 11,
                      }}
                      formatter={(value) => [
                        `${typeof value === 'number' ? value.toFixed(2) : value} ${selectedUnit}`,
                        selectedMetric.label,
                      ]}
                      labelFormatter={(hour) => `${hour}H`}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name={selectedMetric.label}
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

          <section className="summary-section">
            <div className="chart-heading">
              <strong>24-HOUR SUMMARY</strong>
              <span>[ REAL DATA ]</span>
            </div>
            <div className="summary-grid">
              <div className="summary-item">
                <span>TEMPERATURE</span>
                <strong>{request.variable === 'temperature' && hourlyStats ? `${hourlyStats.min.toFixed(1)} → ${hourlyStats.max.toFixed(1)} °C` : `${forecast.forecast.temperature.toFixed(1)} °C`}</strong>
                <small>{request.variable === 'temperature' ? 'HOURLY MIN → MAX' : 'POINT FORECAST'}</small>
              </div>
              <div className="summary-item">
                <span>PRECIPITATION</span>
                <strong>{request.variable === 'rainfall' && hourlyStats ? `${hourlyStats.total.toFixed(1)} mm` : `${forecast.forecast.rainfall.toFixed(1)} mm`}</strong>
                <small>{request.variable === 'rainfall' ? 'HOURLY TOTAL' : 'POINT FORECAST'}</small>
              </div>
              <div className="summary-item">
                <span>WIND</span>
                <strong>{request.variable === 'wind_speed' && hourlyStats ? `${hourlyStats.max.toFixed(1)} m/s` : `${forecast.forecast.wind_speed.toFixed(1)} m/s`}</strong>
                <small>{request.variable === 'wind_speed' ? 'HOURLY MAXIMUM' : 'POINT FORECAST'}</small>
              </div>
              <div className="summary-item">
                <span>RISK</span>
                <strong>{forecast.extremes.risk_level.toUpperCase()}</strong>
                <small>{forecast.regime} REGIME</small>
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
                  PRECIPITATION
                </strong>

                <span>
                  [ mm ]
                </span>
              </div>

              <div className="metric-content">
                <Sparkline points={precipitationPoints} />
                <strong>{forecast.forecast.rainfall.toFixed(2)} mm TOTAL</strong>
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

              <div className="metric-content">
                <Sparkline points={windPoints} />
                <strong>{forecast.forecast.wind_speed.toFixed(2)} m/s MAX</strong>
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

              <div className="metric-content">
                <Sparkline points={temperaturePoints} />
                <strong>
                  {seriesStats(temperaturePoints)
                    ? `${seriesStats(temperaturePoints)!.min.toFixed(2)} → ${seriesStats(temperaturePoints)!.max.toFixed(2)} °C`
                    : `${forecast.forecast.temperature.toFixed(2)} °C`}
                </strong>
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

              <div className="insight-content">
                <div><span>REGIME</span><strong>{forecast.regime}</strong></div>
                <div><span>TEMPERATURE TREND</span><strong>{trendFor(request.variable === 'temperature' ? hourlyPoints : [])}</strong></div>
                <div><span>PRECIPITATION SUMMARY</span><strong>{request.variable === 'rainfall' && hourlyStats ? `${hourlyStats.total.toFixed(1)} mm TOTAL` : `${forecast.forecast.rainfall.toFixed(1)} mm POINT`}</strong></div>
                <div><span>WIND SUMMARY</span><strong>{request.variable === 'wind_speed' && hourlyStats ? `${hourlyStats.max.toFixed(1)} m/s MAX` : `${forecast.forecast.wind_speed.toFixed(1)} m/s POINT`}</strong></div>
                <div><span>EXTREME CONDITIONS</span><strong>{forecast.extremes.heavy_rain ? 'HEAVY RAIN' : forecast.extremes.heat_wave ? 'HEAT WAVE' : forecast.extremes.high_wind ? 'HIGH WIND' : 'NONE FLAGGED'}</strong></div>
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
                {forecast.model_source ?? 'W-CAST (Adaptive)'}
              </div>

              <p>
                {forecast.model_source === 'Global GFS + GEFS'
                  ? 'Global GFS + GEFS baseline forecast.'
                  : <>Adaptive forecast blending using<br />dynamically determined model<br />weights.</>}
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