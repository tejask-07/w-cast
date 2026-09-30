import { Rectangle, Tooltip } from 'react-leaflet'

import type { SpatialRiskCell } from '../../types/forecast'

function riskColor(score: number): string {
  if (score <= 30) return '#41836a'
  if (score <= 60) return '#c29a2e'
  if (score <= 80) return '#cf6938'
  return '#a93635'
}

function RiskMapLayer({ cells }: { cells: SpatialRiskCell[] }) {
  return (
    <>
      {cells.map((cell, index) => (
        <Rectangle
          key={`${cell.south}-${cell.west}-${index}`}
          bounds={[[cell.south, cell.west], [cell.north, cell.east]]}
          pathOptions={{
            color: riskColor(cell.risk_score),
            weight: 0.5,
            fillColor: riskColor(cell.risk_score),
            fillOpacity: 0.58,
          }}
        >
          <Tooltip sticky direction="top">
            <div className="risk-map-tooltip">
              <strong>RISK SCORE {Math.round(cell.risk_score)}</strong>
              <span>TEMPERATURE {cell.temperature.toFixed(1)} °C</span>
              <span>WIND {(cell.wind_speed * 3.6).toFixed(1)} km/h</span>
              <span>PRECIPITATION {cell.rainfall.toFixed(1)} mm</span>
              <span>MODEL AGREEMENT {Math.round(cell.confidence)}%</span>
            </div>
          </Tooltip>
        </Rectangle>
      ))}
    </>
  )
}

export default RiskMapLayer