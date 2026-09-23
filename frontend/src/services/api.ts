import type {
	Extremes,
	ForecastQuery,
	ForecastResponse,
	WeightResponse,
} from '../types/forecast'
import type { VerificationResponse } from '../types/verification'

const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')
	?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
	status: number

	constructor(message: string, status = 0) {
		super(message)
		this.name = 'ApiError'
		this.status = status
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null
}

function isNumber(value: unknown): value is number {
	return typeof value === 'number' && Number.isFinite(value)
}

function isForecastResponse(value: unknown): value is ForecastResponse {
	if (!isRecord(value) || !isRecord(value.location) || !isRecord(value.forecast)
		|| !isRecord(value.weights) || !isRecord(value.extremes)) return false
	return (typeof value.location.name === 'string' || value.location.name === null)
		&& isNumber(value.location.lat) && isNumber(value.location.lon)
		&& isNumber(value.lead_hours)
		&& isNumber(value.forecast.temperature) && isNumber(value.forecast.rainfall)
		&& isNumber(value.forecast.wind_speed)
		&& isNumber(value.weights.gfs) && isNumber(value.weights.gefs) && isNumber(value.weights.baseline)
		&& typeof value.regime === 'string'
		&& typeof value.extremes.heavy_rain === 'boolean'
		&& typeof value.extremes.heat_wave === 'boolean'
		&& typeof value.extremes.high_wind === 'boolean'
		&& typeof value.extremes.risk_level === 'string'
}

function isWeightResponse(value: unknown): value is WeightResponse {
	if (!isRecord(value) || !isRecord(value.location) || !isRecord(value.weights)) return false
	return isNumber(value.location.lat) && isNumber(value.location.lon) && isNumber(value.lead_hours)
		&& isNumber(value.weights.gfs) && isNumber(value.weights.gefs) && isNumber(value.weights.baseline)
		&& typeof value.regime === 'string'
}

function isExtremes(value: unknown): value is Extremes {
	return isRecord(value)
		&& typeof value.heavy_rain === 'boolean'
		&& typeof value.heat_wave === 'boolean'
		&& typeof value.high_wind === 'boolean'
		&& typeof value.risk_level === 'string'
}

function isVerificationResponse(value: unknown): value is VerificationResponse {
	if (!isRecord(value) || typeof value.variable !== 'string' || !isNumber(value.lead_hours)
		|| !isRecord(value.metrics)) return false
	const metrics = value.metrics
	return ['gfs', 'gefs', 'adaptive_blend'].every((model) => {
		const metric = metrics[model]
		return isRecord(metric) && isNumber(metric.mae) && isNumber(metric.rmse) && isNumber(metric.bias)
	})
}

async function request<T>(path: string, validate: (value: unknown) => value is T): Promise<T> {
	let response: Response
	try {
		response = await fetch(`${API_URL}${path}`)
	} catch {
		throw new ApiError('Unable to reach the W-CAST backend.')
	}

	let payload: unknown
	try {
		payload = await response.json()
	} catch {
		throw new ApiError('The backend returned an invalid response.', response.status)
	}

	if (!response.ok) {
		const detail = isRecord(payload) && typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status}).`
		throw new ApiError(detail, response.status)
	}
	if (!validate(payload)) throw new ApiError('The backend returned an unexpected response.', response.status)
	return payload
}

function query(params: Record<string, string | number>): string {
	return new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)])).toString()
}

export function getForecast(params: ForecastQuery): Promise<ForecastResponse> {
	return request(`/api/forecast?${query(params)}`, isForecastResponse)
}

export function getWeights(params: Omit<ForecastQuery, 'variable'>): Promise<WeightResponse> {
	return request(`/api/weights?${query(params)}`, isWeightResponse)
}

export function getExtremes(params: Omit<ForecastQuery, 'variable'>): Promise<Extremes> {
	return request(`/api/extremes?${query(params)}`, isExtremes)
}

export function getVerification(params: ForecastQuery): Promise<VerificationResponse> {
	return request(`/api/verification?${query(params)}`, isVerificationResponse)
}
