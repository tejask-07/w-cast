export type ForecastVariable = 'temperature' | 'rainfall' | 'wind_speed'

export type ForecastLocation = {
	name: string | null
	lat: number
	lon: number
}

export type ForecastValues = {
	temperature: number
	rainfall: number
	wind_speed: number
}

export type ModelWeights = {
	gfs: number
	gefs: number
	baseline: number
}

export type Extremes = {
	heavy_rain: boolean
	heat_wave: boolean
	high_wind: boolean
	risk_level: string
}

export type ForecastResponse = {
	location: ForecastLocation
	lead_hours: number
	forecast: ForecastValues
	weights: ModelWeights
	regime: string
	model_source?: string
	extremes: Extremes
}

export type ForecastQuery = {
	lat: number
	lon: number
	lead_hours: number
	variable: ForecastVariable
}

export type ForecastAOI = {
	north: number
	south: number
	east: number
	west: number
}

export type SpatialForecastQuery = ForecastQuery & Partial<ForecastAOI>

export type WeightResponse = {
	location: {
		lat: number
		lon: number
	}
	lead_hours: number
	weights: ModelWeights
	regime: string
}

export type SpatialForecastResponse = {
	variable: ForecastVariable
	lead_hours: number
	bounds: [[number, number], [number, number]]
	image_url: string
	min_value: number
	max_value: number
	unit: string
}

export type HourlyForecastPoint = {
	hour: number
	value: number
}

export type HourlyForecastResponse = {
	variable: ForecastVariable
	lead_hours: number
	unit: string
	points: HourlyForecastPoint[]
	source?: 'wcast_blend' | 'gfs_fallback' | 'gefs_fallback'
	sources?: {
		gfs: boolean
		gefs: boolean
	}
}
