import { useCallback, useState } from 'react'

import { getForecast, getForecastMap, getHourlyForecast } from '../services/api'
import type {
	ForecastQuery,
	ForecastVariable,
	ForecastResponse,
	HourlyForecastResponse,
	SpatialForecastQuery,
	SpatialForecastResponse,
} from '../types/forecast'

export function useForecast() {
	const [data, setData] = useState<ForecastResponse | null>(null)
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState<string | null>(null)
	const [mapData, setMapData] = useState<SpatialForecastResponse | null>(null)
	const [mapLoading, setMapLoading] = useState(false)
	const [mapError, setMapError] = useState<string | null>(null)
	const [hourlyData, setHourlyData] = useState<HourlyForecastResponse | null>(null)
	const [hourlyDataByVariable, setHourlyDataByVariable] = useState<Partial<Record<ForecastVariable, HourlyForecastResponse>>>({})
	const [hourlyLoading, setHourlyLoading] = useState(false)
	const [hourlyError, setHourlyError] = useState<string | null>(null)

	const fetchForecast = async (params: ForecastQuery) => {
		setLoading(true)
		setError(null)
		try {
			const result = await getForecast(params)
			setData(result)
			return result
		} catch (caught) {
			const message = caught instanceof Error ? caught.message : 'Unable to load the forecast.'
			setError(message)
			throw caught
		} finally {
			setLoading(false)
		}
	}

	const fetchForecastMap = useCallback(async (params: SpatialForecastQuery) => {
		setMapLoading(true)
		setMapError(null)
		setMapData(null)
		try {
			const result = await getForecastMap(params)
			setMapData(result)
			return result
		} catch (caught) {
			const message = caught instanceof Error ? caught.message : 'Unable to load the spatial forecast.'
			setMapError(message)
			throw caught
		} finally {
			setMapLoading(false)
		}
	}, [])

	const fetchHourlyForecast = useCallback(async (params: ForecastQuery) => {
		setHourlyLoading(true)
		setHourlyError(null)
		try {
			const result = await getHourlyForecast(params)
			setHourlyData(result)
			setHourlyDataByVariable((current) => ({ ...current, [params.variable]: result }))
			return result
		} catch (caught) {
			const message = caught instanceof Error ? caught.message : 'Unable to load the hourly forecast.'
			setHourlyError(message)
			throw caught
		} finally {
			setHourlyLoading(false)
		}
	}, [])

	return {
		data,
		loading,
		error,
		fetchForecast,
		mapData,
		mapLoading,
		mapError,
		fetchForecastMap,
		hourlyData,
		hourlyDataByVariable,
		hourlyLoading,
		hourlyError,
		fetchHourlyForecast,
	}
}
