import { useState } from 'react'

import { getForecast } from '../services/api'
import type { ForecastQuery, ForecastResponse } from '../types/forecast'

export function useForecast() {
	const [data, setData] = useState<ForecastResponse | null>(null)
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState<string | null>(null)

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

	return { data, loading, error, fetchForecast }
}
