import { useState } from 'react'

import { getWeights } from '../services/api'
import type { WeightResponse } from '../types/forecast'

export function useWeights() {
	const [data, setData] = useState<WeightResponse | null>(null)
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState<string | null>(null)

	const fetchWeights = async (params: Parameters<typeof getWeights>[0]) => {
		setLoading(true)
		setError(null)
		try {
			const result = await getWeights(params)
			setData(result)
			return result
		} catch (caught) {
			const message = caught instanceof Error ? caught.message : 'Unable to load model weights.'
			setError(message)
			throw caught
		} finally {
			setLoading(false)
		}
	}

	return { data, loading, error, fetchWeights }
}
