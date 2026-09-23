import { useState } from 'react'

import { getVerification } from '../services/api'
import type { VerificationResponse } from '../types/verification'

export function useVerification() {
	const [data, setData] = useState<VerificationResponse | null>(null)
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState<string | null>(null)

	const fetchVerification = async (params: Parameters<typeof getVerification>[0]) => {
		setLoading(true)
		setError(null)
		try {
			const result = await getVerification(params)
			setData(result)
			return result
		} catch (caught) {
			const message = caught instanceof Error ? caught.message : 'Unable to load verification metrics.'
			setError(message)
			throw caught
		} finally {
			setLoading(false)
		}
	}

	return { data, loading, error, fetchVerification }
}
