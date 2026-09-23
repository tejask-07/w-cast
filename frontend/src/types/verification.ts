export type VerificationMetric = {
	mae: number
	rmse: number
	bias: number
}

export type VerificationResponse = {
	variable: string
	lead_hours: number
	metrics: {
		gfs: VerificationMetric
		gefs: VerificationMetric
		adaptive_blend: VerificationMetric
	}
}
