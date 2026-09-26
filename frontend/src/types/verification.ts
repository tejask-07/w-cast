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
		equal_blend?: VerificationMetric
		wcast?: VerificationMetric
		improvement_vs_gfs?: number | null
		improvement_vs_gefs?: number | null
	}
}
