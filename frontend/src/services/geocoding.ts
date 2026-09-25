import type { ForecastLocation } from '../types/forecast'

type NominatimResult = {
  lat: string
  lon: string
  display_name: string
  address?: {
    city?: string
    town?: string
    municipality?: string
    village?: string
  }
}

function resultName(result: NominatimResult): string {
  return result.address?.city
    ?? result.address?.town
    ?? result.address?.municipality
    ?? result.address?.village
    ?? result.display_name
}

export async function searchLocations(
  query: string,
  signal: AbortSignal,
): Promise<ForecastLocation[]> {
  const params = new URLSearchParams({
    q: query,
    format: 'jsonv2',
    addressdetails: '1',
    limit: '3',
    featureType: 'city',
  })
  const response = await fetch(
    `https://nominatim.openstreetmap.org/search?${params.toString()}`,
    {
      signal,
      headers: { Accept: 'application/json' },
    },
  )

  if (!response.ok) {
    throw new Error(`Location search failed (${response.status})`)
  }

  const results = await response.json() as NominatimResult[]
  return results
    .map((result) => ({
      name: resultName(result),
      lat: Number(result.lat),
      lon: Number(result.lon),
    }))
    .filter((location) => Number.isFinite(location.lat) && Number.isFinite(location.lon))
}