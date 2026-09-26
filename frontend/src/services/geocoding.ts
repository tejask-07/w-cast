export interface LocationSuggestion {
  id: string
  name: string
  secondaryText: string
  displayName: string
  center: [number, number]
  zoom: number
}

type PhotonFeature = {
  geometry?: { coordinates?: number[] }
  properties?: {
    osm_id?: number | string
    name?: string
    city?: string
    county?: string
    state?: string
    country?: string
    extent?: number[]
    type?: string
  }
}

type NominatimResult = {
  place_id?: number | string
  name?: string
  display_name?: string
  lat?: string
  lon?: string
  boundingbox?: string[]
  type?: string
  class?: string
}

function calculateZoomFromSpan(maxSpan: number): number {
  if (maxSpan > 25) return 4
  if (maxSpan > 12) return 5
  if (maxSpan > 5) return 7
  if (maxSpan > 2) return 8
  if (maxSpan > 0.8) return 10
  if (maxSpan > 0.2) return 11
  if (maxSpan > 0.05) return 12
  return 13
}

function calculateZoom(extent?: number[], type?: string): number {
  if (extent?.length === 4 && extent.every(Number.isFinite)) {
    const longitudeSpan = Math.abs(extent[2] - extent[0])
    const latitudeSpan = Math.abs(extent[3] - extent[1])
    return calculateZoomFromSpan(Math.max(longitudeSpan, latitudeSpan))
  }
  if (type === 'country') return 5
  if (type === 'state' || type === 'region') return 7
  if (type === 'county' || type === 'district') return 9
  if (type === 'city' || type === 'town') return 11
  return 11
}

function validCenter(latitude: number, longitude: number): boolean {
  return Number.isFinite(latitude)
    && Number.isFinite(longitude)
    && latitude >= -90
    && latitude <= 90
    && longitude >= -180
    && longitude <= 180
}

function photonSuggestion(feature: PhotonFeature, query: string, index: number): LocationSuggestion | null {
  const properties = feature.properties ?? {}
  const coordinates = feature.geometry?.coordinates
  if (!coordinates || coordinates.length < 2) return null

  const longitude = Number(coordinates[0])
  const latitude = Number(coordinates[1])
  if (!validCenter(latitude, longitude)) return null

  const name = properties.name?.trim() || query
  const contexts = [properties.city, properties.county, properties.state, properties.country]
    .filter((part): part is string => Boolean(part?.trim()))
    .map((part) => part.trim())
    .filter((part, partIndex, parts) =>
      part.toLocaleLowerCase() !== name.toLocaleLowerCase()
      && parts.findIndex((candidate) => candidate.toLocaleLowerCase() === part.toLocaleLowerCase()) === partIndex,
    )
  const secondaryText = contexts.join(', ')

  return {
    id: `photon-${properties.osm_id ?? index}-${longitude}-${latitude}`,
    name,
    secondaryText,
    displayName: secondaryText ? `${name}, ${secondaryText}` : name,
    center: [latitude, longitude],
    zoom: calculateZoom(properties.extent, properties.type),
  }
}

function nominatimSuggestion(item: NominatimResult, query: string, index: number): LocationSuggestion | null {
  const latitude = Number(item.lat)
  const longitude = Number(item.lon)
  if (!validCenter(latitude, longitude)) return null

  const displayParts = (item.display_name ?? '').split(',').map((part) => part.trim()).filter(Boolean)
  const name = item.name?.trim() || displayParts[0] || query
  const secondaryText = displayParts
    .slice(1)
    .filter((part) => !/^\d{4,6}$/.test(part))
    .join(', ')

  let zoom = 11
  if (item.boundingbox?.length === 4) {
    const [south, north, west, east] = item.boundingbox.map(Number)
    if ([south, north, west, east].every(Number.isFinite)) {
      zoom = calculateZoomFromSpan(Math.max(Math.abs(north - south), Math.abs(east - west)))
    }
  } else if (item.type === 'administrative' || item.class === 'boundary') {
    zoom = 7
  }

  return {
    id: `nom-${item.place_id ?? index}-${latitude}-${longitude}`,
    name,
    secondaryText,
    displayName: secondaryText ? `${name}, ${secondaryText}` : name,
    center: [latitude, longitude],
    zoom,
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

export async function searchLocations(
  query: string,
  signal?: AbortSignal,
): Promise<LocationSuggestion[]> {
  const trimmed = query.trim()
  if (trimmed.length < 2) return []

  try {
    const response = await fetch(
      `https://photon.komoot.io/api/?q=${encodeURIComponent(trimmed)}&limit=5`,
      { method: 'GET', headers: { Accept: 'application/json' }, signal },
    )
    if (response.ok) {
      const data = await response.json() as { features?: PhotonFeature[] }
      const suggestions = (Array.isArray(data?.features) ? data.features : [])
        .slice(0, 5)
        .map((feature, index) => photonSuggestion(feature, trimmed, index))
        .filter((suggestion): suggestion is LocationSuggestion => suggestion !== null)
      if (suggestions.length > 0) return suggestions
    }
  } catch (error) {
    if (isAbortError(error)) throw error
  }

  try {
    const params = new URLSearchParams({
      q: trimmed,
      format: 'jsonv2',
      addressdetails: '1',
      limit: '5',
    })
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?${params.toString()}`,
      { method: 'GET', headers: { Accept: 'application/json' }, signal },
    )
    if (!response.ok) throw new Error(`Nominatim request failed (${response.status})`)
    const data = await response.json() as NominatimResult[]
    return (Array.isArray(data) ? data : [])
      .slice(0, 5)
      .map((item, index) => nominatimSuggestion(item, trimmed, index))
      .filter((suggestion): suggestion is LocationSuggestion => suggestion !== null)
  } catch (error) {
    if (isAbortError(error)) throw error
    console.warn('Geocoding lookup failed:', error)
    throw error
  }
}
