import axios from 'axios'

const env = import.meta.env
export const apiConfig = {
  baseURL: (env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, ''),
  detectPath: env.VITE_DETECT_PATH || '/detect',
  potholesPath: env.VITE_POTHOLES_PATH || '/potholes',
  savePath: env.VITE_SAVE_REPORT_PATH || '/potholes',
  uploadField: env.VITE_UPLOAD_FIELD || 'file',
  withCredentials: env.VITE_WITH_CREDENTIALS === 'true'
}

export const api = axios.create({
  baseURL: apiConfig.baseURL,
  timeout: 120000,
  withCredentials: apiConfig.withCredentials
})

const pickImage = (data) => {
  const raw = data?.annotated_image || data?.result_image || data?.image || data?.visualization || data?.annotatedImage
  if (!raw) return null
  if (typeof raw !== 'string') return null
  if (raw.startsWith('data:') || raw.startsWith('http://') || raw.startsWith('https://') || raw.startsWith('/')) return raw
  return `data:image/jpeg;base64,${raw}`
}

export function normalizeDetection(data) {
  const rows = data?.detections || data?.potholes || data?.predictions || data?.results || []
  const detections = (Array.isArray(rows) ? rows : []).map((d, i) => ({
    id: d.id || `det-${i + 1}`,
    confidence: Number(d.confidence ?? d.score ?? d.conf ?? 0),
    severity: String(d.severity || d.level || 'Unclassified').toLowerCase(),
    bbox: d.bbox || d.box || d.xyxy || null,
    area: d.area ?? d.area_px ?? null,
    label: d.label || d.class_name || d.class || 'pothole'
  }))
  return {
    raw: data,
    image: pickImage(data),
    detections,
    count: Number(data?.count ?? data?.total_detections ?? detections.length),
    inferenceMs: Number(data?.inference_time_ms ?? data?.processing_time_ms ?? data?.latency_ms ?? 0),
    model: data?.model || data?.model_name || 'Pothole detection model'
  }
}

export async function detectImage(file, location) {
  const form = new FormData()
  form.append(apiConfig.uploadField, file)
  if (location?.lat != null && location?.lng != null) {
    form.append('latitude', String(location.lat))
    form.append('longitude', String(location.lng))
    form.append('lat', String(location.lat))
    form.append('lon', String(location.lng))
  }
  const { data } = await api.post(apiConfig.detectPath, form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return normalizeDetection(data)
}

export function normalizePothole(p, i = 0) {
  const lat = Number(p.latitude ?? p.lat ?? p.location?.lat)
  const lng = Number(p.longitude ?? p.lon ?? p.lng ?? p.location?.lng)
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null
  return {
    id: String(p.id ?? p.report_id ?? `api-${i}`),
    lat, lng,
    severity: String(p.severity || p.level || 'medium').toLowerCase(),
    status: String(p.status || 'reported').toLowerCase(),
    confidence: Number(p.confidence ?? p.score ?? 0),
    description: p.description || p.address || p.road_name || 'Reported road hazard',
    createdAt: p.created_at || p.timestamp || new Date().toISOString(),
    source: p.source || 'API'
  }
}

export async function fetchPotholes({ lat, lng, radiusKm = 5 } = {}) {
  const params = { radius_km: radiusKm }
  if (lat != null && lng != null) Object.assign(params, { lat, lon: lng, latitude: lat, longitude: lng })
  const { data } = await api.get(apiConfig.potholesPath, { params })
  const rows = Array.isArray(data) ? data : (data?.potholes || data?.reports || data?.items || [])
  return rows.map(normalizePothole).filter(Boolean)
}

export async function savePothole(payload) {
  const { data } = await api.post(apiConfig.savePath, payload)
  return data
}

export function friendlyError(error) {
  if (error?.code === 'ECONNABORTED') return 'The model took too long to respond. Try a smaller image.'
  if (error?.response) return error.response.data?.detail || error.response.data?.message || `API error (${error.response.status})`
  if (error?.request) return 'Could not reach your FastAPI server. Check the API URL, deployment status and CORS settings.'
  return error?.message || 'Something went wrong.'
}