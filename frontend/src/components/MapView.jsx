import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
import { useEffect } from 'react'
function Recenter({center,zoom}) { const map=useMap(); useEffect(()=>{map.setView(center,zoom)},[center,zoom,map]); return null }
const color = s => s==='high'?'#E05252':s==='medium'?'#E6A23C':s==='low'?'#299B78':'#64748b'
export default function MapView({reports=[],center=[31.252,75.705],zoom=13,compact=false,onSelect}) {
 return <MapContainer center={center} zoom={zoom} scrollWheelZoom={!compact} className={`map-canvas ${compact?'map-compact':''}`}>
  <Recenter center={center} zoom={zoom}/>
  <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/>
  {reports.map(r=><CircleMarker key={r.id} center={[r.lat,r.lng]} radius={r.severity==='high'?9:7} pathOptions={{color:color(r.severity),fillColor:color(r.severity),fillOpacity:.82,weight:2}} eventHandlers={{click:()=>onSelect?.(r)}}>
   <Popup><div className="min-w-40"><div className="font-bold capitalize">{r.severity} severity</div><div className="text-xs">{r.description}</div><div className="mt-1 text-[10px] text-slate-500">Status: {r.status?.replace('_',' ')}</div></div></Popup>
  </CircleMarker>)}
 </MapContainer>
}