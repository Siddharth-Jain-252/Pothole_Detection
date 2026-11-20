import { Bell, Menu, Search, Wifi, WifiOff } from 'lucide-react'
import { useLocation } from 'react-router-dom'
const titles = {'/':'Overview','/detect':'AI inspection','/map':'GIS explorer','/reports':'Pothole reports','/analytics':'Road analytics','/alerts':'Proximity alerts','/settings':'Settings'}
export default function Topbar({onMenu,apiOnline}) {
 const {pathname}=useLocation()
 return <header className="topbar">
  <div className="flex min-w-0 items-center gap-3"><button onClick={onMenu} className="icon-btn lg:hidden"><Menu size={19}/></button><div><div className="text-[10px] font-bold uppercase tracking-[.16em] text-slate-400">Workspace / <span className="text-brand">{titles[pathname]||'Workspace'}</span></div><div className="font-display text-lg font-extrabold text-ink">{titles[pathname]||'Workspace'}</div></div></div>
  <div className="flex items-center gap-2 sm:gap-3">
   <div className={`hidden items-center gap-1.5 rounded-full px-3 py-2 text-[10px] font-bold sm:flex ${apiOnline?'bg-emerald-50 text-emerald-700':'bg-amber-50 text-amber-700'}`}>{apiOnline?<Wifi size={13}/>:<WifiOff size={13}/>} {apiOnline?'API connected':'API not checked'}</div>
   <button className="icon-btn hidden sm:grid"><Search size={17}/></button><button className="icon-btn relative"><Bell size={17}/><i className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-orange-500 ring-2 ring-white"/></button>
   <div className="avatar hidden sm:grid">SJ</div>
  </div>
 </header>
}