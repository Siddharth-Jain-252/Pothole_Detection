import { NavLink } from 'react-router-dom'
import { Activity, Bell, CircleHelp, Gauge, LayoutDashboard, Map, ScanSearch, Settings, ShieldCheck, Siren, ClipboardList, X } from 'lucide-react'

const groups = [
  { label:'WORKSPACE', items:[
    {to:'/', label:'Overview', icon:LayoutDashboard},
    {to:'/detect', label:'AI inspection', icon:ScanSearch, badge:'AI'},
    {to:'/map', label:'GIS explorer', icon:Map},
    {to:'/reports', label:'Pothole reports', icon:ClipboardList, count:'24'}
  ]},
  { label:'INSIGHTS', items:[
    {to:'/analytics', label:'Road analytics', icon:Gauge},
    {to:'/alerts', label:'Proximity alerts', icon:Bell}
  ]}
]
export default function Sidebar({open,onClose}) {
  return <>
    {open && <button className="fixed inset-0 z-30 bg-slate-950/40 lg:hidden" onClick={onClose} aria-label="Close menu" />}
    <aside className={`sidebar ${open?'sidebar-open':''}`}>
      <div className="flex items-center justify-between px-6 pt-6 pb-8">
        <div className="flex items-center gap-3">
          <div className="brand-mark"><ShieldCheck size={21}/></div>
          <div><div className="font-display text-[17px] font-extrabold tracking-tight text-ink">RoadSight<span className="text-brand">.</span></div><div className="text-[10px] font-bold tracking-[.16em] text-slate-400">ROAD INTELLIGENCE</div></div>
        </div>
        <button className="lg:hidden text-slate-400" onClick={onClose}><X size={19}/></button>
      </div>
      <div className="px-3">
        {groups.map(g=><div className="mb-7" key={g.label}>
          <div className="nav-label">{g.label}</div>
          {g.items.map(({to,label,icon:Icon,badge,count})=><NavLink end={to==='/'} onClick={onClose} key={to} to={to} className={({isActive})=>`nav-item ${isActive?'nav-active':''}`}>
            <Icon size={17}/><span>{label}</span>{badge&&<span className="ml-auto rounded-md bg-emerald-50 px-1.5 py-0.5 text-[9px] font-extrabold text-brand">{badge}</span>}{count&&<span className="ml-auto text-[10px] font-bold text-slate-400">{count}</span>}
          </NavLink>)}
        </div>)}
      </div>
      <div className="mt-auto px-4 pb-5">
        <div className="rounded-2xl bg-[#F1F7F5] p-4">
          <div className="mb-2 flex items-center gap-2 text-brand"><Siren size={16}/><span className="text-xs font-bold">Safety pulse</span></div>
          <p className="text-xs leading-5 text-slate-600">Stay informed about road hazards reported around your selected area.</p>
          <NavLink to="/alerts" className="mt-3 inline-flex text-xs font-bold text-brand">Configure alerts <span className="ml-1">→</span></NavLink>
        </div>
        <NavLink to="/settings" className="nav-item mt-3"><Settings size={17}/>Settings</NavLink>
        <div className="mt-4 flex items-center gap-3 border-t border-line pt-4">
          <div className="avatar">SJ</div><div className="min-w-0 flex-1"><div className="truncate text-xs font-bold text-ink">Road inspector</div><div className="text-[10px] text-slate-400">Local workspace</div></div><CircleHelp size={16} className="text-slate-400"/>
        </div>
      </div>
    </aside>
  </>
}