import { ArrowDownRight, ArrowUpRight } from 'lucide-react'
export default function StatCard({label,value,delta,caption,icon:Icon,tone='green'}) {
 return <div className="panel p-5">
  <div className="flex items-start justify-between"><div className="text-xs font-semibold text-slate-500">{label}</div><div className={`stat-icon stat-${tone}`}><Icon size={18}/></div></div>
  <div className="mt-4 flex items-end justify-between"><div className="font-display text-[30px] font-extrabold tracking-tight text-ink">{value}</div>{delta&&<div className="mb-1 flex items-center gap-1 text-[10px] font-bold text-emerald-700"><ArrowUpRight size={13}/>{delta}</div>}</div>
  <div className="mt-1 text-[10px] text-slate-400">{caption}</div>
 </div>
}