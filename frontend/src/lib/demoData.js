// Clearly labelled sample data for the map when no GIS endpoint is connected.
export const demoReports = [
  { id:'demo-01', lat:31.2524, lng:75.7052, severity:'high', status:'reported', confidence:.91, description:'Phagwara — GT Road service lane', createdAt:'2026-11-18T09:20:00Z', source:'Sample data' },
  { id:'demo-02', lat:31.2495, lng:75.7040, severity:'medium', status:'in_progress', confidence:.87, description:'Phagwara — near bus stand', createdAt:'2026-11-17T14:10:00Z', source:'Sample data' },
  { id:'demo-03', lat:31.2551, lng:75.7115, severity:'low', status:'fixed', confidence:.79, description:'Phagwara — residential road', createdAt:'2026-11-15T11:30:00Z', source:'Sample data' },
  { id:'demo-04', lat:31.2428, lng:75.7131, severity:'high', status:'reported', confidence:.94, description:'Phagwara — highway approach', createdAt:'2026-11-19T07:45:00Z', source:'Sample data' },
  { id:'demo-05', lat:31.2582, lng:75.6974, severity:'medium', status:'reported', confidence:.83, description:'Phagwara — local connector road', createdAt:'2026-11-16T08:00:00Z', source:'Sample data' },
  { id:'demo-06', lat:31.2461, lng:75.7001, severity:'low', status:'in_progress', confidence:.76, description:'Phagwara — market access road', createdAt:'2026-11-14T16:30:00Z', source:'Sample data' }
]
export const activity = [
  { title:'High severity pothole detected', detail:'GT Road service lane', time:'12 min ago', type:'high' },
  { title:'Repair status updated', detail:'Residential road · Marked fixed', time:'1 hr ago', type:'fixed' },
  { title:'New inspection uploaded', detail:'3 detections · 2.1 km from you', time:'3 hrs ago', type:'new' },
  { title:'Duplicate report merged', detail:'Market access road', time:'Yesterday', type:'merge' }
]