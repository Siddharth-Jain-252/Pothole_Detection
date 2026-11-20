/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: { extend: {
    colors: { ink: '#101828', muted: '#667085', brand: '#176B5B', canvas: '#F5F7F8', line: '#E7ECEB' },
    boxShadow: { soft: '0 8px 30px rgba(16,24,40,.06)' }
  }},
  plugins: []
}