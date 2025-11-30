# Frontend Setup & Development

This document covers the React frontend in `frontend/` used for the real-time dashboard.

## Prerequisites
- Node.js 18+ (or recommended LTS)
- npm or bun (project uses npm by default in scripts)

## Install & Run (development)
```bash
cd frontend
npm install
npm run dev
# Open: http://localhost:5173 (Vite)
```

## Build for production
```bash
cd frontend
npm run build
# Output directory: frontend/dist (or build)
```

## Key scripts
- `npm run dev` — Run dev server
- `npm run build` — Create production build
- `npm run preview` — Preview build locally

## Development notes
- WebSocket endpoint defaults to `http://localhost:5000` in development; change to Pi IP when testing on device.
- Main hooks:
  - `useSystemSocket.ts` — manages Socket.IO connection
  - `use-mobile.tsx` — mobile responsive helper
- Components:
  - `SimProcessing` — overlays detections on live feed
  - `Performance` — charts and real-time metrics

## Deploying frontend on Pi
- Build locally or on Pi:
```bash
cd frontend
npm install
npm run build
``` 
- Copy build assets into Flask `static/` or serve via a small static server.

## Environment variables
- For dev, adjust `VITE_API_URL` or similar in `.env` at `frontend/` root.

## Troubleshooting
- If the live overlay is slow, reduce frame frequency or lower stream quality in `.env`.

