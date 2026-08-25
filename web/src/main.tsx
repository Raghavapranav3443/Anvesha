import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'

// theme: light default, persisted choice respected
const saved = localStorage.getItem('anvesha-theme')
document.documentElement.dataset.theme = saved === 'dark' ? 'dark' : 'light'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
