import { BrowserRouter, Routes, Route } from 'react-router-dom'

import Landing from './pages/Landing'
import Forecast from './pages/Forecast'
import Results from './pages/Results'

function App() {
  return (
    <BrowserRouter>
      <Routes>

        <Route
          path="/"
          element={<Landing />}
        />

        <Route
          path="/forecast"
          element={<Forecast />}
        />

        <Route
          path="/forecast/results"
          element={<Results />}
        />

      </Routes>
    </BrowserRouter>
  )
}

export default App