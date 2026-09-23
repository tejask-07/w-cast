import './Landing.css'
import { useNavigate } from 'react-router-dom'

function Landing() {
  const navigate = useNavigate()

  return (
    <main className="landing">
      <header className="landing-header">
        <div className="brand">LEFORECAST</div>

        <div className="header-divider" />

        <div className="header-meta">
          INDIA
          <span>/</span>
          WEATHER FORECAST INTELLIGENCE
        </div>

        <div className="header-right">
          REAL DATA. BETTER DECISIONS.
        </div>
      </header>

      <section className="landing-hero">
        <aside className="hero-left-meta">
          <p>
            FORECASTING
            <br />
            A MORE RESILIENT
            <br />
            INDIA.
          </p>

          <span className="meta-line" />
        </aside>

        <aside className="hero-right-meta">
          <span>[ 01 ]</span>

          <p>
            PEOPLE
            <br />
            DATA
            <br />
            CLIMATE
            <br />
            RESILIENCE
          </p>

          <span className="meta-line" />
        </aside>

        <div className="hero-title">
          <div className="wordmark">
            <span>LEF</span>

            <div className="hero-mark">
              <div className="hero-globe-placeholder">
                ◎
              </div>
            </div>

            <span>RECAST</span>
          </div>

          <div className="hero-subtitle">
            ADAPTIVE WEATHER
            <br />
            FORECAST BLENDING
          </div>

          <p className="hero-description">
            Combining multiple numerical weather prediction sources
            <br />
            with adaptive weights based on historical skill, to deliver
            <br />
            more reliable and actionable weather forecasts for Indian cities.
          </p>

          <button
            type="button"
            className="explore-button"
            onClick={() => navigate('/forecast')}
          >
            <span className="button-arrow">→</span>
            <span>EXPLORE FORECAST</span>
          </button>
        </div>

        <div className="hero-coordinates">
          <span className="coordinate-cross">+</span>

          <div>
            20.5937° N
            <br />
            78.9629° E
          </div>

          <span className="meta-line" />
        </div>
      </section>

      <footer className="landing-footer">
        <span>SAME SKY. A STRONGER TOMORROW.</span>

        <span>
          INDIA
          <b>/</b>
          DATA
          <b>/</b>
          IMPACT
        </span>
      </footer>
    </main>
  )
}

export default Landing