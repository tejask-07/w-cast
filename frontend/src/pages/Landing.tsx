import './Landing.css'
import { useNavigate } from 'react-router-dom'
import heroHandGlobe from '../assets/hero-hand-globe.png'

function Landing() {
  const navigate = useNavigate()

  return (
    <main className="landing">

      {/* ================================================= */}
      {/* HEADER */}
      {/* ================================================= */}

      <header className="landing-header">

        <div className="brand">
          LEFORECAST
        </div>

        <div className="header-divider" />

        <div className="header-meta">
          <span>INDIA</span>
          <b>/</b>
          <span>WEATHER FORECAST INTELLIGENCE</span>
        </div>

        <div className="header-right">
          REAL DATA. BETTER DECISIONS.
        </div>

      </header>


      {/* ================================================= */}
      {/* HERO */}
      {/* ================================================= */}

      <section className="landing-hero">

        {/* ----------------------------------------------- */}
        {/* LEFT META */}
        {/* ----------------------------------------------- */}

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


        {/* ----------------------------------------------- */}
        {/* RIGHT META */}
        {/* ----------------------------------------------- */}

        <aside className="hero-right-meta">

          <span className="meta-index">
            [ 01 ]
          </span>

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




        {/* ================================================= */}
        {/* MAIN HERO CONTENT */}
        {/* ================================================= */}

        <div className="hero-content">

          {/* --------------------------------------------- */}
          {/* WORDMARK */}
          {/* --------------------------------------------- */}

          <div className="wordmark">

            <span className="wordmark-left">
              LEF
            </span>

            <div className="hero-mark">

              <img
                src={heroHandGlobe}
                alt=""
                className="hero-hand-globe"
              />

            </div>

            <span className="wordmark-right">
              RECAST
            </span>

          </div>


          {/* --------------------------------------------- */}
          {/* SUBTITLE */}
          {/* --------------------------------------------- */}

          <div className="hero-subtitle">

            ADAPTIVE WEATHER
            <br />
            FORECAST BLENDING

          </div>


          {/* --------------------------------------------- */}
          {/* DESCRIPTION */}
          {/* --------------------------------------------- */}

          <p className="hero-description">

            Combining multiple numerical weather prediction sources
            <br />

            with adaptive weights based on historical skill, to deliver
            <br />

            more reliable and actionable weather forecasts for Indian cities.

          </p>


          {/* --------------------------------------------- */}
          {/* CTA */}
          {/* --------------------------------------------- */}

          <button
            type="button"
            className="explore-button"
            onClick={() => navigate('/forecast')}
          >

            <span className="button-arrow">
              →
            </span>

            <span className="button-label">
              EXPLORE FORECAST
            </span>

          </button>

        </div>


        {/* ================================================= */}
        {/* COORDINATES */}
        {/* ================================================= */}

        <div className="hero-coordinates">

          <div className="coordinate-marker">

            <span className="coordinate-horizontal" />
            <span className="coordinate-vertical" />

          </div>

          <div className="coordinate-values">

            <span>
              20.5937° N
            </span>

            <span>
              78.9629° E
            </span>

            <span className="meta-line" />

          </div>

        </div>

      </section>


      {/* ================================================= */}
      {/* FOOTER */}
      {/* ================================================= */}

      <footer className="landing-footer">

        <span>
          SAME SKY. A STRONGER TOMORROW.
        </span>

        <span className="footer-right">

          <span>INDIA</span>
          <b>/</b>
          <span>DATA</span>
          <b>/</b>
          <span>IMPACT</span>

        </span>

      </footer>

    </main>
  )
}

export default Landing