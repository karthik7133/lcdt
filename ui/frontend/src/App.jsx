import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, Legend } from 'recharts';
import { Activity, ShieldCheck, Zap, AlertTriangle, TrendingUp, Cpu } from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [summary, setSummary] = useState({ Ct: 1.0, Dt: 0.1, Ht: 0.8, risk_pct: 0, total_interventions: 0 });
  const [latentHistory, setLatentHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [simResults, setSimResults] = useState([]);
  const [baselineResults, setBaselineResults] = useState([]);
  const [simOutcome, setSimOutcome] = useState(null);
  const [activePolicy, setActivePolicy] = useState('baseline');

  // Live data: summary + latent history — refreshes every 10 seconds
  const fetchLiveData = async () => {
    try {
      const summaryRes = await fetch(`${API_BASE}/summary`);
      setSummary(await summaryRes.json());

      const historyRes = await fetch(`${API_BASE}/latent_states`);
      setLatentHistory(await historyRes.json());
    } catch (err) {
      console.error("Fetch Error:", err);
    }
  };

  // Forecast: only refreshes every 60 seconds — expensive computation, slow-changing
  const fetchForecast = async () => {
    try {
      const forecastRes = await fetch(`${API_BASE}/forecast`);
      setForecast(await forecastRes.json());
    } catch (err) {
      console.error("Forecast Fetch Error:", err);
    }
  };

  const runSimulation = async (policy) => {
    setActivePolicy(policy);
    try {
      const res = await fetch(`${API_BASE}/simulation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy })
      });
      const data = await res.json();
      
      // If we are fetching baseline for the first time or updating it
      if (policy === 'baseline') {
        setBaselineResults(data);
      }
      
      setSimResults(data);
      
      if (data.length > 0) {
        const initial = data[0].Risk_Pct;
        const final = data[data.length - 1].Risk_Pct;
        
        // Calculate reduction compared to BASELINE'S final state if possible
        const baselineFinal = baselineResults.length > 0 ? baselineResults[baselineResults.length - 1].Risk_Pct : initial;
        
        setSimOutcome({
          initial,
          final,
          baselineFinal,
          reduction: (baselineFinal - final).toFixed(2),
          improvement: (((baselineFinal - final) / baselineFinal) * 100).toFixed(1)
        });
      }
    } catch (err) {
      console.error("Sim Error:", err);
    }
  };

  const mergedSimData = simResults.map((item, index) => ({
    ...item,
    Baseline_Risk: baselineResults[index]?.Risk_Pct || item.Risk_Pct
  }));

  const policyExplanations = {
    'baseline': 'Current trajectory with no external policy interventions. Natural capacity decay and habits are the primary drivers.',
    'security_training': 'Policy: do(Security Training). High-frequency reinforcement learning spikes ($H_t$) to strengthen the habitual shield against phishing and social engineering.',
    'ui_policy_change': 'Policy: do(UX Optimization). Capping task-switching and notification interruptions to prevent Cognitive Demand ($D_t$) from overloading Capacity.',
    'increased_workload': 'Scenario: do(Increased Workload). Stress-testing the system by forcing high demand. This simulates "Crunch Mode" where mistake probability spikes.'
  };

  useEffect(() => {
    // Initial load: fetch everything immediately
    fetchLiveData();
    fetchForecast();
    runSimulation('baseline');

    // Live data polls every 10s
    const liveInterval = setInterval(fetchLiveData, 10000);
    // Forecast polls every 60s (not every 10s)
    const forecastInterval = setInterval(fetchForecast, 60000);

    return () => {
      clearInterval(liveInterval);
      clearInterval(forecastInterval);
    };
  }, []);


  return (
    <div className="app-container">
      <div className="glass-panel">
        
        {/* Header */}
        <header className="dashboard-header">
          <div>
            <h1 className="dashboard-title"><span className="neon-text-cyan">CYBER</span> WATCHDOG</h1>
            <p style={{color: 'var(--text-dim)', fontSize: '0.9rem'}}>LIFELONG COGNITIVE-CYBER DIGITAL TWIN (V5)</p>
          </div>
          <div className="live-indicator">
            <span className="pulsing-dot"></span>
            <span className="neon-text-red" style={{fontSize: '0.8rem', fontWeight: 'bold'}}>LIVE PREDICTION ACTIVE</span>
          </div>
        </header>

        {/* Top Metrics: Layer 3 & 4 Snapshot */}
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-title"><Zap size={16} inline /> Capacity (Ct)</span>
            <span className="metric-value neon-text-cyan">{(summary.Ct * 100).toFixed(0)}%</span>
            <div style={{height: '4px', background: '#333', borderRadius: '2px', marginTop: '10px'}}>
              <div style={{height: '100%', width: `${summary.Ct * 100}%`, background: 'var(--neon-cyan)', boxShadow: '0 0 10px var(--neon-cyan)'}}></div>
            </div>
          </div>
          
          <div className="metric-card">
            <span className="metric-title"><Activity size={16} /> Demand (Dt)</span>
            <span className="metric-value neon-text-magenta">{(summary.Dt * 100).toFixed(0)}%</span>
            <div style={{height: '4px', background: '#333', borderRadius: '2px', marginTop: '10px'}}>
              <div style={{height: '100%', width: `${summary.Dt * 100}%`, background: 'var(--neon-magenta)', boxShadow: '0 0 10px var(--neon-magenta)'}}></div>
            </div>
          </div>

          <div className="metric-card">
            <span className="metric-title"><ShieldCheck size={16} /> Habits (Ht)</span>
            <span className="metric-value neon-text-green">{(summary.Ht * 100).toFixed(0)}%</span>
            <div style={{height: '4px', background: '#333', borderRadius: '2px', marginTop: '10px'}}>
              <div style={{height: '100%', width: `${summary.Ht * 100}%`, background: 'var(--neon-green)', boxShadow: '0 0 10px var(--neon-green)'}}></div>
            </div>
          </div>

          <div className="metric-card">
            <span className="metric-title"><AlertTriangle size={16} /> Risk P(Mt)</span>
            <span className="metric-value neon-text-red">{summary.risk_pct}%</span>
            <span style={{fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '5px'}}>MISTAKE PROBABILITY</span>
          </div>
        </div>

        {/* Main Analytics Layout */}
        <div className="charts-layout">
          
          {/* Left Column: Historical Dynamics */}
          <div className="section-box">
            <h3 className="chart-title"><TrendingUp size={18} color="var(--neon-cyan)" /> Layer 3: Psychological Dynamics (Ct vs Dt)</h3>
            <div style={{height: '350px'}}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={latentHistory}>
                  <defs>
                    <linearGradient id="colorCt" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--neon-cyan)" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="var(--neon-cyan)" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorDt" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--neon-magenta)" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="var(--neon-magenta)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="var(--text-dim)" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 1]} stroke="var(--text-dim)" fontSize={12} tickLine={false} axisLine={false} />
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <Tooltip contentStyle={{background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', borderRadius: '10px'}} />
                  <Legend />
                  <Area type="monotone" dataKey="Ct" name="Capacity" stroke="var(--neon-cyan)" fillOpacity={1} fill="url(#colorCt)" strokeWidth={3} />
                  <Area type="monotone" dataKey="Dt" name="Demand" stroke="var(--neon-magenta)" fillOpacity={1} fill="url(#colorDt)" strokeWidth={3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Right Column: Prediction & Forecasting */}
          <div className="section-box">
            <h3 className="chart-title"><Cpu size={18} color="var(--neon-red)" /> Layer 4: 7-Day Risk Trajectory</h3>
            <div style={{height: '250px'}}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={forecast}>
                  <XAxis dataKey="day" stroke="var(--text-dim)" fontSize={10} label={{value: 'Days Ahead', position: 'insideBottom', offset: -5, fill: 'var(--text-dim)'}} />
                  <YAxis stroke="var(--text-dim)" fontSize={10} unit="%" />
                  <Tooltip contentStyle={{background: 'var(--glass-bg)', border: '1px solid var(--glass-border)'}} />
                  <Line type="monotone" dataKey="forecasted_risk_pct" name="Risk Forecast" stroke="var(--neon-red)" strokeWidth={4} dot={{r: 4, fill: 'var(--neon-red)'}} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="risk-circle-container" style={{marginTop: '20px'}}>
              <p style={{fontSize: '0.8rem', color: 'var(--text-dim)'}}>LIVE SURVEILLANCE RISK</p>
              <h2 className="risk-percentage neon-text-red">{summary.risk_pct}<span style={{fontSize: '1rem'}}>%</span></h2>
            </div>
          </div>

        </div>

        {/* Bottom Section: Layer 5 Causal Simulation */}
        <div className="section-box" style={{marginTop: '2rem'}}>
          <h3 className="chart-title"><Cpu size={18} color="var(--neon-green)" /> Layer 5: Counterfactual Simulation [ E(M | do(pi)) ]</h3>
          <p style={{color: 'var(--text-dim)', fontSize: '0.8rem', marginBottom: '1rem'}}>
            Simulate how specific policy interventions will affect your cyber-risk probability over the next 30 days.
          </p>
          
          <div className="sim-controls">
            <button className={`sim-btn ${activePolicy === 'baseline' ? 'active' : ''}`} onClick={() => runSimulation('baseline')}>Baseline (No Intervention)</button>
            <button className={`sim-btn ${activePolicy === 'security_training' ? 'active' : ''}`} onClick={() => runSimulation('security_training')}>do(Security Training)</button>
            <button className={`sim-btn ${activePolicy === 'ui_policy_change' ? 'active' : ''}`} onClick={() => runSimulation('ui_policy_change')}>do(UX Optimization)</button>
            <button className={`sim-btn ${activePolicy === 'increased_workload' ? 'active' : ''}`} onClick={() => runSimulation('increased_workload')}>do(Increased Workload)</button>
          </div>

          {simResults.length > 0 && (
            <>
              <div className="policy-analysis-box">
                <p className="analysis-text">
                  <span className="neon-text-cyan" style={{fontWeight: 'bold', fontSize: '0.8rem'}}>ANALYSIS:</span> {policyExplanations[activePolicy]}
                </p>
              </div>

              <div style={{height: '350px', marginTop: '1rem'}}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mergedSimData}>
                    <defs>
                      <linearGradient id="colorSim" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--neon-cyan)" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="var(--neon-cyan)" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="step" stroke="var(--text-dim)" tick={{fontSize: 10}} label={{value: 'Simulation Days', position: 'insideBottom', offset: -5, fill: 'var(--text-dim)', fontSize: 12}} />
                    <YAxis stroke="var(--text-dim)" tick={{fontSize: 10}} domain={['auto', 'auto']} label={{value: 'Risk %', angle: -90, position: 'insideLeft', fill: 'var(--text-dim)', fontSize: 12}} />
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <Tooltip contentStyle={{background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', borderRadius: '10px'}} />
                    <Legend />
                    {activePolicy !== 'baseline' && (
                      <Line type="monotone" dataKey="Baseline_Risk" name="Baseline (No Intervention)" stroke="var(--text-dim)" strokeDasharray="5 5" strokeWidth={2} dot={false} />
                    )}
                    <Line type="monotone" dataKey="Risk_Pct" name={activePolicy === 'baseline' ? "Baseline Risk %" : "Intervention Risk %"} stroke="var(--neon-cyan)" strokeWidth={4} dot={{r: 4, fill: 'var(--neon-cyan)'}} />
                    <Line type="monotone" dataKey="Capacity" name="Simulated Capacity" stroke="var(--neon-magenta)" strokeDasharray="3 3" strokeWidth={1} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {simOutcome && (
                <div className="impact-summary">
                  <div className="impact-card">
                    <span className="impact-label">BASELINE END-STATE</span>
                    <span className="impact-value">{simOutcome.baselineFinal}%</span>
                  </div>
                  <div className="impact-arrow">→</div>
                  <div className="impact-card">
                    <span className="impact-label">INTERVENTION END-STATE</span>
                    <span className={`impact-value ${simOutcome.reduction > 0 ? 'neon-text-green' : 'neon-text-red'}`}>
                      {simOutcome.final}%
                    </span>
                  </div>
                  <div className="impact-card highlight">
                    <span className="impact-label">TOTAL RISK REDUCTION</span>
                    <span className={`impact-value ${simOutcome.reduction > 0 ? 'neon-text-green' : 'neon-text-magenta'}`}>
                      {simOutcome.reduction > 0 ? '-' : '+'}{Math.abs(simOutcome.reduction)}%
                    </span>
                    <span className="impact-subtext">{simOutcome.improvement}% divergence from baseline</span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;
