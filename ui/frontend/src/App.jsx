import { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, Legend, ReferenceLine,
  BarChart, Bar, Cell
} from 'recharts';
import { Activity, ShieldCheck, Zap, AlertTriangle, TrendingUp, Cpu, Clock, Target } from 'lucide-react';
import './index.css';

const API = '/api';

/* ── helpers ── */
const fmt1 = v => (v ?? 0).toFixed(1);
const pct  = v => `${(v * 100).toFixed(0)}%`;
const clr  = v => v > 70 ? 'var(--neon-red)' : v > 40 ? 'var(--neon-magenta)' : 'var(--neon-green)';

/* ── Policy metadata for all 6 L4 interventions ── */
const POLICY_META = {
  baseline:              { label: 'Baseline',                icon: '📊', desc: 'Current trajectory — no external intervention applied.' },
  security_training:     { label: 'do(Security Training)',   icon: '🛡️', desc: 'Reinforcement learning spike into Habit state Ht.' },
  reduce_notifications:  { label: 'do(No Notifications)',    icon: '🔕', desc: 'Reducing interruptions suppresses Dt by 30%.' },
  improve_sleep:         { label: 'do(Improve Sleep)',       icon: '😴', desc: 'Raises Ct ceiling, reducing burnout risk over time.' },
  reduce_meetings:       { label: 'do(Fewer Meetings)',      icon: '📅', desc: 'Calendar pressure (L1) reduced, Dt drops 20%.' },
  pause_work:            { label: 'do(Take Break)',          icon: '⏸️', desc: 'Hard pause: Dt drops 70%, Ct recovers fastest.' },
  adversarial_drill:     { label: 'do(Adversarial Drill)',   icon: '🎯', desc: 'Controlled exposure to simulated threats.' },
  increased_workload:    { label: 'Stress Test',             icon: '🔥', desc: 'Forces high demand — simulates crunch / burnout.' },
};

export default function App() {
  const [summary,      setSummary]      = useState({ Ct:1, Dt:0.1, Ht:0.8, At:0, CRGt:0.9, risk_pct:0, total_interventions:0 });
  const [latentHist,   setLatentHist]   = useState([]);
  const [forecast,     setForecast]     = useState([]);
  const [hazard,       setHazard]       = useState(null);
  const [counterfacts, setCounterfacts] = useState([]);
  const [simResults,   setSimResults]   = useState([]);
  const [simOutcome,   setSimOutcome]   = useState(null);
  const [activePolicy, setActivePolicy] = useState('baseline');
  const [simLoading,   setSimLoading]   = useState(false);
  const [simError,     setSimError]     = useState(null);  // null | string

  /* ── FIX: useRef so runSim always reads latest baseline (no stale closure) ── */
  const baselineRef = useRef([]);

  /* ── fetchers ── */
  const fetchLive = async () => {
    try {
      const [s, h] = await Promise.all([
        fetch(`${API}/summary`).then(r => r.json()),
        fetch(`${API}/latent_states`).then(r => r.json()),
      ]);
      setSummary(s);
      setLatentHist(h);
    } catch (e) { console.error('Live fetch:', e); }
  };

  const fetchHazardAndForecast = async () => {
    try {
      const [f, hz, cf] = await Promise.all([
        fetch(`${API}/forecast`).then(r => r.json()),
        fetch(`${API}/hazard`).then(r => r.json()),
        fetch(`${API}/counterfactuals`).then(r => r.json()),
      ]);
      setForecast(f);
      setHazard(hz);
      setCounterfacts(cf);
    } catch (e) { console.error('Hazard/forecast fetch:', e); }
  };

  const runSim = async (policy) => {
    setActivePolicy(policy);
    setSimLoading(true);
    setSimError(null);   // clear any previous error
    setSimResults([]);   // clear old chart immediately

    /* ── 60-second timeout to accommodate Monte-Carlo computation ── */
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);

    try {
      const resp = await fetch(`${API}/simulation`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ policy }),
        signal:  controller.signal,
      });
      clearTimeout(timer);

      if (!resp.ok) {
        throw new Error(`Server returned ${resp.status}`);
      }

      const data = await resp.json();

      if (!Array.isArray(data) || data.length === 0) {
        setSimError('Backend returned no data. Is main.py running?');
        setSimLoading(false);
        return;
      }

      /* Save baseline in ref so it is NEVER stale when other policies run */
      if (policy === 'baseline') baselineRef.current = data;
      if (baselineRef.current.length === 0) baselineRef.current = data;

      setSimResults(data);

      const bl    = baselineRef.current;
      const blEnd = bl[bl.length - 1]?.Mean_Risk ?? data[data.length - 1]?.Mean_Risk;
      const ivEnd = data[data.length - 1]?.Mean_Risk;
      setSimOutcome({
        blEnd:          blEnd.toFixed(1),
        ivEnd:          ivEnd.toFixed(1),
        reduction:      (blEnd - ivEnd).toFixed(2),
        burnoutConf:    data[data.length - 1]?.Burnout_Confidence ?? 0,
        burnoutHorizon: data[0]?.Burnout_Horizon_Step,
      });

    } catch (e) {
      clearTimeout(timer);
      if (e.name === 'AbortError') {
        setSimError('Request timed out (60s). Backend may be offline — run python main.py to start it.');
      } else {
        setSimError(`Backend unreachable: ${e.message}. Run python main.py to start the server.`);
      }
    }
    setSimLoading(false);
  };

  /* Merge sim + baseline using ref — always current, never stale */
  const mergedSim = simResults.map((row, i) => ({
    ...row,
    Baseline_Mean: baselineRef.current[i]?.Mean_Risk ?? row.Mean_Risk,
  }));

  useEffect(() => {
    fetchLive();
    fetchHazardAndForecast();
    runSim('baseline');
    const live = setInterval(fetchLive, 10000);
    const slow = setInterval(fetchHazardAndForecast, 60000);
    return () => { clearInterval(live); clearInterval(slow); };
  }, []);

  /* ── Risk colour ── */
  const riskColor = clr(summary.risk_pct);

  return (
    <div className="app-container">
      <div className="glass-panel">

        {/* ── Header ── */}
        <header className="dashboard-header">
          <div>
            <h1 className="dashboard-title"><span className="neon-text-cyan">CYBER</span> WATCHDOG</h1>
            <p style={{ color:'var(--text-dim)', fontSize:'0.85rem' }}>
              Neural CDE · Causal SCM · Bayesian Hazard · Monte-Carlo Twin  |  V6 SCIENTIFIC
            </p>
          </div>
          <div className="live-indicator">
            <span className="pulsing-dot" />
            <span className="neon-text-red" style={{ fontSize:'0.8rem', fontWeight:'bold' }}>LIVE</span>
          </div>
        </header>

        {/* ── Row 1: L2/L3 State Cards ── */}
        <div className="metrics-grid">

          {[
            { label:'Capacity Ct', val: summary.Ct, icon: <Zap size={14}/>, color:'var(--neon-cyan)', tip:'Cognitive health — NCDE L2 output' },
            { label:'Demand Dt',   val: summary.Dt, icon: <Activity size={14}/>, color:'var(--neon-magenta)', tip:'Real-time workload pressure' },
            { label:'Habits Ht',   val: summary.Ht, icon: <ShieldCheck size={14}/>, color:'var(--neon-green)', tip:'Cyber hygiene score' },
            { label:'Threat At',   val: summary.At, icon: <AlertTriangle size={14}/>, color:'var(--neon-red)', tip:'Active adversarial exposure' },
          ].map(({ label, val, icon, color, tip }) => (
            <div key={label} className="metric-card" title={tip}>
              <span className="metric-title">{icon} {label}</span>
              <span className="metric-value" style={{ color }}>{pct(val)}</span>
              <div style={{ height:'4px', background:'#333', borderRadius:'2px', marginTop:'8px' }}>
                <div style={{ height:'100%', width:pct(val), background:color, boxShadow:`0 0 8px ${color}`, borderRadius:'2px' }} />
              </div>
            </div>
          ))}

          {/* Live Risk Card */}
          <div className="metric-card" style={{ gridColumn: 'span 2' }}>
            <span className="metric-title"><Target size={14}/> L4 Causal Risk  P(Mt)</span>
            <span className="metric-value" style={{ color: riskColor, fontSize:'2.2rem' }}>
              {summary.risk_pct}<span style={{ fontSize:'1rem' }}>%</span>
            </span>
            <span style={{ fontSize:'0.7rem', color:'var(--text-dim)' }}>
              Reserve Gap CRGt = {summary.CRGt?.toFixed(3) ?? '—'}
            </span>
          </div>

        </div>

        {/* ── Row 2: L5 Hazard Cards + Chart ── */}
        {hazard && (
          <div className="section-box" style={{ marginTop:'1.5rem' }}>
            <h3 className="chart-title"><Clock size={16} color="var(--neon-magenta)"/> L5 Bayesian Hazard — Probabilistic Risk Windows</h3>
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem', marginTop: '1rem' }}>
              <div className="metrics-grid" style={{ marginTop: '0' }}>
                {[
                  { window:'1 Hour',  data: hazard.hazard_1h  },
                  { window:'24 Hours',data: hazard.hazard_24h },
                  { window:'7 Days',  data: hazard.hazard_7d  },
                ].map(({ window, data }) => (
                  <div key={window} className="metric-card">
                    <span className="metric-title"><Clock size={14}/> P(Mistake in {window})</span>
                    <span className="metric-value" style={{ color: clr(data.mean), fontSize:'1.6rem' }}>
                      {fmt1(data.mean)}<span style={{ fontSize:'0.9rem' }}>%</span>
                    </span>
                    <span style={{ fontSize:'0.7rem', color:'var(--text-dim)' }}>
                      95% CI: [{fmt1(data.lower_5)}%, {fmt1(data.upper_95)}%] ± {fmt1(data.std)}
                    </span>
                  </div>
                ))}
                <div className="metric-card">
                  <span className="metric-title">Interventions Logged</span>
                  <span className="metric-value neon-text-cyan">{summary.total_interventions}</span>
                  <span style={{ fontSize:'0.7rem', color:'var(--text-dim)' }}>Vision AI / Alert Events</span>
                </div>
              </div>
              
              <div style={{ height:'100%', minHeight:'150px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { name: '1 Hour',   'P(Mistake)': hazard.hazard_1h.mean },
                    { name: '24 Hours', 'P(Mistake)': hazard.hazard_24h.mean },
                    { name: '7 Days',   'P(Mistake)': hazard.hazard_7d.mean }
                  ]} layout="vertical" margin={{ top: 0, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false}/>
                    <XAxis type="number" stroke="var(--text-dim)" fontSize={10} unit="%" domain={[0, 100]}/>
                    <YAxis dataKey="name" type="category" stroke="var(--text-dim)" fontSize={10} width={60}/>
                    <Tooltip contentStyle={{ background:'var(--glass-bg)', border:'1px solid var(--glass-border)', borderRadius:'8px' }}/>
                    <Bar dataKey="P(Mistake)" fill="var(--neon-magenta)" barSize={20}>
                      {
                        [hazard.hazard_1h.mean, hazard.hazard_24h.mean, hazard.hazard_7d.mean].map((val, idx) => (
                          <Cell key={`cell-${idx}`} fill={clr(val)} />
                        ))
                      }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* ── Row 3: L2 Dynamics Chart + L5 Forecast Chart ── */}
        <div className="charts-layout" style={{ marginTop:'1.5rem' }}>

          <div className="section-box">
            <h3 className="chart-title"><TrendingUp size={16} color="var(--neon-cyan)"/> L2 Neural CDE — Cognitive Dynamics (Ct vs Dt)</h3>
            <div style={{ height:'300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={latentHist}>
                  <defs>
                    <linearGradient id="gCt" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="var(--neon-cyan)"    stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="var(--neon-cyan)"    stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="gDt" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="var(--neon-magenta)" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="var(--neon-magenta)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="var(--text-dim)" fontSize={10} tickLine={false} axisLine={false}/>
                  <YAxis domain={[0,1]} stroke="var(--text-dim)" fontSize={10} tickLine={false} axisLine={false}/>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false}/>
                  <Tooltip contentStyle={{ background:'var(--glass-bg)', border:'1px solid var(--glass-border)', borderRadius:'8px' }}/>
                  <Legend/>
                  <Area type="monotone" dataKey="Ct" name="Capacity Ct" stroke="var(--neon-cyan)"    fillOpacity={1} fill="url(#gCt)" strokeWidth={2.5}/>
                  <Area type="monotone" dataKey="Dt" name="Demand Dt"   stroke="var(--neon-magenta)" fillOpacity={1} fill="url(#gDt)" strokeWidth={2.5}/>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="section-box">
            <h3 className="chart-title"><Cpu size={16} color="var(--neon-red)"/> L5 Bayesian Hazard — 7-Day Risk Trajectory + CI</h3>
            <div style={{ height:'300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={forecast}>
                  <XAxis dataKey="day" stroke="var(--text-dim)" fontSize={10} label={{ value:'Days', position:'insideBottom', offset:-3, fill:'var(--text-dim)', fontSize:10 }}/>
                  <YAxis stroke="var(--text-dim)" fontSize={10} unit="%"/>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false}/>
                  <Tooltip contentStyle={{ background:'var(--glass-bg)', border:'1px solid var(--glass-border)', borderRadius:'8px' }}/>
                  <Legend/>
                  {/* 95% upper band */}
                  <Line type="monotone" dataKey="upper_bound"    name="95th percentile" stroke="rgba(255,70,70,0.35)"  strokeWidth={1.5} strokeDasharray="4 3" dot={false}/>
                  {/* Mean trajectory */}
                  <Line type="monotone" dataKey="mean_risk"      name="Mean Risk"       stroke="var(--neon-red)"       strokeWidth={3}   dot={{ r:4, fill:'var(--neon-red)' }}/>
                  {/* 5% lower band */}
                  <Line type="monotone" dataKey="lower_bound"    name="5th percentile"  stroke="rgba(255,70,70,0.35)"  strokeWidth={1.5} strokeDasharray="4 3" dot={false}/>
                  {/* P(mistake 24h) */}
                  <Line type="monotone" dataKey="p_mistake_24h"  name="P(mistake/24h)"  stroke="var(--neon-magenta)"  strokeWidth={2}   strokeDasharray="2 2" dot={false}/>
                  <ReferenceLine y={70} stroke="rgba(255,60,60,0.5)" strokeDasharray="6 3" label={{ value:'Burnout', fill:'var(--neon-red)', fontSize:10 }}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>

        {/* ── Row 4: L4 Counterfactuals Ranked Table + Chart ── */}
        {counterfacts.length > 0 && (
          <div className="section-box" style={{ marginTop:'1.5rem' }}>
            <h3 className="chart-title"><ShieldCheck size={16} color="var(--neon-green)"/> L4 Causal SCM — Intervention Ranking (do-calculus)</h3>
            <p style={{ color:'var(--text-dim)', fontSize:'0.8rem', marginBottom:'1rem' }}>
              Ranked by expected risk reduction. Each intervention surgically modifies the target variable.
            </p>
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem' }}>
              
              {/* Left Column: The styled list */}
              <div style={{ display:'flex', flexDirection:'column', gap:'0.5rem' }}>
                {counterfacts.map((cf, i) => {
                  const meta = POLICY_META[cf.intervention] ?? { label: cf.intervention, icon: '⚙️', desc: '' };
                  const bar  = Math.max(0, Math.min(100, cf.reduction));
                  return (
                    <div key={cf.intervention} style={{
                      display:'grid', gridTemplateColumns:'2rem 1fr auto auto',
                      alignItems:'center', gap:'1rem',
                      padding:'0.6rem 1rem', borderRadius:'8px',
                      background:'rgba(255,255,255,0.03)', border:'1px solid rgba(255,255,255,0.06)'
                    }}>
                      <span style={{ fontSize:'1.2rem' }}>{meta.icon}</span>
                      <div>
                        <div style={{ fontSize:'0.85rem', fontWeight:600 }}>{meta.label}</div>
                        <div style={{ fontSize:'0.72rem', color:'var(--text-dim)' }}>{meta.desc}</div>
                        <div style={{ height:'3px', background:'#333', borderRadius:'2px', marginTop:'4px', width:'100%' }}>
                          <div style={{ height:'100%', width:`${bar * 5}%`, background:'var(--neon-green)', borderRadius:'2px', maxWidth:'100%' }}/>
                        </div>
                      </div>
                      <div style={{ textAlign:'right', fontSize:'0.8rem' }}>
                        <div style={{ color:'var(--text-dim)' }}>{cf.original_risk}%</div>
                        <div style={{ fontSize:'0.65rem', color:'var(--text-dim)' }}>→</div>
                        <div style={{ color:'var(--neon-green)', fontWeight:700 }}>{cf.mitigated_risk}%</div>
                      </div>
                      <div style={{
                        padding:'0.3rem 0.7rem', borderRadius:'6px',
                        background: cf.reduction > 0 ? 'rgba(0,255,136,0.1)' : 'rgba(255,70,70,0.1)',
                        color: cf.reduction > 0 ? 'var(--neon-green)' : 'var(--neon-red)',
                        fontWeight:700, fontSize:'0.85rem', whiteSpace:'nowrap'
                      }}>
                        {cf.reduction > 0 ? '-' : '+'}{Math.abs(cf.reduction).toFixed(2)}%
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Right Column: The BarChart */}
              <div style={{ height:'100%', minHeight: '300px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={counterfacts.map(cf => ({
                      name: (POLICY_META[cf.intervention]?.label || cf.intervention).replace('do(', '').replace(')', ''),
                      'Original Risk': cf.original_risk,
                      'Mitigated Risk': cf.mitigated_risk,
                      'Reduction': cf.reduction
                    }))} 
                    layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false}/>
                    <XAxis type="number" stroke="var(--text-dim)" fontSize={10} unit="%"/>
                    <YAxis dataKey="name" type="category" stroke="var(--text-dim)" fontSize={10} width={120}/>
                    <Tooltip contentStyle={{ background:'var(--glass-bg)', border:'1px solid var(--glass-border)', borderRadius:'8px' }}/>
                    <Legend />
                    <Bar dataKey="Original Risk" fill="rgba(255, 70, 70, 0.4)" barSize={12} />
                    <Bar dataKey="Mitigated Risk" fill="var(--neon-green)" barSize={12} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

            </div>
          </div>
        )}

        {/* ── Row 5: L6 Monte-Carlo Simulation ── */}
        <div className="section-box" style={{ marginTop:'1.5rem' }}>
          <h3 className="chart-title"><Cpu size={16} color="var(--neon-cyan)"/> L6 Monte-Carlo Digital Twin  [1000 trajectories · p(Zt:T | Zt)]</h3>
          <p style={{ color:'var(--text-dim)', fontSize:'0.8rem', marginBottom:'1rem' }}>
            Simulates 1000 stochastic futures. Bands show 5–95th percentile uncertainty. Choose a policy to run counterfactual simulation.
          </p>

          <div className="sim-controls">
            {Object.entries(POLICY_META).map(([key, { label, icon }]) => (
              <button
                key={key}
                id={`sim-btn-${key}`}
                className={`sim-btn ${activePolicy === key ? 'active' : ''}`}
                onClick={() => runSim(key)}
                disabled={simLoading}
                style={{ opacity: simLoading && activePolicy !== key ? 0.5 : 1 }}
              >
                {simLoading && activePolicy === key ? '⏳' : icon} {label}
              </button>
            ))}
          </div>

          {/* ── Loading State ── */}
          {simLoading && (
            <div style={{ textAlign:'center', padding:'3rem', color:'var(--text-dim)', fontSize:'0.9rem' }}>
              <div style={{ fontSize:'2rem', marginBottom:'0.75rem' }}>⚙️</div>
              <div>Running <strong style={{color:'var(--neon-cyan)'}}>1000 Monte-Carlo trajectories</strong></div>
              <div style={{ marginTop:'0.4rem' }}>Policy: <strong style={{color:'var(--neon-cyan)'}}>{POLICY_META[activePolicy]?.label}</strong></div>
              <div style={{ fontSize:'0.75rem', marginTop:'0.5rem', color:'rgba(255,255,255,0.35)' }}>This is real computation on your live cognitive state data...</div>
            </div>
          )}

          {/* ── Error State (backend offline / timeout) ── */}
          {!simLoading && simError && (
            <div style={{
              margin:'1.5rem 0', padding:'1.25rem 1.5rem', borderRadius:'12px',
              background:'rgba(255,60,60,0.08)', border:'1px solid rgba(255,60,60,0.3)',
              display:'flex', alignItems:'flex-start', gap:'1rem'
            }}>
              <span style={{ fontSize:'1.5rem' }}>🔌</span>
              <div>
                <div style={{ color:'var(--neon-red)', fontWeight:700, marginBottom:'0.4rem' }}>Backend Offline</div>
                <div style={{ color:'var(--text-dim)', fontSize:'0.82rem', lineHeight:1.5 }}>{simError}</div>
                <div style={{
                  marginTop:'0.75rem', padding:'0.4rem 0.75rem', display:'inline-block',
                  background:'rgba(0,212,255,0.1)', border:'1px solid rgba(0,212,255,0.3)',
                  borderRadius:'6px', color:'var(--neon-cyan)', fontSize:'0.78rem', fontFamily:'monospace'
                }}>
                  python main.py
                </div>
              </div>
            </div>
          )}

          {/* ── Results State (real calculated data from backend) ── */}
          {!simLoading && !simError && simResults.length > 0 && (
            <>
              <div style={{ padding:'0.5rem 0 0.25rem', color:'var(--text-dim)', fontSize:'0.8rem', display:'flex', alignItems:'center', gap:'0.5rem' }}>
                <span style={{ width:'8px', height:'8px', borderRadius:'50%', background:'var(--neon-green)', display:'inline-block', boxShadow:'0 0 6px var(--neon-green)' }}/>
                <span>Live backend data · Showing: <span style={{color:'var(--neon-cyan)', fontWeight:700}}>{POLICY_META[activePolicy]?.label ?? activePolicy}</span></span>
                {activePolicy !== 'baseline' && <span style={{color:'var(--text-dim)'}}> vs Baseline</span>}
              </div>

              <div style={{ height:'360px', marginTop:'0.5rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mergedSim}>
                    <XAxis dataKey="step" stroke="var(--text-dim)" fontSize={10}
                      label={{ value:'Simulation Days', position:'insideBottom', offset:-4, fill:'var(--text-dim)', fontSize:11 }}/>
                    <YAxis stroke="var(--text-dim)" fontSize={10} unit="%" domain={['auto','auto']}/>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false}/>
                    <Tooltip contentStyle={{ background:'var(--glass-bg)', border:'1px solid var(--glass-border)', borderRadius:'8px' }}/>
                    <Legend/>
                    <ReferenceLine y={70} stroke="rgba(255,60,60,0.4)" strokeDasharray="5 3"
                      label={{ value:'Burnout', fill:'var(--neon-red)', fontSize:9 }}/>
                    {/* 95th upper band */}
                    <Line type="monotone" dataKey="Risk_Upper_95" name="95th % CI"
                      stroke="rgba(0,212,255,0.3)" strokeWidth={1.5} strokeDasharray="3 3" dot={false}/>
                    {/* Baseline reference */}
                    {activePolicy !== 'baseline' && (
                      <Line type="monotone" dataKey="Baseline_Mean" name="Baseline"
                        stroke="var(--text-dim)" strokeDasharray="5 5" strokeWidth={2} dot={false}/>
                    )}
                    {/* Intervention mean */}
                    <Line type="monotone" dataKey="Mean_Risk" name="Mean Risk"
                      stroke="var(--neon-cyan)" strokeWidth={3.5} dot={{ r:3, fill:'var(--neon-cyan)' }}/>
                    {/* 5th lower band */}
                    <Line type="monotone" dataKey="Risk_Lower_5" name="5th % CI"
                      stroke="rgba(0,212,255,0.3)" strokeWidth={1.5} strokeDasharray="3 3" dot={false}/>
                    {/* Capacity */}
                    <Line type="monotone" dataKey="Mean_Capacity" name="Mean Capacity"
                      stroke="var(--neon-magenta)" strokeDasharray="4 2" strokeWidth={1.5} dot={false}/>
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {simOutcome && (
                <div className="impact-summary" style={{ marginTop:'1rem' }}>
                  <div className="impact-card">
                    <span className="impact-label">BASELINE END-STATE</span>
                    <span className="impact-value">{simOutcome.blEnd}%</span>
                  </div>
                  <div className="impact-arrow">→</div>
                  <div className="impact-card">
                    <span className="impact-label">INTERVENTION END-STATE</span>
                    <span className={`impact-value ${simOutcome.reduction > 0 ? 'neon-text-green' : 'neon-text-red'}`}>
                      {simOutcome.ivEnd}%
                    </span>
                  </div>
                  <div className="impact-card highlight">
                    <span className="impact-label">RISK REDUCTION</span>
                    <span className={`impact-value ${simOutcome.reduction > 0 ? 'neon-text-green' : 'neon-text-magenta'}`}>
                      {simOutcome.reduction > 0 ? '-' : '+'}{Math.abs(simOutcome.reduction)}%
                    </span>
                  </div>
                  <div className="impact-card">
                    <span className="impact-label">BURNOUT CONFIDENCE</span>
                    <span className="impact-value" style={{ color: clr(simOutcome.burnoutConf) }}>
                      {simOutcome.burnoutConf}%
                    </span>
                    <span className="impact-subtext">
                      {simOutcome.burnoutHorizon
                        ? `Horizon: Day ${simOutcome.burnoutHorizon}`
                        : 'No burnout in window'}
                    </span>
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
