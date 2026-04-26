import requests
import pandas as pd
import json
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== Buscando dados das APIs ===\n")

def fetch_bcb(serie, label, start='01/01/2015', end='31/12/2025'):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={start}&dataFinal={end}"
    print(f"[BCB] {label} | série {serie}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            df = df.dropna(subset=['valor']).sort_values('data').reset_index(drop=True)
            print(f"  OK | {len(df)} obs | {df['data'].min().strftime('%m/%Y')} até {df['data'].max().strftime('%m/%Y')}")
            return df
        print(f"  ERRO {r.status_code}")
    except Exception as e:
        print(f"  ERRO: {e}")
    return None

ipca        = fetch_bcb(433,   "IPCA mensal (%)")
inpc        = fetch_bcb(188,   "INPC mensal (%)")
usd         = fetch_bcb(3698,  "USD/BRL fim de período mensal")
exportacoes = fetch_bcb(22708, "Exportações FOB (US$ milhões)")
importacoes = fetch_bcb(22704, "Importações FOB (US$ milhões)")
desemprego  = fetch_bcb(24369, "Taxa de desocupação mensal (%)")

# --- Regressão OLS: IPCA ~ Desemprego ---
print("\n=== Regressão OLS: IPCA ~ Desemprego ===")
reg = {}
scatter_pts = []

if ipca is not None and desemprego is not None:
    df_m = ipca.rename(columns={'valor':'ipca'}).merge(
        desemprego.rename(columns={'valor':'desemprego'}), on='data', how='inner'
    ).dropna()
    print(f"  Observações pareadas: {len(df_m)}")
    X = sm.add_constant(df_m['desemprego'])
    y = df_m['ipca']
    modelo = sm.OLS(y, X).fit()
    print(modelo.summary())
    reg = {
        'n_obs':           int(len(df_m)),
        'r2':              round(float(modelo.rsquared), 4),
        'r2_adj':          round(float(modelo.rsquared_adj), 4),
        'coef_const':      round(float(modelo.params['const']), 4),
        'coef_desemprego': round(float(modelo.params['desemprego']), 4),
        'pval_const':      round(float(modelo.pvalues['const']), 6),
        'pval_desemprego': round(float(modelo.pvalues['desemprego']), 6),
        'f_stat':          round(float(modelo.fvalue), 4),
        'f_pval':          round(float(modelo.f_pvalue), 6),
        'stderr_desemp':   round(float(modelo.bse['desemprego']), 4),
    }
    scatter_pts = [
        {'x': round(float(r['desemprego']),2),
         'y': round(float(r['ipca']),2),
         'label': r['data'].strftime('%b/%Y')}
        for _, r in df_m.iterrows()
    ]

def to_records(df):
    if df is None or df.empty:
        return []
    return [{'d': row['data'].strftime('%Y-%m-%d'), 'v': round(float(row['valor']), 4)}
            for _, row in df.iterrows()]

payload = {
    'ipca':        to_records(ipca),
    'inpc':        to_records(inpc),
    'usd':         to_records(usd),
    'exportacoes': to_records(exportacoes),
    'importacoes': to_records(importacoes),
    'desemprego':  to_records(desemprego),
    'regressao':   reg,
    'scatter':     scatter_pts,
    'atualizado':  datetime.now().strftime('%d/%m/%Y %H:%M'),
}

print("\n=== Gerando HTML ===")

def calc_trend(pts):
    if not pts or not reg: return []
    xs = [p['x'] for p in pts]
    a, b = reg['coef_const'], reg['coef_desemprego']
    return [{'x': round(min(xs),2), 'y': round(a+b*min(xs),4)},
            {'x': round(max(xs),2), 'y': round(a+b*max(xs),4)}]

def sig_stars(pval):
    if pval < 0.001: return '***'
    if pval < 0.01:  return '**'
    if pval < 0.05:  return '*'
    if pval < 0.1:   return '.'
    return ''

trend = calc_trend(scatter_pts)
j      = json.dumps(payload, ensure_ascii=False, separators=(',',':'))
j_trend = json.dumps(trend, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Painel Macroeconômico Brasil</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root {{
    --bg:#0f1117;--card:#1a1d27;--border:#2a2d3e;
    --text:#e2e8f0;--muted:#8892a4;--accent:#4f8ef7;
    --green:#34d399;--red:#f87171;--yellow:#fbbf24;--purple:#a78bfa;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}}
  header{{background:linear-gradient(135deg,#1a1d27,#0f1117);border-bottom:1px solid var(--border);padding:18px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}}
  header h1{{font-size:1.35rem;font-weight:700}}
  header h1 span{{color:var(--accent)}}
  .badge{{display:inline-flex;align-items:center;background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.3);color:var(--accent);border-radius:20px;padding:3px 10px;font-size:.72rem;font-weight:500}}
  .container{{max-width:1300px;margin:0 auto;padding:22px 18px}}
  .filter-bar{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
  .filter-bar label{{font-size:.82rem;color:var(--muted);white-space:nowrap}}
  .filter-bar select{{background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:7px;padding:5px 10px;font-size:.82rem;cursor:pointer;outline:none}}
  .filter-bar select:focus{{border-color:var(--accent)}}
  .filter-bar .sep{{color:var(--border);font-size:1.2rem;padding:0 2px}}
  .filter-bar button{{background:var(--accent);color:#fff;border:none;border-radius:7px;padding:6px 16px;font-size:.82rem;cursor:pointer;transition:opacity .2s}}
  .filter-bar button.sec{{background:transparent;border:1px solid var(--border);color:var(--muted)}}
  .filter-bar button:hover{{opacity:.82}}
  #filtroInfo{{font-size:.78rem;color:var(--accent);margin-left:auto;font-weight:500}}
  .tabs{{display:flex;gap:4px;margin-bottom:18px;flex-wrap:wrap}}
  .tab{{background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:7px 16px;font-size:.83rem;cursor:pointer;transition:all .2s}}
  .tab.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
  .tab:hover:not(.active){{border-color:var(--accent);color:var(--accent)}}
  .section{{display:none}}
  .section.active{{display:block}}
  .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}}
  .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}}
  @media(max-width:900px){{.grid4{{grid-template-columns:1fr 1fr}}.grid3{{grid-template-columns:1fr 1fr}}}}
  @media(max-width:500px){{.grid4,.grid3{{grid-template-columns:1fr}}}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px}}
  .card-full{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:18px}}
  .card h3{{font-size:.88rem;font-weight:600;margin-bottom:3px}}
  .card .sub{{font-size:.7rem;color:var(--muted);margin-bottom:12px}}
  .card .tag12{{font-size:.68rem;color:var(--accent);background:rgba(79,142,247,.1);border-radius:4px;padding:1px 5px;margin-left:5px}}
  .kpi{{font-size:1.8rem;font-weight:700}}
  .kpi-label{{font-size:.7rem;color:var(--muted);margin-top:3px}}
  .kpi.up{{color:var(--red)}}.kpi.down{{color:var(--green)}}.kpi.neutral{{color:var(--accent)}}
  canvas{{max-height:300px}}
  .reg-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
  @media(max-width:768px){{.reg-grid{{grid-template-columns:1fr}}}}
  .reg-table{{width:100%;border-collapse:collapse;font-size:.83rem}}
  .reg-table th{{text-align:left;color:var(--muted);font-weight:500;padding:7px 10px;border-bottom:1px solid var(--border)}}
  .reg-table td{{padding:7px 10px;border-bottom:1px solid var(--border)}}
  .reg-table tr:last-child td{{border-bottom:none}}
  .sig{{color:var(--yellow);font-weight:600;margin-left:4px}}
  .interp{{background:rgba(79,142,247,.07);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:13px 15px;font-size:.83rem;line-height:1.6;margin-top:14px}}
  .interp strong{{color:var(--accent)}}
  .stat-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}}
  .stat-chip{{background:rgba(255,255,255,.05);border:1px solid var(--border);border-radius:7px;padding:7px 12px;font-size:.8rem}}
  .stat-chip span{{color:var(--muted);font-size:.72rem;display:block}}
  .sources{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin-top:22px}}
  .sources h4{{font-size:.8rem;font-weight:600;color:var(--muted);margin-bottom:10px;letter-spacing:.05em;text-transform:uppercase}}
  .sources-grid{{display:flex;flex-wrap:wrap;gap:8px}}
  .source-tag{{display:flex;align-items:center;gap:7px;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:9px;padding:7px 12px}}
  .source-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
  .source-tag .sname{{font-size:.78rem;font-weight:600}}
  .source-tag .sdesc{{font-size:.7rem;color:var(--muted)}}
</style>
</head>
<body>
<header>
  <div>
    <h1>Painel Macroeconômico <span>Brasil</span></h1>
    <p style="font-size:.75rem;color:var(--muted);margin-top:4px">Jan/2015 – Dez/2025 · dados mensais</p>
  </div>
  <div style="text-align:right">
    <div style="font-size:.72rem;color:var(--muted)">Atualizado: {payload['atualizado']}</div>
    <div style="display:flex;gap:5px;margin-top:5px;justify-content:flex-end;flex-wrap:wrap">
      <span class="badge">BCB – Banco Central</span>
      <span class="badge">IBGE · PNAD Contínua</span>
    </div>
  </div>
</header>

<div class="container">

  <!-- FILTRO -->
  <div class="filter-bar">
    <label>De:</label>
    <select id="fM1" onchange="aplicarFiltro()"></select>
    <select id="fA1" onchange="aplicarFiltro()"></select>
    <span class="sep">→</span>
    <label>Até:</label>
    <select id="fM2" onchange="aplicarFiltro()"></select>
    <select id="fA2" onchange="aplicarFiltro()"></select>
    <button onclick="resetFiltro()" class="sec">Todos</button>
    <span id="filtroInfo"></span>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('inflacao',this)">Inflação</button>
    <button class="tab" onclick="showTab('cambio',this)">Câmbio</button>
    <button class="tab" onclick="showTab('desemprego',this)">Desemprego</button>
    <button class="tab" onclick="showTab('comercio',this)">Comércio Exterior</button>
    <button class="tab" onclick="showTab('regressao',this)">Regressão</button>
  </div>

  <!-- INFLAÇÃO -->
  <div id="tab-inflacao" class="section active">
    <div class="grid4">
      <div class="card"><h3>IPCA mensal</h3><div class="sub">Mês selecionado · BCB 433</div><div class="kpi up" id="kpi-ipca">—</div><div class="kpi-label">% ao mês</div></div>
      <div class="card"><h3>INPC mensal</h3><div class="sub">Mês selecionado · BCB 188</div><div class="kpi up" id="kpi-inpc">—</div><div class="kpi-label">% ao mês</div></div>
      <div class="card"><h3>IPCA acum.<span class="tag12">12m</span></h3><div class="sub">Até o mês selecionado</div><div class="kpi up" id="kpi-ipca12">—</div><div class="kpi-label">% acumulado</div></div>
      <div class="card"><h3>INPC acum.<span class="tag12">12m</span></h3><div class="sub">Até o mês selecionado</div><div class="kpi up" id="kpi-inpc12">—</div><div class="kpi-label">% acumulado</div></div>
    </div>
    <div class="card-full">
      <h3>IPCA e INPC – Variação Mensal (%)</h3>
      <div class="sub">Banco Central do Brasil · séries 433 (IPCA) e 188 (INPC)</div>
      <canvas id="chart-inflacao-mensal"></canvas>
    </div>
    <div class="card-full">
      <h3>IPCA e INPC – Acumulado 12 meses (%)</h3>
      <div class="sub">Calculado dinamicamente sobre as séries mensais do BCB</div>
      <canvas id="chart-inflacao-acum"></canvas>
    </div>
  </div>

  <!-- CÂMBIO -->
  <div id="tab-cambio" class="section">
    <div class="grid4">
      <div class="card"><h3>USD/BRL</h3><div class="sub">Mês selecionado · BCB 3698</div><div class="kpi neutral" id="kpi-usd">—</div><div class="kpi-label">R$ por US$</div></div>
      <div class="card"><h3>Mín. no período</h3><div class="sub"></div><div class="kpi down" id="kpi-usd-min">—</div><div class="kpi-label">R$</div></div>
      <div class="card"><h3>Máx. no período</h3><div class="sub"></div><div class="kpi up" id="kpi-usd-max">—</div><div class="kpi-label">R$</div></div>
      <div class="card"><h3>Média<span class="tag12">12m</span></h3><div class="sub">Até o mês selecionado</div><div class="kpi neutral" id="kpi-usd-med12">—</div><div class="kpi-label">R$ média mensal</div></div>
    </div>
    <div class="card-full">
      <h3>Taxa de Câmbio – USD/BRL</h3>
      <div class="sub">Banco Central do Brasil · série 3698 · fim de período mensal</div>
      <canvas id="chart-cambio"></canvas>
    </div>
  </div>

  <!-- DESEMPREGO -->
  <div id="tab-desemprego" class="section">
    <div class="grid4">
      <div class="card"><h3>Desocupação</h3><div class="sub">Mês selecionado · BCB 24369</div><div class="kpi up" id="kpi-des">—</div><div class="kpi-label">% força de trabalho</div></div>
      <div class="card"><h3>Mín. no período</h3><div class="sub"></div><div class="kpi down" id="kpi-des-min">—</div><div class="kpi-label">%</div></div>
      <div class="card"><h3>Máx. no período</h3><div class="sub"></div><div class="kpi up" id="kpi-des-max">—</div><div class="kpi-label">%</div></div>
      <div class="card"><h3>Média<span class="tag12">12m</span></h3><div class="sub">Até o mês selecionado</div><div class="kpi up" id="kpi-des-med12">—</div><div class="kpi-label">% média mensal</div></div>
    </div>
    <div class="card-full">
      <h3>Taxa de Desocupação Mensal (%)</h3>
      <div class="sub">Banco Central do Brasil · série 24369 · PNAD Contínua</div>
      <canvas id="chart-desemprego"></canvas>
    </div>
  </div>

  <!-- COMÉRCIO EXTERIOR -->
  <div id="tab-comercio" class="section">
    <div class="grid3">
      <div class="card"><h3>Exportações</h3><div class="sub">Mês selecionado · BCB 22708</div><div class="kpi neutral" id="kpi-exp">—</div><div class="kpi-label">US$ milhões (FOB)</div></div>
      <div class="card"><h3>Importações</h3><div class="sub">Mês selecionado · BCB 22704</div><div class="kpi neutral" id="kpi-imp">—</div><div class="kpi-label">US$ milhões (FOB)</div></div>
      <div class="card"><h3>Saldo comercial</h3><div class="sub">Exp. − Imp. · mês selecionado</div><div class="kpi neutral" id="kpi-saldo">—</div><div class="kpi-label">US$ milhões</div></div>
    </div>
    <div class="card-full">
      <h3>Exportações e Importações (US$ milhões FOB)</h3>
      <div class="sub">Banco Central do Brasil · séries 22708 e 22704</div>
      <canvas id="chart-comercio"></canvas>
    </div>
    <div class="card-full">
      <h3>Saldo da Balança Comercial (US$ milhões)</h3>
      <div class="sub">Exportações − Importações</div>
      <canvas id="chart-saldo"></canvas>
    </div>
  </div>

  <!-- REGRESSÃO -->
  <div id="tab-regressao" class="section">
    <div class="card-full">
      <h3>Regressão OLS: IPCA ~ Taxa de Desocupação</h3>
      <div class="sub">Mínimos quadrados ordinários · {reg.get('n_obs','—')} observações mensais (série completa 2015–2025)</div>
      <div class="interp">
        <strong>O que o modelo testa:</strong> se existe relação linear entre desemprego e inflação mensal.<br>
        A <strong>Curva de Phillips</strong> prevê coeficiente negativo (mais desemprego → menos inflação).
        Coeficiente positivo indica <strong>estagflação</strong> — padrão do Brasil pós-2015, quando desemprego e inflação subiram juntos.
      </div>
      <div class="stat-row" style="margin-top:14px">
        <div class="stat-chip"><span>R²</span>{reg.get('r2','—')}</div>
        <div class="stat-chip"><span>R² ajustado</span>{reg.get('r2_adj','—')}</div>
        <div class="stat-chip"><span>F-statistic</span>{reg.get('f_stat','—')}</div>
        <div class="stat-chip"><span>p-valor (F)</span>{reg.get('f_pval','—')}</div>
        <div class="stat-chip"><span>Observações</span>{reg.get('n_obs','—')}</div>
      </div>
      <div class="reg-grid">
        <div>
          <table class="reg-table">
            <thead><tr><th>Variável</th><th>Coeficiente</th><th>Erro padrão</th><th>p-valor</th><th></th></tr></thead>
            <tbody>
              <tr><td>Constante (α)</td><td>{reg.get('coef_const','—')}</td><td>—</td><td>{reg.get('pval_const','—')}</td><td class="sig">{sig_stars(reg.get('pval_const',1))}</td></tr>
              <tr><td>Desemprego (β)</td><td>{reg.get('coef_desemprego','—')}</td><td>{reg.get('stderr_desemp','—')}</td><td>{reg.get('pval_desemprego','—')}</td><td class="sig">{sig_stars(reg.get('pval_desemprego',1))}</td></tr>
            </tbody>
          </table>
          <div style="font-size:.72rem;color:var(--muted);margin-top:7px">*** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05 &nbsp;. p&lt;0.1</div>
          <div class="interp" style="margin-top:12px">
            <strong>α = {reg.get('coef_const','—')}</strong>: IPCA esperado quando desemprego = 0 (intercepto teórico).<br><br>
            <strong>β = {reg.get('coef_desemprego','—')}</strong>: para cada +1 p.p. de desemprego, o IPCA mensal varia em média {reg.get('coef_desemprego','—')} p.p.
            {"→ relação positiva (estagflação, padrão Brasil 2015-2017)." if reg.get('coef_desemprego',0) > 0 else "→ relação negativa (Curva de Phillips clássica)."}<br><br>
            <strong>R² = {reg.get('r2','—')}</strong>: desemprego explica {round(float(reg.get('r2',0))*100,1) if reg.get('r2') else '—'}% da variação do IPCA mensal — baixo, pois inflação é multideterminada (câmbio, energia, política monetária, expectativas).
          </div>
        </div>
        <div><canvas id="chart-scatter"></canvas></div>
      </div>
    </div>
  </div>

  <!-- FONTES -->
  <div class="sources">
    <h4>Fontes dos dados</h4>
    <div class="sources-grid">
      <div class="source-tag"><div class="source-dot" style="background:#4f8ef7"></div><div><div class="sname">BCB – Banco Central do Brasil</div><div class="sdesc">api.bcb.gov.br · séries 433, 188, 3698, 22708, 22704, 24369</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#34d399"></div><div><div class="sname">IBGE · PNAD Contínua</div><div class="sdesc">Taxa de desocupação mensal adaptada</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#fbbf24"></div><div><div class="sname">IPCA & INPC</div><div class="sdesc">Índices de inflação oficiais · BCB séries 433 e 188</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#fb923c"></div><div><div class="sname">Comércio Exterior</div><div class="sdesc">BCB · Balança Comercial FOB · séries 22708 e 22704</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#a78bfa"></div><div><div class="sname">Câmbio USD/BRL</div><div class="sdesc">BCB · série 3698 · fim de período mensal</div></div></div>
    </div>
  </div>

</div>

<script>
const RAW   = {j};
const TREND = {j_trend};
const MESES = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];

// Estado do filtro
let fIni = {{ano:2015,mes:1}};
let fFim = {{ano:2025,mes:12}};

// Inicializa dropdowns
(function init() {{
  ['fM1','fM2'].forEach(id => {{
    const s = document.getElementById(id);
    MESES.forEach((m,i) => s.add(new Option(m, i+1)));
  }});
  ['fA1','fA2'].forEach(id => {{
    const s = document.getElementById(id);
    for (let a=2015;a<=2025;a++) s.add(new Option(a,a));
  }});
  document.getElementById('fM1').value = 1;
  document.getElementById('fA1').value = 2015;
  document.getElementById('fM2').value = 12;
  document.getElementById('fA2').value = 2025;
}})();

function aplicarFiltro() {{
  const a1=+document.getElementById('fA1').value, m1=+document.getElementById('fM1').value;
  const a2=+document.getElementById('fA2').value, m2=+document.getElementById('fM2').value;
  if (a1*100+m1 > a2*100+m2) {{ document.getElementById('fA2').value=a1; document.getElementById('fM2').value=m1; }}
  fIni = {{ano:+document.getElementById('fA1').value, mes:+document.getElementById('fM1').value}};
  fFim = {{ano:+document.getElementById('fA2').value, mes:+document.getElementById('fM2').value}};
  document.getElementById('filtroInfo').textContent =
    `${{MESES[fIni.mes-1]}}/${{fIni.ano}} → ${{MESES[fFim.mes-1]}}/${{fFim.ano}}`;
  atualizarTodos();
}}

function resetFiltro() {{
  fIni={{ano:2015,mes:1}}; fFim={{ano:2025,mes:12}};
  document.getElementById('fM1').value=1; document.getElementById('fA1').value=2015;
  document.getElementById('fM2').value=12; document.getElementById('fA2').value=2025;
  document.getElementById('filtroInfo').textContent='';
  atualizarTodos();
}}

// Helpers de data
function dk(ano,mes){{ return ano*100+mes; }}
function pk(p){{ return parseInt(p.d.slice(0,4))*100+parseInt(p.d.slice(5,7)); }}

// Filtra pela janela [fIni, fFim]
function filtrar(arr){{
  const ini=dk(fIni.ano,fIni.mes), fim=dk(fFim.ano,fFim.mes);
  return arr.filter(p=>{{const k=pk(p);return k>=ini&&k<=fim;}});
}}

// Últimos N registros até fFim (para cálculos de 12m)
function ultimos(arr,n){{
  const fim=dk(fFim.ano,fFim.mes);
  return arr.filter(p=>pk(p)<=fim).slice(-n);
}}

// Inflação acumulada (produto encadeado)
function acumInflacao(pts){{
  if(pts.length<1) return null;
  let a=1.0; pts.forEach(p=>a*=(1+p.v/100));
  return r2((a-1)*100);
}}

// Média simples
function media(pts){{
  if(!pts.length) return null;
  return r2(pts.reduce((s,p)=>s+p.v,0)/pts.length);
}}

// Série acumulada 12m para gráfico (dinâmica, respeitando fFim)
function calcAcum12(arr){{
  const fim=dk(fFim.ano,fFim.mes);
  const src=arr.filter(p=>pk(p)<=fim);
  const out=[];
  for(let i=11;i<src.length;i++){{
    let a=1.0;
    for(let j=i-11;j<=i;j++) a*=(1+src[j].v/100);
    out.push({{x:src[i].d, y:r4((a-1)*100)}});
  }}
  return out;
}}

function r2(v){{ return Math.round(v*100)/100; }}
function r4(v){{ return Math.round(v*10000)/10000; }}
function last(a){{ return a.length?a[a.length-1]:null; }}
function minV(a){{ return a.length?Math.min(...a.map(p=>p.v)):null; }}
function maxV(a){{ return a.length?Math.max(...a.map(p=>p.v)):null; }}
function fmt(v,d=2){{ return v!=null?v.toFixed(d):'—'; }}
function set(id,val){{ const e=document.getElementById(id); if(e)e.textContent=val; }}
function toXY(arr){{ return arr.map(p=>({{x:p.d,y:p.v}})); }}

const PAL = {{
  ipca:'#f87171',inpc:'#fbbf24',usd:'#4f8ef7',
  exp:'#34d399',imp:'#f87171',des:'#a78bfa',
  gp:'#34d399',gn:'#f87171',
}};

function makeChart(id, type, datasets, extra={{}}) {{
  const ctx=document.getElementById(id);
  if(!ctx) return;
  if(ctx._ch) ctx._ch.destroy();
  ctx._ch = new Chart(ctx, {{
    type,
    data:{{datasets}},
    options:{{
      responsive:true,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{labels:{{color:'#8892a4',font:{{size:11}}}}}},
        tooltip:{{backgroundColor:'#1a1d27',borderColor:'#2a2d3e',borderWidth:1,titleColor:'#e2e8f0',bodyColor:'#8892a4'}},
        ...extra.plugins
      }},
      scales:{{
        x:{{type:'time',time:{{unit:'year',tooltipFormat:'MMM/yyyy'}},grid:{{color:'#1e2130'}},ticks:{{color:'#8892a4',maxTicksLimit:12}}}},
        y:{{grid:{{color:'#1e2130'}},ticks:{{color:'#8892a4'}}}},
        ...extra.scales
      }},
      ...extra
    }}
  }});
}}

// ---- KPIs ----
function atualizarKPIs(ipca,inpc,usd,des,exp_,imp_){{
  const li=last(ipca),ln=last(inpc),lu=last(usd),ld=last(des),le=last(exp_),lim=last(imp_);

  if(li) set('kpi-ipca', fmt(li.v)+'%');
  if(ln) set('kpi-inpc', fmt(ln.v)+'%');

  // Acumulado 12m até fFim
  const i12=acumInflacao(ultimos(RAW.ipca,12));
  const n12=acumInflacao(ultimos(RAW.inpc,12));
  if(i12!=null) set('kpi-ipca12', fmt(i12)+'%');
  if(n12!=null) set('kpi-inpc12', fmt(n12)+'%');

  // Câmbio
  if(lu) set('kpi-usd','R$ '+fmt(lu.v));
  if(usd.length){{
    set('kpi-usd-min','R$ '+fmt(minV(usd)));
    set('kpi-usd-max','R$ '+fmt(maxV(usd)));
  }}
  const u12=media(ultimos(RAW.usd,12));
  if(u12!=null) set('kpi-usd-med12','R$ '+fmt(u12));

  // Desemprego
  if(ld) set('kpi-des',fmt(ld.v)+'%');
  if(des.length){{
    set('kpi-des-min',fmt(minV(des))+'%');
    set('kpi-des-max',fmt(maxV(des))+'%');
  }}
  const d12=media(ultimos(RAW.desemprego,12));
  if(d12!=null) set('kpi-des-med12',fmt(d12)+'%');

  // Comércio
  if(le) set('kpi-exp',fmt(le.v,0));
  if(lim) set('kpi-imp',fmt(lim.v,0));
  if(le&&lim){{
    const s=le.v-lim.v;
    const el=document.getElementById('kpi-saldo');
    if(el){{ el.textContent=(s>=0?'+':'')+fmt(s,0); el.className='kpi '+(s>=0?'down':'up'); }}
  }}
}}

// ---- Gráficos ----
function atualizarTodos(){{
  const ipca=filtrar(RAW.ipca), inpc=filtrar(RAW.inpc), usd=filtrar(RAW.usd);
  const des=filtrar(RAW.desemprego), exp_=filtrar(RAW.exportacoes), imp_=filtrar(RAW.importacoes);

  atualizarKPIs(ipca,inpc,usd,des,exp_,imp_);

  makeChart('chart-inflacao-mensal','line',[
    {{label:'IPCA (%)',data:toXY(ipca),borderColor:PAL.ipca,backgroundColor:PAL.ipca+'22',tension:.3,pointRadius:0,fill:true}},
    {{label:'INPC (%)',data:toXY(inpc),borderColor:PAL.inpc,backgroundColor:PAL.inpc+'11',tension:.3,pointRadius:0,borderDash:[4,4]}}
  ]);

  makeChart('chart-inflacao-acum','line',[
    {{label:'IPCA acum. 12m (%)',data:calcAcum12(RAW.ipca),borderColor:PAL.ipca,tension:.3,pointRadius:0}},
    {{label:'INPC acum. 12m (%)',data:calcAcum12(RAW.inpc),borderColor:PAL.inpc,tension:.3,pointRadius:0,borderDash:[4,4]}}
  ]);

  makeChart('chart-cambio','line',[
    {{label:'USD/BRL (R$)',data:toXY(usd),borderColor:PAL.usd,backgroundColor:PAL.usd+'18',tension:.3,pointRadius:0,fill:true}}
  ]);

  makeChart('chart-desemprego','line',[
    {{label:'Desocupação (%)',data:toXY(des),borderColor:PAL.des,backgroundColor:PAL.des+'18',tension:.3,pointRadius:0,fill:true}}
  ]);

  makeChart('chart-comercio','line',[
    {{label:'Exportações (US$ mi)',data:toXY(exp_),borderColor:PAL.exp,tension:.3,pointRadius:0}},
    {{label:'Importações (US$ mi)',data:toXY(imp_),borderColor:PAL.imp,tension:.3,pointRadius:0,borderDash:[4,4]}}
  ]);

  const saldo=exp_.map(p=>{{const im=imp_.find(m=>m.d===p.d);return im?{{x:p.d,y:p.v-im.v}}:null;}}).filter(Boolean);
  makeChart('chart-saldo','bar',[
    {{label:'Saldo (US$ mi)',data:saldo,
      backgroundColor:saldo.map(p=>p.y>=0?PAL.gp+'88':PAL.gn+'88'),
      borderColor:saldo.map(p=>p.y>=0?PAL.gp:PAL.gn),
      borderWidth:1}}
  ]);
}}

// ---- Scatter (estático) ----
(function(){{
  const ctx=document.getElementById('chart-scatter');
  if(!ctx) return;
  new Chart(ctx,{{
    type:'scatter',
    data:{{datasets:[
      {{label:'Observações',data:RAW.scatter.map(p=>({{x:p.x,y:p.y,lb:p.label}})),backgroundColor:'#4f8ef755',borderColor:'#4f8ef7',pointRadius:4}},
      {{label:'Tendência OLS',type:'line',data:TREND,borderColor:'#f87171',backgroundColor:'transparent',pointRadius:0,borderWidth:2}}
    ]}},
    options:{{
      responsive:true,
      plugins:{{
        legend:{{labels:{{color:'#8892a4',font:{{size:11}}}}}},
        tooltip:{{
          backgroundColor:'#1a1d27',borderColor:'#2a2d3e',borderWidth:1,
          callbacks:{{label:c=>{{const r=c.raw;return r.lb?`${{r.lb}}: Des ${{r.x}}% · IPCA ${{r.y}}%`:`(${{r.x}},${{r.y}})`; }}}}
        }}
      }},
      scales:{{
        x:{{title:{{display:true,text:'Desocupação (%)',color:'#8892a4'}},grid:{{color:'#1e2130'}},ticks:{{color:'#8892a4'}}}},
        y:{{title:{{display:true,text:'IPCA Mensal (%)',color:'#8892a4'}},grid:{{color:'#1e2130'}},ticks:{{color:'#8892a4'}}}}
      }}
    }}
  }});
}})();

function showTab(name,btn){{
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
  atualizarTodos();
}}

atualizarTodos();
</script>
</body>
</html>"""

out = "painel_economico_brasil.html"
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"HTML gerado: {out}  ({len(html)//1024} KB)")
