import requests
import pandas as pd
import json
import statsmodels.api as sm
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("=== Buscando dados das APIs ===\n")

def fetch_bcb(serie, label, start='01/01/2015', end='31/12/2025'):
    url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
           f"?formato=json&dataInicial={start}&dataFinal={end}")
    print(f"[BCB] {label} | série {serie}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            df = df.dropna(subset=['valor']).sort_values('data').reset_index(drop=True)
            print(f"  OK | {len(df)} obs | "
                  f"{df['data'].min().strftime('%m/%Y')} a {df['data'].max().strftime('%m/%Y')}")
            return df
        print(f"  ERRO {r.status_code}")
    except Exception as e:
        print(f"  ERRO: {e}")
    return None

# Séries principais
ipca        = fetch_bcb(433,   "IPCA mensal (%)")
inpc        = fetch_bcb(188,   "INPC mensal (%)")
usd         = fetch_bcb(3698,  "USD/BRL fim de período")
exportacoes = fetch_bcb(22708, "Exportações FOB total (US$ mi)")
importacoes = fetch_bcb(22704, "Importações FOB total (US$ mi)")
desemprego  = fetch_bcb(24369, "Desocupação mensal (%)")

# Categorias de exportação – fator agregado (BCB)
exp_basicos   = fetch_bcb(22707, "Exp. básicos (US$ mi)")
exp_semimanuf = fetch_bcb(22709, "Exp. semimanufaturados (US$ mi)")
exp_manuf     = fetch_bcb(22710, "Exp. manufaturados (US$ mi)")
exp_op_espec  = fetch_bcb(22711, "Exp. operações especiais (US$ mi)")

# Categorias de importação – uso econômico (BCB)
imp_bk   = fetch_bcb(22701, "Imp. bens de capital (US$ mi)")
imp_mp   = fetch_bcb(22702, "Imp. matérias-primas (US$ mi)")
imp_bc   = fetch_bcb(22703, "Imp. bens de consumo (US$ mi)")
imp_comb = fetch_bcb(22705, "Imp. combustíveis/lubrificantes (US$ mi)")

# Índices de quantum (volume físico)
q_exp = fetch_bcb(4447, "Índice quantum exportações (2006=100)")
q_imp = fetch_bcb(4448, "Índice quantum importações (2006=100)")

# ── Helpers ──────────────────────────────────────────────────────────────────

def to_records(df):
    if df is None or df.empty:
        return []
    return [{'d': row['data'].strftime('%Y-%m-%d'), 'v': round(float(row['valor']), 4)}
            for _, row in df.iterrows()]

def align_merge(df_a, df_b, col_a, col_b):
    return (df_a.rename(columns={'valor': col_a})
            .merge(df_b.rename(columns={'valor': col_b}), on='data', how='inner')
            .dropna())

def ols_model(df_merged, xcol, ycol, xlabel, ylabel):
    X = sm.add_constant(df_merged[xcol])
    y = df_merged[ycol]
    m = sm.OLS(y, X).fit()
    xs = df_merged[xcol].values
    a, b = float(m.params['const']), float(m.params[xcol])
    stats = {
        'n_obs':    int(len(df_merged)),
        'r2':       round(float(m.rsquared), 4),
        'r2_adj':   round(float(m.rsquared_adj), 4),
        'coef_a':   round(a, 4),
        'coef_b':   round(b, 4),
        'pval_a':   round(float(m.pvalues['const']), 6),
        'pval_b':   round(float(m.pvalues[xcol]), 6),
        'f_stat':   round(float(m.fvalue), 4),
        'f_pval':   round(float(m.f_pvalue), 6),
        'stderr_b': round(float(m.bse[xcol]), 4),
        'xlabel':   xlabel,
        'ylabel':   ylabel,
    }
    scatter = [{'x': round(float(row[xcol]), 3), 'y': round(float(row[ycol]), 3),
                'lb': row['data'].strftime('%b/%Y')}
               for _, row in df_merged.iterrows()]
    xmin, xmax = float(xs.min()), float(xs.max())
    trend = [{'x': round(xmin, 3), 'y': round(a + b * xmin, 4)},
             {'x': round(xmax, 3), 'y': round(a + b * xmax, 4)}]
    return stats, scatter, trend

def smooth_12m_acum(df):
    df = df.copy().reset_index(drop=True)
    out = []
    for i in range(11, len(df)):
        w = df.iloc[i-11:i+1]['valor']
        out.append({'data': df.iloc[i]['data'],
                    'valor': round(((1 + w / 100).prod() - 1) * 100, 4)})
    return pd.DataFrame(out)

def smooth_12m_mean(df):
    df = df.copy().reset_index(drop=True)
    out = []
    for i in range(11, len(df)):
        out.append({'data': df.iloc[i]['data'],
                    'valor': round(df.iloc[i-11:i+1]['valor'].mean(), 4)})
    return pd.DataFrame(out)

def first_diff(df):
    df = df.copy().reset_index(drop=True)
    df['valor'] = df['valor'].diff()
    return df.dropna().reset_index(drop=True)

def pct_change_df(df):
    df = df.copy().reset_index(drop=True)
    df['valor'] = df['valor'].pct_change() * 100
    return df.dropna().reset_index(drop=True)

# ── Regressões OLS ────────────────────────────────────────────────────────────
print("\n=== Regressoes OLS ===")

# Reg1: IPCA mensal ~ variação % mensal do USD (pass-through cambial)
reg1, scatter1, trend1 = {}, [], []
if ipca is not None and usd is not None:
    usd_ret = pct_change_df(usd)
    dm = align_merge(ipca, usd_ret, 'ipca', 'usd_ret')
    reg1, scatter1, trend1 = ols_model(dm, 'usd_ret', 'ipca',
                                        'Variacao mensal USD (%)', 'IPCA mensal (%)')
    print(f"  Reg1 IPCA~dUSD%   | R2={reg1['r2']} | n={reg1['n_obs']}")

# Reg2: IPCA acum 12m ~ USD medio 12m (nivel de cambio e inflacao acumulada)
reg2, scatter2, trend2 = {}, [], []
if ipca is not None and usd is not None:
    dm = align_merge(smooth_12m_acum(ipca), smooth_12m_mean(usd), 'ipca12', 'usd12')
    reg2, scatter2, trend2 = ols_model(dm, 'usd12', 'ipca12',
                                        'USD/BRL medio 12m (R$)', 'IPCA acum. 12m (%)')
    print(f"  Reg2 IPCA12~USD12 | R2={reg2['r2']} | n={reg2['n_obs']}")

# Reg3: primeiras diferencas IPCA ~ primeiras diferencas USD (impacto contemporaneo)
reg3, scatter3, trend3 = {}, [], []
if ipca is not None and usd is not None:
    dm = align_merge(first_diff(ipca), first_diff(usd), 'dipca', 'dusd')
    reg3, scatter3, trend3 = ols_model(dm, 'dusd', 'dipca',
                                        'delta USD/BRL (R$)', 'delta IPCA mensal (%)')
    print(f"  Reg3 dIPCA~dUSD   | R2={reg3['r2']} | n={reg3['n_obs']}")

# ── Payload ───────────────────────────────────────────────────────────────────
payload = {
    'ipca': to_records(ipca), 'inpc': to_records(inpc), 'usd': to_records(usd),
    'exportacoes': to_records(exportacoes), 'importacoes': to_records(importacoes),
    'desemprego':  to_records(desemprego),
    'exp_basicos': to_records(exp_basicos), 'exp_semimanuf': to_records(exp_semimanuf),
    'exp_manuf':   to_records(exp_manuf),   'exp_op_espec':  to_records(exp_op_espec),
    'imp_bk': to_records(imp_bk), 'imp_mp': to_records(imp_mp),
    'imp_bc': to_records(imp_bc), 'imp_comb': to_records(imp_comb),
    'q_exp': to_records(q_exp), 'q_imp': to_records(q_imp),
    'reg1': reg1, 'scatter1': scatter1, 'trend1': trend1,
    'reg2': reg2, 'scatter2': scatter2, 'trend2': trend2,
    'reg3': reg3, 'scatter3': scatter3, 'trend3': trend3,
    'atualizado': datetime.now().strftime('%d/%m/%Y %H:%M'),
}

print("\n=== Gerando HTML ===")

# ── HTML helpers ──────────────────────────────────────────────────────────────
def sig_stars(pval):
    if pval is None: return ''
    try: pval = float(pval)
    except: return ''
    if pval < 0.001: return '***'
    if pval < 0.01:  return '**'
    if pval < 0.05:  return '*'
    if pval < 0.1:   return '.'
    return ''

def r2_color(r2):
    if r2 > 0.5:  return '#34d399'
    if r2 > 0.3:  return '#fbbf24'
    if r2 > 0.1:  return '#fb923c'
    return '#f87171'

def reg_html_block(r, coef_label, interp_html):
    if not r:
        return '<p style="color:#8892a4">Dados insuficientes.</p>'
    r2v  = r.get('r2', 0) or 0
    r2c  = r2_color(r2v)
    r2p  = round(r2v * 100, 1)
    sa   = sig_stars(r.get('pval_a'))
    sb   = sig_stars(r.get('pval_b'))
    return f"""
    <div class="stat-row">
      <div class="stat-chip"><span>R²</span>{r.get('r2','—')}</div>
      <div class="stat-chip"><span>R² ajustado</span>{r.get('r2_adj','—')}</div>
      <div class="stat-chip"><span>F-statistic</span>{r.get('f_stat','—')}</div>
      <div class="stat-chip"><span>p-valor (F)</span>{r.get('f_pval','—')}</div>
      <div class="stat-chip"><span>Observações</span>{r.get('n_obs','—')}</div>
    </div>
    <div style="margin-bottom:14px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
        <span style="font-size:.75rem;color:var(--muted)">Poder explicativo (R²)</span>
        <span style="font-weight:700;color:{r2c}">{r2p}%</span>
      </div>
      <div style="background:#1e2130;border-radius:6px;height:10px;overflow:hidden">
        <div style="background:{r2c};height:100%;width:{r2p}%;border-radius:6px"></div>
      </div>
    </div>
    <table class="reg-table">
      <thead><tr><th>Variável</th><th>Coef.</th><th>Erro padrão</th><th>p-valor</th><th></th></tr></thead>
      <tbody>
        <tr><td>Constante α</td><td>{r.get('coef_a','—')}</td><td>—</td>
            <td>{r.get('pval_a','—')}</td><td class="sig">{sa}</td></tr>
        <tr><td>{coef_label} β</td><td>{r.get('coef_b','—')}</td>
            <td>{r.get('stderr_b','—')}</td>
            <td>{r.get('pval_b','—')}</td><td class="sig">{sb}</td></tr>
      </tbody>
    </table>
    <div style="font-size:.72rem;color:var(--muted);margin-top:7px">
      *** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05 &nbsp;. p&lt;0.1
    </div>
    <div class="interp">{interp_html}</div>"""

# Texts for each regression
cb1 = reg1.get('coef_b', '—')
r2_1 = reg1.get('r2', 0) or 0
interp1 = (
    f"<strong>Pass-through cambial (variação % mensal do dólar):</strong> "
    f"quando o dólar sobe 1%, o IPCA mensal varia em média <strong>{cb1} p.p.</strong> "
    f"(R² = {reg1.get('r2','—')} — {round(r2_1*100,1)}% da variância do IPCA explicada).<br><br>"
    f"Usar a <em>variação %</em> do câmbio (não o nível) é a especificação correta: "
    f"a inflação responde ao movimento mensal do dólar, não ao seu valor absoluto. "
    f"{'R² razoável — pass-through cambial detectável na frequência mensal.' if r2_1 > 0.1 else 'R² baixo em frequência mensal — o câmbio afeta inflação com defasagem de 1-3 meses (efeito maior em janelas trimestrais).'}"
)

cb2 = reg2.get('coef_b', '—')
r2_2 = reg2.get('r2', 0) or 0
interp2 = (
    f"<strong>Câmbio médio 12m vs. IPCA acumulado 12m:</strong> "
    f"séries suavizadas revelam a tendência estrutural — ruído mensal eliminado.<br><br>"
    f"β = {cb2}: para cada R$+1 na média do dólar nos últimos 12 meses, o IPCA acumulado 12m "
    f"varia {cb2} p.p. em média. "
    f"{'Relação positiva: câmbio mais alto associado a inflação acumulada maior — padrão esperado.' if (reg2.get('coef_b') or 0) > 0 else 'Relacao inesperada — pode refletir períodos de câmbio alto com deflação de commodities.'}<br><br>"
    f"R² = {reg2.get('r2','—')} — {'poder explicativo relevante: câmbio de longo prazo é um dos principais drivers da inflação acumulada.' if r2_2 > 0.2 else 'relação moderada — outros fatores (política monetária, preços de energia) também determinam a inflação acumulada.'}"
)

cb3 = reg3.get('coef_b', '—')
r2_3 = reg3.get('r2', 0) or 0
interp3 = (
    f"<strong>Primeiras diferenças (delta IPCA ~ delta USD):</strong> "
    f"elimina tendências comuns e testa causalidade de curto prazo sem risco de regressão espúria.<br><br>"
    f"β = {cb3}: uma variação de R$+1 no câmbio em t associa-se a {cb3} p.p. de variação "
    f"adicional no IPCA no mesmo mês.<br><br>"
    f"R² = {reg3.get('r2','—')} — {'impacto contemporâneo detectável.' if r2_3 > 0.05 else 'impacto imediato fraco — o câmbio tipicamente afeta inflação com 1-3 meses de defasagem.'}"
)

reg1_block = reg_html_block(reg1, 'Var. % USD mensal', interp1)
reg2_block = reg_html_block(reg2, 'USD medio 12m', interp2)
reg3_block = reg_html_block(reg3, 'delta USD/BRL', interp3)

# ── CSS (regular string — no f-string escaping for braces) ───────────────────
css = """<style>
  :root {
    --bg:#0f1117;--card:#1a1d27;--border:#2a2d3e;
    --text:#e2e8f0;--muted:#8892a4;--accent:#4f8ef7;
    --green:#34d399;--red:#f87171;--yellow:#fbbf24;--purple:#a78bfa;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
  header{background:linear-gradient(135deg,#1a1d27,#0f1117);border-bottom:1px solid var(--border);
    padding:18px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
  header h1{font-size:1.35rem;font-weight:700}
  header h1 span{color:var(--accent)}
  .badge{display:inline-flex;align-items:center;background:rgba(79,142,247,.12);
    border:1px solid rgba(79,142,247,.3);color:var(--accent);border-radius:20px;
    padding:3px 10px;font-size:.72rem;font-weight:500}
  .container{max-width:1300px;margin:0 auto;padding:22px 18px}
  .filter-bar{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .filter-bar label{font-size:.82rem;color:var(--muted);white-space:nowrap}
  .filter-bar select{background:var(--bg);border:1px solid var(--border);color:var(--text);
    border-radius:7px;padding:5px 10px;font-size:.82rem;cursor:pointer;outline:none}
  .filter-bar select:focus{border-color:var(--accent)}
  .filter-bar .sep{color:var(--border);font-size:1.2rem;padding:0 2px}
  .filter-bar button{background:var(--accent);color:#fff;border:none;border-radius:7px;
    padding:6px 16px;font-size:.82rem;cursor:pointer;transition:opacity .2s}
  .filter-bar button.sec{background:transparent;border:1px solid var(--border);color:var(--muted)}
  .filter-bar button:hover{opacity:.82}
  #filtroInfo{font-size:.78rem;color:var(--muted);margin-left:4px;font-weight:500}
  #modo-badge{margin-left:auto;font-size:.75rem;padding:4px 12px;font-weight:700;transition:all .3s}
  .tabs{display:flex;gap:4px;margin-bottom:18px;flex-wrap:wrap}
  .tab{background:transparent;border:1px solid var(--border);color:var(--muted);
    border-radius:8px;padding:7px 16px;font-size:.83rem;cursor:pointer;transition:all .2s}
  .tab.active{background:var(--accent);border-color:var(--accent);color:#fff}
  .tab:hover:not(.active){border-color:var(--accent);color:var(--accent)}
  .section{display:none}
  .section.active{display:block}
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}
  .grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:18px}
  @media(max-width:900px){.grid4{grid-template-columns:1fr 1fr}.grid3{grid-template-columns:1fr 1fr}}
  @media(max-width:500px){.grid4,.grid3,.grid2{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
  .card-full{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:18px;margin-bottom:18px}
  .card h3{font-size:.88rem;font-weight:600;margin-bottom:3px}
  .card .sub{font-size:.7rem;color:var(--muted);margin-bottom:12px}
  .card .tag12{font-size:.68rem;color:var(--accent);background:rgba(79,142,247,.1);
    border-radius:4px;padding:1px 5px;margin-left:5px}
  .kpi{font-size:1.8rem;font-weight:700}
  .kpi-label{font-size:.7rem;color:var(--muted);margin-top:3px}
  .kpi.up{color:var(--red)}.kpi.down{color:var(--green)}.kpi.neutral{color:var(--accent)}
  /* snapshot-specific */
  .snap-header{background:linear-gradient(90deg,rgba(79,142,247,.08),transparent);
    border:1px solid rgba(79,142,247,.2);border-radius:10px;padding:12px 18px;
    margin-bottom:16px;display:flex;align-items:center;gap:10px}
  .snap-header .snap-label{font-size:1rem;font-weight:700;color:var(--accent)}
  .snap-header .snap-sub{font-size:.78rem;color:var(--muted)}
  .snap-kpi{font-size:2.4rem;font-weight:700;line-height:1}
  .snap-kpi-label{font-size:.72rem;color:var(--muted);margin-top:4px}
  .snap-mini{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:16px 18px;margin-bottom:14px}
  .snap-mini h4{font-size:.82rem;color:var(--muted);margin-bottom:10px;font-weight:500}
  /* category filter bar */
  .cat-bar{background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:10px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .cat-bar label{font-size:.8rem;color:var(--muted);white-space:nowrap}
  .cat-bar select{background:var(--bg);border:1px solid var(--border);color:var(--text);
    border-radius:7px;padding:4px 10px;font-size:.8rem;cursor:pointer;outline:none}
  .cat-bar select:focus{border-color:var(--accent)}
  canvas{max-height:300px}
  /* regression */
  .reg-sub-tabs{display:flex;gap:4px;margin-bottom:18px;flex-wrap:wrap}
  .reg-sub-tab{background:transparent;border:1px solid var(--border);color:var(--muted);
    border-radius:8px;padding:6px 14px;font-size:.8rem;cursor:pointer;transition:all .2s}
  .reg-sub-tab.active{background:rgba(167,139,250,.15);border-color:var(--purple);color:var(--purple)}
  .reg-sub-tab:hover:not(.active){border-color:var(--purple);color:var(--purple)}
  .reg-panel{display:none}
  .reg-panel.active{display:block}
  .reg-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:768px){.reg-grid{grid-template-columns:1fr}}
  .reg-table{width:100%;border-collapse:collapse;font-size:.83rem;margin-top:14px}
  .reg-table th{text-align:left;color:var(--muted);font-weight:500;padding:7px 10px;
    border-bottom:1px solid var(--border)}
  .reg-table td{padding:7px 10px;border-bottom:1px solid var(--border)}
  .reg-table tr:last-child td{border-bottom:none}
  .sig{color:var(--yellow);font-weight:600;margin-left:4px}
  .interp{background:rgba(79,142,247,.07);border-left:3px solid var(--accent);
    border-radius:0 8px 8px 0;padding:13px 15px;font-size:.83rem;line-height:1.7;margin-top:14px}
  .interp strong{color:var(--accent)}
  .stat-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
  .stat-chip{background:rgba(255,255,255,.05);border:1px solid var(--border);
    border-radius:7px;padding:7px 12px;font-size:.8rem}
  .stat-chip span{color:var(--muted);font-size:.72rem;display:block}
  .sources{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:18px 22px;margin-top:22px}
  .sources h4{font-size:.8rem;font-weight:600;color:var(--muted);margin-bottom:10px;
    letter-spacing:.05em;text-transform:uppercase}
  .sources-grid{display:flex;flex-wrap:wrap;gap:8px}
  .source-tag{display:flex;align-items:center;gap:7px;background:rgba(255,255,255,.04);
    border:1px solid var(--border);border-radius:9px;padding:7px 12px}
  .source-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
  .source-tag .sname{font-size:.78rem;font-weight:600}
  .source-tag .sdesc{font-size:.7rem;color:var(--muted)}
  .ncm-note{font-size:.72rem;color:var(--muted);font-style:italic;padding:4px 0}
</style>"""

# ── JavaScript (regular string — embeds JSON via concatenation) ───────────────
j = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

js_code = "const DATA = " + j + """;
const MESES = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'];

let fIni = {ano:2015,mes:1};
let fFim = {ano:2025,mes:12};

(function initDropdowns() {
  ['fM1','fM2'].forEach(id => {
    const s = document.getElementById(id);
    MESES.forEach((m,i) => s.add(new Option(m, i+1)));
  });
  ['fA1','fA2'].forEach(id => {
    const s = document.getElementById(id);
    for (let a=2015;a<=2025;a++) s.add(new Option(a,a));
  });
  document.getElementById('fM1').value = 1;
  document.getElementById('fA1').value = 2015;
  document.getElementById('fM2').value = 12;
  document.getElementById('fA2').value = 2025;
})();

function aplicarFiltro() {
  const a1=+document.getElementById('fA1').value, m1=+document.getElementById('fM1').value;
  const a2=+document.getElementById('fA2').value, m2=+document.getElementById('fM2').value;
  if (a1*100+m1 > a2*100+m2) {
    document.getElementById('fA2').value=a1; document.getElementById('fM2').value=m1;
  }
  fIni = {ano:+document.getElementById('fA1').value, mes:+document.getElementById('fM1').value};
  fFim = {ano:+document.getElementById('fA2').value, mes:+document.getElementById('fM2').value};
  atualizarTodos();
}

function resetFiltro() {
  fIni={ano:2015,mes:1}; fFim={ano:2025,mes:12};
  document.getElementById('fM1').value=1; document.getElementById('fA1').value=2015;
  document.getElementById('fM2').value=12; document.getElementById('fA2').value=2025;
  atualizarTodos();
}

function isModoMes() {
  return fIni.ano === fFim.ano && fIni.mes === fFim.mes;
}

function dk(a,m){ return a*100+m; }
function pk(p){ return parseInt(p.d.slice(0,4))*100+parseInt(p.d.slice(5,7)); }

function filtrar(arr) {
  if(!arr||!arr.length) return [];
  const ini=dk(fIni.ano,fIni.mes), fim=dk(fFim.ano,fFim.mes);
  return arr.filter(p=>{ const k=pk(p); return k>=ini&&k<=fim; });
}

function ultimos(arr,n) {
  if(!arr||!arr.length) return [];
  const fim=dk(fFim.ano,fFim.mes);
  return arr.filter(p=>pk(p)<=fim).slice(-n);
}

function acumInflacao(pts) {
  if(!pts||!pts.length) return null;
  let a=1.0; pts.forEach(p=>a*=(1+p.v/100));
  return r2((a-1)*100);
}
function media(pts){ return pts&&pts.length?r2(pts.reduce((s,p)=>s+p.v,0)/pts.length):null; }

function calcAcum12(arr) {
  const fim=dk(fFim.ano,fFim.mes);
  const src=arr.filter(p=>pk(p)<=fim);
  const out=[];
  for(let i=11;i<src.length;i++){
    let a=1.0;
    for(let j=i-11;j<=i;j++) a*=(1+src[j].v/100);
    out.push({x:src[i].d, y:r4((a-1)*100)});
  }
  return out;
}

function r2(v){ return Math.round(v*100)/100; }
function r4(v){ return Math.round(v*10000)/10000; }
function last(a){ return a&&a.length?a[a.length-1]:null; }
function minV(a){ return a&&a.length?Math.min(...a.map(p=>p.v)):null; }
function maxV(a){ return a&&a.length?Math.max(...a.map(p=>p.v)):null; }
function fmt(v,d=2){ return v!=null?v.toFixed(d):'—'; }
function set(id,val){ const e=document.getElementById(id); if(e) e.textContent=val; }
function toXY(arr){ return arr?arr.map(p=>({x:p.d,y:p.v})):[] }

const PAL = {
  ipca:'#f87171', inpc:'#fbbf24', usd:'#4f8ef7',
  exp:'#34d399', imp:'#f87171', des:'#a78bfa',
  gp:'#34d399',  gn:'#f87171',  qexp:'#34d399', qimp:'#60a5fa',
};

function makeChart(id, type, datasets, extra={}) {
  const ctx=document.getElementById(id);
  if(!ctx) return null;
  if(ctx._ch) ctx._ch.destroy();
  ctx._ch = new Chart(ctx, {
    type,
    data:{datasets},
    options:{
      responsive:true,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#8892a4',font:{size:11}}},
        tooltip:{backgroundColor:'#1a1d27',borderColor:'#2a2d3e',borderWidth:1,
                 titleColor:'#e2e8f0',bodyColor:'#8892a4'},
        ...(extra.plugins||{})
      },
      scales:{
        x:{type:'time',time:{unit:'year',tooltipFormat:'MMM/yyyy'},
           grid:{color:'#1e2130'},ticks:{color:'#8892a4',maxTicksLimit:12}},
        y:{grid:{color:'#1e2130'},ticks:{color:'#8892a4'}},
        ...(extra.scales||{})
      },
      ...(extra.opts||{})
    }
  });
  return ctx._ch;
}

function getExpData(){
  const v=document.getElementById('catExp').value;
  const m={total:DATA.exportacoes,basicos:DATA.exp_basicos,semimanuf:DATA.exp_semimanuf,
           manuf:DATA.exp_manuf,op_espec:DATA.exp_op_espec};
  return filtrar(m[v]||DATA.exportacoes);
}
function getImpData(){
  const v=document.getElementById('catImp').value;
  const m={total:DATA.importacoes,bk:DATA.imp_bk,mp:DATA.imp_mp,bc:DATA.imp_bc,comb:DATA.imp_comb};
  return filtrar(m[v]||DATA.importacoes);
}
function getExpLabel(){
  const v=document.getElementById('catExp').value;
  return {total:'Exportações totais',basicos:'Básicos',semimanuf:'Semimanufaturados',
          manuf:'Manufaturados',op_espec:'Op. Especiais'}[v]||'Exportações';
}
function getImpLabel(){
  const v=document.getElementById('catImp').value;
  return {total:'Importações totais',bk:'Bens de Capital',mp:'Matérias-Primas',
          bc:'Bens de Consumo',comb:'Combustíveis'}[v]||'Importações';
}

function atualizarKPIs(ipca,inpc,usd,des,exp_,imp_) {
  const li=last(ipca),ln=last(inpc),lu=last(usd),ld=last(des),le=last(exp_),lim=last(imp_);
  if(li) set('kpi-ipca', fmt(li.v)+'%');
  if(ln) set('kpi-inpc', fmt(ln.v)+'%');
  const i12=acumInflacao(ultimos(DATA.ipca,12));
  const n12=acumInflacao(ultimos(DATA.inpc,12));
  if(i12!=null) set('kpi-ipca12', fmt(i12)+'%');
  if(n12!=null) set('kpi-inpc12', fmt(n12)+'%');
  if(lu) set('kpi-usd','R$ '+fmt(lu.v));
  if(usd.length){ set('kpi-usd-min','R$ '+fmt(minV(usd))); set('kpi-usd-max','R$ '+fmt(maxV(usd))); }
  const u12=media(ultimos(DATA.usd,12));
  if(u12!=null) set('kpi-usd-med12','R$ '+fmt(u12));
  if(ld) set('kpi-des',fmt(ld.v)+'%');
  if(des.length){ set('kpi-des-min',fmt(minV(des))+'%'); set('kpi-des-max',fmt(maxV(des))+'%'); }
  const d12=media(ultimos(DATA.desemprego,12));
  if(d12!=null) set('kpi-des-med12',fmt(d12)+'%');
  if(le) set('kpi-exp',fmt(le.v,0));
  if(lim) set('kpi-imp',fmt(lim.v,0));
  if(le&&lim){
    const s=le.v-lim.v;
    const el=document.getElementById('kpi-saldo');
    if(el){ el.textContent=(s>=0?'+':'')+fmt(s,0); el.className='kpi '+(s>=0?'down':'up'); }
  }
  // Snapshot KPIs (same values, different elements for larger display)
  const ml = MESES[fFim.mes-1]+'/'+fFim.ano;
  document.querySelectorAll('.snap-mes-label').forEach(e=>e.textContent=ml);
  if(li) set('snap-ipca', fmt(li.v)+'%');
  if(ln) set('snap-inpc', fmt(ln.v)+'%');
  if(i12!=null) set('snap-ipca12', fmt(i12)+'%');
  if(n12!=null) set('snap-inpc12', fmt(n12)+'%');
  if(lu) set('snap-usd', 'R$ '+fmt(lu.v));
  if(u12!=null) set('snap-usd12', 'R$ '+fmt(u12));
  if(ld) set('snap-des', fmt(ld.v)+'%');
  if(d12!=null) set('snap-des12', fmt(d12)+'%');
  if(le) set('snap-exp', fmt(le.v,0));
  if(lim) set('snap-imp', fmt(lim.v,0));
  if(le&&lim){ const s=le.v-lim.v; set('snap-saldo',(s>=0?'+':'')+fmt(s,0)); }
}

function drawPeriodCharts(ipca,inpc,usd,des,exp_,imp_) {
  makeChart('chart-inflacao-mensal','line',[
    {label:'IPCA (%)',data:toXY(ipca),borderColor:PAL.ipca,backgroundColor:PAL.ipca+'22',tension:.3,pointRadius:0,fill:true},
    {label:'INPC (%)',data:toXY(inpc),borderColor:PAL.inpc,backgroundColor:PAL.inpc+'11',tension:.3,pointRadius:0,borderDash:[4,4]}
  ]);
  makeChart('chart-inflacao-acum','line',[
    {label:'IPCA acum. 12m (%)',data:calcAcum12(DATA.ipca),borderColor:PAL.ipca,tension:.3,pointRadius:0},
    {label:'INPC acum. 12m (%)',data:calcAcum12(DATA.inpc),borderColor:PAL.inpc,tension:.3,pointRadius:0,borderDash:[4,4]}
  ]);
  makeChart('chart-cambio','line',[
    {label:'USD/BRL (R$)',data:toXY(usd),borderColor:PAL.usd,backgroundColor:PAL.usd+'18',tension:.3,pointRadius:0,fill:true}
  ]);
  makeChart('chart-desemprego','line',[
    {label:'Desocupação (%)',data:toXY(des),borderColor:PAL.des,backgroundColor:PAL.des+'18',tension:.3,pointRadius:0,fill:true}
  ]);
  makeChart('chart-comercio','line',[
    {label:getExpLabel()+' (US$ mi)',data:toXY(exp_),borderColor:PAL.exp,tension:.3,pointRadius:0},
    {label:getImpLabel()+' (US$ mi)',data:toXY(imp_),borderColor:PAL.imp,tension:.3,pointRadius:0,borderDash:[4,4]}
  ]);
  const saldo=exp_.map(p=>{const im=imp_.find(m=>m.d===p.d);return im?{x:p.d,y:p.v-im.v}:null;}).filter(Boolean);
  makeChart('chart-saldo','bar',[
    {label:'Saldo (US$ mi)',data:saldo,
     backgroundColor:saldo.map(p=>p.y>=0?PAL.gp+'88':PAL.gn+'88'),
     borderColor:saldo.map(p=>p.y>=0?PAL.gp:PAL.gn),borderWidth:1}
  ]);
  // Quantum chart
  const qexp=filtrar(DATA.q_exp), qimp=filtrar(DATA.q_imp);
  if(qexp.length||qimp.length){
    makeChart('chart-quantum','line',[
      {label:'Quantum Exportações (2006=100)',data:toXY(qexp),borderColor:PAL.qexp,tension:.3,pointRadius:0},
      {label:'Quantum Importações (2006=100)',data:toXY(qimp),borderColor:PAL.qimp,tension:.3,pointRadius:0,borderDash:[4,4]}
    ]);
  }
  // Value vs Volume scatter
  const expFob=filtrar(DATA.exportacoes);
  const expQ=filtrar(DATA.q_exp);
  const scVV=expFob.map(p=>{const q=expQ.find(m=>m.d===p.d);return q?{x:q.v,y:p.v,lb:p.d.slice(0,7)}:null;}).filter(Boolean);
  if(scVV.length){
    const vvCtx=document.getElementById('chart-vol-fob');
    if(vvCtx){
      if(vvCtx._ch) vvCtx._ch.destroy();
      vvCtx._ch = new Chart(vvCtx,{
        type:'scatter',
        data:{datasets:[{label:'Exp. FOB vs Quantum',data:scVV.map(p=>({x:p.x,y:p.y,lb:p.lb})),
          backgroundColor:PAL.exp+'55',borderColor:PAL.exp,pointRadius:4}]},
        options:{
          responsive:true,
          plugins:{
            legend:{labels:{color:'#8892a4',font:{size:11}}},
            tooltip:{backgroundColor:'#1a1d27',borderColor:'#2a2d3e',borderWidth:1,
              callbacks:{label:c=>{const r=c.raw;return r.lb?`${r.lb}: Q=${r.x} | FOB=US$${r.y}mi`:''}}}
          },
          scales:{
            x:{title:{display:true,text:'Índice Quantum (2006=100)',color:'#8892a4'},grid:{color:'#1e2130'},ticks:{color:'#8892a4'}},
            y:{title:{display:true,text:'Exportações FOB (US$ mi)',color:'#8892a4'},grid:{color:'#1e2130'},ticks:{color:'#8892a4'}}
          }
        }
      });
    }
  }
}

function drawSnapCharts() {
  const opts = {opts:{animation:false}};
  const i12=ultimos(DATA.ipca,12), n12=ultimos(DATA.inpc,12);
  makeChart('chart-snap-infl','line',[
    {label:'IPCA (%)',data:toXY(i12),borderColor:PAL.ipca,tension:.3,pointRadius:2,borderWidth:2},
    {label:'INPC (%)',data:toXY(n12),borderColor:PAL.inpc,tension:.3,pointRadius:2,borderDash:[4,4],borderWidth:2}
  ], opts);
  makeChart('chart-snap-usd','line',[
    {label:'USD/BRL',data:toXY(ultimos(DATA.usd,12)),borderColor:PAL.usd,fill:true,backgroundColor:PAL.usd+'18',tension:.3,pointRadius:2,borderWidth:2}
  ], opts);
  makeChart('chart-snap-des','line',[
    {label:'Desocupação (%)',data:toXY(ultimos(DATA.desemprego,12)),borderColor:PAL.des,fill:true,backgroundColor:PAL.des+'18',tension:.3,pointRadius:2,borderWidth:2}
  ], opts);
  makeChart('chart-snap-com','line',[
    {label:'Exportações',data:toXY(ultimos(DATA.exportacoes,12)),borderColor:PAL.exp,tension:.3,pointRadius:2,borderWidth:2},
    {label:'Importações',data:toXY(ultimos(DATA.importacoes,12)),borderColor:PAL.imp,tension:.3,pointRadius:2,borderDash:[4,4],borderWidth:2}
  ], opts);
}

// Regression scatters – static (full series, not filter-dependent)
function drawRegScatters() {
  function drawSc(canvasId, scatter, trend, xlabel, ylabel) {
    const ctx=document.getElementById(canvasId);
    if(!ctx||!scatter||!scatter.length) return;
    if(ctx._ch) ctx._ch.destroy();
    ctx._ch = new Chart(ctx,{
      type:'scatter',
      data:{datasets:[
        {label:'Observações',data:scatter.map(p=>({x:p.x,y:p.y,lb:p.lb})),
         backgroundColor:'#4f8ef755',borderColor:'#4f8ef7',pointRadius:4,pointHoverRadius:6},
        {label:'Tendência OLS',type:'line',data:trend,
         borderColor:'#f87171',backgroundColor:'transparent',pointRadius:0,borderWidth:2}
      ]},
      options:{
        responsive:true,
        plugins:{
          legend:{labels:{color:'#8892a4',font:{size:11}}},
          tooltip:{backgroundColor:'#1a1d27',borderColor:'#2a2d3e',borderWidth:1,
            callbacks:{label:c=>{const r=c.raw;return r.lb?`${r.lb}: (${r.x}, ${r.y})`:'';}}}
        },
        scales:{
          x:{title:{display:true,text:xlabel,color:'#8892a4'},grid:{color:'#1e2130'},ticks:{color:'#8892a4'}},
          y:{title:{display:true,text:ylabel,color:'#8892a4'},grid:{color:'#1e2130'},ticks:{color:'#8892a4'}}
        }
      }
    });
  }
  if(DATA.reg1&&DATA.scatter1) drawSc('chart-scatter1',DATA.scatter1,DATA.trend1,DATA.reg1.xlabel||'X',DATA.reg1.ylabel||'Y');
  if(DATA.reg2&&DATA.scatter2) drawSc('chart-scatter2',DATA.scatter2,DATA.trend2,DATA.reg2.xlabel||'X',DATA.reg2.ylabel||'Y');
  if(DATA.reg3&&DATA.scatter3) drawSc('chart-scatter3',DATA.scatter3,DATA.trend3,DATA.reg3.xlabel||'X',DATA.reg3.ylabel||'Y');
}

function atualizarTodos() {
  const modo = isModoMes();
  const mb=document.getElementById('modo-badge');
  if(mb){
    mb.textContent = modo ? 'MÊS' : 'PERÍODO';
    mb.style.background = modo?'rgba(52,211,153,.15)':'rgba(79,142,247,.12)';
    mb.style.color = modo?'#34d399':'#4f8ef7';
    mb.style.borderColor = modo?'rgba(52,211,153,.3)':'rgba(79,142,247,.3)';
  }
  set('filtroInfo', modo?
    MESES[fFim.mes-1]+'/'+fFim.ano :
    MESES[fIni.mes-1]+'/'+fIni.ano+' → '+MESES[fFim.mes-1]+'/'+fFim.ano);

  document.querySelectorAll('.snap').forEach(el=>el.style.display=modo?'':'none');
  document.querySelectorAll('.prd').forEach(el=>el.style.display=modo?'none':'');

  const ipca=filtrar(DATA.ipca), inpc=filtrar(DATA.inpc), usd=filtrar(DATA.usd);
  const des=filtrar(DATA.desemprego);
  const exp_=getExpData(), imp_=getImpData();

  atualizarKPIs(ipca,inpc,usd,des,exp_,imp_);

  if(modo) drawSnapCharts();
  else     drawPeriodCharts(ipca,inpc,usd,des,exp_,imp_);
}

function showTab(name,btn){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
  atualizarTodos();
}

function showReg(idx,btn){
  document.querySelectorAll('.reg-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.reg-sub-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('reg-panel-'+idx).classList.add('active');
  btn.classList.add('active');
}

drawRegScatters();
atualizarTodos();
"""

# ── Full HTML ─────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Painel Macroeconômico Brasil</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
{css}
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
    <span id="modo-badge" class="badge">PERÍODO</span>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('inflacao',this)">Inflação</button>
    <button class="tab" onclick="showTab('cambio',this)">Câmbio</button>
    <button class="tab" onclick="showTab('desemprego',this)">Desemprego</button>
    <button class="tab" onclick="showTab('comercio',this)">Comércio Exterior</button>
    <button class="tab" onclick="showTab('regressao',this)">Regressões</button>
  </div>

  <!-- ═══ INFLAÇÃO ═══ -->
  <div id="tab-inflacao" class="section active">

    <!-- Snapshot: mês único -->
    <div class="snap" style="display:none">
      <div class="snap-header">
        <span class="snap-label" id="snap-mes-infl"></span>
        <span class="snap-sub snap-mes-label"></span>
        <span class="snap-sub" style="margin-left:8px">· Selecione um intervalo para ver gráficos</span>
      </div>
      <div class="grid2" style="margin-bottom:14px">
        <div class="card" style="padding:20px 22px">
          <h3>IPCA mensal</h3><div class="sub">BCB série 433</div>
          <div class="snap-kpi up" id="snap-ipca">—</div>
          <div class="snap-kpi-label">% ao mês</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>INPC mensal</h3><div class="sub">BCB série 188</div>
          <div class="snap-kpi up" id="snap-inpc">—</div>
          <div class="snap-kpi-label">% ao mês</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>IPCA acumulado <span class="tag12">12m</span></h3>
          <div class="sub">Produto encadeado dos últimos 12 meses</div>
          <div class="snap-kpi up" id="snap-ipca12">—</div>
          <div class="snap-kpi-label">% acumulado</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>INPC acumulado <span class="tag12">12m</span></h3>
          <div class="sub">Produto encadeado dos últimos 12 meses</div>
          <div class="snap-kpi up" id="snap-inpc12">—</div>
          <div class="snap-kpi-label">% acumulado</div>
        </div>
      </div>
      <div class="snap-mini">
        <h4>Últimos 12 meses até <span class="snap-mes-label"></span></h4>
        <canvas id="chart-snap-infl" style="max-height:200px"></canvas>
      </div>
    </div>

    <!-- Period: range -->
    <div class="prd">
      <div class="grid4">
        <div class="card"><h3>IPCA mensal</h3><div class="sub">Último mês do período · BCB 433</div>
          <div class="kpi up" id="kpi-ipca">—</div><div class="kpi-label">% ao mês</div></div>
        <div class="card"><h3>INPC mensal</h3><div class="sub">Último mês do período · BCB 188</div>
          <div class="kpi up" id="kpi-inpc">—</div><div class="kpi-label">% ao mês</div></div>
        <div class="card"><h3>IPCA acum.<span class="tag12">12m</span></h3>
          <div class="sub">Até o fim do período</div>
          <div class="kpi up" id="kpi-ipca12">—</div><div class="kpi-label">% acumulado</div></div>
        <div class="card"><h3>INPC acum.<span class="tag12">12m</span></h3>
          <div class="sub">Até o fim do período</div>
          <div class="kpi up" id="kpi-inpc12">—</div><div class="kpi-label">% acumulado</div></div>
      </div>
      <div class="card-full">
        <h3>IPCA e INPC – Variação Mensal (%)</h3>
        <div class="sub">Banco Central do Brasil · séries 433 (IPCA) e 188 (INPC)</div>
        <canvas id="chart-inflacao-mensal"></canvas>
      </div>
      <div class="card-full">
        <h3>IPCA e INPC – Acumulado 12 meses (%)</h3>
        <div class="sub">Calculado sobre séries mensais do BCB</div>
        <canvas id="chart-inflacao-acum"></canvas>
      </div>
    </div>
  </div>

  <!-- ═══ CÂMBIO ═══ -->
  <div id="tab-cambio" class="section">

    <div class="snap" style="display:none">
      <div class="snap-header">
        <span class="snap-label">USD/BRL</span>
        <span class="snap-sub snap-mes-label" style="margin-left:8px"></span>
      </div>
      <div class="grid2" style="margin-bottom:14px">
        <div class="card" style="padding:20px 22px">
          <h3>Câmbio do mês</h3><div class="sub">Fim de período · BCB 3698</div>
          <div class="snap-kpi neutral" id="snap-usd">—</div>
          <div class="snap-kpi-label">R$ por US$</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>Média <span class="tag12">12m</span></h3>
          <div class="sub">Média dos últimos 12 meses</div>
          <div class="snap-kpi neutral" id="snap-usd12">—</div>
          <div class="snap-kpi-label">R$ médio</div>
        </div>
      </div>
      <div class="snap-mini">
        <h4>USD/BRL – últimos 12 meses até <span class="snap-mes-label"></span></h4>
        <canvas id="chart-snap-usd" style="max-height:200px"></canvas>
      </div>
    </div>

    <div class="prd">
      <div class="grid4">
        <div class="card"><h3>USD/BRL</h3><div class="sub">Último mês · BCB 3698</div>
          <div class="kpi neutral" id="kpi-usd">—</div><div class="kpi-label">R$ por US$</div></div>
        <div class="card"><h3>Mín. no período</h3><div class="sub"></div>
          <div class="kpi down" id="kpi-usd-min">—</div><div class="kpi-label">R$</div></div>
        <div class="card"><h3>Máx. no período</h3><div class="sub"></div>
          <div class="kpi up" id="kpi-usd-max">—</div><div class="kpi-label">R$</div></div>
        <div class="card"><h3>Média <span class="tag12">12m</span></h3>
          <div class="sub">Até o fim do período</div>
          <div class="kpi neutral" id="kpi-usd-med12">—</div><div class="kpi-label">R$ média</div></div>
      </div>
      <div class="card-full">
        <h3>Taxa de Câmbio – USD/BRL</h3>
        <div class="sub">Banco Central do Brasil · série 3698 · fim de período mensal</div>
        <canvas id="chart-cambio"></canvas>
      </div>
    </div>
  </div>

  <!-- ═══ DESEMPREGO ═══ -->
  <div id="tab-desemprego" class="section">

    <div class="snap" style="display:none">
      <div class="snap-header">
        <span class="snap-label">Desocupação</span>
        <span class="snap-sub snap-mes-label" style="margin-left:8px"></span>
      </div>
      <div class="grid2" style="margin-bottom:14px">
        <div class="card" style="padding:20px 22px">
          <h3>Taxa do mês</h3><div class="sub">PNAD Contínua · BCB 24369</div>
          <div class="snap-kpi up" id="snap-des">—</div>
          <div class="snap-kpi-label">% da força de trabalho</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>Média <span class="tag12">12m</span></h3>
          <div class="sub">Média dos últimos 12 meses</div>
          <div class="snap-kpi up" id="snap-des12">—</div>
          <div class="snap-kpi-label">% médio</div>
        </div>
      </div>
      <div class="snap-mini">
        <h4>Desocupação – últimos 12 meses até <span class="snap-mes-label"></span></h4>
        <canvas id="chart-snap-des" style="max-height:200px"></canvas>
      </div>
    </div>

    <div class="prd">
      <div class="grid4">
        <div class="card"><h3>Desocupação</h3><div class="sub">Último mês · BCB 24369</div>
          <div class="kpi up" id="kpi-des">—</div><div class="kpi-label">% força de trabalho</div></div>
        <div class="card"><h3>Mín. no período</h3><div class="sub"></div>
          <div class="kpi down" id="kpi-des-min">—</div><div class="kpi-label">%</div></div>
        <div class="card"><h3>Máx. no período</h3><div class="sub"></div>
          <div class="kpi up" id="kpi-des-max">—</div><div class="kpi-label">%</div></div>
        <div class="card"><h3>Média <span class="tag12">12m</span></h3>
          <div class="sub">Até o fim do período</div>
          <div class="kpi up" id="kpi-des-med12">—</div><div class="kpi-label">% média</div></div>
      </div>
      <div class="card-full">
        <h3>Taxa de Desocupação Mensal (%)</h3>
        <div class="sub">Banco Central do Brasil · série 24369 · PNAD Contínua</div>
        <canvas id="chart-desemprego"></canvas>
      </div>
    </div>
  </div>

  <!-- ═══ COMÉRCIO EXTERIOR ═══ -->
  <div id="tab-comercio" class="section">

    <!-- Category filter (always visible) -->
    <div class="cat-bar">
      <label>Exportações:</label>
      <select id="catExp" onchange="atualizarTodos()">
        <option value="total">Todas as categorias</option>
        <option value="basicos">Básicos</option>
        <option value="semimanuf">Semimanufaturados</option>
        <option value="manuf">Manufaturados</option>
        <option value="op_espec">Operações Especiais</option>
      </select>
      <label style="margin-left:10px">Importações:</label>
      <select id="catImp" onchange="atualizarTodos()">
        <option value="total">Todas as categorias</option>
        <option value="bk">Bens de Capital</option>
        <option value="mp">Matérias-Primas</option>
        <option value="bc">Bens de Consumo</option>
        <option value="comb">Combustíveis e Lubrificantes</option>
      </select>
      <span class="ncm-note">· Categorias por fator agregado (BCB) — desagregação por NCM disponível quando MDIC/ComexStat retornar</span>
    </div>

    <!-- Snapshot -->
    <div class="snap" style="display:none">
      <div class="snap-header">
        <span class="snap-label">Comércio Exterior</span>
        <span class="snap-sub snap-mes-label" style="margin-left:8px"></span>
      </div>
      <div class="grid3" style="margin-bottom:14px">
        <div class="card" style="padding:20px 22px">
          <h3>Exportações</h3><div class="sub">Mês selecionado · US$ milhões FOB</div>
          <div class="snap-kpi neutral" id="snap-exp">—</div>
          <div class="snap-kpi-label">US$ milhões</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>Importações</h3><div class="sub">Mês selecionado · US$ milhões FOB</div>
          <div class="snap-kpi neutral" id="snap-imp">—</div>
          <div class="snap-kpi-label">US$ milhões</div>
        </div>
        <div class="card" style="padding:20px 22px">
          <h3>Saldo</h3><div class="sub">Exp. − Imp. · mês selecionado</div>
          <div class="snap-kpi neutral" id="snap-saldo">—</div>
          <div class="snap-kpi-label">US$ milhões</div>
        </div>
      </div>
      <div class="snap-mini">
        <h4>Exportações e Importações – últimos 12 meses até <span class="snap-mes-label"></span></h4>
        <canvas id="chart-snap-com" style="max-height:200px"></canvas>
      </div>
    </div>

    <!-- Period -->
    <div class="prd">
      <div class="grid3">
        <div class="card"><h3>Exportações</h3><div class="sub">Último mês do período · BCB 22708</div>
          <div class="kpi neutral" id="kpi-exp">—</div><div class="kpi-label">US$ milhões (FOB)</div></div>
        <div class="card"><h3>Importações</h3><div class="sub">Último mês do período · BCB 22704</div>
          <div class="kpi neutral" id="kpi-imp">—</div><div class="kpi-label">US$ milhões (FOB)</div></div>
        <div class="card"><h3>Saldo comercial</h3><div class="sub">Exp. − Imp. · último mês</div>
          <div class="kpi neutral" id="kpi-saldo">—</div><div class="kpi-label">US$ milhões</div></div>
      </div>
      <div class="card-full">
        <h3>Exportações e Importações (US$ milhões FOB)</h3>
        <div class="sub">Banco Central do Brasil · séries 22708 e 22704 · categoria selecionada acima</div>
        <canvas id="chart-comercio"></canvas>
      </div>
      <div class="card-full">
        <h3>Saldo da Balança Comercial (US$ milhões)</h3>
        <div class="sub">Exportações − Importações (categoria selecionada)</div>
        <canvas id="chart-saldo"></canvas>
      </div>
      <div class="card-full">
        <h3>Índice de Quantum – Volume Físico (2006 = 100)</h3>
        <div class="sub">Banco Central do Brasil · séries 4447 (exp.) e 4448 (imp.) · mede variação de volume excluindo preços</div>
        <canvas id="chart-quantum"></canvas>
      </div>
      <div class="card-full">
        <h3>Valor FOB × Volume (Quantum) – Exportações</h3>
        <div class="sub">Dispersão mensal: eixo X = índice quantum (2006=100) · eixo Y = FOB em US$ mi · útil para identificar ganhos de preço vs. volume</div>
        <canvas id="chart-vol-fob"></canvas>
      </div>
    </div>
  </div>

  <!-- ═══ REGRESSÕES ═══ -->
  <div id="tab-regressao" class="section">
    <div class="reg-sub-tabs">
      <button class="reg-sub-tab active" onclick="showReg(0,this)">1. IPCA ~ Var. Cambio</button>
      <button class="reg-sub-tab" onclick="showReg(1,this)">2. IPCA 12m ~ Cambio 12m</button>
      <button class="reg-sub-tab" onclick="showReg(2,this)">3. Primeiras Diferencas</button>
    </div>

    <!-- Reg 1: IPCA ~ USD -->
    <div id="reg-panel-0" class="reg-panel active">
      <div class="card-full">
        <h3>Regressão OLS 1: IPCA mensal ~ USD/BRL (pass-through cambial)</h3>
        <div class="sub">Série completa 2015–2025 · {reg1.get('n_obs','—')} observações mensais</div>
        <div class="interp" style="margin-bottom:18px">
          <strong>Hipótese:</strong> a desvalorização cambial pressiona a inflação via preços de importados,
          insumos industriais e expectativas. Chamado de <em>pass-through</em> cambial.
        </div>
        <div class="reg-grid">
          <div>{reg1_block}</div>
          <div><canvas id="chart-scatter1"></canvas></div>
        </div>
      </div>
    </div>

    <!-- Reg 2: IPCA 12m ~ Desemprego 12m -->
    <div id="reg-panel-1" class="reg-panel">
      <div class="card-full">
        <h3>Regressão OLS 2: IPCA acum. 12m ~ USD médio 12m (câmbio de longo prazo)</h3>
        <div class="sub">Séries suavizadas em janelas de 12m · {reg2.get('n_obs','—')} observações</div>
        <div class="interp" style="margin-bottom:18px">
          <strong>Por que suavizar?</strong> Séries mensais têm muito ruído. Usar IPCA acumulado 12m
          e média 12m do câmbio elimina sazonalidade e revela se períodos de dólar alto coincidem
          com inflação acumulada elevada — relação estrutural de médio prazo.
        </div>
        <div class="reg-grid">
          <div>{reg2_block}</div>
          <div><canvas id="chart-scatter2"></canvas></div>
        </div>
      </div>
    </div>

    <!-- Reg 3: ΔIPCA ~ ΔUSD -->
    <div id="reg-panel-2" class="reg-panel">
      <div class="card-full">
        <h3>Regressão OLS 3: delta IPCA ~ delta USD (primeiras diferenças)</h3>
        <div class="sub">Variações absolutas mensais · {reg3.get('n_obs','—')} observações · elimina tendências comuns</div>
        <div class="interp" style="margin-bottom:18px">
          <strong>Por que primeiras diferenças?</strong> IPCA e câmbio têm tendências de longo prazo que
          podem produzir <em>regressão espúria</em>. Diferenciar as séries remove a tendência e isola
          o impacto de variações contemporâneas do câmbio sobre a inflação mensal.
        </div>
        <div class="reg-grid">
          <div>{reg3_block}</div>
          <div><canvas id="chart-scatter3"></canvas></div>
        </div>
      </div>
    </div>
  </div>

  <!-- FONTES -->
  <div class="sources">
    <h4>Fontes dos dados</h4>
    <div class="sources-grid">
      <div class="source-tag"><div class="source-dot" style="background:#4f8ef7"></div>
        <div><div class="sname">BCB – Banco Central do Brasil</div>
             <div class="sdesc">api.bcb.gov.br · séries 433, 188, 3698, 22708, 22704, 24369, 22707–22711, 22701–22705, 4447, 4448</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#34d399"></div>
        <div><div class="sname">IBGE · PNAD Contínua</div>
             <div class="sdesc">Taxa de desocupação mensal (via BCB série 24369)</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#fbbf24"></div>
        <div><div class="sname">IPCA &amp; INPC</div>
             <div class="sdesc">Índices de inflação oficiais do IBGE · BCB séries 433 e 188</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#fb923c"></div>
        <div><div class="sname">Comércio Exterior – categorias</div>
             <div class="sdesc">BCB · fator agregado (exp.) e uso econômico (imp.) · NCM via MDIC/ComexStat (indisponível)</div></div></div>
      <div class="source-tag"><div class="source-dot" style="background:#a78bfa"></div>
        <div><div class="sname">Índice de Quantum</div>
             <div class="sdesc">BCB · séries 4447 (exp.) e 4448 (imp.) · volume físico base 2006=100</div></div></div>
    </div>
  </div>

</div>

<script>
{js_code}
</script>
</body>
</html>"""

out = os.path.join(SCRIPT_DIR, 'index.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML gerado: {out}  ({len(html)//1024} KB)")
