"""Page CSS and the light/dark switch, generated from :mod:`spectramr.core.design_tokens`.

Every reporting page (run QC, comparison, contrast, campaign, Sim2Rank) shares this one
stylesheet, so a colour or type change lands everywhere at once. Dark mode is its own
validated token set, applied when the viewer's system asks for it or when they pick it.
"""

from __future__ import annotations

from spectramr.core.design_tokens import FONT_MONO, FONT_SANS, FONT_SERIF, css_variables

_LAYOUT = """
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--sans);
  font-variant-numeric:tabular-nums}
a{color:var(--series);text-underline-offset:2px}
a:hover{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
code{font-family:var(--mono);font-size:.9em;background:var(--surface-alt);padding:1px 4px;border-radius:3px}

.bar{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;
  padding:10px clamp(16px,4vw,40px);background:var(--page);border-bottom:1px solid var(--rule)}
.crumbs{display:flex;flex-wrap:wrap;gap:6px;font-size:13px;color:var(--muted);flex:1;min-width:0}
.crumbs a{color:var(--ink-2);text-decoration:none}
.crumbs a:hover{color:var(--accent)}
.crumbs span+span::before,.crumbs a+span::before,.crumbs span+a::before,.crumbs a+a::before{
  content:"/";margin-right:6px;color:var(--rule-strong)}
.mode{display:inline-flex;border:1px solid var(--rule-strong);border-radius:6px;overflow:hidden}
.mode button{font:500 12px var(--sans);color:var(--ink-2);background:none;border:0;
  padding:5px 10px;cursor:pointer}
.mode button[aria-pressed=true]{background:var(--ink);color:var(--page)}

.head{padding:40px clamp(16px,4vw,40px) 24px;max-width:1320px;margin:0 auto}
.head h1{font:600 clamp(28px,4vw,42px)/1.12 var(--serif);margin:0 0 12px;letter-spacing:-.01em;
  overflow-wrap:anywhere}
.head .lede{max-width:72ch;color:var(--ink-2);margin:0 0 18px;font-size:16px}
.prov{display:flex;flex-wrap:wrap;gap:6px 28px;margin:0;font-size:12.5px}
.prov div{display:flex;gap:8px}
.prov dt{color:var(--muted)}
.prov dd{margin:0;color:var(--ink-2);overflow-wrap:anywhere}

.frame{display:grid;grid-template-columns:200px minmax(0,1fr);gap:40px;max-width:1320px;
  margin:0 auto;padding:8px clamp(16px,4vw,40px) 96px}
.toc{position:sticky;top:64px;align-self:start;font-size:13.5px}
.toc ol{list-style:none;margin:0;padding:0;border-left:1px solid var(--rule)}
.toc a{display:block;padding:5px 0 5px 14px;margin-left:-1px;color:var(--ink-2);
  text-decoration:none;border-left:2px solid transparent}
.toc a:hover,.toc a[aria-current=true]{color:var(--ink);border-left-color:var(--accent)}

main section{margin:0 0 56px;scroll-margin-top:72px}
main h2{font:600 24px/1.25 var(--serif);margin:0 0 6px}
main h3{font:600 15px/1.3 var(--sans);margin:24px 0 8px}
.note{color:var(--ink-2);max-width:78ch;margin:0 0 16px}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:16px 18px}
.howto{color:var(--ink-2);font-size:13px;line-height:1.5;margin:10px 0 0;max-width:80ch}
.howto b{color:var(--ink)}
.meta,.empty{color:var(--muted);font-size:13px}
.empty{font-style:italic}

.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:20px}
figure{margin:0}
figure img{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--rule);
  border-radius:6px}
figcaption{display:flex;justify-content:space-between;gap:12px;font-size:12.5px;color:var(--ink-2);
  margin-top:6px}
figcaption .fmt{display:flex;gap:10px;white-space:nowrap}
.wide{grid-column:1/-1}

.tbl-wrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
table.tbl{border-collapse:collapse;width:100%;font-size:13px}
.tbl th,.tbl td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
.tbl .num{text-align:right}
.tbl thead th{position:sticky;top:0;background:var(--surface-alt);color:var(--ink-2);font-weight:600}
.tbl tbody tr:last-child td{border-bottom:0}
.tbl tbody tr:hover td{background:var(--surface-alt)}
.tbl .best{color:var(--accent);font-weight:600}
.tbl .base td:first-child{color:var(--muted)}
.ok{color:var(--good)} .bad{color:var(--bad)}
.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:999px;
  border:1px solid currentColor}

.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,260px),1fr));gap:14px}
.card{display:block;padding:14px 16px;background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;text-decoration:none;color:var(--ink)}
.card:hover{border-color:var(--accent);color:var(--ink)}
.card b{display:block;font-weight:600;overflow-wrap:anywhere}
.card span{font-size:12.5px;color:var(--ink-2)}

footer{max-width:1320px;margin:0 auto;padding:24px clamp(16px,4vw,40px);color:var(--muted);
  font-size:12px;border-top:1px solid var(--rule)}

@media (max-width:860px){
  .frame{grid-template-columns:minmax(0,1fr);gap:16px}
  .toc{position:static}
  .toc ol{display:flex;gap:4px 16px;flex-wrap:wrap;border-left:0}
  .toc a{padding:2px 0;border-left:0}
}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
@media print{.bar,.toc{display:none}.frame{display:block}}
"""

# Re-inks plotly figures on a mode change: their backgrounds are transparent, so only
# text and gridlines need the new tokens.
_SCRIPT = """
(function(){
  var root=document.documentElement, KEY='spectramr-theme';
  function stored(){try{return localStorage.getItem(KEY)}catch(e){return null}}
  function save(v){try{localStorage.setItem(KEY,v)}catch(e){}}
  function current(){var s=stored(); if(s) return s;
    return window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}
  function token(n){return getComputedStyle(root).getPropertyValue('--'+n).trim()}
  function reink(){
    if(!window.Plotly) return;
    function tok(r){return token(r.replace(/_/g,'-'))}
    var upd={'font.color':token('ink-2'),'title.font.color':token('ink')};
    document.querySelectorAll('.js-plotly-plot').forEach(function(el){
      var lay=el.layout||{}, u=Object.assign({},upd);
      Object.keys(lay).forEach(function(k){ if(/^[xy]axis\\d*$/.test(k)){
        u[k+'.gridcolor']=token('rule'); u[k+'.linecolor']=token('rule-strong');
        u[k+'.zerolinecolor']=token('rule-strong');}});
      (lay.annotations||[]).forEach(function(a,i){u['annotations['+i+'].font.color']=token('ink-2')});
      try{Plotly.relayout(el,u)}catch(e){}
      (el.data||[]).forEach(function(t,i){var m=t.meta; if(!m||!m.marker_roles) return;
        try{Plotly.restyle(el,{'marker.color':[m.marker_roles.map(tok)],
          'marker.line.color':[m.ring_roles.map(tok)]},[i])}catch(e){}});
    });
  }
  function apply(v){root.setAttribute('data-theme',v);
    document.querySelectorAll('.mode button').forEach(function(b){
      b.setAttribute('aria-pressed',String(b.dataset.mode===v))}); reink()}
  document.addEventListener('DOMContentLoaded',function(){
    apply(current());
    document.querySelectorAll('.mode button').forEach(function(b){
      b.addEventListener('click',function(){save(b.dataset.mode);apply(b.dataset.mode)})});
    var links=[].slice.call(document.querySelectorAll('.toc a'));
    if('IntersectionObserver' in window&&links.length){
      var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){
        links.forEach(function(a){a.setAttribute('aria-current',String(a.hash==='#'+e.target.id))})}})},
        {rootMargin:'-20% 0px -70% 0px'});
      links.forEach(function(a){var s=document.querySelector(a.hash); if(s) io.observe(s)});
    }
  });
  root.setAttribute('data-theme',current());
})();
"""


def stylesheet() -> str:
    """The full page stylesheet: tokens for both modes, then the layout rules."""
    fonts = f"--serif:{FONT_SERIF};--sans:{FONT_SANS};--mono:{FONT_MONO};"
    return (
        f":root{{{fonts}{css_variables('light')}color-scheme:light}}"
        f":root[data-theme=dark]{{{css_variables('dark')}color-scheme:dark}}"
        f"@media (prefers-color-scheme:dark){{:root:not([data-theme=light])"
        f"{{{css_variables('dark')}color-scheme:dark}}}}" + _LAYOUT
    )


def script() -> str:
    """The theme switch and section-nav highlighter (no dependencies)."""
    return _SCRIPT
