"""Render the LangGraph as Mermaid + PNG + standalone HTML.

Run: python -m scripts.render_graph
Outputs:
  webui/static/graph.mmd     (raw mermaid source)
  webui/static/graph.png     (PNG, via mermaid.ink — needs internet)
  webui/static/graph.html    (standalone, self-rendering in browser)
"""
from pathlib import Path
from src.graph import compiled_graph

OUT = Path("webui/static")
OUT.mkdir(parents=True, exist_ok=True)


HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Support Triage Agent — Graph</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .sub {{ color: #666; margin-bottom: 2rem; }}
  .mermaid {{ background: #fafafa; padding: 1rem; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Support Triage Agent — Graph Topology</h1>
<p class="sub">Static diagram of nodes + edges (no run data). Rendered client-side via Mermaid.</p>
<pre class="mermaid">
{mermaid}
</pre>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
</script>
</body></html>
"""


def main() -> None:
    with compiled_graph() as g:
        graph = g.get_graph()
        mermaid = graph.draw_mermaid()
        (OUT / "graph.mmd").write_text(mermaid)
        print(f"Wrote {OUT/'graph.mmd'}")

        (OUT / "graph.html").write_text(HTML_TEMPLATE.format(mermaid=mermaid))
        print(f"Wrote {OUT/'graph.html'}  ← open this in your browser")

        try:
            png_bytes = graph.draw_mermaid_png()
            (OUT / "graph.png").write_bytes(png_bytes)
            print(f"Wrote {OUT/'graph.png'}")
        except Exception as e:
            print(f"PNG render skipped ({e}). The HTML file works offline-of-LangGraph anyway.")


if __name__ == "__main__":
    main()
