import pandas as pd, plotly.express as px
a = pd.read_csv("eval/out/ablation.csv"); b = pd.read_csv("eval/out/buckets.csv")
lab = "label" if "label" in a.columns else a.columns[0]
am = a.melt(id_vars=lab, value_vars=[c for c in ["precision","recall","f1","scenario_recall"] if c in a.columns], var_name="metric", value_name="value")
f1 = px.bar(am, x=lab, y="value", color="metric", barmode="group", text_auto=".2f", title="Ablation: A rules -> B behavior/anomaly -> C graph -> D provenance")
f2 = px.bar(b, x="bucket", y="recall", color="model", barmode="group", text_auto=".2f", title="Recall by amount bucket")
f3 = px.bar(b, x="bucket", y="precision", color="model", barmode="group", text_auto=".2f", title="Precision by amount bucket")
for f in (f1,f2,f3): f.update_layout(height=430)
css = "body{font-family:Segoe UI,Arial;margin:30px auto;max-width:1100px;color:#173b73}table{border-collapse:collapse}td,th{border:1px solid #cbd5e1;padding:6px 10px;font-size:14px}"
html = "<html><head><meta charset='utf-8'><title>RupeeTrace Report</title><style>"+css+"</style></head><body>"
html += "<h1>RupeeTrace: Fraud and AML Risk Engine Results</h1><p>Synthetic data, reduced test split (9,350 transactions). Provenance (D) gives the highest precision and the lowest false-positive rate.</p>"
html += "<h2>Ablation table</h2>" + a.round(3).to_html(index=False)
html += f1.to_html(full_html=False, include_plotlyjs=True)
html += "<h2>Amount-bucket table</h2>" + b.round(3).to_html(index=False)
html += f2.to_html(full_html=False, include_plotlyjs=False) + f3.to_html(full_html=False, include_plotlyjs=False)
html += "</body></html>"
open("eval/out/report.html","w",encoding="utf-8").write(html); print("done")
