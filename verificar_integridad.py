from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
root = Path(__file__).resolve().parent
for line in (root / "SHA256SUMS.txt").read_text().splitlines():
    digest, name = line.split("  ", 1)
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"SHA-256 incorrecto: {name}")
print("Integridad de referencias: OK")
out = root / "resultados_reproducidos"
if not out.exists():
    print("Evaluación nueva pendiente")
else:
    a = pd.read_csv(out / "resultados_validacion_definitiva.csv")
    b = pd.read_csv(out / "resultados_tipologias_definitiva.csv")
    c = pd.read_csv(out / "resultados_sensibilidad_p90_p95_p97.csv")
    keys = ["semilla", "escenario", "contexto_atc", "modelo"]
    for df, n, k in [(a,60,keys),(b,360,keys+["tipo"]),(c,180,keys+["percentil"])]:
        assert len(df)==n and not df.duplicated(k).any()
    metrics = ["precision","recall","f1"]
    p95 = c[c.percentil==95].set_index(keys).sort_index()
    actual = a.set_index(keys).sort_index()
    assert actual.index.equals(p95.index)
    np.testing.assert_allclose(actual[metrics],p95[metrics],atol=1e-8,rtol=0)
    ref = pd.read_csv(root / "resultados/resultados_validacion_tfm_if_ampliado.csv").set_index(keys).sort_index()
    assert actual.index.equals(ref.index)
    cols = metrics+["pr_auc","roc_auc","precision_50","recall_50","r_precision"]
    np.testing.assert_allclose(actual[cols],ref[cols],atol=1e-8,rtol=0)
    print("Métricas principales y coherencia P95: OK")
