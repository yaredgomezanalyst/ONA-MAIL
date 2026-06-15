"""
ONA: Organigrama formal vs comunidades reales de comunicacion
Dataset: email-Eu-core (SNAP, Stanford)
- 1.005 empleados de una institucion de investigacion europea
- 25.571 relaciones de email entre ellos
- Etiqueta de departamento (42 departamentos) para cada empleado
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGS = BASE / "figures"
RESULTS = BASE / "results"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SEED = 42

# 1. Carga de datos

edges = pd.read_csv(DATA / "email-Eu-core.txt", sep=" ", names=["source", "target"])
labels = pd.read_csv(
    DATA / "email-Eu-core-department-labels.txt", sep=" ", names=["node", "dept"]
)
dept = dict(zip(labels.node, labels.dept))

# Grafo no dirigido (nos interesa "quien habla con quien"), sin autobucles
G = nx.from_pandas_edgelist(edges, "source", "target")
G.remove_edges_from(nx.selfloop_edges(G))

# Componente gigante (empleados conectados entre si)
gc = max(nx.connected_components(G), key=len)
G = G.subgraph(gc).copy()

stats = {
    "n_empleados": G.number_of_nodes(),
    "n_relaciones": G.number_of_edges(),
    "densidad": round(nx.density(G), 4),
    "grado_medio": round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 1),
    "n_departamentos": len(set(dept[n] for n in G.nodes())),
}
print("Estadisticas basicas:", stats)

# 2. Deteccion de comunidades (Louvain)

communities = nx.community.louvain_communities(G, seed=SEED)
comm_of = {n: i for i, c in enumerate(communities) for n in c}

nodes = sorted(G.nodes())
y_dept = [dept[n] for n in nodes]
y_comm = [comm_of[n] for n in nodes]

nmi = normalized_mutual_info_score(y_dept, y_comm)
ari = adjusted_rand_score(y_dept, y_comm)
modularity = nx.community.modularity(G, communities)

stats.update(
    {
        "n_comunidades_louvain": len(communities),
        "modularidad": round(modularity, 3),
        "NMI_dept_vs_comunidades": round(nmi, 3),
        "ARI_dept_vs_comunidades": round(ari, 3),
    }
)
print(f"Comunidades Louvain: {len(communities)} | Modularidad: {modularity:.3f}")
print(f"NMI organigrama vs comunidades: {nmi:.3f} | ARI: {ari:.3f}")

# 3. Que departamentos se mezclan? Matriz departamento x comunidad
df = pd.DataFrame({"dept": y_dept, "comm": y_comm}, index=nodes)

# Pureza de cada departamento: % de sus miembros en su comunidad mas frecuente
purity = (
    df.groupby("dept")["comm"]
    .agg(lambda s: s.value_counts().iloc[0] / len(s))
    .sort_values()
)
dept_sizes = df.groupby("dept").size()
big = dept_sizes[dept_sizes >= 10].index
purity_big = purity[purity.index.isin(big)]
print("\nDepartamentos (>=10 personas) MENOS cohesionados:")
print(purity_big.head(5).round(2).to_string())
print("\nDepartamentos (>=10 personas) MAS cohesionados:")
print(purity_big.tail(5).round(2).to_string())

# 4. Brokers: gente que conecta departamentos 
bet = nx.betweenness_centrality(G, seed=SEED)
deg = dict(G.degree())
top_brokers = sorted(bet, key=bet.get, reverse=True)[:10]
broker_rows = []
for n in top_brokers:
    nb_depts = len(set(dept[v] for v in G.neighbors(n)))
    broker_rows.append(
        {
            "empleado": n,
            "dept": dept[n],
            "betweenness": round(bet[n], 4),
            "contactos": deg[n],
            "depts_alcanzados": nb_depts,
        }
    )
brokers_df = pd.DataFrame(broker_rows)
print("\nTop 10 brokers (mayor intermediacion):")
print(brokers_df.to_string(index=False))

# 5. Figuras
pos = nx.spring_layout(G, seed=SEED, k=0.08)

fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
node_list = list(G.nodes())
for ax, mapping, title in [
    (axes[0], dept, "Organigrama formal (42 departamentos)"),
    (axes[1], comm_of, f"Comunidades reales de email (Louvain, {len(communities)})"),
]:
    colors = [mapping[n] for n in node_list]
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list, node_size=18, node_color=colors,
        cmap="tab20", ax=ax, alpha=0.85
    )
    nx.draw_networkx_edges(G, pos, alpha=0.03, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
fig.suptitle(
    "Coincide el organigrama con como se comunica la gente en realidad?",
    fontsize=16, y=1.0,
)
fig.tight_layout()
fig.savefig(FIGS / "01_organigrama_vs_comunidades.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Heatmap: composicion de cada comunidad por departamento
top_comms = df["comm"].value_counts().head(12).index
top_depts = df["dept"].value_counts().head(15).index
ct = pd.crosstab(df["dept"], df["comm"])
ct_sub = ct.loc[top_depts, top_comms]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(ct_sub.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(top_comms)))
ax.set_xticklabels([f"C{c}" for c in top_comms])
ax.set_yticks(range(len(top_depts)))
ax.set_yticklabels([f"Dept {d}" for d in top_depts])
ax.set_xlabel("Comunidad detectada (Louvain)")
ax.set_ylabel("Departamento oficial")
ax.set_title("Donde acaba cada departamento?\n(empleados por departamento y comunidad)")
for i in range(len(top_depts)):
    for j in range(len(top_comms)):
        v = ct_sub.values[i, j]
        if v > 0:
            ax.text(j, i, v, ha="center", va="center", fontsize=8,
                    color="white" if v > ct_sub.values.max() * 0.5 else "black")
fig.colorbar(im, label="n empleados")
fig.tight_layout()
fig.savefig(FIGS / "02_heatmap_dept_comunidad.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Pureza departamental
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#d62728" if v < 0.5 else "#2ca02c" for v in purity_big.values]
ax.barh(
    [f"Dept {d} ({dept_sizes[d]}p)" for d in purity_big.index],
    purity_big.values, color=colors,
)
ax.axvline(0.5, ls="--", c="gray")
ax.set_xlabel("Cohesion: % del dept. en su comunidad principal")
ax.set_title(
    "Cohesion de los departamentos (>=10 personas)\n"
    "Rojo: el dept. trabaja partido en varias comunidades"
)
fig.tight_layout()
fig.savefig(FIGS / "03_cohesion_departamental.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 6. Resumen
brokers_df.to_csv(RESULTS / "top_brokers.csv", index=False)
purity.round(3).to_csv(RESULTS / "cohesion_departamentos.csv")
with open(RESULTS / "stats.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print("\nFiguras y resultados guardados.")
