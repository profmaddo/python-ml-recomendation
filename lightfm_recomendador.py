import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

# Carregar dados
df_interacoes = pd.read_csv("interacoes.csv")  # user_id, item_id, rating
df_professores = pd.read_csv("professores.csv")  # item_id, nome, idioma, nivel, nota_media

# Transformar rating em interações binárias (rating >= 4 considerado positivo)
df_interacoes["interaction"] = df_interacoes["rating"].apply(lambda x: 1 if x >= 4 else 0)

# Criar dataset LightFM
dataset = Dataset()
dataset.fit(
    df_interacoes["user_id"],
    df_interacoes["item_id"],
    item_features=df_professores["item_id"]
)

# Mapear os atributos dos professores como features (idioma + nivel)
professor_features = [
    (row["item_id"], [row["idioma"], row["nivel"]])
    for idx, row in df_professores.iterrows()
]
dataset.fit_partial(items=df_professores["item_id"], item_features=["Inglês", "Iniciante", "Intermediário", "Avançado"])
item_features = dataset.build_item_features(professor_features)

# Criar matriz de interações
(interactions, weights) = dataset.build_interactions([
    (row["user_id"], row["item_id"], row["interaction"])
    for idx, row in df_interacoes.iterrows()
])

# Criar modelo híbrido com WARP
model = LightFM(loss="warp")
model.fit(interactions, item_features=item_features, epochs=10, num_threads=2)

# Avaliar com precision@5
score = precision_at_k(model, interactions, item_features=item_features, k=5).mean()
print(f"Precision@5: {score:.4f}")

# Recomendações para um usuário
user_id = "aluno1"
user_index = list(dataset.mapping()[0].keys()).index(user_id)

n_items = interactions.shape[1]
scores = model.predict(user_index, np.arange(n_items), item_features=item_features)
top_items = np.argsort(-scores)[:5]

item_id_map = {v: k for k, v in dataset.mapping()[2].items()}

print(f"Top 5 professores recomendados para {user_id}:")
for idx in top_items:
    print(f"  {item_id_map[idx]}")
