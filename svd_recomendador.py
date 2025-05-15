import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Carregar o CSV
df = pd.read_csv('interacoes.csv')  # Certifique-se de que o arquivo está na mesma pasta do script

# Criar Reader com escala de notas
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Dividir entre treino e teste
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Treinar modelo SVD
model = SVD()
model.fit(trainset)

# Avaliar
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Exemplo: recomendação para um aluno específico
user_id = 'aluno1'
professores = df['item_id'].unique()

print(f"Recomendações para o usuário {user_id}:")

# Recomendar professores não avaliados
avaliados = df[df['user_id'] == user_id]['item_id'].tolist()
nao_avaliados = [item for item in professores if item not in avaliados]

recomendacoes = []
for item_id in nao_avaliados:
    pred = model.predict(user_id, item_id)
    recomendacoes.append((item_id, pred.est))

# Mostrar top 5 recomendações
top5 = sorted(recomendacoes, key=lambda x: x[1], reverse=True)[:5]
for item_id, nota in top5:
    print(f"  Professor: {item_id}, Nota estimada: {nota:.2f}")
