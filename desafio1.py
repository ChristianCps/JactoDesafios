#Desafio 1 - Análise dataset de carros
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os


#Baixa o dataset do Kaggle
path = kagglehub.dataset_download("meruvulikith/90000-cars-data-from-1970-to-2024")
print("Dataset baixado em:", path)

# O dataset vem como CSV dentro da pasta baixada
files = os.listdir(path)
print("Arquivos disponíveis:", files)


#Carrega o arquivo CSV

try:
    dataset_file = [f for f in files if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, dataset_file))
    print("Dataset carregado com sucesso!")
except IndexError:
    print("Erro: Nenhum arquivo CSV encontrado no diretório do dataset.")
except FileNotFoundError:
    print("Erro: O arquivo CSV não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado ao carregar o dataset: {e}")



#Marcas mais vendidas (Top 10)
top_marcas = df['Manufacturer'].value_counts().head(10)

plt.figure(figsize=(12,6))
top_marcas.plot(kind='bar', color="green")
plt.title("Top 10 marcas mais vendidas (quantidade de carros)")
plt.xlabel("Marca")
plt.ylabel("Quantidade de carros")
plt.show()


#Quantidade de carros vendidos por ano

carros_por_ano = df['year'].value_counts().sort_index()

plt.figure(figsize=(12,6))
carros_por_ano.plot(kind='bar')
plt.title("Quantidade de carros cadastrados por ano")
plt.xlabel("Ano")
plt.ylabel("Quantidade de carros")
plt.show()

#Preço médio por marca (Top 10)

media_preco_marca = df.groupby("Manufacturer")["price"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
media_preco_marca.plot(kind='bar', color="orange")
plt.title("Top 10 marcas com maior preço médio")
plt.xlabel("Marca")
plt.ylabel("Preço médio")
plt.show()
