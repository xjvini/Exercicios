import numpy as np

# Configuração do Perceptron
class Perceptron:
    def __init__(self, num_atributos, taxa_aprendizagem=0.1):
        self.pesos = np.zeros(num_atributos)
        self.bias = 0
        self.taxa_aprendizagem = taxa_aprendizagem

    def ativacao(self, valor):
        return 1 if valor > 0 else 0

    def predizer(self, inputs):
        soma_ponderada = np.dot(inputs, self.pesos) + self.bias
        return self.ativacao(soma_ponderada)

    def treinar(self, dados_treino, gabarito_esperado, epocas=100):
        print(f".:: Treinando com {len(dados_treino)} alunos e {len(dados_treino[0])} atributos ::.\n")
        print("Legenda: [Trabalho (feito?), Frequência, Prova (foi bem ou mal?), Participação e Plágio]")
        print("Onde: 1 = Sim, 0 = Não\n")
        
        # Exibe os pesos antes do treinamento
        print(f"Pesos iniciais: {self.pesos}\n")

        for epoca in range(epocas):
            total_erros = 0
            for x, y_esperado in zip(dados_treino, gabarito_esperado):
                y_predito = self.predizer(x)
                erro = y_esperado - y_predito
                
                if erro != 0:
                    self.pesos += self.taxa_aprendizagem * erro * x
                    self.bias += self.taxa_aprendizagem * erro
                    total_erros += 1
            
            if total_erros == 0:
                print(f"O Perceptron convergiu na época {epoca + 1}.\n")
                break
        
        # Retorna os pesos arredondados e bias para facilitar a leitura
        print(f"Pesos finais: {np.round(self.pesos, 2)}")
        print(f"Bias final: {round(self.bias, 2)}")
        print("---------------------------------\n")

# .:: Dados de Treinamento ::.
# Cada coluna indica respectivamente: [Trabalhos (feitos), Frequência, Prova (foi bem ou mal?), Participação, Plágio]
# Onde: 1 = Sim (copiou), 0 = Não (honesto)

# Obs: coloquei a coluna de plágio para testar a robustez do perceptron

dados_alunos = np.array([
    # Alunos Originais
    [1, 1, 1, 1, 0],  # Aluno 1: perfeito 
    [0, 1, 0, 0, 0],  # Aluno 2: turista kkkkk
    [1, 0, 1, 0, 0],  # Aluno 3: +/- mas estudioso 
    [1, 1, 1, 1, 1],  # Aluno 4: "perfeito" para identificação de plágio 
    [1, 1, 0, 1, 0]   # Aluno 5: mal na prova, mas esforçado
])

# Gabarito (passou?): 1 = Sim, 0 = Não
gabarito = np.array([1, 0, 1, 0, 1])

# Treinamento do Perceptron
brain = Perceptron(num_atributos=5, taxa_aprendizagem=0.2)
brain.treinar(dados_alunos, gabarito)

# Teste final
print("Resultado da Previsão:")
atributos_nomes = ["Trabalho", "Freq", "Prova", "Partic", "Plágio"]

for i, aluno in enumerate(dados_alunos):
    resultado = "Sim" if brain.predizer(aluno) == 1 else "Não"
    
    # Indica se o aluno praticou plágio
    status_plagio = "Plágio identificado" if aluno[4] == 1 else "Limpo"
    print(f"Aluno {i+1} [{status_plagio}]: Aprovado? -> {resultado}")