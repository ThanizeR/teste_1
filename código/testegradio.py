import gradio as gr
import sqlite3
from PIL import Image
import numpy as np
from keras.models import load_model
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
import pickle

# Função de previsão de pneumonia
def predict_pneumonia(img):
    # Convertendo a imagem para um objeto PIL.Image
    img = Image.fromarray(np.uint8(img))
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,1))
    img = img / 255.0
    model = load_model("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/pneumonia.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    if pred_class == 1:
        pred_label = "Pneumonia"
    else:
        pred_label = "Saudável"
    return pred_label, pred_prob

# Função de previsão de malária
def predict_malaria(img):
    img = Image.fromarray(np.uint8(img))
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,3))
    img = img.astype(np.float64)
    model = load_model("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/malaria.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    if pred_class == 1:
        pred_label = "Infectado"
    else:
        pred_label = "Não está infectado"
    return pred_label, pred_prob

with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/diabetes_model.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

# Função de previsão de diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    user_input = [float(x) for x in user_input]
    diab_prediction = diabetes_model.predict([user_input])
    if diab_prediction[0] == 1:
        diab_diagnosis = 'A pessoa é diabética'
    else:
        diab_diagnosis = 'A pessoa não é diabética'
    return diab_diagnosis

# Função para criar o banco de dados e a tabela
def create_database():
    # Conectar ao banco de dados (ou criá-lo se não existir)
    conn = sqlite3.connect('feedbackGradio.db')
    cursor = conn.cursor()
    
    # Criar a tabela de feedback se ela não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            navegacao INTEGER,
            design_interface INTEGER,
            elementos INTEGER,
            desempenho INTEGER,
            atraso INTEGER,
            utilidade INTEGER,
            expectativas INTEGER,
            sugestoes TEXT,
            recursos TEXT,
            comentarios TEXT,
            nome TEXT,
            email TEXT,
            telefone TEXT,
            mensagem TEXT
        )
    ''')
    
    # Fechar a conexão com o banco de dados
    conn.close()

# Função para processar e armazenar o feedback
def process_feedback(navegacao, design_interface, elementos, desempenho, atraso, utilidade, expectativas,
                     sugestoes, recursos, comentarios, nome, email, telefone, mensagem):
    # Conectar ao banco de dados
    conn = sqlite3.connect('feedbackGradio.db')
    cursor = conn.cursor()
    
    # Inserir os dados do feedback na tabela
    cursor.execute('''
        INSERT INTO feedback (navegacao, design_interface, elementos, desempenho, atraso, utilidade,
                              expectativas, sugestoes, recursos, comentarios, nome, email, telefone, mensagem)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (navegacao, design_interface, elementos, desempenho, atraso, utilidade, expectativas, sugestoes,
          recursos, comentarios, nome, email, telefone, mensagem))
    
    # Commit para salvar as alterações e fechar a conexão com o banco de dados
    conn.commit()
    conn.close()
    
    # Retornar uma mensagem de confirmação
    return "Feedback enviado com sucesso!"

# Criar o banco de dados e a tabela
create_database()

# Interface Gradio com guias (Tabs)
tabbed_interface = gr.TabbedInterface(
    [
        # Guia 1: Previsão de Pneumonia
        gr.Interface(
            predict_pneumonia,
            inputs=gr.inputs.Image(label="Imagem para Predição de Pneumonia"),
            outputs=["text", "text"],  # Texto para a classe prevista e probabilidade
            title="Previsão de Pneumonia",
            description="Faça o upload de uma imagem para prever se há pneumonia."
        ),
        
        # Guia 2: Previsão de Malária
        gr.Interface(
            predict_malaria,
            inputs=gr.inputs.Image(label="Imagem para Predição de Malária"),
            outputs=["text", "text"],  # Texto para a classe prevista e probabilidade
            title="Previsão de Malária",
            description="Faça o upload de uma imagem para prever se há malária."
        ),

        # Guia 3: Previsão de Diabetes
        gr.Interface(
            predict_diabetes,
            inputs=[
                gr.Textbox(label="Número de Gestações"),
                gr.Textbox(label="Nível de Glicose"),
                gr.Textbox(label="Valor da Pressão Arterial"),
                gr.Textbox(label="Valor da Espessura da Pele"),
                gr.Textbox(label="Nível de Insulina"),
                gr.Textbox(label="Valor do IMC"),
                gr.Textbox(label="Valor da Função de Pedigree de Diabetes"),
                gr.Textbox(label="Idade da Pessoa")
            ],
            outputs="text",
            title="Previsão de Diabetes",
            description="Insira os dados do paciente para prever se ele tem diabetes."
        ),
        
        # Guia 4: Feedback dos Usuários
        gr.Interface(
            process_feedback,
            inputs=[
                gr.Slider(1, 10, value=5, step=1, label="Navegação", info="Avalie a facilidade de navegação"),
                gr.Slider(1, 10, value=5, step=1, label="Design da Interface", info="Avalie a atratividade visual da interface"),
                gr.Slider(1, 10, value=5, step=1, label="Elementos de Interface", info="Avalie a intuitividade dos elementos de interface"),
                gr.Slider(1, 10, value=5, step=1, label="Desempenho", info="Avalie o desempenho da aplicação durante o uso"),
                gr.Slider(1, 10, value=5, step=1, label="Atraso/Lentidão", info="Avalie a ocorrência de atrasos ou lentidão na aplicação"),
                gr.Slider(1, 10, value=5, step=1, label="Utilidade", info="Avalie se encontrou as informações ou funcionalidades desejadas"),
                gr.Slider(1, 10, value=5, step=1, label="Expectativas", info="Avalie se a aplicação atendeu às suas expectativas"),
                gr.Textbox(label="Sugestões de Melhoria", placeholder="Deixe suas sugestões de melhoria aqui"),
                gr.Textbox(label="Recursos Adicionais", placeholder="Indique os recursos adicionais que gostaria de ver implementados"),
                gr.Textbox(label="Comentários Adicionais", placeholder="Deixe aqui quaisquer outros comentários"),
                gr.Textbox(label="Nome", placeholder="Seu nome"),
                gr.Textbox(label="E-mail", placeholder="Seu e-mail"),
                gr.Textbox(label="Telefone", placeholder="Seu telefone"),
                gr.Textbox(label="Mensagem", placeholder="Mensagem adicional"),
            ],
            outputs="text",
            title="Feedback dos Usuários",
            description="Deixe o seu feedback sobre a aplicação."
        )
    ]
)

# Lançar a interface com guias
tabbed_interface.launch()
