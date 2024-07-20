import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
import pickle
from PIL import Image
import tensorflow as tf
from streamlit_option_menu import option_menu
import os

def predict_malaria(img):
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,3))
    img = img.astype(np.float64)
    model = load_model("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/malaria.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob

def predict_pneumonia(img):
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,1))
    img = img / 255.0
    model = load_model("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/pneumonia.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob


with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/diabetes_model.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/heart_disease_model.sav', 'rb') as file:
    heart_disease_model = pickle.load(file)

with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/kidney-2.pkl', 'rb') as file:
    kidney_model = pickle.load(file)

with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/liver.pkl', 'rb') as file:
    liver_model = pickle.load(file)

# Função para prever problemas hepáticos
def predict_liver(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                  alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                  albumin, albumin_and_globulin_ratio):
    # Formatar os dados de entrada para a previsão
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                            albumin, albumin_and_globulin_ratio]])
    # Fazer a previsão
    prediction = liver_model.predict(input_data)
    return prediction[0]

with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/breast_cancer_dataset.sav', 'rb') as file:
    breast_cancer_model = pickle.load(file)


logo = Image.open("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/logo/MediScan.png")
# Criação de uma sidebar personalizada com ícones redondos
st.sidebar.image(logo, use_column_width=True)
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Navegação",
    ["🏠 Página Inicial", "🦟 Detecção Malaria", " 🫁 Detecção Pneumonia", "🫀 Problemas Cardíacos", "💧 Problemas Renais", "💉 Detecção Diabetes","🧪 Problemas Hepáticos", "🎗️ Câncer de Mama", "📝 Feedback dos Usuários", "📊 Datasets Disponíveis", "📞 Contato"]
)
# Função para mapear seleção de menu para página correspondente
def get_selected_page(menu):
    if menu == "🏠 Página Inicial":
        return "home"
    elif menu == "🦟 Detecção Malaria":
        return "Malaria"
    elif menu == " 🫁 Detecção Pneumonia":
        return "Pneumonia"
    elif menu == "🫀 Problemas Cardíacos":
        return "Problemas Cardíacos"
    elif menu == "💧 Problemas Renais":
        return "Problemas Renais"
    elif menu == "💉 Detecção Diabetes":
        return "Diabetes"
    elif menu == "🧪 Problemas Hepáticos":
        return "Problemas Hepáticos"
    elif menu == "🎗️ Câncer de Mama":
        return "Câncer de Mama"
    elif menu == "📝 Feedback dos Usuários":
        return "feedback"
    elif menu == "📊 Datasets Disponíveis":
        return "Datasets"
    elif menu == "📞 Contato":
            return "contato"
    
selected_page = get_selected_page(menu)


def main(selected_page):
    # Conteúdo da página selecionada
    if selected_page == "home":
        st.title('Bem-vindo à Aplicação de Previsão de Anomalias Médicas')
        st.write("Este é um projeto de previsão de diversas anomalias médicas usando modelos de deep learning e machine learning.")

        st.write("É importante observar que os modelos utilizados nesta aplicação foram obtidos de repositórios públicos na internet e, portanto, sua confiabilidade pode variar.")

        st.write("Embora tenham sido treinados em grandes conjuntos de dados médicos, é fundamental lembrar que todas as previsões devem ser verificadas por profissionais de saúde qualificados.")

        # Seção de Perguntas Frequentes
        st.subheader("Perguntas Frequentes")

        # Lista de perguntas frequentes e respostas
        faq = [
            {
                "pergunta": "Como a previsão de anomalias é feita?",
                "resposta": "A detecção de pneumonia e malária é feita usando uma rede neural convolucional (CNN), enquanto o restante das anomalias é detectado por um modelo Random Forest. Além disso, a previsão de câncer de mama é realizada por meio de regressão logística.",
            },
            {
                "pergunta": "Os modelos são precisos?",
                "resposta": "Os modelos foram treinados em grandes conjuntos de dados médicos, mas lembre-se de que todas as previsões devem ser verificadas por profissionais de saúde qualificados.",
            },
            {
                "pergunta": "Qual é o propósito desta aplicação?",
                "resposta": "Esta aplicação foi desenvolvida para auxiliar na detecção de diversas anomalias médicas em imagens de diferentes partes do corpo.",
            },
            {
                "pergunta": "Quais tipos de anomalias médicas podem ser detectadas?",
                "resposta": "Os modelos podem detectar várias anomalias, incluindo pneumonia, malária, problemas cardíacos, hepáticos, renais, e diabetes.",
            },
            {
                "pergunta": "Como faço para obter suporte técnico?",
                "resposta": "Você pode obter suporte técnico na seção 'Feedback dos Usuários' preenchendo o formulário e descrevendo seu problema, ou acessando a seção 'Contato' para envio de e-mails.",
            },
        ]

        # Exibição das perguntas frequentes
        for item in faq:
            with st.expander(item["pergunta"]):
                st.write(item["resposta"])


    elif selected_page ==  "Malaria":
        st.header("Previsão de Malária")
        uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de malária", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagem enviada", use_column_width=True)
                pred_class, pred_prob = predict_malaria(img)
                
                if pred_class == 1:
                    st.write("Previsão: Infectado")
                    st.write(f"Probabilidade de Malária: {pred_prob * 100:.2f}%")
                else:
                    st.write("Previsão: Não está infectado")
                    st.write(f"Probabilidade de Saúde: {pred_prob * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Erro ao prever Malária: {str(e)}")

    elif selected_page ==  "Pneumonia":
        st.header("Previsão de Pneumonia")
        uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de pneumonia", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagem enviada", use_column_width=True)
                pred_class, pred_prob = predict_pneumonia(img)
                
                if pred_class == 1:
                    st.write("Previsão: Pneumonia")
                    st.write(f"Probabilidade de Pneumonia: {pred_prob * 100:.2f}%")
                else:
                    st.write("Previsão: Saudável")
                    st.write(f"Probabilidade de Saúde: {pred_prob * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Erro ao prever Pneumonia: {str(e)}")


    elif selected_page ==  "Problemas Cardíacos":
        # Título da página
        st.title('Previsão de Doenças Cardíacas')

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Idade')

        with col2:
            sex = st.text_input('Sexo')

        with col3:
            cp = st.text_input('Tipos de Dor no Peito')

        with col1:
            trestbps = st.text_input('Pressão Arterial de Repouso')

        with col2:
            chol = st.text_input('Colesterol Sérico em mg/dl')

        with col3:
            fbs = st.text_input('Açúcar no Sangue em Jejum > 120 mg/dl')

        with col1:
            restecg = st.text_input('Resultados Eletrocardiográficos em Repouso')

        with col2:
            thalach = st.text_input('Frequência Cardíaca Máxima Alcançada')

        with col3:
            exang = st.text_input('Angina Induzida por Exercício')

        with col1:
            oldpeak = st.text_input('Depressão do ST induzida pelo exercício')

        with col2:
            slope = st.text_input('Inclinação do segmento ST de pico do exercício')

        with col3:
            ca = st.text_input('Principais vasos coloridos por flourosopia')

        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = defeito fixo; 2 = defeito reversível')

        # código para previsão
        heart_diagnosis = ''

        # criando um botão para previsão

        if st.button('Resultado do Teste de Doença Cardíaca'):

            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            user_input = [float(x) for x in user_input]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'A pessoa está com doença cardíaca'
            else:
                heart_diagnosis = 'A pessoa não tem doença cardíaca'

        st.success(heart_diagnosis)

    elif selected_page ==  "Problemas Renais":
        # Título da página
        st.title('Previsão de Doença Renal')

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            age = st.text_input('Idade')

        with col2:
            blood_pressure = st.text_input('Pressão Sanguínea')

        with col3:
            specific_gravity = st.text_input('Gravidade Específica')

        with col4:
            albumin = st.text_input('Albumina')

        with col5:
            sugar = st.text_input('Açúcar')

        with col1:
            red_blood_cells = st.text_input('Glóbulos Vermelhos')

        with col2:
            pus_cell = st.text_input('Células de Pus')

        with col3:
            pus_cell_clumps = st.text_input('Aglomerados de Células de Pus')

        with col4:
            bacteria = st.text_input('Bactérias')

        with col5:
            blood_glucose_random = st.text_input('Glicose no Sangue Aleatória')

        with col1:
            blood_urea = st.text_input('Uréia Sanguínea')

        with col2:
            serum_creatinine = st.text_input('Creatinina Sérica')

        with col3:
            sodium = st.text_input('Sódio')

        with col4:
            potassium = st.text_input('Potássio')

        with col5:
            haemoglobin = st.text_input('Hemoglobina')

        with col1:
            packed_cell_volume = st.text_input('Volume de Hemácias')

        with col2:
            white_blood_cell_count = st.text_input('Contagem de Glóbulos Brancos')

        with col3:
            red_blood_cell_count = st.text_input('Contagem de Glóbulos Vermelhos')

        with col4:
            hypertension = st.text_input('Hipertensão')

        with col5:
            diabetes_mellitus = st.text_input('Diabetes Mellitus')

        with col1:
            coronary_artery_disease = st.text_input('Doença da Artéria Coronária')

        with col2:
            appetite = st.text_input('Apetite')

        with col3:
            peda_edema = st.text_input('Edema Pedal')

        with col4:
            aanemia = st.text_input('Anemia')

        # código para previsão
        kidney_diagnosis = ''

        # criando um botão para previsão    
        if st.button("Resultado do Teste de Doença Renal"):

            user_input = [age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell,
                        pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium,
                        potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count,
                        hypertension, diabetes_mellitus, coronary_artery_disease, appetite, peda_edema,
                        aanemia]

            user_input = [float(x) for x in user_input]

            # Você precisa substituir esta linha pela sua lógica de previsão real para doença renal
            kidney_prediction = kidney_model.predict([user_input])

            if kidney_prediction[0] == 1:
                kidney_diagnosis = "A pessoa tem Doença Renal"
            else:
                kidney_diagnosis = "A pessoa não tem Doença Renal"

        st.success(kidney_diagnosis)

    elif selected_page ==  "Diabetes":
        # Título da página
        st.title('Previsão de Diabetes')

        # obtendo os dados de entrada do usuário
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Número de Gestações')

        with col2:
            Glucose = st.text_input('Nível de Glicose')

        with col3:
            BloodPressure = st.text_input('Valor da Pressão Arterial')

        with col1:
            SkinThickness = st.text_input('Valor da Espessura da Pele')

        with col2:
            Insulin = st.text_input('Nível de Insulina')

        with col3:
            BMI = st.text_input('Valor do IMC')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Valor da Função de Pedigree de Diabetes')

        with col2:
            Age = st.text_input('Idade da Pessoa')


        # código para previsão
        diab_diagnosis = ''

        # criando um botão para previsão

        if st.button('Resultado do Teste de Diabetes'):

            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                        BMI, DiabetesPedigreeFunction, Age]

            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'A pessoa é diabética'
            else:
                diab_diagnosis = 'A pessoa não é diabética'

        st.success(diab_diagnosis)
       
    elif selected_page == "Problemas Hepáticos":
            # Título da página
            st.title("Previsão de Problemas Hepáticos")

            # Obtendo os dados de entrada do usuário
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                age = st.text_input('Idade')
            with col2:
                gender = st.radio('Gênero', ['Masculino', 'Feminino'])
            with col3:
                total_bilirubin = st.text_input('Bilirrubina Total')
            with col4:
                direct_bilirubin = st.text_input('Bilirrubina Direta')
            with col5:
                alkaline_phosphotase = st.text_input('Fosfatase Alcalina')
                
            col6, col7, col8, col9, col10 = st.columns(5)
            
            with col6:
                alamine_aminotransferase = st.text_input('Alanina Aminotransferase (ALT)')
            with col7:
                aspartate_aminotransferase = st.text_input('Aspartato Aminotransferase (AST)')
            with col8:
                total_proteins = st.text_input('Proteínas Totais')
            with col9:
                albumin = st.text_input('Albumina')
            with col10:
                albumin_and_globulin_ratio = st.text_input('Razão Albumina/Globulina')

            # Mapear a entrada de gênero para 0 ou 1
            gender_mapping = {'Masculino': 1, 'Feminino': 0}
            gender_code = gender_mapping[gender]

            # Código para a previsão
            liver_diagnosis = ''

            # Criando um botão para previsão
            if st.button('Prever'):
                user_input = [age, gender_code, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                            albumin, albumin_and_globulin_ratio]

                user_input = [float(x) for x in user_input]

                liver_prediction = predict_liver(user_input)

                if liver_prediction[0] == 1:
                    liver_diagnosis = 'O paciente tem um problema hepático.'
                else:
                    liver_diagnosis = 'O paciente não tem um problema hepático.'
                    st.write(liver_diagnosis)

            st.success(liver_diagnosis)

    elif selected_page == "Câncer de Mama":
        # Título da página
        st.title("Previsão de Câncer de Mama")

        # Obtendo os dados de entrada do usuário
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            i1 = st.text_input('Raio Médio', key='raio_medio')
            i6 = st.text_input('Concavidade Média', key='concavidade_media')
            i11 = st.text_input('Textura Pior', key='textura_pior')
            i16 = st.text_input('Concavidade Pior', key='concavidade_pior')
            i21 = st.text_input('Área SE', key='area_se')
            i26 = st.text_input('Concavidade SE', key='concavidade_se')
        with col2:
            i2 = st.text_input('Textura Média', key='textura_media')
            i7 = st.text_input('Pontos Côncavos Médios', key='pontos_concavos_medios')
            i12 = st.text_input('Perímetro SE', key='perimetro_se')
            i17 = st.text_input('Simetria Pior', key='simetria_pior')
            i22 = st.text_input('Área Pior', key='area_pior')
            i27 = st.text_input('Simetria SE', key='simetria_se')
        with col3:
            i3 = st.text_input('Perímetro Médio', key='perimetro_medio')
            i8 = st.text_input('Simetria Média', key='simetria_media')
            i13 = st.text_input('Área Média', key='area_media')
            i18 = st.text_input('Dimensão Fractal Média', key='dimensao_fractal_media')
            i23 = st.text_input('Suavidade Pior', key='suavidade_pior')
            i28 = st.text_input('Dimensão Fractal SE', key='dimensao_fractal_se')
        with col4:
            i4 = st.text_input('Área Média', key='area_media_2')
            i9 = st.text_input('Raio SE', key='raio_se')
            i14 = st.text_input('Suavidade SE', key='suavidade_se')
            i19 = st.text_input('Textura SE', key='textura_se')
            i24 = st.text_input('Simetria Pior', key='simetria_pior_2')
            i29 = st.text_input('Raio Pior', key='raio_pior')
        with col5:
            i5 = st.text_input('Suavidade Média', key='suavidade_media')
            i10 = st.text_input('Área SE', key='area_se_2')
            i15 = st.text_input('Raio Pior', key='raio_pior_2')
            i20 = st.text_input('Suavidade Pior', key='suavidade_pior_2')
            i25 = st.text_input('Concavidade SE', key='concavidade_se_2')
            i30 = st.text_input('Pontos Côncavos Pior', key='pontos_concavos_pior')

         # Código para a previsão
        breast_diagnosis = ''

        # Criando um botão para previsão
        if st.button('Prever'):
            # Convertendo os dados de entrada para ponto flutuante
            user_input = [float(i1), float(i2), float(i3), float(i4), float(i5),
                        float(i6), float(i7), float(i8), float(i9), float(i10),
                        float(i11), float(i12), float(i13), float(i14), float(i15),
                        float(i16), float(i17), float(i18), float(i19), float(i20),
                        float(i21), float(i22), float(i23), float(i24), float(i25),
                        float(i26), float(i27), float(i28), float(i29), float(i30)]

                # Realizando a previsão
            previsao_cancer_mama = breast_cancer_model.predict([user_input])

            if previsao_cancer_mama == 1:
                    breast_diagnosis = "O tumor é maligno."
            else:
                    breast_diagnosis = "O tumor é benigno."
                    st.write(breast_diagnosis)

        st.success(breast_diagnosis)


    elif selected_page == "feedback":
        st.title('Feedback dos Usuários')
        st.subheader("Deixe o seu feedback sobre a aplicação:")

        # Formulário para coletar o feedback
        navegacao = st.slider(" Como você avaliaria a facilidade de encontrar as diferentes funcionalidades na aplicação?", 1, 10, 5)
        design_interface = st.slider("Você achou o design da aplicação visualmente atraente? ", 1, 10, 5)
        elementos = st.slider(" Os elementos de interface, como botões e menus, eram intuitivos? ", 1, 10, 5)
        desempenho = st.slider("A aplicação funcionou sem problemas durante o uso? ", 1, 10, 5)
        atraso = st.slider ("Houve algum atraso ou lentidão ao carregar as páginas ou processar as solicitações?", 1, 5, 10)
        utilidade = st.slider("Você encontrou as informações ou funcionalidades que estava procurando? ", 1, 10, 5)
        expectativas = st.slider(" A aplicação atendeu às suas expectativas em termos de utilidade?", 1, 10, 5)
        sugestoes = st.text_area("Você tem alguma sugestão específica para melhorar a aplicação? ")
        recursos = st.text_area("Existem recursos adicionais que você gostaria de ver implementados? ")
        comentarios = st.text_area("Há mais alguma coisa que você gostaria de compartilhar sobre sua experiência geral com a aplicação?")

        nome = st.text_input("Nome:")
        email = st.text_input("E-mail:")
        telefone = st.text_input("Telefone:")
        mensagem = st.text_area("Mensagem:")

        if st.button("Enviar Feedback"):
            # Processar e armazenar o feedback 
            feedback = {
                "Navegação": navegacao,
                "Design da Interface": design_interface,
                "Elementos": elementos,
                "Desempenho": desempenho,
                "atraso": atraso,
                "Utilidade": utilidade,
                "Expectativas": expectativas,
                "Sugestões de Melhoria": sugestoes,
                "Recursos": recursos,
                "Comentários Adicionais": comentarios,
                "Nome": nome,
                "E-mail": email,
                "Telefone": telefone,
                "Mensagem": mensagem
            }

            # Armazenar o feedback em um banco de dados
            import sqlite3

            # Criar uma conexão com o banco de dados SQLite
            conn = sqlite3.connect('feedbackStreamlit.db')
            cursor = conn.cursor()

            # Criar a tabela de feedback se ela não existir
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    navegacao INT,
                    design_interface INT,
                    elementos INT,
                    desempenho INT,
                    atraso INT,
                    utilidade INT,
                    sugestoes TEXT,
                    recursos TEXT,
                    comentarios TEXT,
                    nome TEXT,
                    email TEXT,
                    telefone TEXT,
                    mensagem TEXT
                )
            ''')

            cursor.execute('''
                INSERT INTO feedback (navegacao, design_interface, desempenho, atraso, utilidade, sugestoes, comentarios, nome, email, telefone, mensagem)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (navegacao, design_interface, desempenho, atraso, utilidade, sugestoes, comentarios, nome, email, telefone, mensagem))

            conn.commit()
            conn.close()

            st.success("Feedback enviado com sucesso!")

    elif selected_page == "Datasets":
        # Título da página
        st.title('Datasets Disponíveis')
        # Introdução
        st.write("Esta página contém links para download e visualização de datasets utilizados na aplicação.")
        
        # Dicionário com URLs dos datasets
        datasets = {
            "Dataset de Malária": "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
            "Dataset de Pneumonia": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
            "Dataset de Doenças Cardíacas": "https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/heart.csv",
            "Dataset de Doenças Renais": "https://www.kaggle.com/datasets/mansoordaku/ckdisease",
            "Dataset de Diabetes": "https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/diabetes.csv",
            "Dataset de Doenças Hepáticas": "https://www.kaggle.com/datasets/uciml/indian-liver-patient-records",
            "Dataset de Câncer de Mama": "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
        }

        # Loop sobre os datasets para exibir links de download e botões para visualização
        for dataset_name, dataset_url in datasets.items():
            st.write(f"**{dataset_name}:**")
            st.markdown(f"[Download {dataset_name}]({dataset_url})")

    elif selected_page == "contato":
        st.title('Entre em Contato Conosco')
        st.write("Você pode entrar em contato conosco através do seguinte e-mail:")
        st.write("E-mail de Contato: contato@exemplo.com")

if __name__ == "__main__":
    main(selected_page)
