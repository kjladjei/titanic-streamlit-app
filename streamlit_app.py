
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data  # Cette ligne dit à Streamlit : "Garde le résultat en mémoire"
def load_data():
    return pd.read_csv("train.csv")
df = load_data()

#1 importation de la base train
#df = pd.read_csv("train.csv")

#2 # titre et sommaire page streamlit
st.title("PROJET DE CLASSIFICATION BINAIRE TITANIC")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

# partie introduction, sur l'app streamlit
if page == pages[0]:
    st.write("### INTRODUCTION")   #affichage de sous titre 'Introduction'
    st.write("#### affichage des premières lignes de notre base de données")
    st.dataframe(df.head())         # affichage des premières lignes de df

    st.write("#### la dimaension de notre base de données")
    st.write(df.shape)              # la dimension de df

    st.write("#### description sommaire résumé ou synthèse de notre base de données")
    st.dataframe(df.describe())     # synthèse de df

    if st.checkbox("Afficher les NA"):
        st.dataframe(df.isna().sum())




#DATA VISUALISATION



#3 écrire data visualisation en haut de la page 
if page == pages[1]:
    st.write("### DATA VISUALIZATION")
#description
    st.write("Nous nous intéressons à la variable cible 'Survived'. Cette variable prend 2 modalités : 0 si l'individu n'a pas survécu \n et 1 si l'individu a survécu.")

    #4 afficher dans un plot la distribution de la variable 'Survived'
    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)


    #Observation sur le résultat de la figure
    st.write(" Nous observons que la majorité des individus du Titanic n'ont pas survécu. \n Les deux classes (0 et 1) sont malgré tout assez équilibrées. Nous faisons maintenant une analyse descriptive des données pour obtenir le profil type d'un passager du Titanic.")
    
    #5 Afficher des plots permettant de décrire les passagers du Titanic. 

    #Repartition des selon le genre des passagers
    st.write(" Répartition selon le 'genre' des passagers ")
    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)

    #Repartition des selon le genre des passagers
    st.write(" Répartition selon les 'classes' des passagers ")
    fig = plt.figure()
    sns.countplot( x='Pclass', data=df)
    plt.title(" Répartition des passagers selon leurs 'classes' " )
    st.pyplot(fig)


    #Repartition des selon le genre des passagers
    st.write(" Répartition selon les 'classes' des passagers ")
    
    fig = plt.figure()
    sns.countplot( x='Age', data=df)
    plt.title(" Répartition des passagers selon leurs 'ages'" )
    st.pyplot(fig)


    #Résultat, interprétation
    st.write(" Nous observons que les passagers sont majoritairement des hommes en Classe 3 dont \n l'âge varie principalement entre 20 et 40 ans. ")

    st.write("  Nous analysons l'impact des différents facteurs sur la survie ou non des passagers. ")

    # Afficher un countplot de la variable cible en fonction du genre
    st.write(" countplot de la variable cible 'Survived' en fonction du genre. ")
    fig = plt.figure()
    sns.countplot( x='Survived', hue='Sex', data=df)
    st.pyplot(fig)

    # Afficher un plot de la variable cible en fonction des classes.
    st.write(" plot de la variable cible 'Survived'en fonction des classes. ")
    fig = sns.catplot( x='Pclass', y='Survived', data=df, kind='point' )
    st.pyplot(fig)

    # Afficher un plot de la variable cible en fonction des âges.
    st.write(" plot de la variable cible 'Survived'en fonction des âges ")
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    #analyse multivariée en regardant les corrélations entre les variables
    #la matrice de corrélation des variables explicatives.
    st.write(" carte de chaleur montrant la correlation entre les variables ")
    fig, ax = plt.subplots()
    # On ajoute numeric_only=True à l'intérieur de corr()
    sns.heatmap(df.corr(numeric_only=True), ax=ax, annot=True, cmap="RdBu")
    st.pyplot(fig) # Utilise st.pyplot(fig) plutôt que st.write(fig) pour les graphiques

    st.write(" le coefficient de corrélation de Pearson, qui varie de -1 à +1 c'est-à-dire [-1,1]. \n  - Proche de +1 (Bleu foncé) : Corrélation positive forte. Si une variable monte, l'autre monte aussi. \n correlation positive entre la variable target et la variable explicative en question ")
    st.write("- Proche de 0 (Beige/Blanc) : Aucune relation. Les variables ne s'influencent pas.")
    st.write("- Proche de -1 (Rouge foncé) : Corrélation négative forte. Si une variable monte, l'autre descend. ")


    st.write("##### Interprétation du Heatmap ")
    st.write(" - Pclass vs Fare (-0.55) : Corrélation négative forte. \n Plus la classe est haute (chiffre 1), plus le prix du billet augmente. \n C'est l'une des relations les plus fortes de ton tableau.   ")
    st.write(" - SibSp vs Parch (0.41) : Corrélation positive.  \n Ceux qui voyageaient avec des frères/sœurs (SibSp) avaient tendance à voyager aussi avec leurs parents/enfants (Parch). Ce sont les familles. ")

    st.write( " La Heatmap révèle que la classe sociale (Pclass) et le prix du billet (Fare) \n sont les variables numériques les plus liées à la survie. \n On observe une préférence marquée pour les passagers de classe supérieure.")
if page == pages[2]:
    st.write("### MODELISATION ")

    #6 suppression des variables non pertinentes (PassengerID, Name, Ticket, Cabin)
    st.write(" suppression des variables non pertinentes (PassengerID, Name, Ticket, Cabin) ")
    df = df.drop( ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    st.dataframe(df.head())
    st.write(" la dimension devient ", df.shape)


    #7 création d'une variable contenant la variable target 'Survived'
    y = df['Survived']
    #7 création Xcat = les variables explicatives catégorielles et Xnum = variables explicatives numériques
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]


    #8 Dans le script Python, remplacer les valeurs manquantes des variables catégorielles 
    #par le mode et remplacer les valeurs manquantes des variables numériques par la médiane.
    #9 Dans le script Python, encoder les variables catégorielles.
    #10 Dans le script Python, concatener les variables explicatives encodées et sans valeurs manquantes 
    #pour obtenir un dataframe X clean
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
        X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
        X = pd.concat([X_cat_scaled, X_num], axis = 1)


    #11 Dans le script Python, séparer les données en un ensemble d'entrainement et un ensemble t
    #est en utilisant la fonction train_test_split du package model_selection de Scikit-Learn.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


    #12 Dans le script Python, standardiser les valeurs numériques en utilisant la fonction StandardScaler 
    #du package Preprocessing de Scikit-Learn.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])


    #13 Dans le script Python, créer une fonction appelée prediction qui prend en argument le nom d'un classifieur et renvoie le classifieur entrainé.
    st.write(" On peut utiliser les classifieurs LogisticRegression, SVC et RandomForestClassifier de la librairie \n Scikit-Learn par exemple. ")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf


    st.write(" Puisque les classes ne sont pas déséquilibrées, il est intéressant de regarder l'accuracy \n des prédictions. Copiez le code suivant dans votre script Python. Il crée une fonction qui renvoie au choix l'accuracy ou la matrice de confusion. ")
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))



    #14 Nous avons effectué le preprocessing et la modélisation. Nous voulons montrer les résultats obtenus de 
    #façon intéractive sur l'application Web Streamlit.
    #Nous créons une "select box" permettant de choisir quel classifieur entrainer.

    #dans le script Python, utiliser la méthode st.selectbox() pour choisir entre le 
    #classifieur RandomForest, le classifieur SVM et le classifieur LogisticRegression. 
    #Puis retourner sur l'application web Streamlit pour visualiser la "select box".
    
    choix = ['Random Forest', 'SVC', 'Logistic Regression']

    option = st.selectbox('choix du modèle', choix)

    st.write(" Le modèle choisi est:", option)


    #Il ne reste maintenant qu'à entrainer le modèle
    st.write(" La méthode 'st.radio( )' permet d'afficher des cases à cocher pour choisir entre plusieurs options.")

    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        # On calcule le score
        score = scores(clf, display)
        
        # On crée deux colonnes pour aérer l'affichage
        col1, col2 = st.columns(2)
        
        with col1:
            # st.metric crée une jolie carte avec un titre et le chiffre en gros
            st.metric(label="Précision du modèle (Accuracy)", value=f"{score:.2%}")
        
        with col2:
            # Tu peux ajouter une petite explication ou un deuxième indicateur ici
            st.write("Ce score indique le pourcentage de prédictions correctes sur les données de test.")

    elif display == 'Confusion matrix':
        st.write("#### Matrice de Confusion")
        st.dataframe(scores(clf, display))


    #option deux ou 2ème méthode de ce dernier bloc de code
    #clf = prediction(option)
    #display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    #if display == 'Accuracy':
        #if display == 'Accuracy':
        #st.write(scores(clf, display))
    #elif display == 'Confusion matrix':
        #st.dataframe(scores(clf, display))"""
    

    

    

