import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from datetime import timedelta
import ast
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics.cluster import adjusted_rand_score
import re # Pour les expressions régulières dans code_segt

# --- Initialize st.session_state ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "À propos de nous" # Default initial choice

# Data import
st.sidebar.title("Navigation")
# Demander à l'utilisateur d'uploader un fichier
uploaded_file = st.sidebar.file_uploader("Importez votre fichier de données (Excel)", type=["xlsx"])

if uploaded_file is not None:
    # Check if the uploaded file is different from the one currently in session state
    # This prevents reprocessing the same file multiple times
    if st.session_state.df is None or uploaded_file.name != st.session_state.last_uploaded_filename:
        try:
            # Traitement selon le type de fichier
            if uploaded_file.name.endswith('.xlsx'):
                df_temp = pd.read_excel(uploaded_file)
                # Calculate 'Montant' and store in session state
                df_temp['Montant'] = df_temp['Quantity'] * df_temp['UnitPrice']
                st.session_state.df = df_temp
                st.session_state.last_uploaded_filename = uploaded_file.name # Store filename to check for changes
            else:
                st.error("Format de fichier non supporté.")
                st.session_state.df = None

            if st.session_state.df is not None:
                st.success("Fichier importé avec succès !")
                # Optionally, set a default menu choice after successful upload
                #st.session_state.menu_choice = "Statistiques Description des données"

        except Exception as e:
            st.error(f"Erreur lors de l'importation : {e}")
            st.session_state.df = None


menu = ["À propos de nous", "Statistiques Description des données", "Visualisation des données", "Modélisation et prédictions", "Résumé"]
# Use on_change to update session_state.menu_choice directly
choice = st.sidebar.selectbox('Menu', menu, key='sidebar_menu_selection', on_change=lambda: st.session_state.update(menu_choice=st.session_state.sidebar_menu_selection))


# --- Functions defined below ---

def description_data():
    df = st.session_state.df
    if df is not None:
        st.subheader("Affichage des données")
        st.dataframe(df.head(10))  # Afficher les 10 premières lignes du DataFrame
        st.write("le nombre total de lignes est :", df.shape[0])
        st.write("le nombre total de colonnes est :", df.shape[1])
        st.write("Valeurs manquantes avant suppression:")
        st.write(df.isna().sum())
        
        # Make a copy to avoid SettingWithCopyWarning if you modify df later
        df_cleaned = df.dropna().copy() 
        st.subheader("Statistiques descriptives")
        numcol=['Quantity', 'UnitPrice', 'Montant']  # Liste des colonnes numériques
        st.write(df_cleaned[numcol].describe())
        st.write("Valeurs manquantes après suppression:")
        st.write(df_cleaned.isna().sum())
        # Update df in session state with the cleaned version if you want to use it downstream
        st.session_state.df = df_cleaned
    else:
        st.warning("Veuillez d'abord importer un fichier de données.")
        

def visualize_data():
    df = st.session_state.df
    if df is not None:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()

        # Préparation des données pour les graphiques
        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(5).reset_index()

        # Création du graphique Plotly
        fig1 = px.bar(
            top_products,
            x="Description",
            y="Quantity",
            orientation='v',
            title="🎯 Top 5 produits vendus",
            color='Description',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Affichage dans Streamlit
        st.plotly_chart(fig1)


        # Extraction du jour de la semaine
        # Ensure 'InvoiceDate' is datetime before operations
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Num_Jour'] = df['InvoiceDate'].dt.dayofweek

        # Groupement et comptage
        order_day = df.groupby('Num_Jour')['InvoiceNo'].nunique()

        # Création du graphique avec matplotlib/seaborn
        st.subheader("📊 Nombre de Transactions par Jour de la Semaine")
        # Initialisation de la figure
        fig2, ax = plt.subplots(figsize=(12, 8))

        # Tracé du graphique
        sns.barplot(x=order_day.index, y=order_day.values, palette="Set3", ax=ax)
        ax.set_title('Nombre de Transactions par Jour', size=20)
        ax.set_xlabel('Jour', size=14)
        ax.set_ylabel('Nombre de Transactions', size=14)
        ax.xaxis.set_tick_params(labelsize=11)
        ax.yaxis.set_tick_params(labelsize=11)
        ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])

        # Affichage dans Streamlit
        st.pyplot(fig2)


        # S'assurer que 'InvoiceDate' est bien au format datetime
        #df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Extraire uniquement la date (sans l'heure)
        df["Date"] = df["InvoiceDate"].dt.date

        # Calcul du chiffre d'affaires par jour
        sales_over_time = df.groupby("Date")["Montant"].sum().reset_index()

        # Création du graphique en courbe
        fig3 = px.line(
            sales_over_time,
            x="Date",
            y="Montant",
            title="📅 Évolution du chiffre d’affaires",
            markers=True,
            line_shape="linear"
        )

        # Affichage dans Streamlit
        st.plotly_chart(fig3)

        # Convertir en chaîne pour de jolis labels
        df["CustomerID"] = df["CustomerID"].astype(str)
        
        # Comptage du nombre d'achats par client (nombre de factures uniques)
        clients_top_achats = (
            df.groupby("CustomerID")["InvoiceNo"]
            .nunique()
            .reset_index()
            .sort_values(by="InvoiceNo", ascending=False)
            .head(5)
        )

        # Création du graphique Plotly
        fig4 = px.bar(
            clients_top_achats,
            x="CustomerID",
            y="InvoiceNo",
            text="InvoiceNo",
            title="👥 Top 5 des clients ayant fait plus d’achats",
            color="CustomerID",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Ajout des étiquettes de valeur
        fig4.update_traces(texttemplate='%{text}', textposition='outside')

        # Mise en forme du graphique
        fig4.update_layout(
            xaxis_title="ID Client",
            yaxis_title="Nombre d'achats",
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )

        # Affichage dans Streamlit
        st.plotly_chart(fig4)
    else:
        st.warning("Veuillez d'abord importer un fichier de données.")


def modeling_and_predictions():
    df = st.session_state.df
    if df is None:
        st.warning("Veuillez d'abord importer un fichier de données pour la modélisation.")
        return
    
    menu1 = ["K-means", "Segmentation RFM", "FP_GROWTH"] 
    choix = st.sidebar.selectbox('Choisisser les méthodes suivants', menu1, key='modeling_menu_selection')

    # Creation of df_invoice (should ideally be done once or based on df update)
  
    df_invoice=df.groupby(['InvoiceNo', 'InvoiceDate', 'Quantity', 'CustomerID']).agg({'Montant': lambda x:x.sum()}).reset_index()
    
    # creation de analysis_date
    analysis_date = max(df_invoice['InvoiceDate']) + timedelta(days= 1)
    
    def kmeans_clustering(df_to_cluster):
        st.subheader("K-means Clustering")
        
        base = df_to_cluster.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'Montant': 'sum'
        }).rename(columns={
            'InvoiceDate': 'Recence',
            'InvoiceNo': 'Frequence',
            'Quantity': 'Quantite_totale',
            'Montant': 'Montant'
        }).reset_index()
        
        # Normalisation des données
        scaler = StandardScaler()
        base_scaled = scaler.fit_transform(base[['Recence', 'Frequence', 'Quantite_totale', 'Montant']])
        
        st.subheader("📉 Méthode du coude pour déterminer le nombre optimal de clusters")

        inertia = []
        cl = 15

        for i in range(1, cl + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10) # Added n_init for KMeans
            kmeans.fit(base_scaled)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 7)
        plt.rcParams['font.size'] = 16

        ax.plot(range(1, cl + 1), inertia, 'o-', color='g')
        ax.set_xticks(np.arange(1, cl + 1, 1.0))
        ax.set_title('Méthode du coude')
        ax.set_xlabel('Nombre de clusters (k)')
        ax.set_ylabel('Inertie (somme des distances au carré)')
        ax.grid(True)

        st.pyplot(fig)
        
        # Application du K-means
        n_clusters_selected = st.slider("Sélectionnez le nombre de clusters (k)", 2, 10, 5) # User can select k

        
        model = KMeans(n_clusters=n_clusters_selected, init='k-means++', max_iter=300, n_init=10, random_state=0)
        model_kmeans = model.fit(base_scaled)
        labels = model_kmeans.labels_
        base['cluster'] = labels
        
        st.subheader("Visualisation des clusters K-means")
        plt.rcParams['font.size'] = 16

        colonnes = ['Montant', 'Recence', 'Frequence', 'Quantite_totale']
        abscisses = st.selectbox("Choisissez la variable pour l'axe des abscisses", colonnes, index=colonnes.index('Recence'))
        ordonnees = st.selectbox("Choisissez la variable pour l'axe des ordonnées", colonnes, index=colonnes.index('Montant'))
        
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 7)

        sns.scatterplot(data=base, x=abscisses, y=ordonnees, hue='cluster', palette='Set1', ax=ax)

        ax.set_title(f'Clusters K-means ({n_clusters_selected} clusters)')
        ax.set_xlabel(abscisses)
        ax.set_ylabel(ordonnees)
        ax.grid(True)

        st.pyplot(fig)

        st.dataframe(base)

        def evaluate_stability(df1, n_clusters, n_init):
            # Convertir la base de données en une matrice numpy
            X = df1

            # Liste pour stocker les indices ARI pour chaque initialisation
            ari_scores = []

            for _ in range(n_init):
                # Initialisation et exécution de K-means
                kmeans = KMeans(n_clusters=n_clusters)
                labels = kmeans.fit_predict(X)

                # Calcul de l'indice ARI
                ari_scores.append(labels)

            # Calcul de la matrice de similarité
            similarity_matrix = np.zeros((n_init, n_init))
            for i in range(n_init):
                for j in range(n_init):
                    similarity_matrix[i, j] = adjusted_rand_score(ari_scores[i], ari_scores[j])

            # Calcul de la stabilité moyenne
            stability = np.mean(similarity_matrix)

            return stability
        
        st.subheader("Stabilité du K-means")
        # Affichage de la stabilité de K-means à l'initialisation
        n_clusters=5
        n_init =40
        stability = evaluate_stability(base_scaled, n_clusters, n_init)
        st.write("Stabilité de k-means à l'initialisation: ", stability)
        st.info("Un score plus proche de 1 indique une meilleure stabilité des clusters.")

        # Contrat de maintenance
        st.subheader("Contrat de maintenance")
        ARI_score = pd.read_csv("ari_scores.csv")
        ARI_scores = ARI_score['ARI_score'].tolist()

        # Paramètres de style
        sns.set(rc={'figure.figsize': (10, 6)})

        # Création de la figure
        fig, ax = plt.subplots()

        # Tracé de la courbe
        sns.lineplot(x=range(1, 12), y=ARI_scores, ax=ax)
        ax.axvline(4, c='red', ls='--')

        # Personnalisation
        ax.set_title('Évolution du score ARI')
        ax.set_xlabel('Semaines')
        ax.set_ylabel('Score ARI')
        ax.set_xticks(range(1, 12))

        # Affichage dans Streamlit
        st.pyplot(fig)


    def segmentation_rfm_func(df_for_rfm): # Renamed to avoid conflict with `segmentation_rfm` string
        rfm = df_for_rfm.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,
            'InvoiceNo': 'count',
            'Montant': 'sum'})
        rfm.rename(columns= {'InvoiceDate':'Recence',
                                'InvoiceNo':'Frequence',
                                'Montant':'Montant'}, inplace = True)

        #Calcul des quartiles
        quartiles=rfm[[ 'Recence', 'Frequence',  'Montant']].quantile([0.25, 0.5, 0.75]).to_dict()
        
        def r_score(x) :
            if x <= quartiles['Recence'][0.25] :
                return 4
            elif quartiles['Recence'][0.25]  < x <= quartiles['Recence'][0.5]:
                return 3
            elif quartiles['Recence'][0.5]  < x <= quartiles['Recence'][0.75]:
                return 2
            else :
                return 1

        def fm_score(x, col) :
            if x <= quartiles[col][0.25] :
                return 1
            elif quartiles[col][0.25]  < x <= quartiles[col][0.5]:
                return 2
            elif quartiles[col][0.5]  < x <= quartiles[col][0.75]:
                return 3
            else :
                return 4
        
        # Application des fonctions de score
        rfm['R_Score']=rfm['Recence'].apply(lambda x:  r_score(x))
        rfm['F_Score']=rfm['Frequence'].apply(lambda x:  fm_score(x, 'Frequence'))
        rfm['M_Score']=rfm['Montant'].apply(lambda x:  fm_score(x, 'Montant'))
        
        # Concatenation des scores RFM
        rfm['RFM_score']=rfm['R_Score'].map(str) + rfm['F_Score'].map(str) +rfm['M_Score'].map(str)
        
        # Affichage des scores RFM
        st.subheader("Segmentation RFM")
        st.write("Scores RFM attribués aux clients :")
    
        # Définition de la carte de segmentation
        code_segt= {
                    r'11': 'Clients en hibernation',
                    r'1[2-3]': 'Clients à risque',
                    r'14': 'Clients à ne pas perdre',
                    r'21': 'Clients presqu\'endormis',
                    r'22': 'Clients à suivre',
                    r'[2-3][3-4]': 'Clients loyaux',
                    r'31': 'Clients prometteurs',
                    r'41': 'Nouveaux clients',
                    r'[3-4]2': 'Clients potentiellement loyaux',
                    r'4[3-4]': 'Très bons clients'
                }

        #Ajout de la colonne "Segment" au dataframe rfm
        rfm['Segment'] = rfm['R_Score'].map(str) + rfm['F_Score'].map(str)
        rfm['Segment'] = rfm['Segment'].replace(code_segt, regex=True)
        st.dataframe(rfm)  
        
        # Visualisation des segments dans Streamlit
        st.subheader("📊 Répartition des Clients par Segment")

        fig, ax = plt.subplots(figsize=(12, 10))

        segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

        norm = plt.Normalize(segments_counts.min(), segments_counts.max())
        colors = cm.Blues(norm(segments_counts.values))

        bars = ax.barh(range(len(segments_counts)),
                        segments_counts,
                        color=colors)

        ax.set_frame_on(False)
        ax.tick_params(left=False,
                        bottom=False,
                        labelbottom=False)
        ax.set_yticks(range(len(segments_counts)))
        ax.set_yticklabels(segments_counts.index)

        for i, bar in enumerate(bars):
            value = bar.get_width()
            ax.text(value,
                    bar.get_y() + bar.get_height() / 2,
                    '{:,} ({:}%)'.format(int(value),
                                            int(value * 100 / segments_counts.sum())),
                    va='center',
                    ha='left')
        

        st.pyplot(fig)


    def fp_growth_func(df_for_fp_growth): # Renamed to avoid conflict with `fp_growth` string
        st.subheader("Analyse des associations avec FP-Growth")
        
        # Placeholder for `regle_association.csv` - ensure this file exists in your deployment
        # For demonstration, let's create a dummy rules dataframe if the file is not found.
        try:
            @st.cache_data
            def load_rules():
                rules = pd.read_csv("regle_association.csv")
                rules['antecedents']=rules['antecedents'].apply(ast.literal_eval)
                rules['consequents']=rules['consequents'].apply(ast.literal_eval)
                return rules
            rules = load_rules()
        except FileNotFoundError:
            st.warning("`regle_association.csv` non trouvé. Création de règles d'association d'exemple.")
            rules_data = {
                'antecedents': [['Product A']],
                'consequents': [['Product B']],
                'support': [0.1],
                'confidence': [0.8],
                'lift': [1.5]
            }
            rules = pd.DataFrame(rules_data)
            # Add more dummy rules for a better example
            rules = pd.concat([rules, pd.DataFrame({
                'antecedents': [['Product C']],
                'consequents': [['Product D']],
                'support': [0.05],
                'confidence': [0.7],
                'lift': [1.2]
            }), pd.DataFrame({
                'antecedents': [['Product B']],
                'consequents': [['Product E']],
                'support': [0.07],
                'confidence': [0.9],
                'lift': [1.8]
            })])
            # Ensure proper types for new dummy data
            rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: frozenset(x))
            st.info("Des règles d'association par défaut sont utilisées. Uploadez `regle_association.csv` pour des résultats réels.")


        st.title(" RECOMMENDATION D'ARTICLES")

        unique_items=sorted(set(item for ant in rules['antecedents'] for item in ant) | set(item for con in rules['consequents'] for item in con))
        
        if not unique_items:
            st.warning("Aucun article trouvé dans les règles d'association. Veuillez vous assurer que le fichier `regle_association.csv` est correct.")
            return

        selected_item=st.selectbox("Sélectionner un produit", unique_items)

        min_conf=st.slider("Confiance minimum", 0.0, 1.0, 0.5, 0.05)
        min_lift=st.slider("Lift minimum", 0.0, 5.0, 1.0, 0.1) # Adjusted lift range, often >1

        # Filter rules based on the selected item
        filtered_rules = rules [
            rules['antecedents'].apply(lambda x : selected_item in x) & 
            (rules['confidence'] >= min_conf) & 
            (rules['lift'] >= min_lift ) ]

        filtered_rules = filtered_rules.sort_values(by= 'confidence', ascending=False)
        filtered_rules['consequents_set'] = filtered_rules['consequents'].apply(lambda x : frozenset(x))
        filtered_rules = filtered_rules.drop_duplicates(subset='consequents_set')

        if not filtered_rules.empty : 
            st.subheader(f"Recommandations pour : {selected_item}")
            for _, row in filtered_rules.iterrows():
                recommended_items= ', '.join(list(row['consequents'])) # Convert frozenset to list for join
                st.markdown(f"""
                - Produit recommandé : **{recommended_items}**
                - Confiance : `{row['confidence']:.2f}`
                - Lift : `{row['lift']:.2f}`
                """)
        else :
            st.warning(f"Aucune recommandation trouvée pour '{selected_item}' avec les critères spécifiés.")


    if choix == "K-means":
        kmeans_clustering(df_invoice) # Pass df_invoice directly
    elif choix == "Segmentation RFM":
        segmentation_rfm_func(df_invoice) # Pass df_invoice directly
    else: # FP_GROWTH
        fp_growth_func(df) # FP-Growth typically uses the original transaction data, not aggregated RFM



# Resumé

def Summary():
    st.header("Analyse des Ventes - Résumé Exécutif")
    st.subheader("Voici Votre Analyse des Ventes")
    st.divider()

# Ou avec le style "star"
st.header("Voici Votre Analyse des Ventes", divider="star")

if 'df' in st.session_state and st.session_state['df'] is not None:
        df_selection = st.session_state['df'].copy() # Travailler sur une copie

        # Assurez-vous que 'InvoiceDate' est bien un type datetime
       # df_selection['InvoiceDate'] = pd.to_datetime(df_selection['InvoiceDate'])

        # Calcul des ventes totales à partir de la colonne 'Montant'
        total_sales = int(df_selection["Montant"].sum())
        
        # Coûts totaux (remplacez par un calcul réel si vous avez une colonne de coûts)
        # Pour l'instant, c'est un placeholder.
        total_costs = 0 
        total_profit = total_sales - total_costs

        # Vente moyenne par transaction (en utilisant 'Montant' et 'InvoiceNo' comme identifiant unique de transaction)
        # S'assurer que InvoiceNo n'est pas nul si vous l'utilisez pour regrouper les transactions
        avg_sales_per_transaction = round(df_selection.groupby('InvoiceNo')['Montant'].sum().mean(), 2)
        
        # Nombre total de clients uniques
        # Filtrer les valeurs NaN de CustomerID avant de compter les uniques
        num_unique_customers = df_selection['CustomerID'].dropna().nunique()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Montant Totales:")
            st.subheader(f" {total_sales:,.0f}")

        with col2:
            st.subheader("Nombre Total de Clients :")
            st.subheader(f"{num_unique_customers:,}")
            
        with col3:
            st.subheader("Prix Moyenne par Transaction:")
            st.subheader(f"{avg_sales_per_transaction:,.2f}") 

        st.markdown("---") # Séparateur

        # Ventes par Description de Produit (Top 5)
        sales_by_description = df_selection.groupby(by=["Description"])[["Montant"]].sum().sort_values(
            by="Montant", ascending=False).head(5)
        fig_description_sales = px.bar(
            sales_by_description,
            y="Montant",
            x=sales_by_description.index,
            orientation="v",
            title="<b>Top 5 des Produits ayant  plus de chiffre d'affaires</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_description),
            template="plotly_white",
        )
        fig_description_sales.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )
        
        # Ventes par Pays (Top 5)
        sales_by_country = df_selection.groupby(by=["Country"])[["Montant"]].sum().sort_values(
            by="Montant", ascending=False).head(5)
        fig_country_sales_treemap = px.treemap(
            sales_by_country.reset_index(),
            path=["Country"],
            values="Montant",
            title="<b>Top 5 Pays par Ventes</b>",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_white",
        )
        fig_country_sales_treemap.update_layout(
            plot_bgcolor="rgba(200, 200, 200, 0.2)",
        )

        col_charts1, col_charts2 = st.columns(2)
        col_charts1.plotly_chart(fig_description_sales, use_container_width=True)
        col_charts2.plotly_chart(fig_country_sales_treemap, use_container_width=True)




        
        st.markdown("---") # Séparateur

        # Analyse de la fidélité client
        # --- Analyse de la fidélité client (Segmentation RFM) ---
        st.subheader("Segmentation de la Fidélité Client (Méthode RFM)")

        # Calculer RFM pour le DataFrame de la session
        # On utilise analysis_date comme le jour suivant la dernière transaction
        analysis_date = df_selection['InvoiceDate'].max() + timedelta(days=1)

        rfm_df = df_selection.groupby('CustomerID').agg(
            Recence=('InvoiceDate', lambda date: (analysis_date - date.max()).days),
            Frequence=('InvoiceNo', 'nunique'),
            Montant=('Montant', 'sum')
        ).reset_index()

        # Calcul des quartiles pour la segmentation RFM
        quartiles = rfm_df[['Recence', 'Frequence', 'Montant']].quantile([0.25, 0.5, 0.75]).to_dict()

        # Fonctions pour attribuer les scores R, F, M (identiques à votre code)
        def r_score(x):
            if x <= quartiles['Recence'][0.25]:
                return 4
            elif quartiles['Recence'][0.25] < x <= quartiles['Recence'][0.5]:
                return 3
            elif quartiles['Recence'][0.5] < x <= quartiles['Recence'][0.75]:
                return 2
            else:
                return 1

        def fm_score(x, col):
            if x <= quartiles[col][0.25]:
                return 1
            elif quartiles[col][0.25] < x <= quartiles[col][0.5]:
                return 2
            elif quartiles[col][0.5] < x <= quartiles[col][0.75]:
                return 3
            else:
                return 4

        # Application des fonctions de score
        rfm_df['R_Score'] = rfm_df['Recence'].apply(lambda x: r_score(x))
        rfm_df['F_Score'] = rfm_df['Frequence'].apply(lambda x: fm_score(x, 'Frequence'))
        rfm_df['M_Score'] = rfm_df['Montant'].apply(lambda x: fm_score(x, 'Montant'))

        # Définition de la carte de segmentation (identique à votre code)
        code_segt = {
            r'11': 'Clients en hibernation',
            r'1[2-3]': 'Clients à risque',
            r'14': 'Clients à ne pas perdre',
            r'21': 'Clients presqu\'endormis',
            r'22': 'Clients à suivre',
            r'[2-3][3-4]': 'Clients loyaux',
            r'31': 'Clients prometteurs',
            r'41': 'Nouveaux clients',
            r'[3-4]2': 'Clients potentiellement loyaux',
            r'4[3-4]': 'Très bons clients'
        }

        # Fonction pour appliquer la segmentation basée sur R_Score et F_Score avec regex
        def apply_rfm_segment(row):
            r_f_score_str = f"{row['R_Score']}{row['F_Score']}"
            for pattern, segment_name in code_segt.items():
                if re.match(pattern, r_f_score_str):
                    return segment_name
            return 'Non défini' # Fallback if no pattern matches

        rfm_df['Segment'] = rfm_df.apply(apply_rfm_segment, axis=1)  
        
        # Visualisation des segments (identique à votre code)
        st.subheader("📊 Répartition des Clients par Segment")
        fig, ax = plt.subplots(figsize=(12, 10)) # Nécessite 'import matplotlib.pyplot as plt' et 'import matplotlib.cm as cm'
        segments_counts = rfm_df['Segment'].value_counts().sort_values(ascending=True)
        norm = plt.Normalize(segments_counts.min(), segments_counts.max())
        colors = cm.Blues(norm(segments_counts.values))
        bars = ax.barh(range(len(segments_counts)), segments_counts, color=colors)
        ax.set_frame_on(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.set_yticks(range(len(segments_counts)))
        ax.set_yticklabels(segments_counts.index)
        for i, bar in enumerate(bars):
            value = bar.get_width()
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    '{:,} ({:}%)'.format(int(value), int(value * 100 / segments_counts.sum())),
                    va='center', ha='left')
        st.pyplot(fig)


        st.subheader("Meilleurs Clients (Top 5 par Total des Achats)")
        st.dataframe(rfm_df.nlargest(5, 'Montant')) # Utilise 'Montant' de rfm_df


else:
        st.warning("Veuillez d'abord importer un fichier de données pour afficher le résumé des ventes.")

# --- Main Application Logic ---
# Use st.session_state.menu_choice to control which section is displayed
if st.session_state.menu_choice == "À propos de nous":
    st.title("📊 Analyse et Segmentation des clients pour la decision marketing ")
    st.header("📈 Application d'analyse comportementale a partir de données transactionnelles", divider='rainbow')
    try:
        image = Image.open("image_ecommerce.jpg")
        st.image(image, caption="image_ecommerce")
    except FileNotFoundError:
        st.warning("L'image 'image_ecommerce.jpg' n'a pas été trouvée. Veuillez vous assurer qu'elle est dans le même répertoire que votre script Streamlit.")
    st.subheader("À propos de nous")
    st.write("Nous sommes une équipe dédiée à l'analyse des données clients pour améliorer les stratégies commerciales.")
    st.write("Notre objectif est de fournir des insights précieux à partir des données clients pour aider les entreprises à mieux comprendre leurs clients et à optimiser leurs opérations.")

# Create three columns layout
    left_column, middle1_column, middle2_column, right_column = st.columns(4)

# Left column - Email
    left_column.subheader("Nom")
    left_column.markdown("**Aliou Diack**")
    left_column.markdown("**Mamadou Lamarana Diallo**")
# middle1 column - Email
    middle1_column.subheader("📧 Email")
    middle1_column.markdown("[diackaliou4@gmail.com](mailto:diackaliou4@gmail.com)")
    middle1_column.markdown("[mamadoulamaranadiallomld1@gmail.com](mailto:mamadoulamaranadiallomld1@gmail.com)")

# Middle2 column - Phone
    middle2_column.subheader("☎️ Contact ")
    middle2_column.markdown("\n\n\n[+221 782948335](tel:+221782948335)")
    middle2_column.markdown("[+221 771050342](tel:+221771050342)")

# Right column - Linkedin
    right_column.markdown("""<h3><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align: middle;"> LinkedIn</h3> """, unsafe_allow_html=True)
    right_column.markdown("[Linkedin](https://www.linkedin.com/in/aliou-diack-977771241/)")
    right_column.markdown("[Linkedin](https://www.linkedin.com/in/mamadou-lamarana-diallo-937430274/)")

elif st.session_state.menu_choice == "Statistiques Description des données":
    st.subheader("Statistiques Description des données")
    description_data()
elif st.session_state.menu_choice == "Visualisation des données":
    st.subheader("Visualisation des données")
    visualize_data()
elif st.session_state.menu_choice == "Modélisation et prédictions":
    st.subheader("Modélisation et prédictions")
    modeling_and_predictions()
elif st.session_state.menu_choice == "Résumé":
    st.subheader("Résumé des ventes")
    Summary()
else:
    st.error("Veuillez sélectionner une option valide dans le menu.")
