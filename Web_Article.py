import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from scipy.interpolate import griddata

# 1. CONFIGURATION DE LA PAGE STREAMLIT
st.set_page_config(
    page_title="HydroPredict Maradi - Application Scientifique",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-title { font-size: 28px; font-weight: 700; color: #0E4C92; margin-bottom: 10px; }
        .sub-title { font-size: 16px; color: #555555; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# 2. CHARGEMENT ET PREPARATION DES DONNEES (SUPER ROBUSTE)
@st.cache_data
def charger_donnees():
    fichier = 'Caracteristiques_PEM_Maradi.xlsx'
    try:
        xls = pd.ExcelFile(fichier)
        sheet_name = 'Caractéristiques' if 'Caractéristiques' in xls.sheet_names else xls.sheet_names[0]
        df_raw = pd.read_excel(fichier, sheet_name=sheet_name)
        
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        cols = df_raw.columns.tolist()
        
        def trouver_col(mots_cles):
            for c in cols:
                for m in mots_cles:
                    if m in c.lower():
                        return c
            return None

        c_dept = trouver_col(['departement', 'département', 'dept'])
        c_comm = trouver_col(['commune', 'com'])
        c_lat = trouver_col(['lat'])
        c_lon = trouver_col(['lon'])
        c_alt = trouver_col(['alt'])
        c_prof = trouver_col(['prof_f', 'profondeur_foree', 'foree', 'forée'])
        c_equi = trouver_col(['prof_eq', 'profondeur_equipee', 'equipee', 'équipée'])
        c_diam = trouver_col(['diam', 'diametre'])
        c_stat = trouver_col(['niveau_statique', 'statique', 'niv'])
        c_debit = trouver_col(['debit', 'débit'])

        df_clean = pd.DataFrame()
        df_clean['departement'] = df_raw[c_dept] if c_dept else "Inconnu"
        df_clean['commune'] = df_raw[c_comm] if c_comm else "Inconnu"
        df_clean['latitude'] = pd.to_numeric(df_raw[c_lat], errors='coerce') if c_lat else 0.0
        df_clean['longitude'] = pd.to_numeric(df_raw[c_lon], errors='coerce') if c_lon else 0.0
        df_clean['altitude'] = pd.to_numeric(df_raw[c_alt], errors='coerce') if c_alt else 0.0
        df_clean['profondeur_foree'] = pd.to_numeric(df_raw[c_prof], errors='coerce') if c_prof else 0.0
        df_clean['profondeur_equipee'] = pd.to_numeric(df_raw[c_equi], errors='coerce') if c_equi else 0.0
        df_clean['diametre'] = pd.to_numeric(df_raw[c_diam], errors='coerce') if c_diam else 0.0
        df_clean['niveau_statique'] = pd.to_numeric(df_raw[c_stat], errors='coerce') if c_stat else 0.0
        df_clean['debit'] = pd.to_numeric(df_raw[c_debit], errors='coerce') if c_debit else 0.0

        df_clean = df_clean.dropna(subset=['debit', 'profondeur_foree', 'latitude', 'longitude']).copy()
        df_clean = df_clean[(df_clean['debit'] > 0) & (df_clean['profondeur_foree'] > 0)]
        
        df_clean['Zone_Hydro'] = df_clean['departement'].apply(
            lambda x: 'Socle (Madarounfa)' if 'MADAROUNFA' in str(x).upper() else 'Sédimentaire (Guidan Roumdji)'
        )
        return df_clean
    except Exception as e:
        st.error(f"Détail de l'erreur : {e}")
        return None

df_clean = charger_donnees()

if df_clean is None or len(df_clean) == 0:
    st.error("⚠️ Impossible de charger les données.")
    st.stop()

# 3. BARRE LATERALE (NAVIGATION & PARAMÈTRES)
st.sidebar.image("https://img.icons8.com/color/96/hydrology.png", width=70)
st.sidebar.title("HydroPredict Maradi")
st.sidebar.markdown("Plateforme d'analyse et de modélisation par Machine Learning des forages.")

menu = st.sidebar.selectbox(
    "Navigation Principale",
    [
        "Accueil & Synthèse",
        "Cartographie & Interpolation",
        "Statistiques & Corrélations",
        "Modélisation Machine Learning",
        "Évaluation & Résidus",
        "Simulation / Prédiction en Direct",
        "À propos de l'auteur"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Paramètres du Modèle")
n_est = st.sidebar.slider("Nombre d'arbres (n_estimators)", 50, 300, 150, 25)
max_d = st.sidebar.slider("Profondeur maximale (max_depth)", 3, 20, 10)

communes_dispo = ["Toutes"] + sorted(df_clean['commune'].dropna().unique().tolist())
commune_filtre = st.sidebar.selectbox("Filtrer par Commune", communes_dispo)

df_travail = df_clean.copy()
if commune_filtre != "Toutes":
    df_travail = df_travail[df_travail['commune'] == commune_filtre]


# 4. PAGES DE L'APPLICATION

if menu == "Accueil & Synthèse":
    st.markdown('<p class="main-title">💧 Système d\'Aide à la Décision Hydrogéologique</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Région de Maradi (Niger) - Comparaison Socle Cristallin / Bassin Sédimentaire</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Forages Validés", len(df_travail))
    with c2: st.metric("Débit Moy.", f"{df_travail['debit'].mean():.2f} m³/h")
    with c3: st.metric("Profondeur Moy.", f"{df_travail['profondeur_foree'].mean():.1f} m")
    with c4: st.metric("Niveau Statique Moy.", f"{df_travail['niveau_statique'].mean():.1f} m")
    
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎯 Objectifs de l'application")
        st.markdown("""
        Cette application interactive centralise les résultats de recherche et de modélisation prédictive :
        * **Analyse comparative** rigoureuse entre Madarounfa (Socle) et Guidan Roumdji (Sédimentaire).
        * **Cartographie spatiale** interactive des ouvrages et interpolation piézométrique.
        * **Modélisation non-linéaire** par *Random Forest* pour la prédiction des débits.
        * **Simulation en direct** pour l'optimisation des implantations futures.
        """)
    with col_b:
        st.subheader("📋 Aperçu des Données Filtrées")
        st.dataframe(df_travail[['departement', 'commune', 'profondeur_foree', 'niveau_statique', 'debit']].head(6), use_container_width=True)
        
        csv_data = df_travail.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv_data,
            file_name='donnees_hydrogeologiques_maradi.csv',
            mime='text/csv'
        )

elif menu == "Cartographie & Interpolation":
    st.markdown('<p class="main-title">🗺️ Cartographie Spatiale et Interpolation</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Répartition des Forages", "Interpolation du Niveau Statique"])

    with tab1:
        st.subheader("Distribution Spatiale des Forages par Zone Hydrogéologique")
        if len(df_travail) > 0:
            fig_map = px.scatter(
                df_travail, x="longitude", y="latitude", color="Zone_Hydro",
                size="debit", hover_name="commune", hover_data=["profondeur_foree", "niveau_statique"],
                color_discrete_map={'Socle (Madarounfa)': 'teal', 'Sédimentaire (Guidan Roumdji)': 'darkorange'},
                labels={"longitude": "Longitude (°)", "latitude": "Latitude (°)"}
            )
            fig_map.update_layout(margin={"r":20,"t":30,"l":20,"b":20}, legend_title="Zone Hydrogéologique")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible.")

    with tab2:
        st.subheader("Carte d'Interpolation Spatiale : Niveau Statique")
        if len(df_travail) > 5:
            fig, ax = plt.subplots(figsize=(9, 6))
            x, y, z = df_travail['longitude'].values, df_travail['latitude'].values, df_travail['niveau_statique'].values
            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((x, y), z, (XI, YI), method='linear')
            
            contour = ax.contourf(XI, YI, ZI, levels=30, cmap='YlGnBu_r', alpha=0.85)
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Niveau Statique (m)')
            ax.scatter(x, y, c='black', s=12, alpha=0.6, label='Forages')
            ax.set_title("Interpolation Spatiale du Niveau Statique", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("Pas assez de points pour l'interpolation.")

elif menu == "Statistiques & Corrélations":
    st.markdown('<p class="main-title">📊 Statistiques Descriptives & Corrélations</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Statistiques Comparatives", "Nuage de Points Bivarié", "Matrice de Corrélation"])

    with tab1:
        st.subheader("Tableau de Synthèse Statistique par Contexte")
        vars_num = ['profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique', 'debit']
        if len(df_travail) > 0:
            stats_summary = df_travail.groupby('Zone_Hydro')[vars_num].agg(['mean', 'median', 'std', 'min', 'max']).T
            st.dataframe(stats_summary, use_container_width=True)

    with tab2:
        st.subheader("Relation : Profondeur Forée vs Débit")
        if len(df_travail) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.scatterplot(
                data=df_travail, x='profondeur_foree', y='debit', hue='Zone_Hydro',
                palette={'Socle (Madarounfa)': 'teal', 'Sédimentaire (Guidan Roumdji)': 'darkorange'},
                alpha=0.8, s=60, ax=ax
            )
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

    with tab3:
        st.subheader("Matrice de Corrélation")
        cols_corr = ['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique', 'debit']
        if len(df_travail) > 2:
            corr_matrix = df_travail[cols_corr].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)

elif menu == "Modélisation Machine Learning":
    st.markdown('<p class="main-title">🤖 Modélisation Prédictive (Random Forest)</p>', unsafe_allow_html=True)
    zone_choisie = st.selectbox("Sélectionner la zone géologique à modéliser :", ["Socle (Madarounfa)", "Sédimentaire (Guidan Roumdji)"])
    df_contexte = df_travail[df_travail['Zone_Hydro'] == zone_choisie].copy()
    
    if len(df_contexte) > 10:
        num_cols = ['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique']
        imputer = SimpleImputer(strategy='median')
        df_contexte[num_cols] = imputer.fit_transform(df_contexte[num_cols])
        
        X = df_contexte[['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique']]
        y = df_contexte['debit']
        y_log = np.log1p(y)
        
        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
        rf.fit(X_train, y_train_log)
        
        y_pred_log = rf.predict(X_test)
        r2 = r2_score(y_test_log, y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
        mae = mean_absolute_error(y_test_log, y_pred_log)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Coefficient R² (Test)", f"{r2:.3f}")
        col2.metric("RMSE (Test)", f"{rmse:.3f}")
        col3.metric("MAE (Test)", f"{mae:.3f}")
        
        st.divider()
        st.subheader("🏆 Importance des Variables")
        imp_df = pd.DataFrame({'Variable': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
        fig = px.bar(imp_df, x='Importance', y='Variable', orientation='h', color='Importance', color_continuous_scale='Teal')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pas assez d'échantillons dans cette zone.")

elif menu == "Évaluation & Résidus":
    st.markdown('<p class="main-title">📈 Évaluation des Modèles & Analyse des Résidus</p>', unsafe_allow_html=True)
    zone_eval = st.selectbox("Zone d'évaluation :", ["Socle (Madarounfa)", "Sédimentaire (Guidan Roumdji)"])
    df_eval = df_travail[df_travail['Zone_Hydro'] == zone_eval].copy()
    
    if len(df_eval) > 10:
        num_cols = ['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique']
        imputer = SimpleImputer(strategy='median')
        df_eval[num_cols] = imputer.fit_transform(df_eval[num_cols])
        
        X = df_eval[['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique']]
        y = df_eval['debit']
        y_log = np.log1p(y)
        
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
        rf.fit(X, y_log)
        y_pred = np.expm1(rf.predict(X))
        df_eval['residu'] = df_eval['debit'] - y_pred
        
        tab1, tab2 = st.tabs(["Carte des Résidus Spatiaux", "Courbes de Fréquence Cumulative (ECDF)"])
        with tab1:
            fig_res = px.scatter(
                df_eval, x="longitude", y="latitude", color="residu",
                color_continuous_scale="RdBu", hover_name="commune", hover_data=["debit"],
                labels={"longitude": "Longitude (°)", "latitude": "Latitude (°)"}
            )
            fig_res.update_layout(margin={"r":20,"t":30,"l":20,"b":20})
            st.plotly_chart(fig_res, use_container_width=True)
        with tab2:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.ecdfplot(data=df_clean, x='debit', hue='Zone_Hydro', palette={'Socle (Madarounfa)': 'teal', 'Sédimentaire (Guidan Roumdji)': 'darkorange'}, linewidth=2, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Données insuffisantes pour l'évaluation.")

elif menu == "Simulation / Prédiction en Direct":
    st.markdown('<p class="main-title">🔮 Simulation et Prédiction de Débit en Direct</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sim_prof_foree = st.slider("Profondeur Forée (m)", 10.0, 150.0, 45.0)
        sim_prof_equip = st.slider("Profondeur Équipée (m)", 5.0, 140.0, 40.0)
        sim_diametre = st.slider("Diamètre", 4.0, 12.0, 6.5)
    with col2:
        sim_niveau_stat = st.slider("Niveau Statique (m)", 1.0, 80.0, 15.0)
        sim_altitude = st.slider("Altitude (m)", 250.0, 500.0, 350.0)
        sim_zone = st.selectbox("Contexte Géologique", ["Socle (Madarounfa)", "Sédimentaire (Guidan Roumdji)"])

    if st.button("Lancer la Prédiction du Débit"):
        df_sim = df_clean[df_clean['Zone_Hydro'] == sim_zone].copy()
        if len(df_sim) > 5:
            num_cols = ['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique']
            imputer = SimpleImputer(strategy='median')
            df_sim[num_cols] = imputer.fit_transform(df_sim[num_cols])
            
            X = df_sim[['latitude', 'longitude', 'altitude', 'profondeur_foree', 'profondeur_equipee', 'diametre', 'niveau_statique']]
            y_log = np.log1p(df_sim['debit'])
            
            rf_sim = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf_sim.fit(X, y_log)
            
            lat_m = X['latitude'].mean()
            lon_m = X['longitude'].mean()
            ligne_test = pd.DataFrame([[lat_m, lon_m, sim_altitude, sim_prof_foree, sim_prof_equip, sim_diametre, sim_niveau_stat]], columns=X.columns)
            
            pred_log = rf_sim.predict(ligne_test)
            debit_estime = np.expm1(pred_log[0])
            st.success(f"🎯 **Débit estimé pour ce forage :** `{debit_estime:.2f} m³/h`")
        else:
            st.error("Données insuffisantes pour la simulation.")

elif menu == "À propos de l'auteur":
    st.markdown('<p class="main-title">👨🎓 À propos de l\'Auteur et du Projet</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Remplace 'ma_photo.jpg' par le nom exact de ton fichier photo dans le dossier si tu l'as ajouté
        try:
            st.image("ma_photo.jpg", width=160, caption="IRO ALKALI Moustapha")
        except:
            st.image("ALKALI.jpg", width=160, caption="IRO ALKALI Moustapha")
    with col2:
        st.markdown("""
        * **Nom & Prénom :** IRO ALKALI Moustapha  
        * **Date et lieu de naissance :** 30 Avril 1996 à Dan Issa (Madarounfa, Maradi, Niger)  
        * **Contact Email :** `moustphairoalkali12@gmail.com`  
        * **Téléphone :** +227 89 57 14 88 / 93 42 11 77 / 95 88 43 32  
        * **Langues :** Français, Haoussa (Lu, parlé, écrit)
        """)
    
    st.divider()
    st.subheader("🎓 Parcours Académique & Mémoire de Recherche")
    st.markdown("""
    * **Master en Terre, Eau et Environnement** — Université André Salifou de Zinder (Niger).  
    * **Sujet de Mémoire :** Évaluation des risques hydrogéologiques et géotechniques des forages et puits privés dans la zone de Maradi (cas du quartier Ali Dan Sofo).  
    * **Licence en Géosciences et Environnement** — Université André Salifou de Zinder.
    """)

    st.divider()
    st.subheader("💼 Expériences Professionnelles et Entrepreneuriat")
    st.markdown("""
    * **Directeur Général (PDG) :** Fondateur de l'entreprise *SAHEL INNOVATION ET DIVERSES TECHNOLOGIES (S.I.DT SARL)* basée à Maradi.
    * **Animateur Local ATPC :** Expérience auprès de l'ONG ADESP dans la commune rurale de Dan Issa (gestion des points d'eau et sensibilisation communautaire).
    """)

    st.subheader("🤝 Engagements Associatifs et Leadership")
    st.markdown("""
    * **Président :** Association des Jeunes pour la Cohésion Sociale et le Développement Local (**AJ-COSODEL**) à Dan Issa.
    """)

    st.divider()
    st.subheader("💻 Compétences Techniques & Projets Numériques")
    st.markdown("""
    * **Science des données & SIG :** Python, ArcGIS, QGIS, Mobile Topographer, FieldMove Clino.
    * **Développement d'applications :** Conception de l'application mobile *HydraData* (via WinDev Mobile) pour la collecte des données hydrogéologiques.
    * **Communication & Média :** Créateur de la chaîne éducative TikTok `@ChiffresQuiParlent` dédiée à la vulgarisation des données numériques.
    """)
