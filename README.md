# KodjoEnglish Sales Dashboard

Application Streamlit pour suivre les métriques de ventes (leads, offres, liens envoyés, closes, revenus) à partir de l'API GoHighLevel (GHL). Fallbacks: Google Sheets (optionnel) puis JSON locaux pour le debug.

## Démarrage rapide
- Prérequis: Python 3.11+ (3.12 OK)
- Installation: `pip install -r requirements.txt`
- Configuration locale: créez `./.streamlit/secrets.toml` (non versionné) à partir de l'exemple ci‑dessous.
- Lancement: `streamlit run app.py`

## Configuration `.streamlit/secrets.toml`
Exemple minimal à adapter:

```toml
[ghl]
api_key = "VOTRE_JETON_GHL"
base_url = "https://rest.gohighlevel.com"
timezone = "Africa/Porto-Novo"  # ou votre TZ
history_days = 30                 # période historique par défaut

# Optionnels mais recommandés
location_id = "VOTRE_LOCATION_ID"      # sinon déduit du JWT si possible
pipeline_ids = ["PIPELINE_ID_1"]
ready_to_pay_stage_ids = ["STAGE_ID_READY"]

[ghl.stage_ids]
offers_made = ["STAGE_ID_OFFER"]

[ghl.product_ids]
first = ["PRODUCT_ID_FIRST"]
upsell = ["PRODUCT_ID_UPSELL"]
downsell = ["PRODUCT_ID_DOWNSELL"]

[ghl.user_mapping]
"USER_ID" = "Nom lisible"

[ghl]
# Optionnels pour le filtrage par closers et les sélections par défaut
closers = ["USER_ID_CLOSER_1", "Bohdan"]     # IDs (mappés via user_mapping) ou noms déjà mappés
default_users = ["Bohdan"]                     # pré‑sélection dans le filtre des Setters

[ghl.source_mapping]
facebook = "From_FB"
instagram = "From_Insta"
tiktok = "From_TikTok"
linkedin = "From_LinkedIn"
forms = "From_Forms"
direct = "From_Direct"

# (Optionnel) Chargement GitHub de JSON (mode debug/sans API)
[github]
# Soit un préfixe raw commun (recommandé)
# Exemple: https://raw.githubusercontent.com/<user>/<repo>/<branch>/data
raw_base = ""
# Ou des URLs explicites (prioritaires si définies)
# contacts_url = "https://raw.githubusercontent.com/.../contacts.json"
# opportunities_url = "https://raw.githubusercontent.com/.../opportunities.json"
# forms_url = "https://raw.githubusercontent.com/.../forms.json"
# messages_url = "https://raw.githubusercontent.com/.../messages.json"
# payments_url = "https://raw.githubusercontent.com/.../payments.json"
# calls_url = "https://raw.githubusercontent.com/.../calls.json"
```

- Google Sheets (facultatif): ajoutez `gcp_service_account` et `google_sheet_id` si vous souhaitez un mode secours.

## Utilisation
- Panneau latéral: coche "Mode rapide (30 jours)" pour limiter les appels API à 30 jours récents. Décochez pour utiliser `history_days` (30 par défaut) ou les dates du filtre.
- Bouton "Actualiser les données" vide le cache (15 min) et relance un chargement.
- "Diagnostics" affiche source, période, en‑têtes, compteurs et erreurs API récentes.
- Source des données: "Auto", "GHL API", "Google Sheets", "GitHub JSON", "Local JSON". Le mode "GitHub JSON" lit des fichiers JSON depuis des URLs raw GitHub définies dans `[github]`.
- Filtre closers: si `ghl.closers` est défini, cochez "Limiter aux closers" pour ne montrer que ces utilisateurs. Utilisez `ghl.default_users` pour pré‑sélectionner (ex: ["Bohdan"]).

## Sécurité
- Ne commitez jamais `./.streamlit/secrets.toml` ni `./.streamlit/Info` (protégés via `.gitignore`).
- Faites tourner/rotater tout secret ayant déjà figuré dans l'historique.
- Évitez de versionner des `*.json` contenant des PII.

## Dépannage
- Message "Configuration GHL absente ou incomplète" → renseignez `ghl.api_key` (et idéalement `location_id`) dans `secrets.toml`.
- Beaucoup de 429 → augmentez `request_pause`, réduisez `page_size`, ou limitez la période.
- Revenus à zéro → vérifiez la colonne `amount`/`total` et le statut de paiement.

## Licence
Usage interne KodjoEnglish.

## Nouveautés UI
- Sélecteur de source (Auto, GHL, Google Sheets, GitHub JSON, Local JSON) dans la barre latérale
- Bouton d’actualisation qui vide les caches de chargement
- Onglets: Vue d’ensemble, Sources, Setters, Pipeline, Diagnostics
- Nouveaux graphiques: donut de sources, activité quotidienne, performance par setter, heatmap d’activité (jour × heure)
- Mode sombre: activez "Mode sombre" dans la barre latérale (applique un thème foncé et un template Plotly sombre).

## Branding
- Logo affiché dans l’en‑tête et utilisé comme icône de page via `st.set_page_config(page_icon="kodjoenglish_logo.jpeg")`.
 - Logo également visible dans la barre latérale.
