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

[ghl.source_mapping]
facebook = "From_FB"
instagram = "From_Insta"
tiktok = "From_TikTok"
linkedin = "From_LinkedIn"
forms = "From_Forms"
direct = "From_Direct"
```

- Google Sheets (facultatif): ajoutez `gcp_service_account` et `google_sheet_id` si vous souhaitez un mode secours.

## Utilisation
- Panneau latéral: coche "Mode rapide (30 jours)" pour limiter les appels API à 30 jours récents. Décochez pour utiliser `history_days` (30 par défaut) ou les dates du filtre.
- Bouton "Actualiser les données" vide le cache (15 min) et relance un chargement.
- "Diagnostics" affiche source, période, en‑têtes, compteurs et erreurs API récentes.

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
