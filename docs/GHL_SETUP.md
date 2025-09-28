# Configuration de l'intégration GoHighLevel

## 1. Récupérer les identifiants nécessaires
- **Clé API** : dans GoHighLevel, ouvrez `Settings → Company → API Keys` puis générez un token `Agency API`.
- **Fuseau horaire & période** : choisissez le fuseau horaire principal (ex. `Africa/Abidjan`) et la profondeur historique (30 jours recommandés).
- **Pipelines et étapes** : pour chaque pipeline setters/closers, notez l'identifiant du pipeline et des étapes clés (offre envoyée, ready-to-pay, win). Les IDs sont visibles dans l'URL `.../pipelines/<PIPELINE_ID>/stages/<STAGE_ID>`.
- **Formulaires** : ouvrez chaque formulaire (`Sites → Forms`) et récupérez son `FORM_ID` dans l'URL (`.../forms/<FORM_ID>`).
- **Produits / Offres** : relevez les `productId` (ou `funnelId`) des offres (`Payments → Products`).
- **Utilisateurs** : notez l'ID de chaque setter/closer (`Settings → Team → Users`) afin d'afficher des noms lisibles.

## 2. Compléter `.streamlit/secrets.toml`
Ajoutez (ou adaptez) la section suivante **sans** la versionner :

```toml
[ghl]
api_key = "GHL_API_TOKEN"
base_url = "https://rest.gohighlevel.com"
timezone = "Africa/Abidjan"
history_days = 30
pipeline_ids = ["PIPELINE_ID"]
ready_to_pay_stage_ids = ["STAGE_ID_READY"]
custom_source_field_id = ""  # optionnel, identifiant d'un champ personnalisé
link_indicators = ["checkout", "pay", "stripe"]
direct_inbound_sources = ["direct", "inbound"]

[ghl.stage_ids]
offers_made = ["STAGE_ID_OFFER"]

[ghl.product_ids]
first = ["PRODUCT_ID_FIRST"]
upsell = ["PRODUCT_ID_UPSELL"]
downsell = ["PRODUCT_ID_DOWNSELL"]

[ghl.user_mapping]
"USER_ID_1" = "Setter 1"
"USER_ID_2" = "Setter 2"

[ghl.source_mapping]
facebook = "From_FB"
instagram = "From_Insta"
tiktok = "From_TikTok"
linkedin = "From_LinkedIn"
forms = "From_Forms"
direct = "From_Direct"
```
> Conservez les sections `gcp_service_account` et `google_sheet_id` si vous souhaitez garder le mode secours Google Sheets.

## 3. Lancer le dashboard
1. Installez les dépendances : `pip install -r requirements.txt` (ajout de `requests`).
2. Placez un fichier `.streamlit/secrets.toml` local avec vos identifiants GHL.
3. Démarrez l'application : `streamlit run app.py`.
4. Utilisez le bouton **Refresh Data** pour forcer un rafraîchissement GHL. En cas d'échec (erreur d'API, quota), l'application bascule automatiquement sur Google Sheets si la configuration est présente.

## 4. Ajustements utiles
- **Sources personnalisées** : si vos leads sont tracés via un champ personnalisé, renseignez `custom_source_field_id` et adaptez `source_mapping`.
- **Détection des liens** : ajustez `link_indicators` pour vos URLs de paiement (Stripe, PayPal, ThriveCart...).
- **Durée d'historique** : modifiez `history_days` pour charger plus ou moins de jours lors d'un refresh.
- **Vérification** : après la première synchronisation, comparez les totaux leads/closing/revenue avec les rapports GHL pour confirmer le mapping.
