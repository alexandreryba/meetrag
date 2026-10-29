---
title: Revue d'architecture de la plateforme data
date: 2026-04-21
type: review
language: fr
participants:
  - Marc Lefèvre | CIO | Voltaria
  - Thomas Girard | Data Platform Lead | Voltaria
  - Léa Moreau | Data Engineer | Consulting team
  - Hugo Lambert | Engagement Manager | Consulting team
---
[00:00:05] Hugo Lambert: Bonjour Marc, bonjour Thomas. Léa va présenter l'architecture cible de la plateforme data, et on a besoin de ta validation, Marc.
[00:00:18] Léa Moreau: Alors, on propose une architecture en médaillon sur BigQuery : une couche bronze avec les données brutes, une couche silver nettoyée et dédoublonnée, et une couche gold avec les tables prêtes pour le modèle et les tableaux de bord.
[00:00:42] Léa Moreau: Les transformations sont écrites en dbt. Pour l'orchestration on avait regardé Cloud Composer, mais c'est surdimensionné et trop cher pour notre volume.
[00:00:58] Thomas Girard: Oui, Composer c'est plusieurs centaines d'euros par mois rien que pour l'environnement.
[00:01:06] Léa Moreau: Donc on propose de faire tourner dbt Core dans des jobs Cloud Run, déclenchés par Cloud Scheduler. C'est simple et ça coûte presque rien.
[00:01:20] Marc Lefèvre: Et les données personnelles ? Notre politique de sécurité classe les données clients en Restricted.
[00:01:31] Léa Moreau: Justement, on pseudonymise toutes les données Restricted dès la couche silver : les identifiants clients sont remplacés par des clés hachées, et les noms, emails et téléphones ne passent jamais en gold.
[00:01:50] Marc Lefèvre: Ça me va sur le principe. Mais euh je dois faire valider l'approche de pseudonymisation par le RSSI. Je m'en occupe d'ici le 30 avril.
[00:02:05] Thomas Girard: Et le streaming des données compteurs ?
[00:02:10] Hugo Lambert: On propose de le reporter en phase 2. Pour le modèle, le mensuel suffit, on l'a décidé en atelier.
[00:02:20] Marc Lefèvre: D'accord. Donc je valide l'architecture médaillon sur BigQuery avec dbt Core sur Cloud Run, sous réserve de l'avis du RSSI sur la pseudonymisation. Le streaming compteur est reporté en phase 2.
[00:02:41] Léa Moreau: Je monte le projet dbt avec sa CI d'ici le 5 mai.
[00:02:48] Hugo Lambert: Le risque principal, c'est un délai sur l'avis du RSSI, qui bloquerait la mise en production. Merci à vous deux.
