# modelisation-embouteillage

INSA Lyon P2I - 7 Sujet 26 - Référent : Michel Perez

Simulateur de trafic routier par une approche **discrète** (microscopique) :
chaque voiture est un agent, sur une route **circulaire** à une ou plusieurs voies.

## Fichiers

| Fichier | Rôle |
|---|---|
| `simulateur_trafic.py` | Le moteur : classes (`Parametres`, `ProfilConducteur`, `Voiture`, `Simulation`) et le schéma numérique. Aucun affichage. |
| `simulation.ipynb` | Notebook : définit les paramètres, lance la simulation, trace les courbes et l'animation. |

## Utilisation

```bash
# Démo rapide en ligne de commande
python3 simulateur_trafic.py
```

Sinon, ouvrir `simulation.ipynb` (dans le même dossier que le `.py`), ajuster la
cellule **Paramètres**, puis exécuter les cellules.

## Modèle

Pour la voiture `i` suivant la voiture `j` (écart pare-chocs `d`, pas de temps `dt`) :

```
ecart_prevu = d + (v_j - v_i) * dt
a_i = mu_i * (ecart_prevu - d_sec_i)        # bridée dans [a_min, a_max]
```

Intégration par schéma d'**Euler explicite**, avec un garde-fou anticollision :

```
v_i <- clip(v_i + a_i * dt, 0, v_max_i)
x_i <- (x_i + v_i * dt) mod L
```

Les trois coefficients réglables par conducteur :
- `mu` — sensibilité de l'accélération (force de la réaction à l'écart) ;
- `d_sec` — distance de sécurité visée ;
- `v_max` — vitesse maximale.

## Profils de conducteur

| Profil | Particularité |
|---|---|
| `prudent` | grande distance de sécurité, vitesse plus faible |
| `normal` | valeurs de référence |
| `fou` | rapide, agressif ; **ignore la voiture arrière** quand il déboîte pour doubler |
| `camion` | lent ; **interdit de 3e voie** |

## Règles de circulation (européennes)

- Dépassement par la **gauche** uniquement, et seulement si on se rapproche
  trop de la voiture de devant et que la voie de gauche est libre (devant et,
  sauf pour le `fou`, derrière).
- Rabattement à **droite** dès qu'il y a la place.

## Historique

`Simulation.simuler()` enregistre tout l'historique dans des tableaux de forme
`(nb_pas, N)` : `sim.X` (positions), `sim.V` (vitesses), `sim.A` (accélérations),
`sim.VOIE` (voies), `sim.temps`. De quoi rejouer ou animer la simulation.
