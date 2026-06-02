# modelisation-embouteillage

INSA Lyon P2I - 7 Sujet 26 - Référent : Michel Perez

Simulateur de trafic routier par une approche **discrète** (microscopique) :
chaque voiture est un agent, sur une route **circulaire** à une ou plusieurs voies.

## Fichiers

| Fichier | Rôle |
|---|---|
| `simulateur_trafic.py` | Le moteur : classes (`Parametres`, `ProfilConducteur`, `Voiture`, `Simulation`) et le schéma numérique. Aucun affichage. |
| `animation_pygame.py` | Animation **temps réel, fluide et interactive** dans une fenêtre pygame. |
| `simulation.ipynb` | Notebook : définit les paramètres, lance la simulation, trace les courbes et l'animation (matplotlib en ligne + lancement de la fenêtre pygame). |

## Utilisation

```bash
# Démo rapide en ligne de commande
python3 simulateur_trafic.py

# Animation pygame interactive (installer pygame : pip install pygame)
python3 animation_pygame.py
python3 animation_pygame.py --duree 60 --voitures 90 --voies 3 --graine 1

# Voitures trop petites / trop grosses ? Régler leur taille avec --zoom :
python3 animation_pygame.py --zoom 2.5      # plus grosses (se touchent dans les bouchons)
python3 animation_pygame.py --zoom 1.3      # plus petites, jamais de chevauchement
```

La fenêtre s'ajuste automatiquement à l'écran (`--taille 0` par défaut ; passer
une valeur en pixels pour forcer). Les voitures sont dessinées à `--zoom` × leur
longueur réelle (2 par défaut) : assez grosses pour être suivies à l'œil, tout en
ne se chevauchant que dans les bouchons les plus serrés (pare-chocs contre
pare-chocs, ce qui est réaliste).

Sinon, ouvrir `simulation.ipynb` (dans le même dossier que les `.py`), ajuster la
cellule **Paramètres**, puis exécuter les cellules. La section 5 affiche une
animation matplotlib **dans** le notebook ; la section 6 lance la fenêtre pygame.

## Animation

Deux choses rendent le mouvement agréable à suivre :

- **Changements de voie progressifs.** La décision de voie reste discrète, mais
  chaque voiture porte une position latérale *continue* (`sim.YLAT`) qui rejoint
  la voie cible en `tps_changement_voie` secondes (0,8 s par défaut). La voiture
  *glisse* d'une voie à l'autre au lieu de sauter. C'est réglable dans
  `Parametres(tps_changement_voie=...)`.
- **Interpolation temps réel** (pygame). La lecture est pilotée par l'horloge et
  les positions sont interpolées entre les pas enregistrés : le rendu reste
  fluide quels que soient `dt` et la cadence.

Commandes de la fenêtre pygame :

| Touche | Action |
|---|---|
| `Espace` | pause / lecture |
| `↑` / `↓` | accélérer / ralentir la lecture |
| `←` / `→` | reculer / avancer de 2 s |
| `R` | revenir au début |
| `Échap` / `Q` | quitter |

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

## Temps de réaction

Le conducteur ne calcule pas son accélération sur l'état **instantané** du
trafic, mais sur l'état **perçu il y a `temps_reaction` secondes** (0,8 s par
défaut, réglable ; `0` = réaction instantanée). Concrètement, le freinage réagit
à la position et à la vitesse du véhicule de devant telles qu'elles étaient
`temps_reaction` plus tôt.

Conséquence : quand un `fou` se rabat brusquement devant une voiture, celle-ci ne
le « voit » pas tout de suite et **freine en retard** — collage au pare-chocs
pendant ~`temps_reaction`, puis freinage brutal. Cela crée des ondes de bouchon
beaucoup plus marquées (stop-and-go). Le garde-fou anticollision reste, lui, basé
sur l'état réel : il représente le **freinage d'urgence** et empêche tout
chevauchement même en cas de réaction trop tardive.

```python
Parametres(temps_reaction=0.8)   # 0.8 s ; mettre 0.0 pour l'ancien comportement
```

```bash
python3 animation_pygame.py --treaction 1.2   # conducteurs plus lents à réagir
python3 animation_pygame.py --treaction 0     # réaction instantanée
```

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
`sim.VOIE` (voie cible, entier), `sim.YLAT` (position latérale continue, pour le
rendu fluide) et `sim.temps`. De quoi rejouer ou animer la simulation.
