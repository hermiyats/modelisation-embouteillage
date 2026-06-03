"""
Simulateur de trafic — modèle microscopique discret (suivi de véhicule).
=========================================================================

Route CIRCULAIRE à une ou plusieurs voies. Chaque voiture est un agent
(classe `Voiture`) ; la classe `Simulation` contient le code principal qui
résout le schéma numérique pas à pas et enregistre tout l'historique.

Modèle d'accélération (poursuite type Helly)
--------------------------------------------
Pour la voiture i, on note :
    - d         : écart pare-chocs à pare-chocs avec la voiture de devant
    - v_devant  : vitesse de la voiture de devant
    - v         : sa propre vitesse
    - d_arret   : écart résiduel à l'arrêt (`distance_securite` du profil)
    - T         : temps inter-véhiculaire (`temps_inter` du profil)
    - mu        : sensibilité à l'écart (profil)
    - lambda    : sensibilité à la vitesse d'approche (`sensibilite_vitesse`)

L'écart DÉSIRÉ croît avec la vitesse (règle des t secondes) :

    d_desiree(v) = d_arret + T * v

et l'accélération combine deux termes :

    a = mu * (d - d_desiree(v)) + lambda * (v_devant - v)   (bridée [a_min, a_max])

Le 1er terme vise le bon écart (croissant avec la vitesse) ; le 2e fait freiner
dès qu'on se RAPPROCHE vite d'un véhicule plus lent, même si l'écart est encore
grand — ce qui manquait à un modèle purement basé sur la distance.

Schéma numérique (Euler explicite)
----------------------------------
    v(t+dt) = clip( v(t) + a * dt , 0 , v_max )
    x(t+dt) = ( x(t) + v(t+dt) * dt )  modulo L

avec un garde-fou anticollision sur la vitesse pour empêcher tout
chevauchement dû à la discrétisation.

Changement de voie progressif
------------------------------
La décision de voie reste discrète (un entier `voie`), mais chaque voiture
porte aussi une position latérale CONTINUE `pos_laterale` qui rejoint la voie
cible à vitesse constante (paramètre `tps_changement_voie`). Toute la logique
(voisins, anticollision) travaille sur l'entier `voie` ; `pos_laterale` ne sert
qu'au rendu, pour que les changements de voie soient fluides et non instantanés.

Tout est purement algorithmique : aucun affichage ici. Les visualisations se
font à partir des historiques `X`, `V`, `A`, `VOIE` et `YLAT` (latéral continu).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# =========================================================================
#  Profils de conducteur
# =========================================================================
@dataclass
class ProfilConducteur:
    """Décrit un comportement de conduite.

    voie_max : numéro de la voie la plus à gauche que ce profil peut emprunter
               (None = pas de limite, donc jusqu'à la voie la plus à gauche).
               Le camion a voie_max = 2 : il ne peut pas utiliser la 3e voie.
    ignore_arriere_au_depassement : si vrai, le conducteur ne tient pas compte
               de la voiture qui arrive derrière sur la voie de gauche quand il
               déboîte pour doubler (conducteur « fou »).
    """

    nom: str
    mu: float                       # sensibilité de l'accélération
    distance_securite: float        # écart pare-chocs visé À L'ARRÊT (m, en bouchon)
    vitesse_max: float              # vitesse maximale (m/s)
    temps_inter: float = 1.4        # temps inter-véhiculaire (s) : marge ajoutée à
                                    #   l'écart visé, proportionnelle à la vitesse
    couleur: str = "tab:green"      # couleur pour l'animation
    voie_max: int | None = None     # voie la plus à gauche autorisée
    ignore_arriere_au_depassement: bool = False


# =========================================================================
#  Paramètres globaux (modifiables depuis le notebook)
# =========================================================================
@dataclass
class Parametres:
    # --- Route ---
    L: float = 1000.0          # longueur de la route circulaire (m)
    N: int = 75                # nombre de voitures
    N_voie: int = 3            # nombre de voies (1 = droite, ..., N_voie = gauche)

    # --- Temps ---
    dt: float = 0.01           # pas de temps de calcul (s)
    T: float = 120.0           # durée de la simulation (s)

    # --- Physique ---
    a_max: float = 3.0         # accélération maximale (m/s^2)
    a_min: float = -7.0        # freinage de CONFORT maximal du conducteur (m/s^2)
    a_urgence: float = -9.0    # freinage d'URGENCE maximal (limite physique pneus)
    v_min: float = 0.0         # vitesse minimale (m/s)
    longueur_voiture: float = 3.0   # longueur d'une voiture (m)
    distance_min: float = 1.0       # interstice minimal pare-chocs en bouchon (m)

    # --- Comportement ---
    temps_reaction: float = 0.8     # temps de réaction du conducteur (s) : il agit
                                    # sur l'état du trafic perçu il y a `temps_reaction`
                                    # secondes (0 = réaction instantanée). Rend les
                                    # rabattements serrés dangereux (freinage tardif).
    freinage_securite: float = 4.0  # décélération max (m/s^2) qu'on accepte d'imposer
                                    # à la voiture arrière en déboîtant (critère MOBIL)
    tps_min_changement_voie: float = 2.0  # temps minimal entre deux changements de
                                          # voie d'une même voiture (anti-papillonnage)

    # --- Valeurs de base (profil « normal ») dont dérivent les autres profils ---
    mu: float = 0.4                       # sensibilité à l'écart
    distance_securite: float = 2.0        # écart pare-chocs À L'ARRÊT / en bouchon (m)
    temps_inter_vehiculaire: float = 1.4  # « règle des t secondes » : écart visé =
                                          #   distance_securite + t * vitesse  (s)
    sensibilite_vitesse: float = 0.6      # lambda : réaction à la vitesse d'approche
    vitesse_max: float = 30.0

    # --- Population ---
    proportions: dict = field(default_factory=lambda: {
        "prudent": 0.20, "normal": 0.50, "fou": 0.20, "camion": 0.10})
    vitesse_initiale: float = 22.0   # vitesse de départ des voitures (m/s)
    nb_arret_initial: int = 3        # voitures à l'arrêt au départ (onde de bouchon)
    graine: int | None = None        # graine aléatoire (None = non reproductible)

    # --- Rendu / animation ---
    tps_changement_voie: float = 0.8  # durée d'un changement de voie (s) ; n'agit
                                      # que sur la position latérale de rendu, pas
                                      # sur la décision (qui reste instantanée)

    @property
    def nb_iterations(self) -> int:
        """Nombre de pas de temps de la simulation."""
        return int(self.T / self.dt)

    @property
    def distance_centre_min(self) -> float:
        """Distance centre à centre minimale (anticollision)."""
        return self.longueur_voiture + self.distance_min

    @property
    def n_pas_reaction(self) -> int:
        """Temps de réaction exprimé en nombre de pas de temps (>= 0)."""
        return max(0, round(self.temps_reaction / self.dt))


def construire_profils(p: Parametres) -> dict[str, ProfilConducteur]:
    """Construit les quatre profils à partir des valeurs de base de `p`.

    Les profils dérivent du profil « normal » : si l'on change `mu`,
    `distance_securite` ou `vitesse_max` dans les paramètres, tous les profils
    se décalent de façon cohérente. On crée aussi un écart de vitesse net entre
    les comportements (prudent < normal < fou, camion le plus lent).
    """
    base_t = p.temps_inter_vehiculaire
    return {
        "prudent": ProfilConducteur(
            nom="prudent",
            mu=p.mu - 0.15,
            distance_securite=p.distance_securite + 1.5,
            temps_inter=base_t + 0.5,              # garde une grande marge temporelle
            vitesse_max=p.vitesse_max - 5.0,
            couleur="tab:blue",
        ),
        "normal": ProfilConducteur(
            nom="normal",
            mu=p.mu,
            distance_securite=p.distance_securite,
            temps_inter=base_t,
            vitesse_max=p.vitesse_max,
            couleur="tab:green",
        ),
        "fou": ProfilConducteur(
            nom="fou",
            mu=p.mu + 0.30,
            distance_securite=max(0.5, p.distance_securite - 1.0),
            temps_inter=max(0.4, base_t - 0.7),    # colle au pare-chocs (tailgating)
            vitesse_max=p.vitesse_max + 4.0,
            couleur="tab:red",
            ignore_arriere_au_depassement=True,   # ignore la voiture arrière
        ),
        "camion": ProfilConducteur(
            nom="camion",
            mu=max(0.05, p.mu - 0.20),
            distance_securite=p.distance_securite + 2.0,
            temps_inter=base_t + 0.4,
            vitesse_max=p.vitesse_max - 10.0,
            couleur="black",
            voie_max=2,                            # interdit de 3e voie
        ),
    }


# =========================================================================
#  La voiture = un agent
# =========================================================================
class Voiture:
    """Un véhicule. Porte son état dynamique et son profil de conducteur."""

    def __init__(self, identifiant: int, x: float, v: float,
                 voie: int, profil: ProfilConducteur):
        self.id = identifiant
        self.x = float(x)        # position le long de la route (m, modulo L)
        self.v = float(v)        # vitesse (m/s)
        self.a = 0.0             # accélération courante (m/s^2)
        self.voie = int(voie)    # 1 = voie de droite, ..., N_voie = voie de gauche
        # Position latérale CONTINUE (en « numéro de voie » fractionnaire) qui
        # rejoint progressivement `voie`. Sert uniquement au rendu fluide.
        self.pos_laterale = float(voie)
        self.profil = profil

    # Raccourcis pratiques vers les paramètres du profil
    @property
    def mu(self) -> float:
        return self.profil.mu

    @property
    def distance_securite(self) -> float:
        return self.profil.distance_securite

    @property
    def vitesse_max(self) -> float:
        return self.profil.vitesse_max

    @property
    def temps_inter(self) -> float:
        return self.profil.temps_inter

    def distance_desiree(self, v: float | None = None) -> float:
        """Écart pare-chocs souhaité à la vitesse `v` (défaut : vitesse actuelle).

        Modèle du « temps inter-véhiculaire » (règle des t secondes) :

            d_desiree = distance_securite + temps_inter * v

        L'écart visé CROÎT avec la vitesse (réaliste) au lieu d'être constant ;
        `distance_securite` est l'écart résiduel à l'arrêt (bouchon).
        """
        v = self.v if v is None else v
        return self.distance_securite + self.temps_inter * max(0.0, v)

    def acceleration_souhaitee(self, ecart: float, v_devant: float,
                               a_min: float, a_max: float,
                               sensibilite_vitesse: float) -> float:
        """Modèle de poursuite type Helly (linéaire, deux termes) :

            a = mu * (ecart - d_desiree(v)) + lambda * (v_devant - v)

        - 1er terme : on vise l'écart désiré, qui croît avec la vitesse ;
        - 2e terme  : on réagit à la VITESSE D'APPROCHE (lambda = sensibilité).
          C'est lui qui fait freiner quand on se rapproche vite d'un véhicule
          plus lent, MÊME si l'écart est encore grand (réaliste).
        Résultat bridé dans [a_min, a_max].
        """
        a = (self.mu * (ecart - self.distance_desiree())
             + sensibilite_vitesse * (v_devant - self.v))
        return max(a_min, min(a_max, a))

    def __repr__(self) -> str:
        return (f"Voiture(id={self.id}, x={self.x:.1f}, v={self.v:.1f}, "
                f"voie={self.voie}, profil={self.profil.nom})")


# =========================================================================
#  La simulation = le code principal qui résout le schéma numérique
# =========================================================================
class Simulation:
    """Orchestre les voitures sur la route et avance le schéma numérique."""

    def __init__(self, parametres: Parametres | None = None,
                 profils: dict[str, ProfilConducteur] | None = None):
        self.p = parametres or Parametres()
        self.profils = profils or construire_profils(self.p)
        self._rng = np.random.default_rng(self.p.graine)

        self.voitures: list[Voiture] = []
        self._initialiser_voitures()

        # Métadonnées utiles pour l'analyse / l'animation
        self.profils_voitures = [v.profil.nom for v in self.voitures]
        self.couleurs = [v.profil.couleur for v in self.voitures]

        # Mémoire de perception pour le temps de réaction : on garde les
        # `n_pas_reaction` derniers états (positions, vitesses, voies). Le
        # conducteur décide son accélération d'après l'état le plus ancien
        # encore en mémoire (donc perçu il y a `temps_reaction` secondes).
        self._perception: deque = deque(maxlen=self.p.n_pas_reaction + 1)

        # Comptage des collisions (freinage d'urgence insuffisant). `_en_collision`
        # évite de recompter chaque pas un même choc (front montant uniquement).
        self.nb_collisions = 0
        self._en_collision = [False] * self.p.N
        self._cooldown_voie = np.zeros(self.p.N, dtype=int)   # anti-papillonnage

        # Historiques (remplis par simuler())
        self.X = self.V = self.A = self.VOIE = self.YLAT = self.temps = None

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------
    def _initialiser_voitures(self) -> None:
        """Voitures réparties uniformément sur la voie de droite, quelques-unes
        à l'arrêt au départ pour déclencher une onde de bouchon."""
        p = self.p
        positions = np.linspace(0, p.L, p.N, endpoint=False)

        noms = list(p.proportions.keys())
        probas = np.array([p.proportions[n] for n in noms], dtype=float)
        probas = probas / probas.sum()
        tirage = self._rng.choice(noms, size=p.N, p=probas)

        for i in range(p.N):
            profil = self.profils[tirage[i]]
            v0 = 0.0 if i < p.nb_arret_initial else p.vitesse_initiale
            v0 = min(v0, profil.vitesse_max)      # pas plus vite que sa vitesse max
            # Voitures réparties sur TOUTES les voies (round-robin), en respectant
            # la voie max du profil. Cela part d'un état proche de l'équilibre
            # plutôt que toutes entassées sur une voie. Les voitures arrêtées du
            # départ restent sur la voie de droite pour créer un bouchon net.
            if i < p.nb_arret_initial:
                voie0 = 1
            else:
                voie0 = (i % p.N_voie) + 1
                if profil.voie_max is not None:
                    voie0 = min(voie0, profil.voie_max)
            self.voitures.append(Voiture(i, positions[i], v0, voie=voie0, profil=profil))

    # ------------------------------------------------------------------
    #  Recherche de voisins sur l'anneau
    # ------------------------------------------------------------------
    def _voie_max(self, voiture: Voiture) -> int:
        """Voie la plus à gauche réellement accessible à cette voiture."""
        vm = voiture.profil.voie_max
        return self.p.N_voie if vm is None else min(self.p.N_voie, vm)

    def voiture_devant(self, i, voie_visee, positions, voies):
        """Voiture la plus proche DEVANT i sur `voie_visee`.

        Retourne (indice ou None, distance centre à centre). Si aucune voiture
        n'est trouvée, retourne (None, L) — route libre. Version vectorisée
        (numpy) : équivalente à une boucle `0 < d < meilleure` mais plus rapide.
        """
        L = self.p.L
        pos = np.asarray(positions, dtype=float)
        voi = np.asarray(voies)
        d = (pos - pos[i]) % L                    # distance vers l'avant (cyclique)
        cand = (voi == voie_visee) & (d > 0)      # même voie, devant (exclut i : d[i]=0)
        if not cand.any():
            return None, L
        d_cand = np.where(cand, d, np.inf)
        k = int(np.argmin(d_cand))                # plus proche ; ex æquo -> plus petit indice
        return k, float(d_cand[k])

    def voiture_derriere(self, i, voie_visee, positions, voies):
        """Voiture la plus proche DERRIÈRE i sur `voie_visee` (vectorisée)."""
        L = self.p.L
        pos = np.asarray(positions, dtype=float)
        voi = np.asarray(voies)
        d = (pos[i] - pos) % L                    # distance vers l'arrière (cyclique)
        cand = (voi == voie_visee) & (d > 0)
        if not cand.any():
            return None, L
        d_cand = np.where(cand, d, np.inf)
        k = int(np.argmin(d_cand))
        return k, float(d_cand[k])

    # ------------------------------------------------------------------
    #  Décision de changement de voie (règles européennes)
    # ------------------------------------------------------------------
    def _decel_requise(self, v_arriere, v_avant, ecart, tau) -> float:
        """Décélération que la voiture arrière devra fournir pour ne pas percuter
        celle de devant, EN TENANT COMPTE de son temps de réaction `tau`.

        Pendant `tau` la voiture arrière ne réagit pas : l'écart se referme de
        `approche * tau`. Sur l'écart restant, il faut résorber la vitesse
        d'approche : decel = approche^2 / (2 * écart_restant). Si l'écart est
        déjà consommé avant même de réagir, c'est un choc certain (+inf).
        """
        approche = v_arriere - v_avant
        if approche <= 0.0:
            return 0.0                                    # ne rattrape pas
        ecart_restant = ecart - approche * tau
        if ecart_restant <= 0.1:
            return float("inf")                           # rattrapé avant de réagir
        return approche * approche / (2.0 * ecart_restant)

    def _creneau_libre(self, i, voie_cible, positions, voies,
                       seuil_avant, seuil_arriere) -> bool:
        """Vrai si la voiture i peut s'insérer sur `voie_cible` sans forcer
        un freinage trop fort, NI sur elle-même (vs la voiture de devant), NI
        sur la voiture de derrière (critère de type MOBIL, dépendant des
        vitesses ET du temps de réaction). Les seuils tolérés diffèrent : un
        conducteur « fou » accepte des freinages d'urgence (seuil élevé).
        """
        p = self.p
        moi = self.voitures[i]
        tau = p.temps_reaction
        lng = p.longueur_voiture

        f, d_centre_f = self.voiture_devant(i, voie_cible, positions, voies)
        if f is not None:
            ecart_f = d_centre_f - lng
            if ecart_f <= p.distance_min:
                return False
            if self._decel_requise(moi.v, self.voitures[f].v, ecart_f, tau) > seuil_avant:
                return False

        r, d_centre_r = self.voiture_derriere(i, voie_cible, positions, voies)
        if r is not None:
            ecart_r = d_centre_r - lng
            if ecart_r <= p.distance_min:
                return False
            if self._decel_requise(self.voitures[r].v, moi.v, ecart_r, tau) > seuil_arriere:
                return False
        return True

    def _decider_voie(self, i, positions, voies) -> int:
        """Choisit la voie de la voiture i au pas suivant (règles européennes).

        1) On se rabat le plus à droite possible dès qu'il y a un créneau sûr
           (avant ET arrière), avec une petite marge supplémentaire devant
           (hystérésis anti-oscillation). Le rabattement reste « de confort ».
        2) Sinon, on double UNIQUEMENT par la gauche si l'on est gêné (écart
           avant < écart désiré) ET que le créneau de gauche est sûr.

        La sûreté d'un créneau (`_creneau_libre`) est une décélération imposée
        bornée, qui dépend des VITESSES et du TEMPS DE RÉACTION — pas seulement
        d'une distance fixe. Le conducteur « fou » tolère des freinages
        d'urgence (il coupe la route), le conducteur normal reste sur du confort.
        """
        p = self.p
        voiture = self.voitures[i]
        voie_actuelle = voies[i]
        lng = p.longueur_voiture
        d_des = voiture.distance_desiree()                # écart désiré à sa vitesse
        confort = p.freinage_securite
        # Le « fou » accepte d'imposer/subir un freinage d'urgence en déboîtant.
        agressif = (-p.a_urgence) if voiture.profil.ignore_arriere_au_depassement \
            else confort

        # 1) Se rabattre à droite (toujours en confort)
        if voie_actuelle > 1:
            for voie_droite in range(1, voie_actuelle):
                _, d_dev = self.voiture_devant(i, voie_droite, positions, voies)
                if (d_dev - lng) > 1.2 * d_des and \
                   self._creneau_libre(i, voie_droite, positions, voies, confort, confort):
                    return voie_droite

        # 2) Doubler par la gauche (si gêné devant)
        voie_gauche = voie_actuelle + 1
        if voie_gauche <= self._voie_max(voiture):
            _, d_dev_actuel = self.voiture_devant(i, voie_actuelle, positions, voies)
            if (d_dev_actuel - lng) < d_des and \
               self._creneau_libre(i, voie_gauche, positions, voies, agressif, agressif):
                return voie_gauche

        return voie_actuelle

    # ------------------------------------------------------------------
    #  Un pas de temps du schéma numérique
    # ------------------------------------------------------------------
    def etape(self) -> None:
        p = self.p
        N = p.N
        positions = np.array([v.x for v in self.voitures])
        vitesses = np.array([v.v for v in self.voitures])
        voies = [v.voie for v in self.voitures]

        # --- 1) Changements de voie (décidés sur l'état au temps t) ---
        # Mise à jour incrémentale : une voiture déjà déplacée est vue par les
        # suivantes, ce qui limite les conflits de déboîtement simultané. Un
        # TEMPS MINIMAL entre deux changements (cooldown) empêche le papillonnage
        # irréaliste (une voiture ne change pas de voie à chaque pas de temps).
        n_cooldown = max(1, int(p.tps_min_changement_voie / p.dt))
        voies_t1 = np.array(voies, dtype=int)
        for i in range(N):
            if self._cooldown_voie[i] > 0:
                self._cooldown_voie[i] -= 1
                continue
            nouvelle = self._decider_voie(i, positions, voies_t1)
            if nouvelle != voies_t1[i]:
                voies_t1[i] = nouvelle
                self._cooldown_voie[i] = n_cooldown
        for i in range(N):
            self.voitures[i].voie = int(voies_t1[i])

        # --- 1bis) Glissement latéral continu vers la voie cible ---
        # La voie (entier) est déjà fixée ci-dessus et pilote toute la logique.
        # `pos_laterale` la rejoint à vitesse constante : un changement d'une
        # voie prend `tps_changement_voie` secondes, ce qui rend la manœuvre
        # progressive à l'écran au lieu d'un saut instantané.
        if p.tps_changement_voie > 0:
            pas_lateral = p.dt / p.tps_changement_voie      # voies par pas de temps
            for voit in self.voitures:
                ecart = float(voit.voie) - voit.pos_laterale
                if abs(ecart) <= pas_lateral:
                    voit.pos_laterale = float(voit.voie)
                else:
                    voit.pos_laterale += math.copysign(pas_lateral, ecart)
        else:
            for voit in self.voitures:                      # changement instantané
                voit.pos_laterale = float(voit.voie)

        # --- 1ter) Mémoriser l'état courant pour la perception retardée ---
        # On enregistre l'état (positions, vitesses, voies) APRÈS les changements
        # de voie : c'est ce que les autres conducteurs « verront », mais avec un
        # retard `temps_reaction`. Le plus ancien élément du tampon date donc de
        # `temps_reaction` secondes (ou moins pendant l'amorçage initial).
        self._perception.append((positions, vitesses, voies_t1))
        pos_perc, vit_perc, voies_perc = self._perception[0]

        # --- 2) Accélération (sur l'état PERÇU) puis vitesse provisoire ---
        # Le conducteur réagit à ce qu'il a perçu il y a `temps_reaction` s : si
        # un « fou » vient de se rabattre devant lui, il ne le « voit » pas encore
        # et freine donc en retard, ce qui rend les rabattements serrés dangereux.
        vitesses_prov = np.zeros(N)
        leaders = [None] * N
        d_centre_leaders = [p.L] * N
        for i in range(N):
            voit = self.voitures[i]
            # Leader PERÇU (état retardé) -> accélération souhaitée
            jp, d_centre_p = self.voiture_devant(i, voies_perc[i], pos_perc, voies_perc)
            ecart = d_centre_p - p.longueur_voiture
            v_devant = vit_perc[jp] if jp is not None else voit.v
            voit.a = voit.acceleration_souhaitee(ecart, v_devant, p.a_min, p.a_max,
                                                 p.sensibilite_vitesse)
            v_new = voit.v + voit.a * p.dt
            vitesses_prov[i] = min(max(v_new, p.v_min), voit.vitesse_max)
            # Leader RÉEL (état courant) -> mémorisé pour le garde-fou anticollision
            jc, d_centre_c = self.voiture_devant(i, voies_t1[i], positions, voies_t1)
            leaders[i] = jc
            d_centre_leaders[i] = d_centre_c

        # --- 3) Garde-fou anticollision + freinage borné physiquement ---
        # Vitesse SÛRE = vitesse maximale depuis laquelle on peut encore s'arrêter
        # dans l'espace disponible en freinant à |a_urgence| (cinématique) :
        #     v_sure = v_leader + sqrt(2 * b * dispo)
        # Tant que le conducteur reste sous v_sure, aucune intervention (il a la
        # place de freiner progressivement). Sinon on freine au MAXIMUM PHYSIQUE
        # (a_urgence) ; si même cela ne suffit pas, c'est une COLLISION (réaction
        # trop tardive ou créneau coupé) : on la compte une fois par épisode et on
        # borne la vitesse pour éviter le chevauchement réel (la simu continue).
        b = -p.a_urgence
        for i in range(N):
            voit = self.voitures[i]
            j = leaders[i]
            if j is None:
                self._en_collision[i] = False
                if vitesses_prov[i] < 0:
                    vitesses_prov[i] = 0.0
                continue
            dispo = d_centre_leaders[i] - p.distance_centre_min
            v_sure = vitesses_prov[j] + (math.sqrt(2.0 * b * dispo) if dispo > 0 else 0.0)
            if vitesses_prov[i] > v_sure:
                v_plancher = voit.v + p.a_urgence * p.dt        # freinage physique max
                vitesses_prov[i] = max(v_sure, v_plancher)
                if v_sure < v_plancher:                         # choc inévitable
                    if not self._en_collision[i]:
                        self.nb_collisions += 1
                        self._en_collision[i] = True
                    v_sans_chevauchement = vitesses_prov[j] + dispo / p.dt
                    vitesses_prov[i] = min(vitesses_prov[i], v_sans_chevauchement)
                else:
                    self._en_collision[i] = False
            else:
                self._en_collision[i] = False
            if vitesses_prov[i] < 0:
                vitesses_prov[i] = 0.0

        # --- 4) Mise à jour des vitesses et des positions ---
        for i in range(N):
            voit = self.voitures[i]
            voit.v = vitesses_prov[i]
            voit.x = (voit.x + voit.v * p.dt) % p.L

    # ------------------------------------------------------------------
    #  Boucle de simulation complète
    # ------------------------------------------------------------------
    def simuler(self, verbeux: bool = False) -> "Simulation":
        """Lance la simulation et enregistre l'historique complet.

        Historiques de forme (nb_iterations, N) :
            X[n, i]    position de la voiture i au pas n
            V[n, i]    vitesse
            A[n, i]    accélération appliquée pendant le pas n -> n+1
            VOIE[n, i] voie cible (entier)
            YLAT[n, i] position latérale continue (float, pour le rendu fluide)
        """
        p = self.p
        nb = p.nb_iterations
        self.nb_collisions = 0
        self._en_collision = [False] * p.N
        self._cooldown_voie = np.zeros(p.N, dtype=int)
        X = np.empty((nb, p.N))
        V = np.empty((nb, p.N))
        A = np.empty((nb, p.N))
        VOIE = np.empty((nb, p.N), dtype=int)
        YLAT = np.empty((nb, p.N))

        for n in range(nb):
            for i, voit in enumerate(self.voitures):
                X[n, i] = voit.x
                V[n, i] = voit.v
                VOIE[n, i] = voit.voie
                YLAT[n, i] = voit.pos_laterale
            self.etape()
            for i, voit in enumerate(self.voitures):
                A[n, i] = voit.a
            if verbeux and nb >= 10 and n % (nb // 10) == 0:
                print(f"  pas {n:>6}/{nb}  (t = {n * p.dt:6.1f} s)")

        self.X, self.V, self.A, self.VOIE, self.YLAT = X, V, A, VOIE, YLAT
        self.temps = np.arange(nb) * p.dt
        return self


# =========================================================================
#  Démonstration en ligne de commande
# =========================================================================
if __name__ == "__main__":
    import collections

    params = Parametres(T=10.0, graine=42)     # courte démo reproductible
    sim = Simulation(params).simuler(verbeux=True)

    print("\nFormes des historiques :", sim.X.shape, sim.VOIE.shape)
    print("Vitesse moyenne finale :", round(float(sim.V[-1].mean()), 2), "m/s")
    print("Collisions détectées   :", sim.nb_collisions)
    print("Profils tirés          :", dict(collections.Counter(sim.profils_voitures)))
    print("Répartition par voie   :", dict(collections.Counter(sim.VOIE[-1].tolist())))
