"""
Simulateur de trafic — modèle microscopique discret (suivi de véhicule).
=========================================================================

Route CIRCULAIRE à une ou plusieurs voies. Chaque voiture est un agent
(classe `Voiture`) ; la classe `Simulation` contient le code principal qui
résout le schéma numérique pas à pas et enregistre tout l'historique.

Modèle d'accélération
---------------------
Pour la voiture i, on note :
    - d        : écart pare-chocs à pare-chocs avec la voiture de devant
    - v_devant : vitesse de la voiture de devant
    - dt       : pas de temps
    - d_sec    : distance de sécurité (paramètre du profil)
    - mu       : coefficient de sensibilité de l'accélération (profil)

On calcule un écart PRÉVU au pas suivant puis l'accélération :

    ecart_prevu = d + (v_devant - v) * dt
    a = mu * (ecart_prevu - d_sec)              (puis bridée dans [a_min, a_max])

mu contrôle donc la force avec laquelle la voiture accélère (si elle a de la
marge) ou freine (si elle se rapproche trop) par rapport à sa distance de
sécurité.

Schéma numérique (Euler explicite)
----------------------------------
    v(t+dt) = clip( v(t) + a * dt , 0 , v_max )
    x(t+dt) = ( x(t) + v(t+dt) * dt )  modulo L

avec un garde-fou anticollision sur la vitesse pour empêcher tout
chevauchement dû à la discrétisation.

Tout est purement algorithmique : aucun affichage ici. Les visualisations se
font dans le notebook à partir des historiques `X`, `V`, `A`, `VOIE`.
"""

from __future__ import annotations

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
    distance_securite: float        # distance de sécurité visée (m, pare-chocs)
    vitesse_max: float              # vitesse maximale (m/s)
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
    a_min: float = -7.0        # décélération maximale / freinage (m/s^2)
    v_min: float = 0.0         # vitesse minimale (m/s)
    longueur_voiture: float = 3.0   # longueur d'une voiture (m)
    distance_min: float = 1.0       # interstice minimal pare-chocs en bouchon (m)

    # --- Valeurs de base (profil « normal ») dont dérivent les autres profils ---
    mu: float = 0.4
    distance_securite: float = 10.0
    vitesse_max: float = 30.0

    # --- Population ---
    proportions: dict = field(default_factory=lambda: {
        "prudent": 0.20, "normal": 0.50, "fou": 0.20, "camion": 0.10})
    vitesse_initiale: float = 22.0   # vitesse de départ des voitures (m/s)
    nb_arret_initial: int = 3        # voitures à l'arrêt au départ (onde de bouchon)
    graine: int | None = None        # graine aléatoire (None = non reproductible)

    @property
    def nb_iterations(self) -> int:
        """Nombre de pas de temps de la simulation."""
        return int(self.T / self.dt)

    @property
    def distance_centre_min(self) -> float:
        """Distance centre à centre minimale (anticollision)."""
        return self.longueur_voiture + self.distance_min


def construire_profils(p: Parametres) -> dict[str, ProfilConducteur]:
    """Construit les quatre profils à partir des valeurs de base de `p`.

    Les profils dérivent du profil « normal » : si l'on change `mu`,
    `distance_securite` ou `vitesse_max` dans les paramètres, tous les profils
    se décalent de façon cohérente. On crée aussi un écart de vitesse net entre
    les comportements (prudent < normal < fou, camion le plus lent).
    """
    return {
        "prudent": ProfilConducteur(
            nom="prudent",
            mu=p.mu - 0.15,
            distance_securite=p.distance_securite + 5.0,
            vitesse_max=p.vitesse_max - 5.0,
            couleur="tab:blue",
        ),
        "normal": ProfilConducteur(
            nom="normal",
            mu=p.mu,
            distance_securite=p.distance_securite,
            vitesse_max=p.vitesse_max,
            couleur="tab:green",
        ),
        "fou": ProfilConducteur(
            nom="fou",
            mu=p.mu + 0.30,
            distance_securite=max(2.0, p.distance_securite - 3.0),
            vitesse_max=p.vitesse_max + 4.0,
            couleur="tab:red",
            ignore_arriere_au_depassement=True,   # ignore la voiture arrière
        ),
        "camion": ProfilConducteur(
            nom="camion",
            mu=max(0.05, p.mu - 0.20),
            distance_securite=p.distance_securite + 10.0,
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

    def acceleration_souhaitee(self, ecart: float, v_devant: float,
                               dt: float, a_min: float, a_max: float) -> float:
        """Loi du modèle : a = mu * (ecart_prevu - distance_securite).

        `ecart` est l'écart pare-chocs actuel avec la voiture de devant ;
        `v_devant` sa vitesse. Le résultat est bridé dans [a_min, a_max].
        """
        ecart_prevu = ecart + (v_devant - self.v) * dt
        a = self.mu * (ecart_prevu - self.distance_securite)
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

        # Historiques (remplis par simuler())
        self.X = self.V = self.A = self.VOIE = self.temps = None

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
            self.voitures.append(Voiture(i, positions[i], v0, voie=1, profil=profil))

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
        n'est trouvée, retourne (None, L) — route libre.
        """
        L = self.p.L
        meilleure = L
        indice = None
        xi = positions[i]
        for j in range(self.p.N):
            if j == i or voies[j] != voie_visee:
                continue
            d = (positions[j] - xi) % L          # distance vers l'avant (cyclique)
            if 0 < d < meilleure:
                meilleure = d
                indice = j
        return indice, meilleure

    def voiture_derriere(self, i, voie_visee, positions, voies):
        """Voiture la plus proche DERRIÈRE i sur `voie_visee`."""
        L = self.p.L
        meilleure = L
        indice = None
        xi = positions[i]
        for j in range(self.p.N):
            if j == i or voies[j] != voie_visee:
                continue
            d = (xi - positions[j]) % L          # distance vers l'arrière (cyclique)
            if 0 < d < meilleure:
                meilleure = d
                indice = j
        return indice, meilleure

    # ------------------------------------------------------------------
    #  Décision de changement de voie (règles européennes)
    # ------------------------------------------------------------------
    def _decider_voie(self, i, positions, voies) -> int:
        """Choisit la voie de la voiture i au pas suivant.

        1) On se rabat le plus à droite possible dès qu'il y a la place
           (devant ET derrière), avec une marge un peu plus large que la
           distance de sécurité (hystérésis pour éviter les oscillations).
        2) Sinon, on double UNIQUEMENT par la gauche, et seulement si :
             - on se rapproche trop de la voiture de devant (écart < d_sec) ;
             - la voie de gauche est libre devant ;
             - la voie de gauche est libre derrière (sauf conducteur « fou »).
           Un camion ne peut jamais aller au-delà de sa voie_max (3e interdite).
        """
        p = self.p
        voiture = self.voitures[i]
        voie_actuelle = voies[i]
        lng = p.longueur_voiture
        d_sec = voiture.distance_securite

        # 1) Se rabattre à droite
        distance_rabattement = 1.2 * d_sec
        if voie_actuelle > 1:
            for voie_droite in range(1, voie_actuelle):
                _, d_dev = self.voiture_devant(i, voie_droite, positions, voies)
                _, d_der = self.voiture_derriere(i, voie_droite, positions, voies)
                if (d_dev - lng) > distance_rabattement and \
                   (d_der - lng) > distance_rabattement:
                    return voie_droite

        # 2) Doubler par la gauche
        voie_gauche = voie_actuelle + 1
        if voie_gauche <= self._voie_max(voiture):
            _, d_dev_actuel = self.voiture_devant(i, voie_actuelle, positions, voies)
            if (d_dev_actuel - lng) < d_sec:               # trop proche devant
                _, d_dev_g = self.voiture_devant(i, voie_gauche, positions, voies)
                place_devant = (d_dev_g - lng) > d_sec
                _, d_der_g = self.voiture_derriere(i, voie_gauche, positions, voies)
                if voiture.profil.ignore_arriere_au_depassement:
                    # Le « fou » ignore la distance de SÉCURITÉ arrière (il
                    # coupe la route), mais garde l'interstice physique minimal :
                    # il ne peut pas déboîter sur une voiture.
                    place_derriere = d_der_g > p.distance_centre_min
                else:
                    place_derriere = (d_der_g - lng) > d_sec
                if place_devant and place_derriere:
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
        # suivantes, ce qui limite les conflits de déboîtement simultané.
        voies_t1 = list(voies)
        for i in range(N):
            voies_t1[i] = self._decider_voie(i, positions, voies_t1)
        for i in range(N):
            self.voitures[i].voie = voies_t1[i]

        # --- 2) Accélération puis vitesse provisoire ---
        vitesses_prov = np.zeros(N)
        leaders = [None] * N
        d_centre_leaders = [p.L] * N
        for i in range(N):
            voit = self.voitures[i]
            j, d_centre = self.voiture_devant(i, voit.voie, positions, voies_t1)
            leaders[i] = j
            d_centre_leaders[i] = d_centre
            ecart = d_centre - p.longueur_voiture
            v_devant = vitesses[j] if j is not None else voit.v
            voit.a = voit.acceleration_souhaitee(ecart, v_devant, p.dt, p.a_min, p.a_max)
            v_new = voit.v + voit.a * p.dt
            vitesses_prov[i] = min(max(v_new, p.v_min), voit.vitesse_max)

        # --- 3) Garde-fou anticollision ---
        for i in range(N):
            j = leaders[i]
            if j is not None:
                distance_disponible = d_centre_leaders[i] - p.distance_centre_min
                v_limite = vitesses_prov[j] + 0.5 * distance_disponible
                if vitesses_prov[i] > v_limite:
                    vitesses_prov[i] = v_limite
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
            VOIE[n, i] voie (entier)
        """
        p = self.p
        nb = p.nb_iterations
        X = np.empty((nb, p.N))
        V = np.empty((nb, p.N))
        A = np.empty((nb, p.N))
        VOIE = np.empty((nb, p.N), dtype=int)

        for n in range(nb):
            for i, voit in enumerate(self.voitures):
                X[n, i] = voit.x
                V[n, i] = voit.v
                VOIE[n, i] = voit.voie
            self.etape()
            for i, voit in enumerate(self.voitures):
                A[n, i] = voit.a
            if verbeux and nb >= 10 and n % (nb // 10) == 0:
                print(f"  pas {n:>6}/{nb}  (t = {n * p.dt:6.1f} s)")

        self.X, self.V, self.A, self.VOIE = X, V, A, VOIE
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
    print("Profils tirés          :", dict(collections.Counter(sim.profils_voitures)))
    print("Répartition par voie   :", dict(collections.Counter(sim.VOIE[-1].tolist())))
