"""
Animation pygame du simulateur de trafic.
==========================================

Fenêtre temps réel, fluide et interactive, qui rejoue une `Simulation` déjà
calculée (historiques `X`, `V`, `VOIE`, `YLAT`).

Deux ingrédients rendent le mouvement agréable à l'œil :

1. Changements de voie progressifs — on dessine la position latérale CONTINUE
   `YLAT` (et non la voie entière `VOIE`), si bien que la voiture glisse d'une
   voie à l'autre au lieu de sauter.
2. Interpolation entre les pas enregistrés — la lecture est pilotée par le temps
   horloge, et la position de chaque voiture est interpolée entre deux pas de la
   simulation. L'animation reste fluide quels que soient `dt` et la cadence.

Commandes
---------
    Espace            pause / lecture
    Flèches ↑ / ↓     accélérer / ralentir la lecture
    Flèches ← / →     reculer / avancer de 2 s
    R                 revenir au début
    Échap  /  Q       quitter

Lancement
---------
    python3 animation_pygame.py                 # paramètres par défaut
    python3 animation_pygame.py --duree 60 --voitures 90 --graine 1
"""

from __future__ import annotations

import argparse
import collections
import math
import os

import numpy as np

from simulateur_trafic import Parametres, Simulation


# =========================================================================
#  Palette
# =========================================================================
FOND        = (15, 16, 20)        # fenêtre
ROUTE       = (46, 48, 56)        # bitume
BORD_ROUTE  = (78, 80, 92)        # bords intérieur/extérieur de la chaussée
SEPARATEUR  = (150, 152, 165)     # pointillés entre voies
CONTOUR     = (228, 230, 240)     # liseré des voitures
TEXTE       = (232, 233, 240)
TEXTE_FAIBLE = (150, 152, 165)
ACCENT      = (90, 170, 255)


def _nom_couleur_vers_rgb(nom: str) -> tuple[int, int, int]:
    """Convertit un nom de couleur matplotlib (« tab:blue », « black »…) en RGB.

    Les voitures noires (camions) sont éclaircies pour rester visibles sur le
    bitume sombre ; leur liseré clair les distingue malgré tout.
    """
    try:
        from matplotlib.colors import to_rgb
        r, g, b = (int(round(c * 255)) for c in to_rgb(nom))
    except Exception:                       # repli si matplotlib indisponible
        table = {"tab:blue": (31, 119, 180), "tab:green": (44, 160, 44),
                 "tab:red": (214, 39, 40), "black": (0, 0, 0)}
        r, g, b = table.get(nom, (200, 200, 200))
    if max(r, g, b) < 40:                   # quasi noir -> « camion » sombre lisible
        return (34, 36, 44)
    return (r, g, b)


# =========================================================================
#  Animateur
# =========================================================================
class AnimateurPygame:
    """Rejoue une simulation dans une fenêtre pygame fluide et interactive."""

    SUR_ECHANT = 2          # super-échantillonnage du fond (anti-crénelage)

    def __init__(self, sim: Simulation, taille: int | None = None, fps: int = 60,
                 vitesse_lecture: float = 1.0, zoom: float = 2.0):
        if sim.X is None:
            raise ValueError("La simulation doit être lancée (sim.simuler()) avant l'animation.")
        self.sim = sim
        self.p = sim.p
        self.taille = taille          # None => ajustée à l'écran dans _init_pygame
        self.fps = fps
        self.vitesse_lecture = vitesse_lecture
        # Taille des voitures à l'écran = `zoom` × longueur réelle. Le modèle
        # garde un écart suffisant pour qu'à zoom ≈ 2 elles ne se chevauchent
        # que dans les bouchons les plus serrés (aspect pare-chocs contre
        # pare-chocs réaliste). Réglable avec --zoom.
        self.zoom = zoom

        # --- Historiques en tableaux numpy (lecture rapide) ---
        self.X = np.asarray(sim.X)
        self.V = np.asarray(sim.V)
        self.YLAT = np.asarray(sim.YLAT)
        self.couleurs_rgb = [_nom_couleur_vers_rgb(c) for c in sim.couleurs]
        self.profils = list(sim.profils_voitures)
        self.nb_pas = self.X.shape[0]
        self.duree_totale = self.nb_pas * self.p.dt

        # Géométrie de l'anneau : calculée dans _calc_geometrie() une fois la
        # taille de fenêtre connue (elle peut être déduite de l'écran).
        self.cx = self.cy = self.rayon_ext = self.rayon_int = self.largeur_voie = 0.0

        # --- Lecture ---
        self.t_sim = 0.0          # temps de simulation courant (s)
        self.en_pause = False

        self._pygame = None       # importé tardivement (dans _init_pygame)
        self.ecran = None
        self.horloge = None
        self.fond = None
        self.sprites: dict[str, "pygame.Surface"] = {}
        self.police = self.police_titre = self.police_petite = None

    # ------------------------------------------------------------------
    #  Géométrie (dépend de la taille de fenêtre, connue tardivement)
    # ------------------------------------------------------------------
    def _taille_auto(self, sans_fenetre):
        """Côté de la fenêtre carrée : ~90 % du plus petit côté de l'écran."""
        if sans_fenetre:
            return 920
        try:
            w, h = self._pygame.display.get_desktop_sizes()[0]
        except Exception:
            try:
                info = self._pygame.display.Info()
                w, h = info.current_w, info.current_h
            except Exception:
                return 1000
        return int(max(640, min(1500, 0.9 * min(w, h))))

    def _calc_geometrie(self):
        """Centre et rayons des voies (voie 1 = extérieur, N_voie = intérieur)."""
        t = self.taille
        self.cx = self.cy = t / 2
        self.rayon_ext = t * 0.40
        self.rayon_int = t * 0.24
        if self.p.N_voie > 1:
            self.largeur_voie = (self.rayon_ext - self.rayon_int) / (self.p.N_voie - 1)
        else:
            self.largeur_voie = t * 0.28
            self.rayon_int = self.rayon_ext

    # ------------------------------------------------------------------
    #  Conversion position simulation -> écran
    # ------------------------------------------------------------------
    def _rayon(self, lat):
        """Rayon écran (px) pour une position latérale continue `lat` ∈ [1, N_voie]."""
        if self.p.N_voie <= 1:
            return np.full_like(np.asarray(lat, dtype=float), self.rayon_ext)
        frac = (np.asarray(lat, dtype=float) - 1.0) / (self.p.N_voie - 1)
        return self.rayon_ext - frac * (self.rayon_ext - self.rayon_int)

    def _etat_interpole(self, t_sim):
        """Positions/vitesses interpolées à l'instant `t_sim` (s).

        Renvoie (xs, ys, angles_deg, v_moyenne, indice_pas). L'interpolation
        gère l'enroulement de la route (les voitures avancent toujours).
        """
        L = self.p.L
        pos = t_sim / self.p.dt
        f = int(pos) % self.nb_pas
        g = (f + 1) % self.nb_pas
        frac = pos - math.floor(pos)

        xf, xg = self.X[f], self.X[g]
        dx = (xg - xf) % L                      # distance vers l'avant (cyclique)
        x = (xf + frac * dx) % L
        lat = self.YLAT[f] * (1 - frac) + self.YLAT[g] * frac

        theta = 2 * np.pi * x / L
        r = self._rayon(lat)
        sx = self.cx + r * np.cos(theta)
        sy = self.cy - r * np.sin(theta)        # y écran vers le bas -> sens trigo à l'écran

        # Cap (heading) tangent à l'anneau, dans le sens de la marche
        hx, hy = -np.sin(theta), -np.cos(theta)
        angles = np.degrees(np.arctan2(-hy, hx))

        v_moy = float(self.V[f].mean())
        return sx, sy, angles, r, v_moy, f

    # ------------------------------------------------------------------
    #  Construction des éléments graphiques
    # ------------------------------------------------------------------
    def _faire_sprite(self, couleur, longueur, largeur):
        """Sprite de voiture orienté vers +x (avant à droite), fond transparent."""
        pg = self._pygame
        longueur, largeur = int(longueur), int(largeur)
        surf = pg.Surface((longueur, largeur), pg.SRCALPHA)
        rect = surf.get_rect()
        rayon = max(2, largeur // 3)
        pg.draw.rect(surf, couleur, rect, border_radius=rayon)
        pg.draw.rect(surf, CONTOUR, rect, width=1, border_radius=rayon)
        # pare-brise près de l'avant (indique le sens)
        pare_brise = pg.Rect(int(longueur * 0.60), int(largeur * 0.20),
                             max(2, int(longueur * 0.20)), max(2, int(largeur * 0.60)))
        pg.draw.rect(surf, (210, 224, 240), pare_brise, border_radius=2)
        return surf

    # Sprite de référence dessiné en grand puis réduit par voiture (rendu net).
    BASE_LONG = 56            # longueur (px) du sprite de référence

    def _construire_sprites(self):
        """Un sprite « de référence » par profil, mis à l'échelle à l'affichage.

        La taille écran d'une voiture vaut `zoom` × sa longueur réelle. Comme le
        modèle garde un écart confortable la plupart du temps, à `zoom` ≈ 2 les
        voitures restent bien visibles et ne se chevauchent que dans les bouchons
        les plus serrés (pare-chocs contre pare-chocs). Toutes ont la même
        longueur dans le modèle ; le camion se distingue par sa couleur.
        """
        # Longueur/largeur « physiques » de référence (m) ; le facteur `zoom` est
        # appliqué au moment du dessin (voir _dessiner_voitures).
        self.long_voiture_m = self.p.longueur_voiture
        self.larg_voiture_m = 0.6 * self.p.longueur_voiture
        base_larg = self.BASE_LONG * self.larg_voiture_m / self.long_voiture_m
        for nom in self.p.proportions:
            couleur = _nom_couleur_vers_rgb(self.sim.profils[nom].couleur)
            self.sprites[nom] = self._faire_sprite(couleur, self.BASE_LONG, base_larg)

    def _construire_fond(self):
        """Fond statique (bitume, séparateurs, titre) super-échantillonné puis réduit."""
        pg = self._pygame
        S = self.SUR_ECHANT
        grand = pg.Surface((self.taille * S, self.taille * S))
        grand.fill(FOND)
        cx, cy = self.cx * S, self.cy * S

        r_ext_route = (self.rayon_ext + self.largeur_voie / 2) * S
        r_int_route = (self.rayon_int - self.largeur_voie / 2) * S

        # Chaussée = grand disque - disque intérieur
        pg.draw.circle(grand, ROUTE, (cx, cy), r_ext_route)
        pg.draw.circle(grand, FOND, (cx, cy), max(1, r_int_route))
        # Bords de la chaussée
        pg.draw.circle(grand, BORD_ROUTE, (cx, cy), r_ext_route, width=max(1, 2 * S))
        pg.draw.circle(grand, BORD_ROUTE, (cx, cy), max(1, r_int_route), width=max(1, 2 * S))

        # Séparateurs de voies en pointillés
        for k in range(1, self.p.N_voie):
            r_sep = self._rayon(k + 0.5)[()] * S
            self._cercle_pointille(grand, cx, cy, r_sep)

        grand = pg.transform.smoothscale(grand, (self.taille, self.taille))

        # Titre (après réduction, pour un texte net)
        titre = self.police_titre.render("Route circulaire — simulation de trafic", True, TEXTE)
        grand.blit(titre, titre.get_rect(midtop=(self.taille // 2, 12)))
        self.fond = grand

    def _cercle_pointille(self, surf, cx, cy, rayon, pas_deg=4.0, plein=0.55):
        """Trace un cercle en pointillés (dans le repère super-échantillonné)."""
        pg = self._pygame
        ang = 0.0
        while ang < 360.0:
            a0 = math.radians(ang)
            a1 = math.radians(ang + pas_deg * plein)
            p0 = (cx + rayon * math.cos(a0), cy - rayon * math.sin(a0))
            p1 = (cx + rayon * math.cos(a1), cy - rayon * math.sin(a1))
            pg.draw.line(surf, SEPARATEUR, p0, p1, width=max(1, self.SUR_ECHANT))
            ang += pas_deg

    # ------------------------------------------------------------------
    #  Dessin d'une image
    # ------------------------------------------------------------------
    def _dessiner_voitures(self, surface, sx, sy, angles, r):
        pg = self._pygame
        # Échelle locale (px/m) = circonférence / L = 2*pi*r / L, propre à chaque
        # voiture selon sa voie (les voies intérieures, plus courtes, donnent des
        # voitures plus petites). Longueur écran = zoom * long_voiture_m * px/m.
        k = self.zoom * self.long_voiture_m * 2 * math.pi / (self.p.L * self.BASE_LONG)
        echelle_min = 3.0 / self.BASE_LONG          # au moins ~3 px, pour rester visible
        for i in range(len(sx)):
            echelle = max(echelle_min, k * r[i])
            sprite = pg.transform.rotozoom(self.sprites[self.profils[i]], angles[i], echelle)
            surface.blit(sprite, sprite.get_rect(center=(sx[i], sy[i])))

    def _dessiner_hud(self, surface, t_sim, v_moy):
        pg = self._pygame
        h = self.taille

        # --- En haut à gauche : temps, vitesse, lecture ---
        lignes = [
            (f"t = {t_sim:5.1f} / {self.duree_totale:.0f} s", TEXTE),
            (f"v moyenne = {v_moy * 3.6:5.1f} km/h  ({v_moy:4.1f} m/s)", TEXTE),
            (f"lecture ×{self.vitesse_lecture:.2f}" + ("   [PAUSE]" if self.en_pause else ""),
             ACCENT if not self.en_pause else (255, 170, 90)),
        ]
        y = 44
        for txt, col in lignes:
            surface.blit(self.police.render(txt, True, col), (18, y))
            y += 24

        # --- Jauge de fluidité (vitesse moyenne / vitesse max de référence) ---
        v_ref = max(1e-6, self.p.vitesse_max)
        frac = float(np.clip(v_moy / v_ref, 0, 1))
        larg, haut = 190, 12
        x0, y0 = 18, y + 6
        pg.draw.rect(surface, (40, 42, 50), (x0, y0, larg, haut), border_radius=6)
        couleur_jauge = (int(220 * (1 - frac) + 70 * frac),
                         int(70 * (1 - frac) + 200 * frac), 90)
        pg.draw.rect(surface, couleur_jauge, (x0, y0, int(larg * frac), haut), border_radius=6)
        surface.blit(self.police_petite.render("fluidité", True, TEXTE_FAIBLE), (x0, y0 + haut + 3))

        # --- En bas à gauche : légende avec effectifs ---
        compte = collections.Counter(self.profils)
        y = h - 18 - 20 * len(self.p.proportions)
        for nom in self.p.proportions:
            couleur = _nom_couleur_vers_rgb(self.sim.profils[nom].couleur)
            pg.draw.rect(surface, couleur, (18, y + 3, 14, 11), border_radius=3)
            pg.draw.rect(surface, CONTOUR, (18, y + 3, 14, 11), width=1, border_radius=3)
            surface.blit(self.police_petite.render(f"{nom} ({compte.get(nom, 0)})", True, TEXTE),
                         (40, y))
            y += 20

        # --- En bas à droite : aide ---
        aide = ["Espace : pause", "↑ ↓ : vitesse", "← → : ±2 s", "R : début", "Échap : quitter"]
        y = h - 18 - 18 * len(aide)
        for txt in aide:
            r = self.police_petite.render(txt, True, TEXTE_FAIBLE)
            surface.blit(r, r.get_rect(topright=(self.taille - 16, y)))
            y += 18

    def _dessiner(self):
        sx, sy, angles, r, v_moy, _ = self._etat_interpole(self.t_sim)
        self.ecran.blit(self.fond, (0, 0))
        self._dessiner_voitures(self.ecran, sx, sy, angles, r)
        self._dessiner_hud(self.ecran, self.t_sim, v_moy)

    # ------------------------------------------------------------------
    #  Initialisation pygame
    # ------------------------------------------------------------------
    def _init_pygame(self, sans_fenetre=False):
        if sans_fenetre:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self._pygame = pygame
        pygame.init()
        pygame.font.init()
        if self.taille is None:                 # ajuste la fenêtre à l'écran
            self.taille = self._taille_auto(sans_fenetre)
        self._calc_geometrie()
        self.police = pygame.font.SysFont("menlo,consolas,monospace", 18)
        self.police_titre = pygame.font.SysFont("helvetica,arial", 22, bold=True)
        self.police_petite = pygame.font.SysFont("menlo,consolas,monospace", 14)
        flags = 0 if sans_fenetre else pygame.DOUBLEBUF
        self.ecran = pygame.display.set_mode((self.taille, self.taille), flags)
        pygame.display.set_caption("Simulateur de trafic — animation")
        self.horloge = pygame.time.Clock()
        self._construire_sprites()
        self._construire_fond()

    # ------------------------------------------------------------------
    #  Boucle principale interactive
    # ------------------------------------------------------------------
    def lancer(self):
        self._init_pygame()
        pg = self._pygame
        en_cours = True
        while en_cours:
            dt_horloge = self.horloge.tick(self.fps) / 1000.0
            for evt in pg.event.get():
                if evt.type == pg.QUIT:
                    en_cours = False
                elif evt.type == pg.KEYDOWN:
                    en_cours = self._gerer_touche(evt.key)

            if not self.en_pause:
                self.t_sim = (self.t_sim + dt_horloge * self.vitesse_lecture) % self.duree_totale

            self._dessiner()
            pg.display.flip()
        pg.quit()

    def _gerer_touche(self, touche):
        pg = self._pygame
        if touche in (pg.K_ESCAPE, pg.K_q):
            return False
        elif touche == pg.K_SPACE:
            self.en_pause = not self.en_pause
        elif touche == pg.K_UP:
            self.vitesse_lecture = min(8.0, self.vitesse_lecture * 1.25)
        elif touche == pg.K_DOWN:
            self.vitesse_lecture = max(0.1, self.vitesse_lecture / 1.25)
        elif touche == pg.K_RIGHT:
            self.t_sim = (self.t_sim + 2.0) % self.duree_totale
        elif touche == pg.K_LEFT:
            self.t_sim = (self.t_sim - 2.0) % self.duree_totale
        elif touche == pg.K_r:
            self.t_sim = 0.0
        return True

    # ------------------------------------------------------------------
    #  Rendu hors-écran (vérification / aperçu PNG, sans fenêtre)
    # ------------------------------------------------------------------
    def apercu_png(self, instants, prefixe="apercu"):
        """Rend quelques images à des instants (s) donnés, en PNG, sans fenêtre."""
        self._init_pygame(sans_fenetre=True)
        chemins = []
        for t in instants:
            self.t_sim = t % self.duree_totale
            self._dessiner()
            chemin = f"{prefixe}_{t:05.1f}s.png"
            self._pygame.image.save(self.ecran, chemin)
            chemins.append(chemin)
        self._pygame.quit()
        return chemins


# =========================================================================
#  Construction d'une simulation (mêmes valeurs par défaut que le notebook)
# =========================================================================
def faire_simulation(duree=120.0, voitures=75, voies=3, graine=42,
                     tps_changement_voie=0.8, temps_reaction=0.8,
                     verbeux=True) -> Simulation:
    params = Parametres(
        L=1000.0, N=voitures, N_voie=voies,
        dt=0.01, T=duree,
        a_max=3.0, a_min=-7.0,
        longueur_voiture=3.0, distance_min=1.0,
        mu=0.4, distance_securite=10.0, vitesse_max=30.0,
        proportions={"prudent": 0.20, "normal": 0.50, "fou": 0.20, "camion": 0.10},
        vitesse_initiale=22.0, nb_arret_initial=3,
        graine=graine, tps_changement_voie=tps_changement_voie,
        temps_reaction=temps_reaction,
    )
    if verbeux:
        print(f"Simulation : {params.nb_iterations} pas, {params.N} voitures, "
              f"{params.N_voie} voies…")
    return Simulation(params).simuler(verbeux=verbeux)


def _args():
    ap = argparse.ArgumentParser(description="Animation pygame du simulateur de trafic.")
    ap.add_argument("--duree", type=float, default=120.0, help="durée simulée (s)")
    ap.add_argument("--voitures", type=int, default=75, help="nombre de voitures")
    ap.add_argument("--voies", type=int, default=3, help="nombre de voies")
    ap.add_argument("--graine", type=int, default=42, help="graine aléatoire")
    ap.add_argument("--tcv", type=float, default=0.8,
                    help="durée d'un changement de voie à l'écran (s)")
    ap.add_argument("--treaction", type=float, default=0.8,
                    help="temps de réaction du conducteur (s) ; 0 = instantané")
    ap.add_argument("--taille", type=int, default=0,
                    help="taille de la fenêtre (px) ; 0 = ajustée à l'écran")
    ap.add_argument("--zoom", type=float, default=2.0,
                    help="taille des voitures (× longueur réelle) ; ↑ = plus grosses")
    ap.add_argument("--fps", type=int, default=60, help="images par seconde")
    ap.add_argument("--vitesse", type=float, default=1.0, help="vitesse de lecture initiale")
    return ap.parse_args()


if __name__ == "__main__":
    a = _args()
    sim = faire_simulation(duree=a.duree, voitures=a.voitures, voies=a.voies,
                           graine=a.graine, tps_changement_voie=a.tcv,
                           temps_reaction=a.treaction)
    AnimateurPygame(sim, taille=(a.taille or None), fps=a.fps,
                    vitesse_lecture=a.vitesse, zoom=a.zoom).lancer()
