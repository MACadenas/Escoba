import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from itertools import combinations, chain

class EscobaEnv(gym.Env):
    def __init__(self):
        super(EscobaEnv, self).__init__()

        # Baraja de 40 cartas (sin 8 ni 9)
        self.todas_las_cartas = [(p, v) for p in range(4) for v in [1,2,3,4,5,6,7,10,11,12]]
        self.reset()

        # Espacio de observación (40 posibles cartas en mano + 40 en mesa)
        self.observation_space = spaces.MultiBinary(80)

        # Espacio de acción: índice de carta en mano (0, 1, 2)
        # Más combinaciones posibles de mesa (como simplificación inicial: elegimos índice del subconjunto)
        self.action_space = spaces.Discrete(20)  # Se ajustará luego dinámicamente

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mazo = self.todas_las_cartas.copy()
        random.shuffle(self.mazo)
        self.mesa = [self.mazo.pop() for _ in range(4)]
        self.jugador = [ [self.mazo.pop() for _ in range(3)], [] ]  # mano, capturas
        self.oponente = [ [self.mazo.pop() for _ in range(3)], [] ]
        self.turno = 0  # 0: IA, 1: bot
        self.done = False
        return self._get_state(), {}  # obs, info

    def _get_state(self):
        state = np.zeros(80, dtype=np.int8)
        for carta in self.jugador[0]:
            idx = self.todas_las_cartas.index(carta)
            state[idx] = 1
        for carta in self.mesa:
            idx = self.todas_las_cartas.index(carta)
            state[40 + idx] = 1
        return state

    def _valor(self, carta):
        return carta[1] if carta[1] <= 7 else carta[1] - 2

    def _jugadas_validas(self, carta):
        combos = chain.from_iterable(combinations(self.mesa, r) for r in range(1, len(self.mesa)+1))
        return [list(c) for c in combos if self._valor(carta) + sum(self._valor(x) for x in c) == 15]

    def step(self, action):
        reward = 0
        info = {}
        done = False

        mano = self.jugador[0]
        if action >= len(mano):
            action = 0  # fallback

        carta = mano.pop(action)
        jugadas = self._jugadas_validas(carta)

        if jugadas:
            jugada = jugadas[0]
            for c in jugada:
                self.mesa.remove(c)
            self.jugador[1].extend(jugada + [carta])
            if not self.mesa:
                reward += 1  # escoba
        else:
            self.mesa.append(carta)

        # Turno del bot (jugador simplificado)
        carta_bot = self.oponente[0].pop(0)
        jugadas_bot = self._jugadas_validas(carta_bot)
        if jugadas_bot:
            jugada_bot = jugadas_bot[0]
            for c in jugada_bot:
                self.mesa.remove(c)
            self.oponente[1].extend(jugada_bot + [carta_bot])
        else:
            self.mesa.append(carta_bot)

        # Fin de ronda
        if not self.jugador[0]:
            self.done = True
            reward += self._evaluar_partida()

        return self._get_state(), reward, self.done, False, {}  # obs, reward, terminated, truncated, info

    def _evaluar_partida(self):
        puntos = 0
        capturas = len(self.jugador[1])
        oros = len([c for c in self.jugador[1] if c[0] == 0])
        siete_bello = (0, 7) in self.jugador[1]

        if capturas > len(self.oponente[1]):
            puntos += 1
        if oros > len([c for c in self.oponente[1] if c[0] == 0]):
            puntos += 1
        if siete_bello:
            puntos += 1

        return puntos
