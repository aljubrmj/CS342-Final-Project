from mod_utils import Player
from mod_utils import Tournament
from argparse import ArgumentParser
import importlib
import numpy as np
import pystk


class DummyPlayer:
    def __init__(self, player, team=0):
        self.team = team
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        self.kart = np.random.choice(all_players)

    @property
    def config(self):
        return pystk.PlayerConfig(
            controller=pystk.PlayerConfig.Controller.AI_CONTROL,
            team=self.team,kart = self.kart)
    
    def __call__(self, image, player_info, state):
        return dict()

