import pystk
import numpy as np
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
import importlib

from agent.models import *
from torchvision import transforms
import torch

# import pdb
def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()
        self.CNN_Classifier = load_model('cnn_classifier').eval()
        self.CNN_Regressor = load_model('cnn_regressor').eval()
        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # print(self.device)

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        # state.update()

        # #### Example of how to change the initial location of the karts and/or puck ####
        # # # Change initial location of the puck e

        # state.set_ball_location(position=[-6.7, 0.3, 0.0])
        # state.set_kart_location(kart_id=0, 
        #                                   position=[-6.7, 0.3, -56],
        #                                   speed=0.0,
        #                                   rotation=[0.0, 0.0, 0.0, 1.0]
        #                                  )
        # state.set_kart_location(kart_id=1, 
        #                                   position=[-6.7, 0.3, 56],
        #                                   speed=0.0,
        #                                   rotation=[0.0, 1.0, 0.0, 0.0]
        #                                  )

        
        if save is not None:
            import PIL.Image
            import os
            shutil.rmtree(save)
            os.makedirs(save)

        for t in range(max_frames):
            # print('\rframe %d' % t, end='\r')

            state.update()

            list_actions = []
            for i, p in enumerate(self.active_players):
                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                puck_location = state.soccer.ball.location
                # kart_location_1 = state.players[1-i].kart.location
                kart_location_2 = state.players[i].kart.location
                goal_line_1 = state.soccer.goal_line[1-i][0]
                goal_line_2 = state.soccer.goal_line[1-i][1]
                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T

                if save is not None:
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    # fig, ax = plt.subplots(1, 1)
                    # ax.imshow(image)

                    # WH2 = np.array([self.graphics_config.screen_width, self.graphics_config.screen_height]) / 2
                    # # ax.add_artist(plt.Circle(WH2*(1+to_image(puck_location, proj, view)), 10, ec='g', fill=False, lw=1.5))
                    # # ax.add_artist(plt.Circle(WH2*(1+to_image(kart_location_1, proj, view)), 10, ec='r', fill=False, lw=1.5))
                    # # ax.add_artist(plt.Circle(WH2*(1+to_image(kart_location_2, proj, view)), 10, ec='k', fill=False, lw=1.5))
                    # # ax.add_artist(plt.Circle(WH2*(1+to_image(player.kart.front, proj, view)), 10, ec='y', fill=False, lw=1.5))
                    # # ax.add_artist(plt.Circle(WH2*(2+to_image(goal_line_2, proj, view)+to_image(goal_line_1, proj, view))/2, 10, ec='m', fill=False, lw=1.5))
                    
                    # logit = self.CNN_Classifier(self.transform(image)[None].to(self.device))
                    # puck_is_ahead = logit.argmax(1) == 1.0
                    # if puck_is_ahead:
                    #     # puck_x = output[1].item()
                    #     # puck_y = output[2].item() 
                    #     # ax.add_artist(plt.Circle(WH2*(1+np.array([puck_x, puck_y])), 10, ec='red', fill=False, lw=1.5))
                    #     ax.add_artist(plt.Circle(WH2*(1+self.CNN_Regressor(self.transform(image)[None].to(self.device)).to(torch.device("cpu")).data.numpy()[0]), 10, ec='r', fill=False, lw=1.5))
                    # fig.savefig(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    # plt.close()



            # pdb.set_trace()
            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k



#Level-0
# pos_ball = state.players[0].kart.location
# pos_ball[0] = -5
# pos_ball[1] = 0.9000000002980232
# pos_ball[2] =  45
# state.set_ball_location(position=pos_ball)
# pos_player = state.players[0].kart.location
# pos_player[0] = 20
# pos_player[1] = 0.07000000029802322
# pos_player[2] = 22

#Level-CubicSplineChallenge
# pos_ball = state.players[0].kart.location
# pos_ball[0] = 2.0
# pos_ball[1] = 0.34931060671806335
# pos_ball[2] = 0.0
# state.set_ball_location(position=pos_ball)
# pos_player = state.players[0].kart.location
# pos_player[0] = 2.0+20.0
# pos_player[1] = 0.30
# pos_player[2] = -(0.0+20)

        # # Change initial location of the puck 
#Level-CubicSplineChallenge
        # pos_ball = state.soccer.ball.location
        # pos_ball[0] += +20
        # pos_ball[2] += -35
        # state.set_ball_location(position=pos_ball)
        # pos_player = state.players[0].kart.location
        # pos_player[0] = 2.0+20.0
        # pos_player[1] = 0.30
        # pos_player[2] = -(0.0+20)
