import pystk
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import importlib

def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    # return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)
    return np.array([p[0] / p[-1], -p[1] / p[-1]])
class Player:
    def __init__(self, player, team):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info, state):
        return self.player.act(image, player_info, state)


class Tournament:
    _singleton = None

    def __init__(self, players, game_no, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.game_no = game_no
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
        self.score_1 = 0
        self.score_2 = 0
        self.counter = 0

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        if save is not None:
            import PIL.Image
            import os
            if not os.path.exists(save):
                os.makedirs(save)


        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')
            state.update()

            # if t % 150 == 0:
            #     pos_ball = state.soccer.ball.location
            #     pos_ball[0] = np.random.uniform(low=-40, high=40)
            #     pos_ball[1] = 1.0
            #     pos_ball[2] = np.random.uniform(low=-20, high=20)
            #     state.set_ball_location(position=pos_ball)

            current_score_1, current_score_2 = state.soccer.score
            if current_score_1 > self.score_1: 
                self.score_1 = current_score_1
                self.counter = 40
            if current_score_2 > self.score_2: 
                self.score_2 = current_score_2
                self.counter = 40

            list_actions = []

            for i, p in enumerate(self.active_players):
                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                # Perform 3D vecotr manipulation
                kart_front_vec = np.array(player.kart.front) - np.array(player.kart.location)
                kart_front_vec_norm = kart_front_vec / np.linalg.norm(kart_front_vec)
                kart_puck_vec = np.array(state.soccer.ball.location) - np.array(player.kart.location)
                kart_puck_vec_norm = kart_puck_vec / np.linalg.norm(kart_puck_vec)
                kart_puck_dp = kart_front_vec_norm.dot(kart_puck_vec_norm)

                
                puck_location = state.soccer.ball.location
                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T
                puck_x, puck_y = to_image(puck_location, proj, view)

                classification = 1 if ((-1.03 <= round(puck_x,1) <= 1.03) and (-1.03 <= round(puck_y,1) <= 1.03)) else 0
                action = pystk.Action()
                player_action = p(image, player, state)
                #print("player is {}{}".format(player,player.kart.front))
                #print("ball location is {}".format(state.soccer.ball.location))
                #print("player location is {}".format(player.kart.location))
                
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)



                # print(player.kart.name)
                # print(player.kart.name == 'konqi')
                if self.counter <=0: 
                    if (save is not None) and (player.kart.name == 'Konqi'):
                        PIL.Image.fromarray(image).save(os.path.join(save, f"g{self.game_no}_f{t}_p{i}_c{classification}.png"))    
                        if classification == 1:               
                            with open(os.path.join(save,f"g{self.game_no}_f{t}_p{i}_c{classification}.csv"), 'w') as f:
                                f.write(f"{puck_x}, {puck_y}")
                            f.close()
            
            if self.counter >0: 
                self.counter -= 1
                


                        # f.write('%0.2f,%0.1f,%0.1f' % tuple((kart_puck_dp,puck_location_image[0],puck_location_image[1])))
                    

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                
                dest = os.path.join(save, save+'_player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k


