import pygame
import ui
import player
import board
from config import BLACK, WHITE
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Othello:
    """
    黑白棋类
    """
    def __init__(self):
        """
        初始化UI和棋盘
        获取游戏选项
        """
        # UI
        self.gui = ui.Gui()
        # 棋盘
        self.board = board.Board()
        # 获取用户选项并开始游戏
        self.get_options()

    def get_options(self):
        """
        设置用户选项，如先开始的是谁，后开始是谁
        """
        player1, player2, level = self.gui.show_options()
        if player1 == "human":
            self.now_playing = player.Human(self.gui, BLACK)
            print(type(self.now_playing))
        else:
            self.now_playing = player.Computer(BLACK, level + 6)
        if player2 == "human":
            self.other_player = player.Human(self.gui, WHITE)
        else:
            self.other_player = player.Computer(WHITE, level + 6)

        # 开始游戏
        self.gui.show_game()
        # 更新分数
        self.gui.update(self.board.board, 2, 2, self.now_playing.color)

    # 运行
    def run(self):
        # 时钟对象，可以设置每秒刷新速率
        clock = pygame.time.Clock()
        while True:
            # 每秒刷新60次
            clock.tick(60)
            # 判断游戏是否结束
            if self.board.game_ended():
                whites, blacks, empty = self.board.count_stones()
                if whites > blacks:
                    winner = WHITE
                elif blacks > whites:
                    winner = BLACK
                else:
                    winner = None
                break
            # 更新当前玩家（人类或者电脑）的棋盘
            self.now_playing.get_current_board(self.board)
            # 获取有效的落子点
            if self.board.get_valid_moves(self.now_playing.color) != []:
                now=time.time()
                # 现在下的人作出自己的动作
                score, self.board, (move_x,move_y) = self.now_playing.get_move()
                print(self.now_playing.name+'Search specnt: '+timeSince(now)+' move:{}'.format((move_x+1,move_y+1))+'score: {}'.format(score))
                whites, blacks, empty = self.board.count_stones()
                # 更新UI
                self.gui.update(self.board.board, blacks, whites,
                                self.now_playing.color)
            # 交换
            self.now_playing, self.other_player = self.other_player, self.now_playing

        # 显示胜者
        self.gui.show_winner(winner)
        # 重新开始
        self.restart()

    def restart(self):
        self.board = board.Board()
        self.get_options()
        self.run()

if __name__ == '__main__':
    # 初始化othello
    game = Othello()
    game.run()
