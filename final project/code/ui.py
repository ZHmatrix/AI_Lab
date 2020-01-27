import pygame
import sys
from pygame.locals import *
import time
from config import BLACK, WHITE, DEFAULT_LEVEL, HUMAN, COMPUTER
import os

class Gui:
    def __init__(self):
        # 初始化game
        pygame.init()
        pygame.display.set_caption('黑白棋')
        # 设置rgb颜色
        self.BLACK = (0, 0, 0)
        self.BACKGROUND = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255,0,0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (128, 128, 0)
        # 显示的参数
        self.SCREEN_SIZE = (640, 480)
        self.BOARD_POS = (100, 20)
        # 棋盘左上角的位置
        self.BOARD = (120, 40)
        # 由于是8X8，一个棋子是50像素，因此总的边长为400
        self.BOARD_SIZE = 400
        # 一个方格的边长
        self.SQUARE_SIZE = 50

        # 创建屏幕
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)

        # 黑棋分数展示的位置
        self.BLACK_LAB_POS = (5, self.SCREEN_SIZE[1] / 4)
        # 黑棋分数展示的位置
        self.WHITE_LAB_POS = (560, self.SCREEN_SIZE[1] / 4)
        self.font = pygame.font.SysFont("Times New Roman", 22)
        self.scoreFont = pygame.font.SysFont("Serif", 58)

        # 加载图片文件
        self.board_img = pygame.image.load(os.path.join(
            "ui_img", "board.bmp")).convert()
        self.black_img = pygame.image.load(os.path.join(
            "ui_img", "preta.bmp")).convert()
        self.white_img = pygame.image.load(os.path.join(
            "ui_img", "branca.bmp")).convert()
        self.tip_img = pygame.image.load(os.path.join("ui_img",
                                                      "tip.bmp")).convert()
        self.clear_img = pygame.image.load(os.path.join("ui_img",
                                                        "nada.bmp")).convert()

    # 展示游戏选项并返回
    def show_options(self):
        # 默认值
        player1 = HUMAN
        player2 = COMPUTER
        level = DEFAULT_LEVEL

        while True:
            # 填充背景色
            self.screen.fill(self.BACKGROUND)
            # 设置标题字体
            title_fnt = pygame.font.SysFont("Times New Roman", 34)
            title = title_fnt.render("Othello", True, self.WHITE)
            title_pos = title.get_rect(
                centerx=self.screen.get_width() / 2, centery=60)
            # 渲染开始字体
            start_txt = self.font.render("Start", True, self.WHITE)
            start_pos = start_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=220)
            # 渲染player1字体
            player1_txt = self.font.render("First Player", True, self.WHITE)
            player1_pos = player1_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=260)
            # 渲染player2字体
            player2_txt = self.font.render("Second Player", True, self.WHITE)
            player2_pos = player2_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=300)
            # 电脑的水平
            level_txt = self.font.render("AI Level", True, self.WHITE)
            level_pos = level_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=340)

            # 显示
            self.screen.blit(title, title_pos)
            self.screen.blit(start_txt, start_pos)
            self.screen.blit(player1_txt, player1_pos)
            self.screen.blit(player2_txt, player2_pos)
            self.screen.blit(level_txt, level_pos)

            # 等待点击事件
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    # 碰撞检测
                    if start_pos.collidepoint(mouse_x, mouse_y):
                        return (player1, player2, level)
                    elif player1_pos.collidepoint(mouse_x, mouse_y):
                        player1 = self.get_chosen_player()
                    elif player2_pos.collidepoint(mouse_x, mouse_y):
                        player2 = self.get_chosen_player()
                    elif level_pos.collidepoint(mouse_x, mouse_y):
                        level = self.get_chosen_level()
            # 显示
            pygame.display.flip()

    # 展示winner
    def show_winner(self, player_color):
        font = pygame.font.SysFont("Courier New", 34,bold=True)
        if player_color == WHITE:
            msg = font.render("White player wins", True, self.RED)
        elif player_color == BLACK:
            msg = font.render("Black player wins", True, self.RED)
        else:
            msg = font.render("Tie !", True, self.RED)
        self.screen.blit(
            msg, msg.get_rect(
                centerx=self.screen.get_width() / 2, centery=120))
        pygame.display.flip()
        # 等待用户点击，以重新开始游戏
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    return

    # 展示player选择界面
    def get_chosen_player(self):
        while True:
            self.screen.fill(self.BACKGROUND)
            title_fnt = pygame.font.SysFont("Times New Roman", 34)
            title = title_fnt.render("Othello", True,self.WHITE)
            title_pos = title.get_rect(
                centerx=self.screen.get_width() / 2, centery=60)
            human_txt = self.font.render("Human", True, self.WHITE)
            human_pos = human_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=120)
            comp_txt = self.font.render("AI", True, self.WHITE)
            comp_pos = comp_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=360)

            self.screen.blit(title, title_pos)
            self.screen.blit(human_txt, human_pos)
            self.screen.blit(comp_txt, comp_pos)

            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    if human_pos.collidepoint(mouse_x, mouse_y):
                        return HUMAN
                    elif comp_pos.collidepoint(mouse_x, mouse_y):
                        return COMPUTER

            pygame.display.flip()

    # 获取选择的等级
    def get_chosen_level(self):
        while True:
            self.screen.fill(self.BACKGROUND)
            title_fnt = pygame.font.SysFont("Times New Roman", 34)
            title = title_fnt.render("Othello", True, self.WHITE)
            title_pos = title.get_rect(
                centerx=self.screen.get_width() / 2, centery=60)
            one_txt = self.font.render("Level 1", True, self.WHITE)
            one_pos = one_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=120)
            two_txt = self.font.render("Level 2", True, self.WHITE)
            two_pos = two_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=240)

            three_txt = self.font.render("Level 3", True, self.WHITE)
            three_pos = three_txt.get_rect(
                centerx=self.screen.get_width() / 2, centery=360)

            self.screen.blit(title, title_pos)
            self.screen.blit(one_txt, one_pos)
            self.screen.blit(two_txt, two_pos)
            self.screen.blit(three_txt, three_pos)

            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    if one_pos.collidepoint(mouse_x, mouse_y):
                        return 1
                    elif two_pos.collidepoint(mouse_x, mouse_y):
                        return 2
                    elif three_pos.collidepoint(mouse_x, mouse_y):
                        return 3

            pygame.display.flip()
            time.sleep(.05)

    def show_game(self):
        # 展示初始屏幕
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill(self.BACKGROUND)
        self.score_size = 50
        self.score1 = pygame.Surface((self.score_size, self.score_size))
        self.score2 = pygame.Surface((self.score_size, self.score_size))
        # 在背景上方显示
        self.screen.blit(self.background, (0, 0), self.background.get_rect())
        self.screen.blit(self.board_img, self.BOARD_POS,
                         self.board_img.get_rect())

        self.put_stone((3, 3), WHITE)
        self.put_stone((4, 4), WHITE)
        self.put_stone((3, 4), BLACK)
        self.put_stone((4, 3), BLACK)
        # 显示
        pygame.display.flip()

    # 绘制棋子
    def put_stone(self, pos, color):
        if pos == None:
            return

        # 翻转方向
        pos = (pos[1], pos[0])

        if color == BLACK:
            img = self.black_img
        elif color == WHITE:
            img = self.white_img
        else:
            img = self.tip_img

        x = pos[0] * self.SQUARE_SIZE + self.BOARD[0]
        y = pos[1] * self.SQUARE_SIZE + self.BOARD[1]

        # 把image显示在(x,y)处，（x,y)为左上角的坐标，rect为限定的范围
        self.screen.blit(img, (x, y), img.get_rect())
        pygame.display.flip()

    # 清除指定位置的棋子
    def clear_square(self, pos):
        """
        清除指定位置的棋子
        方法是在指定位置放上网格图片
        """
        pos = (pos[1], pos[0])
        x = pos[0] * self.SQUARE_SIZE + self.BOARD[0]
        y = pos[1] * self.SQUARE_SIZE + self.BOARD[1]
        self.screen.blit(self.clear_img, (x, y), self.clear_img.get_rect())
        pygame.display.flip()

    # 获取鼠标点击的位置，返回的是 pos
    def get_mouse_input(self):
        while True:
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()

                    # 超出边界，继续检测
                    if mouse_x > self.BOARD_SIZE + self.BOARD[0] or \
                       mouse_x < self.BOARD[0] or \
                       mouse_y > self.BOARD_SIZE + self.BOARD[1] or \
                       mouse_y < self.BOARD[1]:
                        continue

                    # 获取位置
                    position = ((mouse_x - self.BOARD[0]) // self.SQUARE_SIZE), \
                               ((mouse_y - self.BOARD[1]) // self.SQUARE_SIZE)
                    # 翻转位置
                    position = (position[1], position[0])
                    return position

                elif event.type == QUIT:
                    sys.exit(0)

            time.sleep(.05)

    # 更新屏幕，显示分数
    # board：棋盘，balcks：黑棋数，whites：白棋数，current_player_color：当前下棋的人的颜色
    def update(self, board, blacks, whites, current_player_color):
        for i in range(8):
            for j in range(8):
                if board[i][j] != 0:
                    self.put_stone((i, j), board[i][j])

        blacks_str = '%02d ' % int(blacks)
        whites_str = '%02d ' % int(whites)
        self.showScore(blacks_str, whites_str, current_player_color)
        pygame.display.flip()

    # 展示黑白棋的棋数
    def showScore(self, blackStr, whiteStr, current_player_color):
        # 黑棋（左边）的背景色
        black_background = self.BACKGROUND
        # 白棋（右边）的背景色
        white_background = self.BACKGROUND
        text = self.scoreFont.render(blackStr, True, self.WHITE,
                                     black_background)
        text2 = self.scoreFont.render(whiteStr, True, self.WHITE,
                                      white_background)

        self.screen.blit(self.black_img, (5,  self.SCREEN_SIZE[1] / 5), self.black_img.get_rect())
        self.screen.blit(self.white_img, (560,  self.SCREEN_SIZE[1] / 5), self.white_img.get_rect())
        self.screen.blit(text,
                         (self.BLACK_LAB_POS[0], self.BLACK_LAB_POS[1] + 40))
        self.screen.blit(text2,
                         (self.WHITE_LAB_POS[0], self.WHITE_LAB_POS[1] + 40))

    def wait_quit(self):
        # 等待用户关闭窗口
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                break
