from evaluator import evaluate
from config import WHITE, BLACK
import random


def change_color(color):
    '''改变颜色
    '''
    if color == BLACK:
        return WHITE
    else:
        return BLACK


class Minimax(object):
    '''实现带有alpha-beta剪枝的minimax搜索
    '''
    INFINITY = 100000
    def __init__(self, evaluate):
        # 这个是一个评估函数
        self.evaluate = evaluate

    def minimax(self, board, depth, player, opponent,
                alfa=-INFINITY, beta=INFINITY):
        '''
        board: 当前棋盘，depth：搜索深度，
        player：当前落子棋子的颜色，opponent：落子棋子的颜色： 
        '''
        bestmove=(0,0)
        bestChild = board
        # 到达了叶子节点
        if depth == 0:
            return ( self.evaluate(player,board), board, bestmove )
        for (child,move) in board.next_states(player):
            # 返回分数，最优子节点，最优分数
            score, newChild,_= self.minimax(
                child, depth - 1, opponent, player, -beta, -alfa)
            score = -score
            if score > alfa:
                alfa= score
                bestChild = child
                bestmove= move
            # 剪枝
            if beta <= alfa:
                break
        return self.evaluate(player, board), bestChild, bestmove


class Human:
    """ 人类玩家 """
    def __init__(self, gui, color="black"):
        self.color = color
        self.gui = gui
        self.name='Human'

    def get_move(self):
        """
         使用GUI去操纵鼠标
        """
        # 获取有效的落子点
        validMoves = self.current_board.get_valid_moves(self.color)
        # 获取鼠标点击事件，直到获得一个有效的落子点
        while True:
            move = self.gui.get_mouse_input()
            if move in validMoves:
                break
        # 落子并翻转对方颜色棋子为己方棋子
        self.current_board.apply_move(move, self.color)
        return 0, self.current_board, move

    def get_current_board(self, board):
        self.current_board = board


class Computer(object):
    # 电脑
    def __init__(self, color, prune=3):
        # 深度限制
        self.depthLimit = prune
        # 评估函数
        self.minimaxObj = Minimax(evaluate)
        self.color = color
        self.name='Computer'

    # 更新当前的棋盘
    def get_current_board(self, board):
        self.current_board = board

    # 返回分数，棋盘
    def get_move(self):
        score,board,move=self.minimaxObj.minimax(self.current_board, self.depthLimit, self.color,
                                       change_color(self.color))
        return score,board,move

# 随机下棋的人
class RandomPlayer (Computer):
    def get_move(self):
        x=0
        if len(self.current_board.get_valid_moves(self.color))>0:
            x = random.sample(self.current_board.get_valid_moves(self.color), 1)[0]
            self.current_board.apply_move(x,self.color)
        return 0,self.current_board,x
