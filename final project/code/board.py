from config import WHITE, BLACK, EMPTY
from copy import deepcopy

class Board:
    def __init__(self):
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[3][3] = WHITE
        self.board[4][4] = WHITE
        self.valid_moves = []

    def __getitem__(self, i, j):
        return self.board[i][j]


    def lookup(self, row, column, color):
        """
        从(row,column)方向去寻找八个方向，是否有可以落子的点
        """
        if color == BLACK:
            other = WHITE
        else:
            other = BLACK

        places = []

        if (row < 0 or row > 7 or column < 0 or column > 7):
            return places

        # 搜索八个方向，寻找可能的落子点
        # 北方
        i = row - 1
        if (i >= 0 and self.board[i][column] == other):
            i = i - 1
            while (i >= 0 and self.board[i][column] == other):
                i = i - 1
            if (i >= 0 and self.board[i][column] == 0):
                places = places + [(i, column)]

        # 东北方向
        i = row - 1
        j = column + 1
        if (i >= 0 and j < 8 and self.board[i][j] == other):
            i = i - 1
            j = j + 1
            while (i >= 0 and j < 8 and self.board[i][j] == other):
                i = i - 1
                j = j + 1
            if (i >= 0 and j < 8 and self.board[i][j] == 0):
                places = places + [(i, j)]

        # 东方
        j = column + 1
        if (j < 8 and self.board[row][j] == other):
            j = j + 1
            while (j < 8 and self.board[row][j] == other):
                j = j + 1
            if (j < 8 and self.board[row][j] == 0):
                places = places + [(row, j)]

        # 东南方向
        i = row + 1
        j = column + 1
        if (i < 8 and j < 8 and self.board[i][j] == other):
            i = i + 1
            j = j + 1
            while (i < 8 and j < 8 and self.board[i][j] == other):
                i = i + 1
                j = j + 1
            if (i < 8 and j < 8 and self.board[i][j] == 0):
                places = places + [(i, j)]

        # 南方
        i = row + 1
        if (i < 8 and self.board[i][column] == other):
            i = i + 1
            while (i < 8 and self.board[i][column] == other):
                i = i + 1
            if (i < 8 and self.board[i][column] == 0):
                places = places + [(i, column)]

        # 西南方向
        i = row + 1
        j = column - 1
        if (i < 8 and j >= 0 and self.board[i][j] == other):
            i = i + 1
            j = j - 1
            while (i < 8 and j >= 0 and self.board[i][j] == other):
                i = i + 1
                j = j - 1
            if (i < 8 and j >= 0 and self.board[i][j] == 0):
                places = places + [(i, j)]

        # 西方
        j = column - 1
        if (j >= 0 and self.board[row][j] == other):
            j = j - 1
            while (j >= 0 and self.board[row][j] == other):
                j = j - 1
            if (j >= 0 and self.board[row][j] == 0):
                places = places + [(row, j)]

        # 西北方
        i = row - 1
        j = column - 1
        if (i >= 0 and j >= 0 and self.board[i][j] == other):
            i = i - 1
            j = j - 1
            while (i >= 0 and j >= 0 and self.board[i][j] == other):
                i = i - 1
                j = j - 1
            if (i >= 0 and j >= 0 and self.board[i][j] == 0):
                places = places + [(i, j)]
        return places


    def get_valid_moves(self, color):
        """
        获取可以落color这个颜色的棋子的地方
        方法是搜索棋盘上每一个点
        返回元素为位置元组的列表
        """

        if color == BLACK:
            other = WHITE
        else:
            other = BLACK
        # 获取可以下的地方
        places = []
        # 遍历棋盘上每一个点
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == color:
                    places = places + self.lookup(i, j, color)
        places = list(set(places))
        self.valid_moves = places
        return places

    def apply_move(self, move, color):
        """
        确定是否行动是正确的以及
        将落得子显示到棋盘上
        """
        # 如果是有效的移动
        if move in self.valid_moves:
            self.board[move[0]][move[1]] = color
            for i in range(1, 9):
                # 向九个方向检测，翻转棋盘
                self.flip(i, move, color)

    def flip(self, direction, position, color):
        """
        将棋盘上direction方向的棋子变换成color颜色的
        """

        if direction == 1:
            # 北方
            row_inc = -1
            col_inc = 0
        elif direction == 2:
            # 东北
            row_inc = -1
            col_inc = 1
        elif direction == 3:
            # 东方
            row_inc = 0
            col_inc = 1
        elif direction == 4:
            # 东南
            row_inc = 1
            col_inc = 1
        elif direction == 5:
            # 南方
            row_inc = 1
            col_inc = 0
        elif direction == 6:
            # 西南方
            row_inc = 1
            col_inc = -1
        elif direction == 7:
            # 西方
            row_inc = 0
            col_inc = -1
        elif direction == 8:
            # 西北
            row_inc = -1
            col_inc = -1

        places = []
        i = position[0] + row_inc
        j = position[1] + col_inc

        if color == WHITE:
            other = BLACK
        else:
            other = WHITE

        if i in range(8) and j in range(8) and self.board[i][j] == other:
            # 确保至少有一个棋子可以翻转
            places = places + [(i, j)]
            i = i + row_inc
            j = j + col_inc
            while i in range(8) and j in range(8) and self.board[i][j] == other:
                # 搜索更多可以翻转的棋子
                places = places + [(i, j)]
                i = i + row_inc
                j = j + col_inc
            if i in range(8) and j in range(8) and self.board[i][j] == color:
                # 遇到了本颜色的棋子，翻转
                for pos in places:
                    # 翻转颜色
                    self.board[pos[0]][pos[1]] = color

    def get_changes(self):
        # 返回棋盘上的黑白棋个数
        whites, blacks, empty = self.count_stones()
        return (self.board, blacks, whites)

    # 判断游戏是否结束
    def game_ended(self):
        # 获取白棋、黑棋、空的数目
        whites, blacks, empty = self.count_stones()
        # 如果没有地方可以下返回True
        if whites == 0 or blacks == 0 or empty == 0:
            return True

        # 如果双方都没有可以下的地方就返回True
        if self.get_valid_moves(BLACK) == [] and \
        self.get_valid_moves(WHITE) == []:
            return True
        return False

    def print_board(self):
        for i in range(8):
            print(i, ' |', end=' ')
            for j in range(8):
                if self.board[i][j] == BLACK:
                    print('B', end=' ')
                elif self.board[i][j] == WHITE:
                    print('W', end=' ')
                else:
                    print(' ', end=' ')
                print('|', end=' ')
            print()

    # 获取棋盘上的白棋数目，黑棋数目，空的数目
    def count_stones(self):
        whites = 0
        blacks = 0
        empty = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == WHITE:
                    whites += 1
                elif self.board[i][j] == BLACK:
                    blacks += 1
                else:
                    empty += 1
        return whites, blacks, empty

    def compare(self, otherBoard):
        """
        返回一个棋盘，这个棋盘只显示那些两个棋盘不同的棋子
        """
        diffBoard = Board()
        diffBoard.board[3][4] = 0
        diffBoard.board[3][3] = 0
        diffBoard.board[4][3] = 0
        diffBoard.board[4][4] = 0
        for i in range(8):
            for j in range(8):
                if otherBoard.board[i][j] != self.board[i][j]:
                    diffBoard.board[i][j] = otherBoard.board[i][j]
        return otherBoard

    def get_adjacent_count(self, color):
        """
            返回color颜色的棋子周围空的棋子数
        """
        adjCount = 0
        for x, y in [(a, b) for a in range(8) for b in range(8) if self.board[a][b] == color]:
            for i, j in [(a, b) for a in [-1, 0, 1] for b in [-1, 0, 1]]:
                if 0 <= x + i <= 7 and 0 <= y + j <= 7:
                    if self.board[x + i][y + j] == EMPTY:
                        adjCount += 1
        return adjCount


    def next_states(self, color):
        """
            给定玩家的颜色，返回所有可以下的点组成的棋盘
        """
        valid_moves = self.get_valid_moves(color)
        for move in valid_moves:
            newBoard = deepcopy(self)
            newBoard.apply_move(move, color)
            yield (newBoard,move)
