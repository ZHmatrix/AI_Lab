from graphics import *

class Gomoku(object):
    """五子棋游戏类

    Attributes:
        player1: list 玩家1(AI)落子的坐标，每个元素为二元组
        player2: list 玩家2(人)落子的坐标，每个元素为二元组
        two_player: list 两名玩家落子的所有坐标
        first_hand: 谁先手
        next_point: 搜索到下一步最应该下的位置
        DEPTH: int 搜索深度,奇数
        shape_score: 棋型和对应分数
        ROW_SPACE,COLUMN,ROW: 绘图行间距 行数 列数
    """    
    def __init__(self, first_hand):
        """初始化
        """
        start1, start2 = [(4,5),(5,6)], [(5,5),(6,5)]
        self.player1 = []  
        self.player2 = [] 
        if first_hand:
            self.player1, self.player2 = start1, start2
        else:
            self.player1, self.player2 = start2, start1
        self.two_player = [(5,5),(6,5),(4,5),(5,6)] 
        self.first_hand = first_hand
        self.next_point = [0, 0]  
        self.DEPTH = 3  
        # 棋型和对应的分数
        self.shape_score = {
            (0, 1, 1, 0, 0) : 50,
            (0, 0, 1, 1, 0) : 50, 
            (1, 1, 0, 1, 0) : 200,
            (0, 0, 1, 1, 1) : 200, 
            (1, 1, 1, 0, 0) : 200, 
            (0, 1, 1, 1, 0) : 200,
            (1, 1, 1, 0, 1) : 1000, 
            (1, 1, 0, 1, 1) : 1000, 
            (1, 0, 1, 1, 1) : 1000, 
            (1, 1, 1, 1, 0) : 1000, 
            (0, 1, 1, 1, 1) : 1000, 
            (1, 1, 1, 1, 1) : 10000, 
            (0, 1, 0, 1, 1, 0) : 1000, 
            (0, 0, 1, 1, 1, 0) : 1000,
            (0, 1, 1, 1, 0, 0) : 1000,  
            (0, 1, 1, 0, 1, 0) : 1000, 
            (0, 1, 1, 1, 1, 0) : 5000, 
        }
        self.ROW_SPACE = 40
        self.COLUMN = 10
        self.ROW = 10

    def one_search(self):
        """一次搜索，搜索depth步后的最优落子点
        Returns:
            元组(x, y)，代表接下来应该下的坐标
        """
        score = self.alpha_beta(True, self.DEPTH, -999999999, 999999999)
        print("得分：", score)
        return self.next_point[0], self.next_point[1]

    def get_children(self):
        """获取子状态节点，即接下来可以落子的点坐标
        Returns:
            列表，每个元素为接下来可以落子的点坐标
        """
        # 整个棋盘的点坐标
        all_points = set([(i, j) for j in range(self.ROW+1) for i in range(self.COLUMN+1)])
        # 找出当前没有落子的点
        accessible_pos = all_points.difference(set(self.two_player))
        return list(accessible_pos)

    def alpha_beta(self, turn, depth, alpha, beta):
        """负值极大算法搜索 + alpha_beta剪枝
        Args:
            turn 轮到谁走
            depth 搜索深度
            alpha beta alpha-beta剪枝的两个值
        Returns:
            当前状态棋盘的评估分数
        """

        # 到达搜索深度or游戏结束返回当前得分
        if depth == 0 or self.win(self.player1) or self.win(self.player2):
            return self.evaluation(turn)

        # 获取子状态节点，即当前可以落子的位置(棋盘上为空的地方)
        next_accessible_pos = self.get_children()
        # 遍历每一个子状态节点
        for next_pos in next_accessible_pos:
            # 忽略所有周围没有相邻棋子的点
            if not self.has_neightnor(next_pos):
                continue
            
            # 假设将子落在next_pos
            if turn:
                self.player1.append(next_pos)
            else:
                self.player2.append(next_pos)
            self.two_player.append(next_pos)

            # 递归获取这个在这个位置落子的分数
            value = - self.alpha_beta(not turn, depth - 1, -beta, -alpha)

            # 还原状态
            if turn:
                self.player1.remove(next_pos)
            else:
                self.player2.remove(next_pos)
            self.two_player.remove(next_pos)

            # alpha-beta剪枝
            if value > alpha:
                alpha = value
                # 当前节点需要落的下一个子的坐标
                if depth == self.DEPTH:
                    self.next_point[0] = next_pos[0]
                    self.next_point[1] = next_pos[1]
                # 剪枝
                if alpha >= beta:
                    return beta

        return alpha

    def has_neightnor(self, point):
        """判断一个点周围八个位置是否有棋子
        Args:
            point 二元组，点坐标
        Returns:
            是或否
        """        
        x, y = point[0], point[1]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (x + dx, y + dy) in self.two_player:
                    return True
        return False

    def evaluation(self, turn):
        """评估函数
        Args:
            turn Boolean 轮到谁
        Returns:
            当前棋盘的分数
        """    
        # 分别计算我方和敌方的棋盘得分
        score1 = self.compute_player_score(self.player1, self.player2)
        score2 = self.compute_player_score(self.player2, self.player1)
        # 我方棋盘得分 - weight×地方得分 此处 weight = 0.5
        return score1 - 0.5*score2 if turn else score2 - 0.5*score1

    def compute_player_score(self, me, enemy):
        """计算一个玩家的得分
        Args:
            me, enemy 坐标列表
        Returns:
            该玩家的得分
        """    
        # 存储已经计算过的得分和形状,避免重复计算
        used_score_shape = set() 
        score = 0
        # 计算我方每个棋子四个方向上的得分之和作为最后的score
        for point in me: 
            x, y = point[0], point[1]
            # 计算横，竖，正斜，反斜四个方向的分数之和
            for (dir_x, dir_y) in [(0,1),(1,0),(1,1),(1,-1)]:
                score += self.compute_dir_score(x, y, dir_x, dir_y, me, enemy, used_score_shape)        
        return score

    def compute_dir_score(self, x, y, dir_x, dir_y, me, enemy, used_score_shape):
        """计算一个棋子给定方向上的得分
        Args:
            x, y 棋子坐标
            dir_x, dir_y 方向 分为横向,纵向,正斜,反斜四个方向,用方向向量表示为
                        (0,1)(1,0)(1,1)(1,-1)
            me, enemy 坐标列表
            used_score_shape 已经计算过的得分和形状 set
        Returns:
            该棋子指定方向上的得分
        """    
        # 判断该棋子在这个方向上是否已有得分
        for sc_points_dir in used_score_shape: 
            _, points, dir = sc_points_dir[0], sc_points_dir[1], sc_points_dir[2]           
            for point in points:
                # 棋子坐标相同且得分方向相同，则忽略
                if (x, y) == (point[0], point[1]) and (dir_x, dir_y) == (dir[0], dir[1]):
                    return 0

        # 该方向上得分的最大值和形状，格式为(分数,(五个棋子的坐标),方向)
        max_sc_points_dir = (0, None, None)
        # 在落子点周围按照指定方向查找棋子形状
        for offset in range(-5, 1):
            shape = []
            for i in range(0, 6):
                if (x + (i + offset) * dir_x, y + (i + offset) * dir_y) in enemy:
                    shape.append(-1)
                elif (x + (i + offset) * dir_x, y + (i + offset) * dir_y) in me:
                    shape.append(1)
                else:
                    shape.append(0)
            # shape5和6分别为该棋子该方向上五个棋子的棋型和六个棋子的棋型
            # 因为得分棋型中分为五子型和六子型
            shape5 = (shape[0], shape[1], shape[2], shape[3], shape[4])
            shape6 = (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5])
            # 判断是否有能够得分的棋型 shape_score是棋型和对应得分的dict
            for shape in self.shape_score:
                score = self.shape_score[shape]
                # 如果有得分棋型，则取该方向上得分最高的棋型
                if shape5 == shape or shape6 == shape:
                    if score > max_sc_points_dir[0]:
                        # max_sc_points_dir格式为(分数,(五个棋子的坐标),方向)
                        max_sc_points_dir = (
                            score, (
                                (x + (0+offset) * dir_x, y + (0+offset) * dir_y),
                                (x + (1+offset) * dir_x, y + (1+offset) * dir_y),
                                (x + (2+offset) * dir_x, y + (2+offset) * dir_y),
                                (x + (3+offset) * dir_x, y + (3+offset) * dir_y),
                                (x + (4+offset) * dir_x, y + (4+offset) * dir_y)
                            ), 
                            (dir_x, dir_y)
                        )

        # 将该得分棋型记为已计算过得分
        if max_sc_points_dir[1] is not None:
            used_score_shape.add(max_sc_points_dir)
        # 返回最大分数
        return max_sc_points_dir[0]

    def win(self, player):
        """判断该玩家是否已经获胜
        Args:
            player 坐标列表 代表一个玩家
        Returns:
            是否获胜 是为true 否为false
        """    
        # 遍历棋子 判断其四个方向上是否能够构成五子
        for x in range(self.COLUMN):
            for y in range(self.ROW):
                # 纵向
                if y < self.ROW - 4:
                    if (x, y) in player and (x, y + 1) in player and (x, y + 2) in player and (
                        x, y + 3) in player and (x, y + 4) in player:
                        return True
                # 横向
                if x < self.COLUMN - 4:
                    if (x, y) in player and (x + 1, y) in player and (x + 2, y) in player and (
                            x + 3, y) in player and (x + 4, y) in player:
                        return True
                # 正斜向
                if x < self.COLUMN - 4 and y < self.ROW - 4:
                    if (x, y) in player and (x + 1, y + 1) in player and (
                            x + 2, y + 2) in player and (x + 3, y + 3) in player and (x + 4, y + 4) in player:
                        return True
                # 反斜向
                if x < self.COLUMN - 4 and y > 3:
                    if (x, y) in player and (x + 1, y - 1) in player and (
                            x + 2, y - 2) in player and (x + 3, y - 3) in player and (x + 4, y - 4) in player:
                        return True
        return False

    def game_window(self):
        """画出游戏窗口
        Returns:
            游戏窗口
        """    
        window = GraphWin("16327143仲逊 五子棋", self.ROW_SPACE * self.COLUMN, self.ROW_SPACE * self.ROW)
        # 棕色棋盘
        bgcolor = color_rgb(200, 100, 0)
        window.setBackground(bgcolor)
        
        i1 = 0
        # 画棋盘
        while i1 <= self.ROW_SPACE * self.COLUMN:
            l = Line(Point(i1, 0), Point(i1, self.ROW_SPACE * self.COLUMN))
            l.draw(window)
            i1 = i1 + self.ROW_SPACE
        
        i2 = 0
        while i2 <= self.ROW_SPACE * self.ROW:
            l = Line(Point(0, i2), Point(self.ROW_SPACE * self.ROW, i2))
            l.draw(window)
            i2 = i2 + self.ROW_SPACE

        for pos in self.player1:
            piece = Circle(Point(self.ROW_SPACE * pos[0], self.ROW_SPACE * pos[1]), 16)
            piece.setFill('white')
            piece.draw(window)
        
        for pos in self.player2:
            piece = Circle(Point(self.ROW_SPACE * pos[0], self.ROW_SPACE * pos[1]), 16)
            piece.setFill('black')
            piece.draw(window)
        
        return window

    def start(self):
        window = self.game_window()
        # 判断谁先手
        change_turn = 0 if self.first_hand else 1
        # 棋局是否还在进行中
        ongoing = True

        while ongoing:

            if change_turn % 2 == 1:
                pos = self.one_search()

                if pos in self.two_player:
                    message = Text(Point(200, 200), "该位置不可落子" + str(pos[0]) + "," + str(pos[1]))
                    message.draw(window)
                    ongoing = False

                self.player1.append(pos)
                self.two_player.append(pos)

                piece = Circle(Point(self.ROW_SPACE * pos[0], self.ROW_SPACE * pos[1]), 16)
                piece.setFill('white')
                piece.draw(window)

                if self.win(self.player1):
                    message = Text(Point(100, 100), "white win.")
                    message.draw(window)
                    ongoing = False
                change_turn = change_turn + 1

            else:
                p2 = window.getMouse()
                if not ((round((p2.getX()) / self.ROW_SPACE), round((p2.getY()) / self.ROW_SPACE)) in self.two_player):

                    a2 = round((p2.getX()) / self.ROW_SPACE)
                    b2 = round((p2.getY()) / self.ROW_SPACE)
                    self.player2.append((a2, b2))
                    self.two_player.append((a2, b2))

                    piece = Circle(Point(self.ROW_SPACE * a2, self.ROW_SPACE * b2), 16)
                    piece.setFill('black')
                    piece.draw(window)
                    if self.win(self.player2):
                        message = Text(Point(100, 100), "black win.")
                        message.draw(window)
                        ongoing = False

                    change_turn = change_turn + 1

        message = Text(Point(100, 120), "Click anywhere to quit.")
        message.draw(window)
        window.getMouse()
        window.close()

if __name__ == "__main__":
    is_first = input("Do you want to first_hand?[y/n]:")
    first_hand = True if is_first == 'y' else False
    test = Gomoku(first_hand)
    test.start()