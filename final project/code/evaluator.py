from config import BLACK, WHITE, EMPTY

# 各个位置的权重
board_weights=[
    [400, -30, 11, 8, 8, 11, -30, 400],
    [-30, -70, -4, 1, 1, -4, -70, -30],
    [11, -4, 2, 2, 2, 2, -4, 11],
    [8, 1, 2, -3, -3, 2, 1, 8],
    [8, 1, 2, -3, -3, 2, 1, 8],
    [11, -4, 2, 2, 2, 2, -4, 11],
    [-30, -70, -4, 1, 1, -4, -70, -30],
    [400, -30, 11, 8, 8, 11, -30, 400]
]

# 方向
direction=[(0, 1), (0, -1), (1, 1), (1, 0), (1, -1), (-1, 0), (-1, 1), (-1, -1)]

# 查看从x,y开始的八个方向中的四周是否都占满了棋子
def is_stable(board,x,y):
    for i in range(8):
        nx= x+direction[i][0]
        ny= y+direction[i][1]
        while 0<=nx<8 and 0<=ny<8:
            if board[nx][ny]==0:
                return False
            nx += direction[i][0]
            ny += direction[i][1]
    return True


def evaluate(color,board2):
    sideVal = [1, 2, 3, 4, 5, 6, 7, 8]
    corner_pos = [(0, 0), (0, 7), (7, 0), (7, 7)]
    # 棋子个数统计
    mystonecount = 0
    opstonecount = 0
    mystable = 0
    oppstable = 0
    # 分数统计
    score = 0
    rateeval = 0
    moveeval = 0
    sidestableeval = 0
    cornereval = 0
    stableeval = 0
    neareval = 0
    opponent_color = color

    best_weights=[0.190776976,0.173278458,0.125868592,0.118276044,0.115696808,0.135493866,0.140609256]

    if color == BLACK:
        opponent_color = WHITE
    else:
        opponent_color = BLACK

    # 计算黑棋和白棋各个点的权重
    for i in range(8):
        for j in range(8):
            if board2.board[i][j] == color:
                score += board_weights[i][j]
                mystonecount += 1
            elif board2.board[i][j] == opponent_color:
                score -= board_weights[i][j]
                opstonecount += 1

    # 根据黑白棋子的比例进行计算分数
    if mystonecount > opstonecount:
        rateeval = 100.0 * mystonecount / (mystonecount + opstonecount)
    elif mystonecount < opstonecount:
        rateeval = -100.0 * opstonecount / (mystonecount + opstonecount)
    else:
        rateeval = 0


    # 看四个角的周围是否有黑白棋
    # mynear表示四个角的周围的color颜色的棋子数
    # oppnear表示四个角的周围opponent_color颜色的棋子数
    # corner_pos存储棋盘四个角的位置
    # direction表示八个方向
    mynear = 0
    oppnear = 0
    for i in range(4):
        x = corner_pos[i][0]
        y = corner_pos[i][1]
        if board2.board[x][y] == 0:
            for j in range(8):
                nx = x + direction[j][0]
                ny = y + direction[j][1]
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if board2.board[nx][ny] == color:
                        mynear += 1
                    elif board2.board[nx][ny] == opponent_color:
                        oppnear += 1
    neareval = -24.5 * (mynear - oppnear)

    # 通过我方和对方可下的地方的数量的比值来估计
    # 我方可以下的地方的数量
    mymove = len(board2.get_valid_moves(color))
    # 对方可以下的地方的数量
    opmove = len(board2.get_valid_moves(opponent_color))
    if mymove == 0:
        moveeval = -450
    elif opmove == 0:
        moveeval = 150
    elif mymove > opmove:
        moveeval = (100.0 * mymove) / (mymove + opmove)
    elif mymove < opmove:
        moveeval = -100.0 * opmove / (mymove + opmove)
    else:
        moveeval = 0

    # 某个棋子的八个方向是否被占满了棋子
    for i in range(8):
        for j in range(8):
            # is_stable判断该枚棋子是否稳定
            if board2.board[i][j] != 0 and is_stable(board2.board, i, j):
                if board2.board[i][j] == color:
                    mystable += 1
                else:
                    oppstable += 1
    stableeval = 12.5 * (mystable - oppstable)

    # 我方边界点数量
    myside = 0
    # 对方边界点数量
    opside = 0
    # 四个角的我方棋子数
    mycorner = 0
    # 四个角对方棋子数
    opcorner = 0
    for i in range(4):
        if board2.board[corner_pos[i][0]][corner_pos[i][1]] == color:
            mycorner += 1
            for j in range(8):
                # 沿着列
                if board2.board[corner_pos[i][0]][j] == color:
                    myside += sideVal[j]
                else:
                    break
            for j in range(8):
                # 沿着行
                if board2.board[j][corner_pos[i][1]] == color:
                    myside += sideVal[j]
                else:
                    break
        elif board2.board[corner_pos[i][0]][corner_pos[i][1]] == opponent_color:
            opcorner += 1
            for j in range(8):
                # 沿着列
                if board2.board[corner_pos[i][0]][j] == opponent_color:
                    opside += sideVal[j]
                else:
                    break
            for j in range(8):
                # 沿着行
                if board2.board[j][corner_pos[i][1]] == opponent_color:
                    opside += sideVal[j]
                else:
                    break
    sidestableeval = 2.5 * (myside - opside)
    cornereval = 25 * (mycorner - opcorner)

    return best_weights[0]*score+best_weights[1]*moveeval+best_weights[2]*sidestableeval+best_weights[3]*cornereval\
           +best_weights[4]*rateeval+best_weights[5]*stableeval+best_weights[6]*neareval
