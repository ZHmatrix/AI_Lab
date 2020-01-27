from config import BLACK, WHITE, EMPTY
import board as board_file
import copy
import player
import random
import pandas as pd

board2=[]

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

# 定义染色体
class chromosome:
    def __init__(self,weight_len):
        # 适应度
        self.adapt=0
        # 向量
        self.weight=[random.random() for i in range(weight_len)]
        # 归一化
        sum_weight=sum(self.weight)
        self.weight=[weight/sum_weight for weight in self.weight]


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

# 遗传算法
def ga(dim1,dim2, dim3, dim4, dim5, dim6, dim7):
    dim=[dim1,dim2,dim3,dim4,dim5,dim6,dim7]

    # 权重中元素个数
    weight_len=7
    # 染色体条数
    chromo_num=10
    # 随机生成10条染色体,并初始化适应度为0
    randweight=[chromosome(weight_len) for i in range(chromo_num)]

    # 选取阈值染色体
    limit_chromo=randweight[0]

    # 进行N轮遗传
    for _ in range(chromo_num):
        # 两两进行比较得分，赢的适应度+1
        for i in range(chromo_num):
            for j in range(i+1,chromo_num):
                res1=sum([randweight[i].weight[k]*dim[k] for k in range(weight_len)])
                res2=sum([randweight[j].weight[k]*dim[k] for k in range(weight_len)])
                if res1 > res2:
                    randweight[i].adapt += 1
                else:
                    randweight[j].adapt += 1

        # 把比阈值染色体差的染色体的适应度减少
        for j in range(chromo_num):
            res1=sum([limit_chromo.weight[k]*dim[k] for k in range(weight_len)])
            res2=sum([randweight[j].weight[k] * dim[k] for k in range(weight_len)])
            if res1>res2:
                randweight[j].adapt-=2

        # 根据适应度大小从大到小排序
        randweight.sort(key=lambda e:e.adapt,reverse=True)

        # 选取适应度最大的作为阈值染色体
        limit_chromo=randweight[0]

        # 最后一代就停止
        if _==chromo_num-1:
            break

        # 新的染色体列表
        new_randweight=[]

        # 选择前copy_num个进行复制
        copy_num=3
        # 以90%的概率复制
        if random.random()<=0.9:
            for i in range(copy_num):
                chromo=randweight[i]
                chromo.adapt=0
                new_randweight.append(chromo)
        else:
            # 否则随机生成 copy_num个
            for i in range(copy_num):
                new_randweight.append(chromosome(weight_len))

        # 交叉部分, 12,13,14,23,24,34 交叉，产生交叉后新的6条染色体
        for i in range(4):
            for j in range(i+1,4):
                tmp_chromo=chromosome(weight_len)
                tmp_chromo.weight=[]
                rand2=random.randint(1,2)
                tmpsum=0
                # 交叉点
                cut=random.randint(0,weight_len-1)
                for k in range(weight_len):
                    if rand2==0:
                        if k<=cut:
                            tmp_chromo.weight.append(randweight[i].weight[k])
                            tmpsum+=randweight[i].weight[k]
                        else:
                            tmp_chromo.weight.append(randweight[j].weight[k])
                            tmpsum += randweight[j].weight[k]
                    else:
                        if k <= cut:
                            tmp_chromo.weight.append(randweight[j].weight[k])
                            tmpsum += randweight[j].weight[k]
                        else:
                            tmp_chromo.weight.append(randweight[i].weight[k])
                            tmpsum += randweight[i].weight[k]
                for k in range(weight_len):
                    tmp_chromo.weight[k]/=tmpsum
                new_randweight.append(tmp_chromo)

        # 变异
        tmp_chromo=randweight[0]
        tmp_chromo.adapt=0
        # 随机生成一个替换的数
        replace_num=random.random()
        # 随机生成一个替换的位置
        replace_index=random.randint(0,6)
        weight_sum=0
        for k in range(weight_len):
            if k==replace_index:
                weight_sum+=replace_num
            else:
                weight_sum+=tmp_chromo.weight[k]
        for k in range(weight_len):
            if k==replace_index:
                tmp_chromo.weight[k]=replace_num/weight_sum
            else:
                tmp_chromo.weight[k]/=weight_sum
        new_randweight.append(tmp_chromo)
        randweight=new_randweight
    # 选取适应度最大的染色体
    return randweight[0]




def train_evaluate(color):
    sideVal=[1,2,3,4,5,6,7,8]
    corner_pos=[(0,0),(0,7),(7,0),(7,7)]
    # 棋子个数统计
    mystonecount=0
    opstonecount=0
    mystable=0
    oppstable=0
    # 分数统计
    score=0
    rateeval=0
    moveeval=0
    sidestableeval=0
    cornereval=0
    stableeval=0
    neareval=0
    opponent_color=color

    if color==BLACK:
        opponent_color=WHITE
    else:
        opponent_color=BLACK

    # 计算黑棋和白棋各个点的权重
    for i in range(8):
        for j in range(8):
            if board2.board[i][j]==color:
                score+=board_weights[i][j]
                mystonecount+=1
            elif board2.board[i][j]==opponent_color:
                score-=board_weights[i][j]
                opstonecount+=1

    # 根据黑白棋子的比例进行计算分数
    if mystonecount>opstonecount:
        rateeval=100.0 * mystonecount/(mystonecount+opstonecount)
    elif mystonecount<opstonecount:
        rateeval=-100.0*opstonecount/(mystonecount+opstonecount)
    else:
        rateeval=0

    mynear=0
    oppnear=0
    # 看四个角的周围是否有黑白棋
    # mynear表示四个角的周围的color颜色的棋子数
    # oppnear表示四个角的周围opponent_color颜色的棋子数
    for i in range(4):
        x=corner_pos[i][0]
        y=corner_pos[i][1]
        if board2.board[x][y]==0:
            for j in range(8):
                nx=x+direction[j][0]
                ny=y+direction[j][1]
                if 0<=nx<8 and 0<=ny<8:
                    if board2.board[nx][ny]==color:
                        mynear+=1
                    elif board2.board[nx][ny]==opponent_color:
                        oppnear+=1

    neareval=-24.5*(mynear-oppnear)

    # 通过我方和对方可下的地方的数量的比值来估计
    # 我方可以下的地方的数量
    mymove=len(board2.get_valid_moves(color))
    # 对方可以下的地方的数量
    opmove=len(board2.get_valid_moves(opponent_color))
    if mymove==0:
        moveeval=-450
    elif opmove==0:
        moveeval=150
    elif mymove>opmove:
        moveeval=(100.0*mymove)/(mymove+opmove)
    elif mymove<opmove:
        moveeval=-100.0*opmove/(mymove+opmove)
    else:
        moveeval=0

    # 某个棋子的八个方向是否被占满了棋子
    for i in range(8):
        for j in range(8):
            if board2.board[i][j]!=0 and is_stable(board2.board,i,j):
                if board2.board[i][j]==color:
                    mystable+=1
                else:
                    oppstable+=1

    stableeval=12.5*(mystable-oppstable)
    myside=0
    opside=0
    # 四个角的我方棋子数
    mycorner=0
    opcorner=0
    for i in range(4):
        if board2.board[corner_pos[i][0]][corner_pos[i][1]]==color:
            mycorner+=1
            for j in range(8):
                if board2.board[corner_pos[i][0]][j]==color:
                    myside+=sideVal[j]
                else:
                    break
            for j in range(8):
                if board2.board[j][corner_pos[i][1]]==color:
                    myside+=sideVal[j]
                else:
                    break
        elif board2.board[corner_pos[i][0]][corner_pos[i][1]]==opponent_color:
            opcorner+=1
            for j in range(8):
                if board2.board[corner_pos[i][0]][j]==opponent_color:
                    opside+=sideVal[j]
                else:
                    break
            for j in range(8):
                if board2.board[j][corner_pos[i][1]]==opponent_color:
                    opside+=sideVal[j]
                else:
                    break
    sidestableeval=2.5*(myside-opside)
    cornereval=25*(mycorner-opcorner)

    return ga(score, moveeval, sidestableeval, cornereval, rateeval, stableeval, neareval)

def ave(weight1,weight2):
    return_weight=[]
    weight_sum=0
    for i in range(len(weight1)):
        weight_sum+=(weight1[i]+weight2[i])/2
        return_weight.append((weight1[i]+weight2[i])/2)
    for i in range(len(return_weight)):
        return_weight[i]/=weight_sum
    return return_weight

def ave2(weights):
    # 向量的数目
    weight_num=len(weights)
    best_weight=[]
    for j in range(len(weights[0])):
        weight_sum = 0
        for i in range(weight_num):
            weight_sum+=weights[i][j]
        best_weight.append(weight_sum/weight_num)
    return best_weight

if __name__=='__main__':
    df=pd.DataFrame(columns=['e1','e2','e3','e4','e5','e6','e7'])
    # 初始化棋盘
    board = board_file.Board()
    # 训练500轮
    train_iter=100

    # 存储每一步返回的最优权重
    weights=[]
    # 先手
    random_player1=player.RandomPlayer(color=BLACK)
    # 后手
    random_player2=player.RandomPlayer(color=WHITE)
    # 获取棋盘
    random_player1.get_current_board(board)
    random_player2.get_current_board(board)

    current_player=random_player1
    weighttmp=[]
    for _ in range(train_iter):
        print('Iter {}'.format(_))
        cnt=0
        # 下60个子
        while not board.game_ended():
            cnt+=1
            # 深拷贝一份棋盘
            board2=copy.deepcopy(board)
            # 每两轮取均值
            if cnt%2==0:
                weights.append(ave(train_evaluate(current_player.color).weight,weighttmp))
            else:
               weighttmp = train_evaluate(current_player.color).weight

            # 进行行动
            score,board,move =current_player.get_move()

            # 交换身份
            if current_player==random_player1:
                current_player=random_player2
            else:
                current_player=random_player1

        now_weights=ave2(weights)
        df=df.append({'e1':now_weights[0],'e2':now_weights[1],'e3':now_weights[2],'e4':now_weights[3],
                      'e5': now_weights[4], 'e6': now_weights[5],'e7':now_weights[6]},ignore_index=True)
        # print('weights: {}'.format(now_weights))
        # 初始化，准备下一轮迭代
        board=board_file.Board()
        random_player1.get_current_board(board)
        random_player2.get_current_board(board)
        current_player=random_player1

    best_weight=ave2(weights)
    print("The best weight is:{}".format(best_weight))

    df.index=df.index.map(lambda x:x+1)
    df.to_excel('weights.xlsx')