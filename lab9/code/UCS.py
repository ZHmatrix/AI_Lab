from queue import PriorityQueue
import time


def get_maze(file_path):
    """
    用给定文件建立迷宫
    Args:
        file_path: 字符串 代表文件路径

    Returns:
        maze: 二维列表 代表迷宫 
              '1'墙 '0'通路 'S'起点 'E'终点
    """
    fp = open(file_path)
    maze = [line.strip() for line in fp.readlines()]
    fp.close()
    maze = [list(line) for line in maze]
    return maze

   
def ucs(maze, start, end):
    """
    给定 迷宫 起点 终点 进行一致代价搜索
    Args:
        maze: 二维列表 代表迷宫
        start: (int, int) 起点坐标
        end: (int, int) 终点坐标
    Returns:
        fathers：字典 键为子节点坐标 值为父节点坐标
                 可以从终点回溯路径直到起点
    """
    # 迷宫高和宽
    h, w = len(maze), len(maze[0])
    fathers = {} 
    # 访问过的节点
    visited = set()
    # 优先队列，按照总路径cost自小到大
    pqueue = PriorityQueue()
    # 插入起始节点
    pqueue.put((0, start))
    fathers[start] = (-1, -1)

    # 直到队列为空
    while pqueue:
        # 取出最小的cost和对应节点
        cost, node = pqueue.get()
        if node not in visited:
            visited.add(node)

            # 到达终点则可以返回
            if node == end:
                return fathers
            # 遍历当前节点的邻居节点(即上下左右四个节点)
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,+1)]:
                next_x, next_y = node[0]+dx, node[1]+dy
                # 保证坐标要合法
                if next_x < h and next_x >= 0 and next_y < w and next_y >= 0:
                    # 保证改坐标不是墙或者已经访问过
                    if maze[next_x][next_y] != '1' and (next_x, next_y) not in visited:
                        # 将邻居节点的父节点设为当前节点
                        fathers[(next_x, next_y)] = (node[0], node[1])
                        # 总路径cost需要加上新的路径cost，此处迷宫全部为1
                        total_cost = cost + 1
                        # 将新的总cost和节点加入优先队列
                        pqueue.put((total_cost, (next_x, next_y)))

def display_path(fathers):
    """
    给定 fathers字典可视化迷宫路径
    Args:
        fathers：字典 键为子节点坐标 值为父节点坐标
                 可以从终点回溯路径直到起点
    """
    fp = open('D:/AI_Lab/lab9/MazeData.txt')
    maze = [line.strip() for line in fp.readlines()]
    fp.close()
    maze = [list(line) for line in maze]
    h, w = len(maze), len(maze[0])

    x, y = end[0], end[1]
    # 路径节点置为黄色的'O'
    while(x > 0):
        maze[x][y] = '\033[1;33mO\033[0m'
        x, y = fathers[(x,y)]
    # 将迷宫有墙的地方置为'W', 通路置为' '
    for x in range(h):
        for y in range(w):
            if(maze[x][y] == '0'):
                maze[x][y] = ' '
            elif(maze[x][y] == '1'):
                maze[x][y] = 'W'

    maze = [' '.join(line) for line in maze]
    for line in maze:
        print(line)


if __name__ == "__main__":
    maze = get_maze('D:/AI_Lab/lab9/MazeData.txt')
    start = (1,34)
    end = (16,1)
    time_start = time.time()
    fathers = ucs(maze, start, end)
    time_end = time.time()
    print('totally cost', time_end - time_start)

    display_path(fathers)
