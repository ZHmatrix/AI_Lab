import heapq
import time

class Node(object):
    """节点类

    Attributes:
        x, y: int 横纵坐标
        g: float 到当前节点的路径成本 
        h: float 当前节点到终点的启发式估计值 
        f: float 起点到终点的成本估计值
        parent: Node节点类 父节点 用于寻路
        accessible: bool 是否可达 
    """
    def __init__(self, x, y, accessible):
        """初始化"""
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        self.accessible = accessible

    def __lt__(self, other):
        """重载函数< 用于优先队列中的比较排序

        Args:
            other: <的右值 Node节点类
        """
        return self.f < other.f


class AStar(object):
    """A*算法

    Attributes:
        maze: 二维list 元素为Node 迷宫
        height: int 迷宫高度
        width: int 迷宫宽度
        open_list: list A*算法维护的open列表，需要堆化为优先队列
        close_list: set A*算法维护的close列表
        start: Node 起点
        end: Node 终点
    """
    def __init__(self, file_path):
        """初始化"""
        self.maze = self.init_maze(file_path)
        self.height = len(self.maze)
        self.width = len(self.maze[0])
        self.open_list = []
        self.close_list = set()
        self.start = self.maze[1][34]
        self.end = self.maze[16][1]

        heapq.heapify(self.open_list)

    def init_maze(self, file_path):
        """根据给定文件初始化迷宫

        Args:
            maze: 二维list 元素为Node 迷宫
        """
        fp = open(file_path)
        maze = [line.strip() for line in fp.readlines()]
        fp.close()
        maze = [list(line) for line in maze]
        h, w = len(maze), len(maze[0])
        res = [[None for j in range(w)] for i in range(h)]

        # '1'为墙不可达，其余可达
        for x in range(h):
            for y in range(w):
                accessible = False if maze[x][y] == '1' else True
                res[x][y] = Node(x, y, accessible)
        return res

    def heuristic(self, node):
        """计算给定节点到终点的启发式估计值h

        Args:
            node 节点
        """
        return abs(node.x - self.end.x) + abs(node.y - self.end.y)

    def get_neighbour(self, node):
        """得到给定节点的相邻节点

        Args:
            node 节点
        Returns:
            neighbours list node的相邻节点列表
        """
        neighbours = []
        # 判断上下左右四个坐标是否合法，是则加入neighbours
        if node.x + 1 < self.height:
            neighbours.append(self.maze[node.x + 1][node.y])
        if node.x > 0:
            neighbours.append(self.maze[node.x - 1][node.y])
        if node.y + 1 < self.width:
            neighbours.append(self.maze[node.x][node.y + 1])
        if node.y > 0:
            neighbours.append(self.maze[node.x][node.y - 1])
        return neighbours

    def get_path(self):
        """从终点回溯得到路径上的节点坐标列表

        Returns:
            neighbours list 路径上的节点坐标
        """
        res = []
        node = self.end
        while node.parent is not self.start:
            node = node.parent
            res.append((node.x, node.y))
        return  res

    def update_node(self, neig, node):
        """更新节点
            将node设为邻居节点neig的father
            更新neig的路径成本 启发式估计值 总估计值
        """        
        neig.parent = node
        neig.g = node.g + 1
        neig.h = self.heuristic(neig)
        neig.f = neig.h + neig.g

    def find_path(self):
        """寻找最短路径
            使用A*算法寻找maze的最短路径
        """ 
        # 由于此处需要open列表为可迭代对象故没有直接使用PriorityQueue
        # 堆化，即建立优先队列，将起点和起点估计值加入open_list中    
        heapq.heappush(self.open_list, (self.start.f, self.start))

        # 直至open_list为空
        while len(self.open_list) > 0:
            # 从open_list取出总估计值f最小的节点加入close_list中
            f, node = heapq.heappop(self.open_list)
            self.close_list.add(node)

            # 到达终点则可以返回
            if (node.x, node.y) == (self.end.x, self.end.y):
                break

            neighbours = self.get_neighbour(node)
            # 遍历其所有邻居节点
            for neig in neighbours:
                # 保证其可达性且不在close_list中
                if neig.accessible and neig not in self.close_list:
                    # 如果已经在open_list中且new_g > old_g + cost则需要更新节点
                    if (neig.f, neig) in self.open_list:
                        if neig.g > node.g + 1:
                            self.update_node(neig, node)
                    # 如果不在open_list中则更新节点并加入open_list中
                    else:
                        self.update_node(neig, node)
                        heapq.heappush(self.open_list, (neig.f, neig))

def display_path(path):
    """
    给定 path列表 可视化迷宫路径
    Args:
        path：list 路径上的坐标
    """
    fp = open('D:/AI_Lab/lab9/MazeData.txt')
    maze = [line.strip() for line in fp.readlines()]
    fp.close()
    maze = [list(line) for line in maze]
    h, w = len(maze), len(maze[0])

    # 路径节点置为黄色的'O'
    for (x,y) in path:
        maze[x][y] = '\033[1;33mO\033[0m'
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
    test = AStar('D:/AI_Lab/lab9/MazeData.txt')

    time_start = time.time()
    test.find_path()
    time_end = time.time()
    print('totally cost', time_end - time_start)

    path = test.get_path()
    display_path(path)
