import numpy as np
import copy

HOLE_BORDER = 1
OUTER_BORDER = 2

# 点是否还在图片里
def CorInImage(p, numrows, numcols):
    if p[1] < 0 \
        or p[0] < 0 \
            or p[1] >= numcols \
                or p[0] >= numrows:
        return False
    else:
        return True

# 顺时钟冲一发
def GoClockwise(current, pivot):
    if current[1] > pivot[1]:
        return (pivot[0]-1, pivot[1])
    elif current[1] < pivot[1]:
        return (pivot[0]+1, pivot[1])
    elif current[0] > pivot[0]:
        return (pivot[0], pivot[1]+1)
    elif current[0] < pivot[0]:
        return (pivot[0], pivot[1]-1)

def GoCounterColokwise(current, pivot):
    if current[1] > pivot[1]:
        return (pivot[0]+1, pivot[1])
    elif current[1] < pivot[1]:
        return (pivot[0]-1, pivot[1])
    elif current[0] > pivot[0]:
        return (pivot[0], pivot[1]-1)
    elif current[0] < pivot[0]:
        return (pivot[0], pivot[1]+1)

# 一边冲一边标记
def MarkPath(mark, center, checked):
    loc = -1
    #     3
	#   2 x 0
	#     1
    if mark[1] > center[1]:
        loc = 0
    elif mark[1] < center[1]:
        loc = 2
    elif mark[0] > center[0]:
        loc = 1
    elif mark[0] < center[0]:
        loc = 3
    
    if loc == -1:
        print("mark fail")
    checked[loc] = True
    

# 边界跟踪
def FollowBorder(image, row, col, p2, NBD, contours):
    numrows = image.shape[0]
    numcols = image.shape[1]
    current = (p2[0], p2[1])
    start = (row, col)
    point_storage = []

    # 从 current 的四邻域找一个非零像素
    while True:
        current = GoClockwise(current, start)
        if current[0] == p2[0] and current[1] == p2[1]:
            image[start[0], start[1]] = -NBD[0]
            point_storage.append(copy.deepcopy(start))
            contours.append(copy.deepcopy(point_storage))
            return
        if not CorInImage(current, numrows, numcols) \
            or image[current[0], current[1]] == 0:
            continue
        else:
            break
    p1 = (current[0], current[1])


    p3 = (start[0], start[1])
    p4 = (-1, -1)
    p2 = (p1[0], p1[1])
    checked = [False]*4
    while True:
        current = (p2[0], p2[1])
        checked = [False]*4

        while True:
            MarkPath(current, p3, checked)
            current = GoCounterColokwise(current, p3)
            if not CorInImage(current, numrows, numcols) \
                or image[current[0], current[1]] == 0:
                continue
            else:
                break
        p4 = (current[0], current[1])

        if (p3[1]+1 >= numcols \
            or image[p3[0], p3[1]+1] == 0) \
                and checked[0]:
            image[p3[0], p3[1]] = -NBD[0]
        elif p3[1]+1 < numcols \
            and image[p3[0], p3[1]] == 1:
            image[p3[0], p3[1]] = NBD[0]
        point_storage.append(p3)

        if p4[0] == start[0] and p4[1] == start[1] \
            and p3[0] == p1[0] and p3[1] == p3[1]:
            contours.append(copy.deepcopy(point_storage))
            return
        p2 = (p3[0], p3[1])
        p3 = (p4[0], p4[1])


# suzuki 算法
def FindContours(image):
    if np.max(image) > 1:
        image = np.divide(image, 255).astype(np.int16)
    
    
    # 图片长宽
    numrows = image.shape[0]
    numcols = image.shape[1]
    
    # 现轮廓和上一个轮廓的信息
    # [ 轮廓编号, 轮廓类型 ]
    NBD = [-1]*2
    LNBD = [-1]*2

    NBD[0] = 1
    NBD[1] = HOLE_BORDER
    LNBD[1] = HOLE_BORDER

    
    # 存放所有轮廓
    # [points] 是轮廓 contours 是[ [point1] ... ]
    contours = []
    
    # 存放轮廓层级关系树，这里参考opencv的存放方法
    # [爸爸轮廓的idx, 第一个儿子轮廓的idx, 同级下一个idx, [轮廓编号, 轮廓类型]]
    hierachy = []
    t_node = [-1, -1, -1, copy.deepcopy(NBD)]
    hierachy.append(copy.deepcopy(t_node))

    p2 = None
    border_start_found = bool()
    for r in range(numrows):
        LNBD[0] = 1
        LNBD[1] = HOLE_BORDER
        for c in range(numcols):
            border_start_found = False

            # 如果遇到外边界
            if image[r, c] == 1 and c-1 < 0 \
                or image[r, c] ==1 and image[r, c-1] == 0:
                NBD[1] = OUTER_BORDER
                NBD[0] += 1
                p2 = (r, c-1)
                border_start_found = True

            # 如果遇到孔边界
            elif c+1 < numcols and image[r, c] >= 1 and image[r, c+1] == 0:
                NBD[1] = HOLE_BORDER
                NBD[0] += 1
                if image[r, c] > 1:
                    LNBD[0] = int(image[r, c]) 
                    LNBD[1] = hierachy[LNBD[0]-1][-1][0]
                p2 = (r, c+1)
                border_start_found = True
            
            if border_start_found:
                t_node[0] = -1
                t_node[1] = -1
                t_node[2] = -1
                # 确定关系
                if NBD[1] == LNBD[1]:
                    t_node[0] = hierachy[LNBD[0]-1][0]
                    t_node[2] = hierachy[t_node[0]-1][1]
                    hierachy[t_node[0]-1][1] = NBD[0]
                    t_node[-1] = copy.deepcopy(NBD)
                    hierachy.append(copy.deepcopy(t_node))
                else:
                    if hierachy[LNBD[0]-1][1] != -1:
                        t_node[2] = hierachy[LNBD[0]-1][1]

                    t_node[0] = LNBD[0]
                    hierachy[LNBD[0]-1][1] = NBD[0]
                    t_node[-1] = copy.deepcopy(NBD)
                    hierachy.append(copy.deepcopy(t_node))

                # 跟踪轮廓！
                FollowBorder(image, r, c, p2, NBD, contours)
                print(image)
            if abs(image[r, c]) > 1:
                LNBD[0] = abs(image[r, c])
                LNBD[1] = hierachy[LNBD[0]-1][-1][1]
    return contours, hierachy