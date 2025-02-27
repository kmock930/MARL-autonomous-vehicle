ACTION_SPACE = dict[int,tuple] = {
        # straight
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1),   #RIGHT
        # diagonals
        4: (-1, -1), #UP-LEFT
        5: (-1, 1), #UP-RIGHT
        6: (1, -1), #DOWN-LEFT
        7: (1, 1), #DOWN-RIGHT
        # stay
        8: (0, 0) #STAY
    }