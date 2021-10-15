from typing import List, Dict, Tuple


class BoundingBox:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def h_flip(self, img_width=1.0):
        return BoundingBox(img_width - self.x - self.w, self.y, self.w, self.h)

    def v_flip(self, img_height=1.0):
        return BoundingBox(self.x, img_height - self.y - self.h, self.w, self.h)

    def rot90(self, img_height=1.0):
        return BoundingBox(self.y, img_height - (self.x + self.w), self.h, self.w)

    def translate(self, t: Tuple[float, float]):
        return BoundingBox(self.x + t[0], self.y + t[1], self.w, self.h)

    def scale(self, s: Tuple[float, float]):
        return BoundingBox(self.x * s[0], self.y * s[1], self.w * s[0], self.h * s[1])

    def round(self):
        return BoundingBox(round(self.x), round(self.y), round(self.w), round(self.h))

    def dist_squared(self, other_center):
        self_center = self.center()
        return (self_center[0] - other_center[0]) ** 2 + (self_center[1] - other_center[1]) ** 2


def save_bbs(bb_dict: Dict[int, List[BoundingBox]], file_path: str):
    with open(file_path, 'w') as f:
        for frame in sorted(bb_dict.keys()):
            bbs = bb_dict[frame]
            for bb in bbs:
                f.write(f'{frame},{round(bb.x, 4)},{round(bb.y, 4)},{round(bb.w, 4)},{round(bb.h, 4)}\n')


def parse_bbs(file_path: str) -> Dict[int, List[BoundingBox]]:
    bbs = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = [float(coord) for coord in line.replace(' ', '').replace('\n', '').split(',')]
        frame = int(data[0])
        if frame not in bbs:
            bbs[frame] = []
        bbs[frame].append(BoundingBox(x=data[1], y=data[2], w=data[3], h=data[4]))
    return bbs
