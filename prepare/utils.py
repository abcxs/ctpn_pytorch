import numpy as np
from shapely.geometry import Polygon


def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]


def orderConvex(p):
    points = Polygon(p).convex_hull
    points = np.array(points.exterior.coords)[:4]
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points


def shrink_poly(poly, r=16):
    # y = kx + b
    x_min = int(np.min(poly[:, 0]))
    x_max = int(np.max(poly[:, 0]))

    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]

    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    res = []
    d = 16
    d -= 1
    p = x_min
    while p < x_max:
        if x_max - p < 4:
            break
        res.append(
            [
                int(p),
                int(k1 * p + b1),
                int(p + d),
                int(k1 * (p + d) + b1),
                int(p + d),
                int(k2 * (p + d) + b2),
                int(p),
                int(k2 * p + b2),
            ]
        )
        p = p + d + 1
    # start = int((x_min // 16 + 1) * 16)
    # end = int((x_max // 16) * 16)

    # p = x_min
    # res.append(
    #     [
    #         p,
    #         int(k1 * p + b1),
    #         p + 15,
    #         int(k1 * (p + 15) + b1),
    #         start - 1,
    #         int(k2 * (p + 15) + b2),
    #         p,
    #         int(k2 * p + b2),
    #     ]
    # )

    # for p in range(start, end, r):
    #     res.append(
    #         [
    #             p,
    #             int(k1 * p + b1),
    #             (p + 15),
    #             int(k1 * (p + 15) + b1),
    #             (p + 15),
    #             int(k2 * (p + 15) + b2),
    #             p,
    #             int(k2 * p + b2),
    #         ]
    #     )
    # last = x_max - 15
    # res.append(
    #     [
    #         last,
    #         int(k1 * last + b1),
    #         x_max,
    #         int(k1 * x_max + b1),
    #         x_max,
    #         int(k2 * x_max + b2),
    #         last,
    #         int(k2 * last + b2),
    #     ]
    # )
    res = np.array(res, dtype=np.int).reshape([-1, 8])
    labels = np.zeros(res.shape[0])
    labels[:2] = 1
    labels[-2:] = 1
    return res, labels
