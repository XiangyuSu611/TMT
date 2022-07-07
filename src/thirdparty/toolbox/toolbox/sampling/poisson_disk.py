from math import ceil
from math import cos
from math import pi
from math import sin
from math import sqrt
from random import randint
from random import uniform

from .datastructures import RandomQueue
from .enhanced_grid import Grid2D


def sqr_dist(p0, p1):
    x0, y0, = p0
    x1, y1 = p1
    return (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)


def sample_poisson_uniform(width, height, r, k, mask):
    """
    :param width: The width of the rectangle to sample.
    :param height: The height of the rectangle to sample.
    :param r:
        The mimum distance between points, in terms of
        rectangle units. For example, in a 10 by 10 grid, a mimum distance of
        10 will probably only give you one sample point.
    :param k:
        The algorithm generates k points around points already
        in the sample, and then check if they are not too close
        to other points. Typically, k = 30 is sufficient. The larger
        k is, the slower th algorithm, but the more sample points
        are produced.
    :return:
    """
    # Convert rectangle (the one to be sampled) coordinates to
    # coordinates in the grid.
    def grid_coordinates(p):
        x, y = p
        return (int(x * inv_cell_size), int(y * inv_cell_size))

    # Puts a sample point in all the algorithm's relevant containers.
    def put_point(p):
        process_list.push(p)
        sample_points.append(p)
        grid[grid_coordinates(p)] = p

    # Generates a point randomly selected around
    # the given point, between r and 2*r units away.
    def generate_random_around(p, r):
        x, y = p
        rr = uniform(r, 2 * r)
        rt = uniform(0, 2 * pi)

        return rr * sin(rt) + x, rr * cos(rt) + y

    # Is the given point in the rectangle to be sampled?
    def in_mask(p):
        x, y = p
        return 0 <= x < width and 0 <= y < height and mask[x, y]

    def in_neighbourhood(p):
        gp = grid_coordinates(p)

        if grid[gp]: return True

        for cell in grid.square_iter(gp, 2):
            if cell and sqr_dist(cell, p) <= r_sqr:
                return True
        return False

    # Create the grid
    cell_size = r / sqrt(2)
    inv_cell_size = 1 / cell_size
    r_sqr = r * r

    grid = Grid2D((int(ceil(width / cell_size)),
                   int(ceil(height / cell_size))))

    process_list = RandomQueue()
    sample_points = []

    # generate the first point
    put_point((randint(0, width - 1), randint(0, height - 1)))

    # generate other points from points in queue.
    while not process_list.empty():
        p = process_list.pop()

        for i in range(k):
            q = generate_random_around(p, r)
            if in_mask(q) and not in_neighbourhood(q):
                put_point(q)

    return sample_points


