import random
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

def generate_random_point_in_polygon(vertices):
    polygon = Polygon(vertices)
    
    while True:
        min_x, min_y, max_x, max_y = polygon.bounds
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        
        if polygon.contains(random_point):
            return random_point

# define your own vertices
vertices = [(0, 0), (5, 0), (5, 5), (3,5), (3, 7.5),  (2, 7.5), (2,5),(0, 5)]

# generation of random points
num_points = 300
random_points = [generate_random_point_in_polygon(vertices) for _ in range(num_points)]

# pick vertices
polygon_x, polygon_y = zip(*vertices)

# pick position of random point
point_x, point_y = zip(*[(point.x, point.y) for point in random_points])

# visualization
plt.figure(figsize=(8, 8))
plt.plot(polygon_x + (polygon_x[0],), polygon_y + (polygon_y[0],), 'b-')  # polygon
plt.scatter(point_x, point_y, color='r', label='Random Points')  # points
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points in Polygon')
plt.legend()
plt.show()

