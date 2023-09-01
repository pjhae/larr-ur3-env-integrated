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

# 다각형의 꼭짓점 정의 (x, y 좌표 쌍)
vertices = [(0, 0), (5, 0), (5, 5), (3,5), (3, 7.5),  (2, 7.5), (2,5),(0, 5)]

# 랜덤 점 생성 및 시각화
num_points = 300
random_points = [generate_random_point_in_polygon(vertices) for _ in range(num_points)]

# 다각형 꼭짓점을 추출하여 시각화를 위한 데이터로 사용
polygon_x, polygon_y = zip(*vertices)

# 랜덤 점의 x, y 좌표 추출
point_x, point_y = zip(*[(point.x, point.y) for point in random_points])

# 시각화
plt.figure(figsize=(8, 8))
plt.plot(polygon_x + (polygon_x[0],), polygon_y + (polygon_y[0],), 'b-')  # 다각형 그리기
plt.scatter(point_x, point_y, color='r', label='Random Points')  # 랜덤 점 그리기
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points in Polygon')
plt.legend()
plt.show()

