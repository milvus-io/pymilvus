import numpy as np
import random

def random_point()->str:
    x = random.uniform(-90, 90)
    y = random.uniform(-180, 180)
    return f"POINT ({x:.3f} {y:.3f})"

def random_linestring(num_points)->str:
    points = ", ".join(f"{random.uniform(-90, 90):.3f} {random.uniform(-180, 180):.3f}" for _ in range(num_points))
    return f"LINESTRING ({points})"

def random_polygon(num_points: int) -> str:
    points = [
        f"{random.uniform(-90, 90):.3f} {random.uniform(-180, 180):.3f}"
        for _ in range(num_points)
    ]
    # 闭合多边形
    points.append(points[0])  # 将第一个点再添加一次
    return f"POLYGON(({', '.join(points)}))"


def generate_data(num):
    data = list()
    for i in range(num):
        if i%3==0:
            data.append(random_point())
        elif i%3==1:
            data.append(random_linestring(random.randint(2,9)))
        else:
            data.append(random_polygon(random.randint(3,9)))
    return data

def main():
    num_entities = 10
    data = generate_data(num_entities)
    for item in data:
        print(item)


if __name__ == "__main__":
    main()