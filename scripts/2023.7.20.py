import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 位姿矩阵
pose_matrix = np.array([[-0.242098, -0.92174815, 0.30293383, -0.16230445],
                        [-0.96608017, 0.25792801, 0.01273739, -0.09484011],
                        [-0.08987579, -0.28957467, -0.95292647, 0.56955565],
                        [0., 0., 0., 1.]])

# 提取位姿的平移向量
translation_vector = pose_matrix[:3, 3]

# 提取位姿的旋转矩阵
rotation_matrix = pose_matrix[:3, :3]

# 设置3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原点
ax.scatter(0, 0, 0, color='red', label='Origin')

# 绘制位姿的平移向量
ax.quiver(0, 0, 0, translation_vector[0], translation_vector[1], translation_vector[2], color='blue', label='Translation')

# 设置坐标轴范围
max_val = max(np.abs(translation_vector))
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 绘制位姿的旋转方向
for i in range(3):
    ax.quiver(0, 0, 0, rotation_matrix[i, 0], rotation_matrix[i, 1], rotation_matrix[i, 2], color='green',
              label=f'Rotation{i+1}', pivot='tail')

# 添加图例
ax.legend()

# 显示图像
plt.savefig("scripts/img/location.png")