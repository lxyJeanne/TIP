from PIL import Image
import numpy as np


def matrices_to_image(red_matrix, green_matrix, blue_matrix, output_image):
    # 将红绿蓝矩阵转换为NumPy数组
    red_array = np.array(red_matrix)
    green_array = np.array(green_matrix)
    blue_array = np.array(blue_matrix)
    # print(red_array)


    # 将三个通道合并成RGB图像
    rgb_array = np.stack([red_array, green_array, blue_array], axis=-1)


    # print(np.shape(red_array))
    # print(np.shape(green_array))
    # print(np.shape(blue_array))
    print(rgb_array.astype(np.uint8))

    # 创建图像对象
    image = Image.fromarray(rgb_array.astype(np.uint8))

    # Display the image for visual inspection
    image.show()

    # 保存图像为JPEG文件
    image.save(output_image)


txt_file_path = 'output.txt'
i=1
# 逐行读取文件并按逗号分隔
with open(txt_file_path, 'r') as txtfile:
    for line in txtfile:
        # 去除行末尾的换行符
        line = line.strip()

        # 使用逗号分隔字符串
        data = line.split(',')

        # 去掉第一个元素
        data = data[1:]
        data = [float(x) * 256 for x in data]

        # 将剩余的元素分割成三个矩阵
        num_elements_per_matrix = 74*74  # 每个矩阵的元素个数
        matrix1 = list(map(int, data[:num_elements_per_matrix]))
        matrix2 = list(map(int, data[num_elements_per_matrix:2*num_elements_per_matrix]))
        matrix3 = list(map(int, data[2*num_elements_per_matrix:]))
        matrix1 = [matrix1[i * 74:(i + 1) * 74] for i in range(74)]
        matrix2 = [matrix2[i * 74:(i + 1) * 74] for i in range(74)]
        matrix3 = [matrix3[i * 74:(i + 1) * 74] for i in range(74)]

        # 指定图像文件
        output_image = 'output' + str(i) + '.jpg'
        i = i+1
        # 将红绿蓝三个矩阵转换为图像并保存
        matrices_to_image(matrix1, matrix2, matrix3, output_image)















