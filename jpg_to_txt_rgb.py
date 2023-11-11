from PIL import Image
import os
import csv


def image_to_grayscale(image_path):
    # 打开图像
    img = Image.open(image_path)
    r, g, b = img.split()
    red_matrix = list(r.getdata())
    green_matrix = list(g.getdata())
    blue_matrix = list(b.getdata())
    # combined_matrix = red_matrix[0] + green_matrix[0] + blue_matrix[0]


    width, height = r.size
    red_matrix = [red_matrix[i * width:(i + 1) * width] for i in range(height)]
    green_matrix = [green_matrix[i * width:(i + 1) * width] for i in range(height)]
    blue_matrix = [blue_matrix[i * width:(i + 1) * width] for i in range(height)]

    # 去掉每个矩阵周围一圈的数值
    red_matrix = [row[1:-1] for row in red_matrix[1:-1]]
    green_matrix = [row[1:-1] for row in green_matrix[1:-1]]
    blue_matrix = [row[1:-1] for row in blue_matrix[1:-1]]

    red_matrix_fin = compression(red_matrix)
    green_matrix_fin = compression(green_matrix)
    blue_matrix_fin = compression(blue_matrix)

    combined_matrix = red_matrix_fin + green_matrix_fin + blue_matrix_fin
    # 将图像转换为灰度
    # grayscale_img = img.convert("L")

    # 获取灰度图像的像素值
    # grayscale_matrix = list(grayscale_img.getdata())

    # 将一维数组转换为二维矩阵
    # width, height = grayscale_img.size
    # grayscale_matrix = [grayscale_matrix[i * width:(i + 1) * width] for i in range(height)]

    # return grayscale_matrix
    return combined_matrix

def compression(matrix):

    # 获取矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0])

    # 创建一个新的矩阵，用于存储压缩后的结果
    compressed_matrix = []
    # 遍历矩阵的每个3x3子矩阵，计算其中位数并存储
    for i in range(0, rows, 3):
        for j in range(0, cols, 3):
            submatrix = [matrix[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            median_value = sorted(submatrix)[4]  # 中值为排序后的第5个元素
            compressed_matrix.append(median_value)

    return compressed_matrix

def load_labels(csv_path):
    # 初始化空字典用于存储标签
    labels = {}

    with open(csv_path, 'r') as csvfile:
        # 使用 csv.reader 读取 CSV 文件
        csv_reader = csv.reader(csvfile)

        # 跳过表头
        next(csv_reader)

        # 处理每一行数据
        for row in csv_reader:
            # 从行中提取文件名和标签
            # data = row[0].split(',')  # 使用逗号分隔文件名和标签
            # print(row[0],row[1])
            filename, label = row[0]+".jpg", row[1]

            # 将文件名和标签添加到字典中
            labels[filename] = label

    # 返回包含文件名和标签的字典
    return labels


def process_images(folder_path, labels_csv, output_txt):
    # 获取文件夹中所有图片的文件名
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # 加载标签数据
    image_labels = load_labels(labels_csv)

    # 打开文本文件进行写入
    with open(output_txt, 'w') as txtfile:
        # 处理每张图片
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            # print(image_file)

            # 获取灰度值矩阵
            grayscale_matrix = image_to_grayscale(image_path)

            # 将灰度值矩阵中的每个元素除以 256
            normalized_grayscale_matrix = [pixel / 256.0 for pixel in grayscale_matrix]

            # 获取标签
            label = image_labels.get(image_file, 'Unknown')

            # 写入文本文件
            # row = [image_file, label] + normalized_grayscale_matrix  # 文件名、标签和灰度值数据
            row = [label] + normalized_grayscale_matrix  # 文件名、标签和灰度值数据
            row_str = ','.join(map(str, row))  # 将列表转换为逗号分隔的字符串
            txtfile.write(row_str + '\n')


# 指定文件夹路径、标签CSV文件路径和输出文本文件名
folder_path = 'train-resizedtest'
labels_csv = 'train-labels.csv'
output_txt = 'output.txt'

# 处理图像并写入文本文件
process_images(folder_path, labels_csv, output_txt)
