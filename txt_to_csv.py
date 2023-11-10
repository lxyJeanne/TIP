import csv

def txt_to_csv(input_txt, output_csv):
    with open(input_txt, 'r') as txtfile:
        # 读取文本文件的每一行
        lines = txtfile.readlines()

    with open(output_csv, 'w', newline='') as csvfile:
        # 创建 CSV writer，指定分隔符为逗号
        csv_writer = csv.writer(csvfile, delimiter=',')

        # 将文本文件的每一行写入 CSV 文件
        for line in lines:
            # 使用 strip() 方法移除每行末尾的换行符
            csv_writer.writerow(line.strip().split(','))

# 指定输入文本文件和输出 CSV 文件
input_txt = 'output.txt'
output_csv = 'output.csv'

# 转换文本文件为 CSV 文件
txt_to_csv(input_txt, output_csv)
