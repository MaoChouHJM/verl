import pyarrow as pa  # 导入 pyarrow 并使用别名 pa
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import io

# 假设你有一个包含图像字节的Parquet文件
# 这里我们创建一个示例Parquet文件，其中包含图像的字节数据
# 在实际应用中，你的Parquet文件可能已经包含了图像的字节数据

# 创建一个示例图像并获取其字节数据
dummy_image = Image.new('RGB', (5, 5), color = 'red')
byte_arr = io.BytesIO()
dummy_image.save(byte_arr, format='PNG')
image_bytes = byte_arr.getvalue()

# 创建一个包含图像字节的DataFrame
df_with_image = pd.DataFrame({
    'id': [1],
    'image_data': [image_bytes]
})

# 将DataFrame写入Parquet文件
table_to_write = pa.Table.from_pandas(df_with_image)
pq.write_table(table_to_write, 'image_example.parquet')

# 从Parquet文件读取数据
table_read = pq.read_table('image_example.parquet')
df_read = table_read.to_pandas()

# 访问图像数据并尝试重新创建PIL图像
if not df_read.empty:
    first_image_bytes = df_read['image_data'].iloc[0]
    print(f'{first_image_bytes=}')
    try:
        reconstructed_image = Image.open(io.BytesIO(first_image_bytes))
        print(f'{reconstructed_image=}')
        print("成功从Parquet文件读取并重建PIL图像。")
        # reconstructed_image.show() # 可以取消注释来显示图像
    except Exception as e:
        print(f"重建PIL图像失败: {e}")
else:
    print("读取的DataFrame为空。")

