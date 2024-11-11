# Import các thư viện cần thiết
import io
import os
import re
import openpyxl
import numpy as np
import pandas as pd
import streamlit as st
from minio import Minio
from minio.error import S3Error
from pyspark.sql import SparkSession
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import col, count, when, mean


# Hàm kiểm tra kết nối MinIO
def check_minio_connection():
    try:
        # Khởi tạo Minio client
        minio_client = Minio(
            "192.168.0.101:9000",
            access_key="admin",    
            secret_key="password", 
            secure=False
        )

        # Kiểm tra kết nối bằng cách liệt kê các bucket
        buckets = minio_client.list_buckets()
        return True, minio_client, buckets  # Trả về client và buckets

    except S3Error as err:
        return False, str(err), None  # Trả về lỗi như một chuỗi



# Bước 1: Khởi tạo Spark session
spark = SparkSession.builder \
    .appName("Titanic Data Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Bước 2: Đọc dữ liệu từ file CSV
df = spark.read.csv("titanic.csv", header=True, inferSchema=True)

# Thay đổi giao diện
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f5;
        }
        .st-emotion-cache-vmpjyt {
            background-color: #9E9E9E;
        }
        .st-emotion-cache-1ny7cjd {
            display: inline-flex;
            -webkit-box-align: center;
            align-items: center;
            -webkit-box-pack: center;
            justify-content: center;
            font-weight: 400;
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            min-height: 2.5rem;
            margin: 0px;
            line-height: 1.6;
            color: inherit;
            width: auto;
            user-select: none;
            background-color: #e7ff95;
            border: 1px solid rgb(255 255 255 / 20%);
            flex-direction: row;
        }
        element.style {
            background: darkseagreen;
            padding-bottom: inherit;
            display: inline-block;
            padding-top: inherit;
        }
        .title {
            background-color: #4caf50; /* Màu nền xanh lá cho tiêu đề chính */
            color: white; /* Màu chữ cho tiêu đề chính */
            font-size: 2.5em;
            font-weight: bold;
            padding: 10px; /* Khoảng cách cho tiêu đề */
            border-radius: 10px; /* Bo góc cho tiêu đề */
            display: inline-block; /* Để nền ôm sát chữ */
            margin-bottom: 20px;
        }
        .subheader {
            background-color: #8B8378; /* Màu nền cho tiêu đề phụ */
            color: white; /* Màu chữ cho tiêu đề phụ */
            font-size: 1.5em;
            font-weight: bold;
            padding: 10px; /* Khoảng cách cho tiêu đề phụ */
            border-radius: 10px; /* Bo góc cho tiêu đề phụ */
            display: inline-block; /* Để nền ôm sát chữ */
            margin-bottom: 20px;
        }
        h1, h2 {
            color: white; /* Màu chữ cho tiêu đề */
            padding: 10px; /* Khoảng cách cho tiêu đề */
            border-radius: 10px; /* Bo góc cho tiêu đề */
            margin-bottom: 20px;
        }
        .streamlit-expanderHeader {
            background-color: #4caf50; /* Màu nền xanh lá cho sidebar */
            color: white; /* Màu chữ cho sidebar */
        }
        .css-1g5y6g3 {
            background-color: #4caf50; /* Màu nền cho sidebar */
            color: white; /* Màu chữ cho sidebar */
        }
        .st-emotion-cache-1gwvy71 h2 {
            background-color: #4caf50 !important; 
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: inline-block;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Hiển thị tiêu đề ứng dụng
st.markdown("<h1 class='title'>🚢 Phân Tích Dữ Liệu Titanic</h1>", unsafe_allow_html=True)

# Hiển thị dữ liệu ban đầu
st.markdown("<h2 class='subheader'>Dữ liệu ban đầu:</h2>", unsafe_allow_html=True)
st.write(df.toPandas().head())  # Hiển thị dữ liệu trên Streamlit

# Tạo không gian ngăn cách
st.markdown("<hr>", unsafe_allow_html=True)

# Bước 3: Xử lý dữ liệu thiếu
st.markdown("<h2 class='subheader'>Dữ liệu thiếu:</h2>", unsafe_allow_html=True)
missing_data = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
st.write(missing_data)

st.markdown("<hr>", unsafe_allow_html=True)

mean_age = df.select(mean(col("Age"))).collect()[0][0]
df = df.na.fill({"Age": mean_age})
df = df.na.drop(subset=["Embarked"])
df = df.na.drop(subset=["Cabin"])

st.markdown("<h2 class='subheader'>Sau khi xử lý:</h2>", unsafe_allow_html=True)
missing_values_after = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()

st.write(missing_values_after)

st.markdown("<hr>", unsafe_allow_html=True)


# Bước 4: Tạo Sidebar cho truy xuất dữ liệu
st.sidebar.header("Tùy chọn phân tích")
option = st.sidebar.selectbox(
    "Chọn loại phân tích thực hiện:",
    ("Tỷ lệ sống sót tổng quan", 
     "Phân tích sống sót theo giới tính", 
     "Phân tích sống sót theo hạng vé", 
     "Dự đoán sống sót"),
    index=None,
    placeholder="Loại phân tích"
)


# Đường dẫn đến thư mục lưu trữ hình ảnh
image_folder = "images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Bước 5: Phân tích dữ liệu dựa trên lựa chọn
if option == "Tỷ lệ sống sót tổng quan":
    survival_rate = df.groupBy("Survived").count().toPandas()
    st.markdown("<h2 class='subheader'>Tỷ lệ sống sót:</h2>", unsafe_allow_html=True)
    st.write(survival_rate)

    total_passengers = survival_rate['count'].sum()
    survived_count = survival_rate[survival_rate['Survived'] == 1]['count'].values[0]
    survival_percentage = (survived_count / total_passengers) * 100

    st.markdown(f"<p><strong>Tổng số hành khách:</strong> {total_passengers}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Tỷ lệ sống sót:</strong> {survival_percentage:.2f}%</p>", unsafe_allow_html=True)

    survived_count = df.toPandas().groupby('Survived').size()
    st.markdown("<h2 class='subheader'>Biểu đồ tỷ lệ sống sót:</h2>", unsafe_allow_html=True)
    chart = survived_count.plot(kind='bar')  # Gán biểu đồ vào biến chart
    st.pyplot(chart.get_figure())  # Hiển thị biểu đồ trong Streamlit

    # Lưu biểu đồ
    if st.button("Lưu biểu đồ"):
        image_path = os.path.join(image_folder, "ty_le_song_tot_tong_quan.png")
        chart.get_figure().savefig(image_path, format="png")
        st.success(f"Biểu đồ đã được lưu: {image_path}")

    # Nhận xét
    st.markdown("""
    <p><strong>Nhận xét:</strong> 
                <p>Tỷ lệ sống sót tổng quan cho thấy chỉ có 38.25% hành khách sống sót sau thảm họa, cho thấy sự tàn khốc của tình huống.</p> Số lượng hành khách không qua khỏi (khoảng 61.75%) phản ánh những thách thức lớn mà họ phải đối mặt trong tình huống khẩn cấp.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

elif option == "Phân tích sống sót theo giới tính":
    sex_survival = df.groupBy("Sex", "Survived").count().toPandas()
    st.markdown("<h2 class='subheader'>Phân tích sống sót theo giới tính:</h2>", unsafe_allow_html=True)
    st.write(sex_survival)

    total_males = sex_survival[sex_survival['Sex'] == 'male']['count'].values[0]
    total_females = sex_survival[sex_survival['Sex'] == 'female']['count'].values[0]
    survived_males = sex_survival[(sex_survival['Sex'] == 'male') & (sex_survival['Survived'] == 1)]['count'].values[0]
    survived_females = sex_survival[(sex_survival['Sex'] == 'female') & (sex_survival['Survived'] == 1)]['count'].values[0]

    male_survival_rate = (survived_males / total_males) * 100
    female_survival_rate = (survived_females / total_females) * 100

    st.markdown(f"<p><strong>Tỷ lệ sống sót của nam:</strong> {male_survival_rate:.2f}%</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Tỷ lệ sống sót của nữ:</strong> {female_survival_rate:.2f}%</p>", unsafe_allow_html=True)

    sex_survived_count = df.toPandas().groupby(['Sex', 'Survived']).size().unstack()
    st.markdown("<h2 class='subheader'>Biểu đồ sống sót theo giới tính:</h2>", unsafe_allow_html=True)
    chart = sex_survived_count.plot(kind='bar')
    st.pyplot(chart.get_figure())

    # Lưu biểu đồ
    if st.button("Lưu biểu đồ"):
        image_path = os.path.join(image_folder, "ty_le_song_tot_theo_gioi_tinh.png")
        chart.get_figure().savefig(image_path, format="png")
        st.success(f"Biểu đồ đã được lưu: {image_path}")

# Nhận xét
    st.markdown("""
    <p><strong>Nhận xét:</strong> 
                <p>Phân tích cho thấy tỷ lệ sống sót của nam là 23.29%, trong khi nữ đạt 100%.</p> Điều này chỉ ra rằng nữ hành khách có khả năng sống sót cao hơn rất nhiều so với nam giới, có thể phản ánh các yếu tố như chính sách cứu hộ ưu tiên phụ nữ và trẻ em trong tình huống khẩn cấp.</p>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

elif option == "Phân tích sống sót theo hạng vé":
    pclass_survival = df.groupBy("Pclass", "Survived").count().toPandas()
    st.markdown("<h2 class='subheader'>Phân tích sống sót theo hạng vé:</h2>", unsafe_allow_html=True)
    # st.write(pclass_survival)

    # Nhóm theo Pclass và Survived, sau đó đếm số lượng
    pclass_survival1 = df.groupBy("Pclass", "Survived").count().toPandas()

    # Chuyển đổi dữ liệu thành định dạng dễ đọc hơn
    pclass_survival1 = pclass_survival1.pivot(index='Pclass', columns='Survived', values='count').fillna(0)
    
    # Đổi tên cột để dễ hiểu hơn
    pclass_survival1.columns = ['Không sống sót', 'Sống sót']

    # Tính tổng số hành khách cho mỗi hạng vé
    pclass_survival1['Tổng cộng'] = pclass_survival1['Sống sót'] + pclass_survival1['Không sống sót']

    # Hiển thị kết quả
    st.write(pclass_survival1)

    # Lấy số lượng hành khách sống sót và tổng số hành khách theo hạng
    total_pclass1 = pclass_survival[pclass_survival['Pclass'] == 1]['count'].values.sum()
    total_pclass2 = pclass_survival[pclass_survival['Pclass'] == 2]['count'].values.sum()
    total_pclass3 = pclass_survival[pclass_survival['Pclass'] == 3]['count'].values.sum()
    
    survived_pclass1 = pclass_survival[(pclass_survival['Pclass'] == 1) & (pclass_survival['Survived'] == 1)]['count'].values.sum()
    survived_pclass2 = pclass_survival[(pclass_survival['Pclass'] == 2) & (pclass_survival['Survived'] == 1)]['count'].values.sum()
    survived_pclass3 = pclass_survival[(pclass_survival['Pclass'] == 3) & (pclass_survival['Survived'] == 1)]['count'].values.sum()

    # Tính tỷ lệ sống sót
    pclass1_survival_rate = (survived_pclass1 / total_pclass1) * 100 if total_pclass1 > 0 else 0
    pclass2_survival_rate = (survived_pclass2 / total_pclass2) * 100 if total_pclass2 > 0 else 0
    pclass3_survival_rate = (survived_pclass3 / total_pclass3) * 100 if total_pclass3 > 0 else 0

    st.markdown(f"<p><strong>Tỷ lệ sống sót hạng 1:</strong> {pclass1_survival_rate:.2f}%</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Tỷ lệ sống sót hạng 2:</strong> {pclass2_survival_rate:.2f}%</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Tỷ lệ sống sót hạng 3:</strong> {pclass3_survival_rate:.2f}%</p>", unsafe_allow_html=True)

    pclass_survived_count = df.toPandas().groupby(['Pclass', 'Survived']).size().unstack()
    st.markdown("<h2 class='subheader'>Biểu đồ sống sót theo hạng vé:</h2>", unsafe_allow_html=True)
    chart = pclass_survived_count.plot(kind='bar')
    st.pyplot(chart.get_figure())

    # Lưu biểu đồ
    if st.button("Lưu biểu đồ"):
        image_path = os.path.join(image_folder, "ty_le_song_tot_theo_hang_ve.png")
        chart.get_figure().savefig(image_path, format="png")
        st.success(f"Biểu đồ đã được lưu: {image_path}")

    #Nhận xét
    st.markdown("""
    <p><strong>Nhận xét:</strong> 
                <p>Phân tích cho thấy tỷ lệ sống sót ở hạng 1 là 62.62%, hạng 2 là 47.28%, và hạng 3 chỉ đạt 24.24%.</p> Sự chênh lệch này cho thấy rằng hành khách ở các hạng vé cao hơn có khả năng sống sót tốt hơn, có thể do vị trí ngồi gần lối thoát hiểm và ưu tiên trong quá trình cứu hộ.</p>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

elif option == "Dự đoán sống sót":
    st.markdown("<h2 class='subheader'>Dự đoán khả năng sống sót</h2>", unsafe_allow_html=True)
    
    # Chọn các đặc điểm để dự đoán
    st.markdown("**Nhập các thông tin dưới đây để dự đoán khả năng sống sót:**")
    
    pclass = st.selectbox("Hạng vé:", (1, 2, 3))
    sex = st.selectbox("Giới tính:", ("male", "female"))
    age = st.number_input("Tuổi:", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Số lượng anh chị em/chồng/vợ đi cùng:", min_value=0, max_value=8, value=0)
    parch = st.number_input("Số lượng cha mẹ/con cái đi cùng:", min_value=0, max_value=6, value=0)

    # Chuyển đổi giới tính thành số
    sex_encoded = 1 if sex == "female" else 0

    # Tạo DataFrame cho đầu vào
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_encoded],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch]
    })

    # Thêm tùy chọn thuật toán
    algorithm = st.selectbox("Chọn thuật toán:", ("Logistic Regression", "Decision Tree", "Random Forest"))

    # Bước 6: Huấn luyện mô hình
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
    df_pd = df.toPandas()
    df_pd["Sex"] = df_pd["Sex"].map({"male": 0, "female": 1})

    X = df_pd[feature_cols]
    y = df_pd["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # Dự đoán
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Khởi tạo file Excel
    results_file = "KetQua\\results.xlsx"
    if not os.path.exists(results_file):
        df_results = pd.DataFrame(columns=["pclass", "sex", "age", "sibsp", "parch", "algorithm", "prediction", "probability"])
        df_results.to_excel(results_file, index=False)

    # Hiển thị kết quả dự đoán
    if st.button("Dự đoán"):
        st.success("Khả năng sống sót: **{:.2f}%**".format(probability[0][1] * 100))
        if prediction[0] == 1:
            st.success("Bạn có khả năng sống sót! 🎉")
        else:
            st.error("Bạn không có khả năng sống sót. 😔")

    is_saved = False
    # Nút lưu dự đoán
    if st.button("Lưu dự đoán"):
        # Lưu thông tin dự đoán vào file Excel
        data = {
            "pclass": [pclass],
            "sex": [sex],
            "age": [age],
            "sibsp": [sibsp],
            "parch": [parch],
            "algorithm": [algorithm],
            "prediction": [prediction[0]],
            "probability": [probability[0][1]]
        }
        df_new_result = pd.DataFrame(data)
        df_results = pd.read_excel(results_file)
        df_results = pd.concat([df_results, df_new_result], ignore_index=True)
        df_results.to_excel(results_file, index=False)
        st.success("Thông tin dự đoán đã được lưu.") 
        is_saved = True
    
      # Hiển thị thông báo
    if is_saved:
        st.success("Thông tin dự đoán đã được lưu.")


st.sidebar.header("Hình ảnh đã lưu")
images = os.listdir(image_folder)
selected_image = st.sidebar.selectbox("Chọn hình ảnh:", images,  index=None, placeholder="Xem ảnh")

if selected_image:
    st.image(os.path.join(image_folder, selected_image))


# Hàm kiểm tra tên bucket hợp lệ
def is_valid_bucket_name(name):
    # Kiểm tra tên bucket theo quy tắc của MinIO
    return bool(re.match(r'^[a-z0-9\-]{3,63}$', name))  # Chỉ cho phép chữ thường, số, và dấu -

# Giao diện Streamlit
st.sidebar.title("MinIO Upload Tool")

# Biến trạng thái
if 'upload_disabled' not in st.session_state:
    st.session_state.upload_disabled = True

if 'minio_client' not in st.session_state:
    st.session_state.minio_client = None

if 'buckets' not in st.session_state:
    st.session_state.buckets = []

if 'selected_bucket' not in st.session_state:
    st.session_state.selected_bucket = None  # Lưu bucket hiện tại

# Tạo button kiểm tra kết nối
if st.sidebar.button("Kết nối MinIO"):
    success, result, buckets = check_minio_connection()
    if success:
        st.sidebar.success("Kết nối thành công!")
        st.session_state.minio_client = result
        st.session_state.buckets = [bucket.name for bucket in buckets]  # Chỉ lưu tên bucket
        st.session_state.upload_disabled = False
    else:
        st.sidebar.error(f"Kết nối thất bại: {result}")
        st.session_state.upload_disabled = True

# Nếu kết nối thành công, hiện thị các nút khác
if not st.session_state.upload_disabled:
    # Tạo danh sách bucket có sẵn
    bucket_names = st.session_state.buckets.copy()  # Sao chép danh sách bucket
    bucket_names.append("Tạo bucket mới")  # Thêm tùy chọn "Tạo bucket mới"

    # Chọn bucket từ danh sách
    st.session_state.selected_bucket = st.sidebar.selectbox("Chọn bucket để tải lên", bucket_names)

    # Trường nhập tên bucket mới nếu "Tạo bucket mới" được chọn
    if st.session_state.selected_bucket == "Tạo bucket mới":
        new_bucket_name = st.sidebar.text_input("Nhập tên bucket mới:")
        
        if st.sidebar.button("Tạo bucket"):  # Nút tạo bucket nằm trong điều kiện này
            if new_bucket_name:
                if is_valid_bucket_name(new_bucket_name):
                    try:
                        minio_client = st.session_state.minio_client
                        minio_client.make_bucket(new_bucket_name)
                        st.sidebar.success(f"Bucket '{new_bucket_name}' đã được tạo thành công!")

                        # Cập nhật danh sách bucket
                        st.session_state.buckets.append(new_bucket_name)  # Lưu tên bucket mới
                        # Cập nhật lại selectbox
                        st.session_state.selected_bucket = new_bucket_name  # Cập nhật bucket đã chọn
                    except S3Error as err:
                        st.sidebar.error(f"Lỗi tạo bucket: {err}")
                else:
                    st.sidebar.warning("Tên bucket không hợp lệ! Vui lòng nhập tên chỉ chứa chữ thường, số và dấu '-' (từ 3-63 ký tự).")
            else:
                st.sidebar.warning("Vui lòng nhập tên bucket mới.")

    # Nút tải file lên
    uploaded_files = st.sidebar.file_uploader("Chọn file để tải lên", accept_multiple_files=True, disabled=st.session_state.upload_disabled)

    if uploaded_files:  # Kiểm tra nếu có tệp đã được chọn
        for uploaded_file in uploaded_files:
            if st.sidebar.button(f"Tải lên {uploaded_file.name}", disabled=st.session_state.upload_disabled):
                try:
                    minio_client = st.session_state.minio_client
                    target_bucket = st.session_state.selected_bucket  # Sử dụng bucket đã chọn

                    if not target_bucket or target_bucket == "Tạo bucket mới":
                        st.sidebar.error("Vui lòng chọn một bucket hợp lệ để tải lên.")
                    else:
                        minio_client.put_object(target_bucket, uploaded_file.name, uploaded_file, uploaded_file.size)
                        st.sidebar.success(f"Tải lên thành công: {uploaded_file.name}")
                except S3Error as err:
                    st.sidebar.error(f"Lỗi tải lên: {err}")

# Đảm bảo nút tải lên bị khóa khi chưa kết nối
if st.session_state.upload_disabled:
    st.sidebar.button("Tải lên", disabled=True)

# Kết thúc Spark session
spark.stop()
