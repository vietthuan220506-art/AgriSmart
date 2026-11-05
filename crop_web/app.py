
import sys
import types
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="AgriSmart - Dự đoán cây trồng",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. THIẾT KẾ GIAO DIỆN
# =========================================================

st.markdown("""
<style>

/* Nền tổng thể */
.stApp {
    background-color: #F4F8F5;
}

/* Nội dung chính */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Tiêu đề */
h1, h2, h3 {
    color: #164E36 !important;
    font-weight: 750 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123F30, #1E6547);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Thẻ thông tin */
.info-card {
    background: white;
    border: 1px solid #E1ECE4;
    border-radius: 15px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(30, 80, 50, 0.05);
}

.info-number {
    font-size: 29px;
    font-weight: 800;
    color: #167348;
    margin-top: 8px;
}

.info-label {
    font-size: 14px;
    color: #64756A;
    font-weight: 600;
}

/* Banner */
.hero {
    background: linear-gradient(120deg, #155D3B, #269B65);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 6px 20px rgba(22, 100, 62, 0.16);
}

.hero h1 {
    color: white !important;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #E6F6EB;
    font-size: 15px;
    margin-bottom: 0;
}

/* Các hộp nội dung */
.panel {
    background: white;
    border: 1px solid #E2ECE5;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

/* Thẻ kết quả */
.crop-card {
    background: white;
    border: 1px solid #DDEBE1;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 3px 10px rgba(20, 80, 45, 0.05);
}

.crop-rank {
    color: #218653;
    font-size: 13px;
    font-weight: 750;
}

.crop-name {
    color: #174D34;
    font-size: 21px;
    font-weight: 800;
    margin: 7px 0;
}

.crop-percent {
    color: #20804D;
    font-size: 25px;
    font-weight: 800;
}

/* Nút */
.stButton > button {
    background: linear-gradient(90deg, #187548, #269B65);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #105D38, #197C4D);
    color: white;
    border: none;
}

/* Nút nhập liệu */
div[data-testid="stNumberInput"] input {
    border-radius: 9px;
}

/* Thông báo */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Đường phân cách */
hr {
    border-color: #DFEAE2;
}

/* Ẩn menu mặc định */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. KẾT NỐI MODULE VỚI FILE PKL
# =========================================================

import thuat_toan

models_package = types.ModuleType("models")
models_package.__path__ = []

N_package = types.ModuleType("models.N")
N_package.__path__ = []

models_package.N = N_package
N_package.thuat_toan = thuat_toan

sys.modules["models"] = models_package
sys.modules["models.N"] = N_package
sys.modules["models.N.thuat_toan"] = thuat_toan


# =========================================================
# 4. TẢI MÔ HÌNH
# =========================================================

DUONG_DAN = Path(__file__).parent / "crop_models.pkl"


@st.cache_resource
def tai_mo_hinh():
    return joblib.load(DUONG_DAN)


try:
    du_lieu = tai_mo_hinh()

    cac_mo_hinh = du_lieu["models"]
    X_train = du_lieu["X_train"]
    features = list(du_lieu["features"])
    labels = list(du_lieu["labels"])

except Exception as e:
    st.error("Không thể tải mô hình.")
    st.exception(e)
    st.stop()


# =========================================================
# 5. THÔNG TIN CHỈ SỐ
# =========================================================

mo_ta_chi_so = {
    "N": {
        "ten": "Nitrogen (Đạm)",
        "don_vi": "mg/kg",
        "mo_ta": "Hàm lượng đạm trong đất, có vai trò quan trọng đối với sự phát triển thân và lá.",
        "icon": "🟢"
    },
    "P": {
        "ten": "Phosphorus (Lân)",
        "don_vi": "mg/kg",
        "mo_ta": "Hàm lượng lân trong đất, hỗ trợ phát triển bộ rễ và quá trình ra hoa.",
        "icon": "🔵"
    },
    "K": {
        "ten": "Potassium (Kali)",
        "don_vi": "mg/kg",
        "mo_ta": "Hàm lượng kali trong đất, hỗ trợ điều hòa nước và tăng khả năng chống chịu của cây.",
        "icon": "🟠"
    },
    "pH": {
        "ten": "Độ pH của đất",
        "don_vi": "pH",
        "mo_ta": "Thể hiện mức độ axit hoặc kiềm của đất, ảnh hưởng đến khả năng hấp thụ dinh dưỡng.",
        "icon": "🟣"
    }
}


gioi_han = {}

for c in features:
    gioi_han[c] = {
        "min": float(X_train[c].min()),
        "max": float(X_train[c].max()),
        "median": float(X_train[c].median())
    }


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🌱 AgriSmart")

    st.markdown("### HỆ THỐNG HỖ TRỢ NÔNG NGHIỆP")

    st.markdown("---")

    st.markdown("### 📌 Giới thiệu")

    st.write(
        "AgriSmart là hệ thống hỗ trợ tham khảo "
        "loại cây trồng dựa trên các chỉ số dinh dưỡng "
        "và độ pH của đất."
    )

    st.markdown("---")

    st.markdown("### ⚙️ Công nghệ sử dụng")

    st.write("🌳 Random Forest")
    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🖥️ Streamlit")

    st.markdown("---")

    st.markdown("### 📋 Hướng dẫn")

    st.write("1. Chọn các chỉ số đất.")
    st.write("2. Nhập giá trị tương ứng.")
    st.write("3. Nhấn nút dự đoán.")
    st.write("4. Xem kết quả và cảnh báo.")

    st.markdown("---")

    st.caption("AgriSmart | Crop Recommendation System")


# =========================================================
# 7. BANNER TRANG CHỦ
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🌱 AGRISMART</h1>
    <p>
        HỆ THỐNG HỖ TRỢ DỰ ĐOÁN CÂY TRỒNG
    </p>
    <p>
        Phân tích các chỉ số đất và gợi ý những loại cây
        có khả năng phù hợp theo mô hình học máy.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. TỔNG QUAN HỆ THỐNG
# =========================================================

st.markdown("## 📊 Tổng quan dữ liệu và mô hình")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">🌾 Số loại cây</div>
        <div class="info-number">{len(labels)}</div>
        <div class="info-label">Lớp phân loại</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">📁 Số mẫu dữ liệu</div>
        <div class="info-number">{len(X_train):,}</div>
        <div class="info-label">Mẫu huấn luyện</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">🧪 Chỉ số đầu vào</div>
        <div class="info-number">{len(features)}</div>
        <div class="info-label">Thông số đất</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">🌳 Mô hình</div>
        <div class="info-number">{len(cac_mo_hinh)}</div>
        <div class="info-label">Tổ hợp đầu vào</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.info(
    "Hệ thống hiện sử dụng các mô hình Random Forest "
    "đã được huấn luyện và lưu trong crop_models.pkl."
)


# =========================================================
# 9. THÔNG TIN VÀ GIỚI HẠN CHỈ SỐ
# =========================================================

with st.expander("📖 Tìm hiểu các chỉ số đất và giới hạn dữ liệu"):

    st.markdown("### Ý nghĩa các chỉ số")

    for c in features:

        if c in mo_ta_chi_so:

            info = mo_ta_chi_so[c]

            st.markdown(
                f"**{info['icon']} {c} - {info['ten']}**"
            )

            st.write(info["mo_ta"])

    st.markdown("### Phạm vi dữ liệu huấn luyện")

    bang_gioi_han = pd.DataFrame([
        {
            "Chỉ số": c,
            "Giá trị nhỏ nhất": gioi_han[c]["min"],
            "Giá trị lớn nhất": gioi_han[c]["max"]
        }
        for c in features
    ])

    st.dataframe(
        bang_gioi_han,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Đây là phạm vi xuất hiện trong dữ liệu huấn luyện, "
        "không phải giới hạn sinh học hoặc tiêu chuẩn nông học "
        "áp dụng cho tất cả các loại đất."
    )


st.divider()


# =========================================================
# 10. KHU VỰC NHẬP DỮ LIỆU
# =========================================================

st.markdown("## 🧪 Phân tích điều kiện đất")

st.write(
    "Hãy lựa chọn những chỉ số bạn có thể đo được. "
    "Hệ thống hỗ trợ nhập từ 1 đến 4 chỉ số."
)

with st.container(border=True):

    st.markdown("### 1. Lựa chọn chỉ số")

    chi_so_chon = st.multiselect(
        "Chọn các chỉ số muốn sử dụng:",
        options=features,
        default=features,
        format_func=lambda x: (
            f"{x} - {mo_ta_chi_so[x]['ten']}"
            if x in mo_ta_chi_so else x
        )
    )

    chi_so_chon = [
        c for c in features if c in chi_so_chon
    ]

    if not chi_so_chon:

        st.warning(
            "Bạn chưa chọn chỉ số nào. "
            "Hãy chọn ít nhất một chỉ số để tiếp tục."
        )

    else:

        st.markdown("---")
        st.markdown("### 2. Nhập giá trị đo được")

        st.caption(
            "Bạn nên sử dụng kết quả đo đất thực tế. "
            "Không nên nhập giá trị ước lượng nếu chưa có căn cứ."
        )

        gia_tri = {}

        with st.form("form_du_doan"):

            cot = st.columns(2)

            for i, c in enumerate(chi_so_chon):

                with cot[i % 2]:

                    info = mo_ta_chi_so.get(c, {})

                    ten = info.get("ten", c)
                    don_vi = info.get("don_vi", "")

                    st.markdown(f"**{info.get('icon', '🧪')} {ten}**")

                    st.caption(info.get("mo_ta", ""))

                    if c == "pH":

                        gia_tri[c] = st.number_input(
                            f"Giá trị {c}",
                            min_value=0.0,
                            max_value=14.0,
                            value=float(gioi_han[c]["median"]),
                            step=0.1,
                            format="%.2f",
                            help="Thang đo pH từ 0 đến 14."
                        )

                    else:

                        gia_tri[c] = st.number_input(
                            f"Giá trị {c} ({don_vi})",
                            min_value=0.0,
                            max_value=max(
                                1000.0,
                                gioi_han[c]["max"] * 2
                            ),
                            value=float(gioi_han[c]["median"]),
                            step=1.0,
                            format="%.2f",
                            help=(
                                "Giá trị cần không âm và phải "
                                "nằm trong phạm vi dữ liệu huấn luyện."
                            )
                        )

            st.markdown("---")

            xac_nhan = st.form_submit_button(
                "🌱 PHÂN TÍCH VÀ DỰ ĐOÁN",
                use_container_width=True
            )


# =========================================================
# 11. HÀM KIỂM TRA ĐẦU VÀO
# =========================================================

def kiem_tra_du_lieu(gia_tri, chi_so_chon):

    loi = []
    canh_bao = []

    for c in chi_so_chon:

        x = gia_tri[c]

        min_dl = gioi_han[c]["min"]
        max_dl = gioi_han[c]["max"]

        if c in ["N", "P", "K"] and x < 0:

            loi.append(f"{c} không được nhỏ hơn 0.")

        if c == "pH" and not 0 <= x <= 14:

            loi.append("pH phải nằm trong khoảng 0 đến 14.")

        if x < min_dl or x > max_dl:

            loi.append(
                f"{c} = {x:.2f} nằm ngoài phạm vi "
                f"dữ liệu huấn luyện ({min_dl:.2f} đến {max_dl:.2f})."
            )

        elif max_dl > min_dl:

            vi_tri = (x - min_dl) / (max_dl - min_dl)

            if vi_tri < 0.05 or vi_tri > 0.95:

                canh_bao.append(
                    f"{c} đang nằm gần biên dữ liệu huấn luyện."
                )

    return loi, canh_bao


# =========================================================
# 12. XỬ LÝ DỰ ĐOÁN
# =========================================================

if xac_nhan and chi_so_chon:

    loi, canh_bao = kiem_tra_du_lieu(
        gia_tri,
        chi_so_chon
    )

    st.divider()

    st.markdown("## 🎯 Kết quả phân tích")

    if loi:

        st.error(
            "Không thể thực hiện dự đoán vì dữ liệu đầu vào "
            "chưa đáp ứng điều kiện."
        )

        for l in loi:
            st.write("❌ " + l)

        st.info(
            "Hãy kiểm tra lại các chỉ số và nhập giá trị "
            "nằm trong phạm vi dữ liệu huấn luyện."
        )

    else:

        for cb in canh_bao:
            st.warning("⚠️ " + cb)

        ten_to_hop = tuple(chi_so_chon)

        if ten_to_hop not in cac_mo_hinh:

            st.error(
                "Không tìm thấy mô hình cho tổ hợp chỉ số "
                "đã chọn. Vui lòng kiểm tra lại dữ liệu mô hình."
            )

        else:

            mo_hinh = cac_mo_hinh[ten_to_hop]

            mau_moi = pd.DataFrame(
                [[gia_tri[c] for c in chi_so_chon]],
                columns=chi_so_chon
            )

            try:

                xac_suat = mo_hinh.predict_proba(mau_moi)[0]

                if not all(pd.notna(xac_suat)):

                    st.error("Mô hình trả về xác suất không hợp lệ.")

                    st.stop()

                if abs(float(sum(xac_suat)) - 1.0) > 1e-6:

                    st.error("Tổng xác suất dự đoán không hợp lệ.")

                    st.stop()

            except Exception as e:

                st.error("Có lỗi khi thực hiện dự đoán.")
                st.exception(e)
                st.stop()

            thu_tu = xac_suat.argsort()[::-1]

            thu_tu = [
                i for i in thu_tu
                if xac_suat[i] > 1e-12
            ]

            if len(thu_tu) == 0:

                st.error("Không tìm thấy kết quả dự đoán.")

            else:

                top3 = thu_tu[:3]

                ti_le_cao_nhat = float(xac_suat[top3[0]])

                if len(top3) >= 2:

                    ti_le_thu_hai = float(xac_suat[top3[1]])

                    khoang_cach = (
                        ti_le_cao_nhat - ti_le_thu_hai
                    )

                else:

                    khoang_cach = ti_le_cao_nhat

                # =========================================
                # 13. TÓM TẮT ĐẦU VÀO
                # =========================================

                st.markdown("### 📋 Thông tin đã phân tích")

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        '<div class="panel">'
                        '<b>🔬 Tổ hợp chỉ số</b><br><br>'
                        + " + ".join(chi_so_chon)
                        + '</div>',
                        unsafe_allow_html=True
                    )

                with c2:

                    st.markdown(
                        '<div class="panel">'
                        '<b>🌳 Thuật toán</b><br><br>'
                        'Random Forest'
                        '</div>',
                        unsafe_allow_html=True
                    )

                bang_dau_vao = pd.DataFrame({
                    "Chỉ số": chi_so_chon,
                    "Giá trị đã nhập": [
                        gia_tri[c] for c in chi_so_chon
                    ],
                    "Đơn vị": [
                        mo_ta_chi_so.get(c, {}).get("don_vi", "")
                        for c in chi_so_chon
                    ]
                })

                st.dataframe(
                    bang_dau_vao,
                    use_container_width=True,
                    hide_index=True
                )

                # =========================================
                # 14. KẾT QUẢ TOP 3
                # =========================================

                st.markdown("### 🏆 Top 3 loại cây được gợi ý")

                st.caption(
                    "Các loại cây được sắp xếp theo xác suất "
                    "mô hình dự đoán từ cao xuống thấp."
                )

                cot_ket_qua = st.columns(3)

                mau_the = [
                    "#E4F5E9",
                    "#E8F1FC",
                    "#FFF3DC"
                ]

                for vi_tri, i in enumerate(top3):

                    ten_cay = str(mo_hinh.classes_[i])
                    ti_le = float(xac_suat[i])

                    with cot_ket_qua[vi_tri]:

                        st.markdown(
                            f"""
                            <div class="crop-card"
                                 style="background:{mau_the[vi_tri]}">
                                <div class="crop-rank">
                                    VỊ TRÍ {vi_tri + 1}
                                </div>
                                <div class="crop-name">
                                    🌾 {ten_cay}
                                </div>
                                <div class="crop-percent">
                                    {ti_le:.2%}
                                </div>
                                <div style="color:#63756A;font-size:12px">
                                    Điểm xác suất mô hình
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.progress(ti_le)

                # =========================================
                # 15. BIỂU ĐỒ XÁC SUẤT
                # =========================================

                st.markdown("### 📊 So sánh xác suất dự đoán")

                du_lieu_bieu_do = pd.DataFrame({
                    "Loại cây": [
                        str(mo_hinh.classes_[i])
                        for i in top3
                    ],
                    "Xác suất (%)": [
                        float(xac_suat[i]) * 100
                        for i in top3
                    ]
                }).set_index("Loại cây")

                st.bar_chart(
                    du_lieu_bieu_do,
                    color="#239B62",
                    horizontal=True
                )

                # =========================================
                # 16. ĐÁNH GIÁ KẾT QUẢ
                # =========================================

                st.markdown("### 🔎 Đánh giá mức độ phân biệt")

                if ti_le_cao_nhat >= 0.70 and khoang_cach >= 0.20:

                    st.success(
                        "Mô hình phân biệt tương đối rõ "
                        "loại cây đứng đầu so với loại cây đứng thứ hai."
                    )

                elif ti_le_cao_nhat >= 0.50:

                    st.warning(
                        "Kết quả có sự khác biệt nhưng chưa thật sự rõ. "
                        "Nên tham khảo thêm các điều kiện canh tác."
                    )

                else:

                    st.error(
                        "Mô hình chưa đưa ra được một lựa chọn nổi bật. "
                        "Nên thận trọng khi tham khảo kết quả."
                    )

                if len(top3) >= 2 and khoang_cach < 0.10:

                    st.warning(
                        "Hai loại cây đứng đầu có xác suất gần nhau. "
                        "Mô hình chưa phân biệt rõ giữa hai loại cây."
                    )

                st.caption(
                    "Các ngưỡng đánh giá trên là quy tắc cảnh báo "
                    "do giao diện đặt ra, chưa phải tiêu chuẩn "
                    "đánh giá khoa học đã được kiểm định."
                )

                # =========================================
                # 17. LƯU Ý VỀ KẾT QUẢ
                # =========================================

                with st.expander("📌 Lưu ý khi sử dụng kết quả"):

                    st.markdown("""
                    **1. Về xác suất dự đoán**

                    Xác suất thể hiện mức độ mô hình phân bổ
                    cho từng loại cây. Xác suất cao không có nghĩa
                    cây chắc chắn phát triển tốt.

                    **2. Về số lượng chỉ số đầu vào**

                    Nếu chỉ nhập 1 hoặc 2 chỉ số, mô hình chỉ sử dụng
                    những thông tin đó. Kết quả có thể thiếu thông tin
                    so với trường hợp nhập đủ 4 chỉ số.

                    **3. Về điều kiện canh tác**

                    Khả năng phát triển của cây còn phụ thuộc vào
                    khí hậu, lượng mưa, nhiệt độ, độ ẩm, loại đất,
                    mùa vụ, giống cây và kỹ thuật chăm sóc.

                    **4. Về dữ liệu**

                    Mô hình chỉ học từ dữ liệu huấn luyện hiện có.
                    Những điều kiện khác biệt với dữ liệu huấn luyện
                    có thể làm giảm độ tin cậy của dự đoán.

                    **5. Khuyến nghị**

                    Nên sử dụng kết quả như thông tin tham khảo,
                    kết hợp với kiểm tra đất thực tế và tư vấn
                    chuyên môn trước khi quyết định gieo trồng.
                    """)


# =========================================================
# 18. CHÂN TRANG
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px; color:#718278;">
    <b style="color:#19734A;">🌱 AGRISMART</b><br>
    Hệ thống hỗ trợ dự đoán cây trồng bằng học máy<br>
    <small>Random Forest | Python | Streamlit</small>
</div>
""", unsafe_allow_html=True)
