# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/09/2026**

**Runtime Colab:** T4-GPU

**Python / PyTorch / Ultralytics:** Python: 3.13.15
PyTorch: 2.11.0+cu128
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
Mô hình phân loại toàn bộ ảnh vào lớp cab với độ tin cậy khoảng 0.511 (51.1%). Đây là prediction ở cấp ảnh, nên nó chỉ cho biết nhãn tổng quát mà mô hình cho là phù hợp nhất với toàn bức ảnh, không chỉ ra vị trí cụ thể của vật thể trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list được định nghĩa bởi taxonomy của bộ dữ liệu dùng để huấn luyện checkpoint. Trong trường hợp này, yolo11n-cls.pt sử dụng taxonomy ImageNet-1K, nên mô hình dự đoán trong tập các lớp của ImageNet-1K.
dataset/taxonomy định nghĩa lớp, model học để chọn lớp.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Vì 3 cái này phục vụ 3 mục đích khác nhau:

class_id: máy xử lý nhanh, ổn định, dễ lưu trong JSON/DB.
class_name: con người đọc hiểu được ngay lớp đó là gì.
taxonomy_name: cho biết class_id và class_name đó thuộc bộ nhãn nào.

Cái quan trọng nhất là class_id = 468 chỉ có ý nghĩa khi biết taxonomy. Ở bài của bạn, taxonomy_name = ImageNet-1K, nên mới hiểu 468 tương ứng với cab. Nếu đổi taxonomy khác thì cùng ID đó có thể mang nghĩa khác.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline cần quy định rõ cách chọn nhãn khi ảnh có nhiều chủ thể, ví dụ ưu tiên chủ thể chính/chiếm diện tích lớn nhất, xử lý thế nào khi có nhiều chủ thể ngang nhau, và khi nào cần chuyển sang multi-label hoặc escalation thay vì ép ảnh vào một class duy nhất.
- Vì sao model score không phải ground truth?
Model score không phải ground truth vì score chỉ thể hiện mức độ tự tin của mô hình đối với prediction. Ground truth là nhãn chuẩn được xác định theo guideline/dữ liệu đã được xác nhận, dùng để đánh giá prediction của model.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
"class_name": "bus",
"score": 0.912558,
"bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
"bbox_width": 129.84,
"bbox_height": 132.96
- Diễn giải vị trí box bằng lời:
Bounding box của bus kéo dài từ (225.61, 132.68) đến (296.09, 220.45) trên ảnh kích thước 640×428 pixel. Box nằm ở vùng trung tâm hơi lệch trái và hơi về phía trên của ảnh, với chiều rộng khoảng 70.48 px và chiều cao khoảng 87.76 px.
- So sánh số prediction ở hai threshold:
Ở threshold 0.20, mô hình phát hiện 17 vật thể; ở threshold 0.35, còn 11 vật thể. Khi tăng threshold, số prediction giảm do các dự đoán có độ tin cậy thấp hơn bị loại bỏ.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?Ở threshold 0.20 có 17 prediction, còn 0.35 chỉ có 11 prediction. Vì vậy threshold thấp cho độ bao phủ cao hơn nhưng làm reviewer phải xem nhiều prediction hơn; threshold cao giảm khối lượng review nhưng có thể bỏ sót vật thể.
- Đề xuất một quy tắc box chặt: Box phải được đặt sát biên ngoài của phần object nhìn thấy, chỉ bao gồm object mục tiêu và hạn chế tối đa background; không được cắt mất phần object còn nhìn thấy.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định cách xử lý object bị che khuất hoặc cắt mép, ví dụ box chỉ bao phần còn nhìn thấy hay ước lượng toàn bộ object, ngưỡng che khuất nào vẫn được gán nhãn và trường hợp nào phải escalation cho reviewer/lead để quyết định thay vì annotator tự suy đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):"instance_id": "traffic-001",
    "instance_id": "traffic-001",
    "class_name": "bus",
    "score": 0.925745,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0]]
- Polygon bổ sung chi tiết gì so với box? Polygon bổ sung thông tin chi tiết về hình dạng và đường biên thực tế của từng object, nên phân biệt chính xác vùng thuộc object với background tốt hơn bounding box. Box chỉ cho biết vùng chữ nhật bao quanh đối tượng, còn polygon bám theo biên của đối tượng.
- `instance_id` dùng để làm gì và không phải loại ID nào?instance_id dùng để định danh từng instance riêng biệt trong ảnh, giúp phân biệt các object khác nhau dù chúng có cùng class. Nó không phải class_id của taxonomy và cũng không mặc định là tracking ID hay ID toàn cục dùng xuyên nhiều ảnh/frame.
- Đề xuất một quy tắc biên mask: Khi ranh giới object nhìn thấy rõ, polygon/mask phải đi sát biên ngoài của object; không tự suy đoán phần bị che khuất và không bao gồm background không thuộc object.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định cách xác định biên mask khi vùng bị mờ, hai object tiếp xúc hoặc object bị che khuất; ví dụ chỉ annotate phần nhìn thấy hay ước lượng phần bị che, cách tách các instance chạm nhau, và ngưỡng mơ hồ nào phải escalation cho reviewer/lead thay vì annotator tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn class cho toàn ảnh | Ảnh có nhiều chủ thể hoặc chủ thể chính không rõ | Gán class theo guideline, không dựa chỉ vào model score | Kiểm tra class có đúng guideline và đúng chủ thể chính không |
| Phát hiện vật thể | Class + bounding box cho từng object | Box có thể dư background, cắt mất object, object bị che khuất/cắt mép | Vẽ box ôm sát object và gán đúng class | Kiểm tra class, độ chặt của box, object bị bỏ sót hoặc box bị trùng |
| Instance segmentation | Class + polygon/mask cho từng instance | Biên mờ, object tiếp xúc nhau, che khuất | Vẽ mask/polygon bám sát phần object nhìn thấy và tách đúng instance | Kiểm tra biên mask, phần background bị lấn, phần object bị thiếu và việc tách instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu đúng phạm vi được phép, không chia sẻ dữ liệu, ảnh, token, password hoặc thông tin nhạy cảm ra ngoài môi trường được quy định.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach/mentor.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
