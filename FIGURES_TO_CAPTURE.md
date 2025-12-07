# Danh sách ảnh cần chụp cho Pipeline Documentation

## 1. Frontend Interface (`figures/frontend_interface.png`)

**Mô tả**: Chụp toàn bộ giao diện frontend khi đang sử dụng

- Main canvas hiển thị base image với red overlay (mask A) đang được vẽ
- Brush tool đang active
- Sidebar hiển thị các controls: model selection (FastSAM/BiRefNet), brush size, generation parameters
- Có thể thấy một số controls khác như prompt input, settings

**Cách chụp**:

- Mở ứng dụng, upload một base image
- Vẽ một số brush strokes để tạo mask A (sẽ thấy red overlay)
- Đảm bảo sidebar và các controls đều visible

---

## 2. Reference Image Editor (`figures/reference_image_editor.png`)

**Mô tả**: Chụp modal Reference Image Editor khi đang mở

- Left panel: reference image với brush strokes (red overlay) đang được vẽ trên object
- Right panel: extracted object với transparent/black background (nếu đã extract)
- Toolbar: model selection buttons (Segmentation/BiRefNet), border adjustment slider, auto-detect button
- Các controls như brush size, clear button

**Cách chụp**:

- Mở Reference Image Editor modal
- Upload một reference image (ví dụ: remote control trên orange mat)
- Vẽ brush strokes trên object
- Có thể extract object để thấy right panel

---

## 3. FastSAM Mask Generation (`figures/fastsam_mask_generation.png`)

**Mô tả**: So sánh brush strokes và kết quả mask từ FastSAM

- Left side: Ảnh gốc với brush strokes (red overlay) mà user đã vẽ
- Right side: Kết quả mask (white = object, black = background) sau khi FastSAM xử lý
- Có thể chụp 2 ảnh riêng rồi ghép lại, hoặc chụp màn hình debug panel

**Cách chụp**:

- Vẽ brush strokes trên một object
- Chọn model "Segmentation" (FastSAM)
- Generate mask
- Chụp: (1) Ảnh với brush strokes, (2) Kết quả mask binary

---

## 4. BiRefNet Mask Generation (`figures/birefnet_mask_generation.png`)

**Mô tả**: So sánh 2-stage refinement của BiRefNet

- Left side: FastSAM initial mask (từ brush strokes)
- Right side: BiRefNet refined mask (sau khi refine trên cropped region)
- Có thể thấy sự khác biệt về độ chính xác của boundaries

**Cách chụp**:

- Vẽ brush strokes trên một object
- Chọn model "BiRefNet"
- Generate mask (sẽ chạy FastSAM trước, rồi BiRefNet)
- Chụp: (1) FastSAM mask result, (2) BiRefNet final mask result
- Hoặc chụp debug panel nếu có hiển thị cả 2 stages

---

## 6. Example Base Image (`figures/example_base_image.png`)

**Mô tả**: Base image - wooden table

- Ảnh một cái bàn gỗ (wooden table)
- Không có object nào trên bàn (hoặc có thể có một số items nhỏ)
- Background đơn giản, lighting tốt

**Cách chụp**: Chụp ảnh thật hoặc dùng stock photo

---

## 7. Example Reference Image (`figures/example_reference_image.png`)

**Mô tả**: Reference image - remote control trên orange mat

- Remote control màu đen
- Đặt trên một tấm mat màu cam (orange)
- Background có thể là wooden table hoặc surface khác
- Object rõ ràng, dễ phân biệt với background

**Cách chụp**: Chụp ảnh thật hoặc dùng stock photo

---

## 8. Example Result (`figures/example_result.png`)

**Mô tả**: Final result - remote control đã được insert vào wooden table

- Remote control xuất hiện trên wooden table (từ example_base_image)
- Lighting và shadows tự nhiên, hòa hợp với background
- Object được integrate một cách seamless

**Cách chụp**:

- Chạy pipeline với example_base_image và example_reference_image
- Chụp kết quả final generated image

---

## 9. Example Mask A (`figures/example_mask_a.png`)

**Mô tả**: Main mask A với red overlay trên base image

- Base image (wooden table) với red semi-transparent overlay
- Overlay chỉ ở vùng mà user muốn đặt object (placement region)
- Có thể là bounding box hoặc brush strokes đã được convert thành mask

**Cách chụp**:

- Vẽ mask A trên base image (wooden table)
- Chụp màn hình với red overlay visible
- Hoặc chụp mask binary (white/black) riêng

---

## 10. Example Mask R (`figures/example_mask_r.png`)

**Mô tả**: Reference mask R isolating remote control

- Reference image với red overlay chỉ trên remote control
- Hoặc mask binary (white = remote control, black = background)
- Mask phải chính xác, follow boundaries của remote control

**Cách chụp**:

- Vẽ mask trên reference image (remote control)
- Chụp với red overlay hoặc mask binary result

---

## 12. Debug Info (`figures/debug_info.png`)

**Mô tả**: Debug panel hiển thị tất cả intermediate images

- Grid layout hiển thị:
  1. Original base image
  2. Main mask A
  3. Masked reference object (extracted object)
  4. Original reference image
  5. Positioned mask R
  6. Final generated image
- Có labels hoặc numbers cho mỗi image
- Có thể có metadata như timestamps, parameters

**Cách chụp**:

- Sau khi generation hoàn thành, mở debug panel/session
- Chụp toàn bộ panel với tất cả images visible
- Hoặc chụp từng image riêng rồi ghép lại

---

## 13. Architecture Diagram (`figures/architecture_diagram.png`)

**Mô tả**: System architecture diagram

- API Gateway ở center/top
- Các services: Segmentation Service (T4 GPU), Image Utils Service (CPU), Job Manager
- A100 GPU workers
- Modal Volumes
- Arrows showing request flow
- Labels cho mỗi component

**Cách chụp**:

- Vẽ diagram bằng tool như draw.io, Lucidchart, hoặc PowerPoint
- Hoặc chụp architecture diagram từ documentation nếu có
- Cần rõ ràng, professional, dễ đọc

---

## 14. Application E-commerce (`figures/application_ecommerce.png`)

**Mô tả**: Example e-commerce application

- Product được insert vào một scene context
- Ví dụ: một sản phẩm (như laptop, phone) được đặt trên desk hoặc trong room setting
- Professional look, suitable cho marketing

**Cách chụp**:

- Chạy pipeline với một product image và scene image
- Hoặc dùng example có sẵn từ hệ thống

---

## 15. Application Interior (`figures/application_interior.png`)

**Mô tả**: Example interior design application

- Furniture hoặc decor item được visualize trong một real space
- Ví dụ: một chiếc ghế được đặt trong living room
- Natural lighting, realistic integration

**Cách chụp**:

- Chạy pipeline với furniture image và room image
- Hoặc dùng example có sẵn

---

## 16. Application Content (`figures/application_content.png`)

**Mô tả**: Example content creation/digital art

- Composite image với elements từ nhiều sources
- Creative/artistic look
- Ví dụ: combine objects từ nhiều ảnh khác nhau

**Cách chụp**:

- Chạy pipeline với creative combination
- Hoặc dùng example có sẵn

---

## Tips cho việc chụp ảnh:

1. **Screenshot tools**: Sử dụng browser DevTools, screenshot extensions, hoặc built-in screenshot tools
2. **Resolution**: Chụp ở resolution cao (ít nhất 1920x1080) để đảm bảo chất lượng khi resize
3. **Consistency**: Giữ cùng một style, color scheme cho tất cả screenshots
4. **Annotations**: Có thể thêm arrows, labels, numbers nếu cần (sẽ làm rõ hơn trong document)
5. **File format**: Lưu dưới dạng PNG để giữ chất lượng tốt
6. **Naming**: Đặt tên file đúng như trong document (ví dụ: `frontend_interface.png`)

## Thứ tự ưu tiên:

**Quan trọng nhất** (cần có để document hoàn chỉnh):

1. frontend_interface.png
2. reference_image_editor.png
3. example_base_image.png
4. example_reference_image.png
5. example_result.png

**Quan trọng** (làm rõ technical details): 6. fastsam_mask_generation.png 7. birefnet_mask_generation.png 8. example_mask_a.png 9. example_mask_r.png 10. debug_info.png

**Bổ sung** (làm document đẹp hơn): 11. architecture_diagram.png 12. application_ecommerce.png 13. application_interior.png 14. application_content.png
