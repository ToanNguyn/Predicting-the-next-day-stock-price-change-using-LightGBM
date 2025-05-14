## I. Giới thiệu về bài làm 
Đây là bài làm miniproject dự đoán chênh lệch giá trong 1 ngày của 4 mã cổ phiểu FPT, VIC, MSN, PNJ trong giai đoạn từ năm 2018 đến 2020. Trong file README.md này, em sẽ trình bày các bước thực hiện, giải thích chi tiết cũng như kết quả của bài làm.

## II. Các folders có trong repo
### 1. Datasets 
Gồm datasets của 4 mã cổ phiếu FPT, MSN, VIC, PNJ.
### 2. Notebooks
Mỗi notebook chứa các bước xử lý cũng như model của mã cổ phiếu tương ứng.
### 3. SRC
Các function dùng để trực quan hóa data (visualization.py), phân chia data (data_segregation.py), tiền xử lý (preprocessing.py), tối ưu hyperparameter (tunning.py).
### 4. File ảnh 
Ảnh các chart trực quan hóa kết quả predicted của 4 mã cổ phiếu tương ứng.

## III. Các bước làm
Các bước xử lý trong mỗi notebook cho từng mã cổ phiếu là giống nhau, điểm khác biệt duy nhất nằm ở phần tuning, nơi sẽ tìm bộ tham số phù hợp riêng cho từng tập dữ liệu.
### 1. Load libraries and datasets
- Import các libraries được sử dụng.
- Import dataset. 
### 2. EDA
#### 2.1 Descriptive Statistics
- Kiểm tra các các thông tin cơ bản của dataset như phân bố, kiểu dữ liệu, dữ liệu bị thiếu, trùng lặp hoặc không hợp lệ qua các hàm: info(), describe(), duplicated().
- Hàm aggregate_intraday_to_daily(): Nhận thấy data trong datasets được cung cấp rời rạc, không liên tục theo phút, giờ. Điều này khiến data xuất hiện nhiều biến động mạnh và kết quả dự đoán sẽ khó chính xác hơn. Cùng với đó việc tạo thêm các features như moving average, shift,... khó hơn và xuất hiện nhiều missing values hơn. Do đó, toàn bộ data một ngày sẽ được gộp lại một dòng duy nhất. Cụ thể: Open là giá trị Open đầu tiên trong ngày, High là giá trị High cao nhất trong ngày, Low là giá trị Low thấp nhất trong ngày, Close là giá trị Close cuối cùng trong ngày, Volume là tổng Volume trong một ngày. Ngoài ra hai features Ticker và Open Interest chỉ chứa một giá trị duy nhất, không mang nhiều giá trị dự đoán sẽ được loại bỏ.
  
#### 2.2. Close price visualization 
- Trực quan hóa Close price theo ngày và đưa ra nhận xét.

#### 2.3. Volume over time 
- Trực quan hóa Volume theo ngày và đưa ra nhận xét.

#### 2.4. Rate of return 
- Hàm return_volatility_target(): Tạo các biến return, return so với 5 ngày gần nhất, volatility 5 ngày gần nhất, và target là sự thay đổi giá trong ngày tiếp theo.
- Trực quan hóa Rate of return theo ngày và đưa ra nhận xét.

#### 2.5. Volatility 
- Trực quan hóa Volatility theo ngày và đưa ra nhận xét.
  
### 3. Preprocessing
#### 3.1. Removing outliers 
- Hàm remove_outliers_and_plot(): Sử dụng IQR method để xác định và loại bỏ các điểm outlier, sau đó trực quan hóa boxplot trước và sau khi loại bỏ outlier, và print số lượng samples còn lại.
- Trực quan hóa distribution của biến target và đưa ra nhận xét.
  
#### 3.2. Features engineering 
Hàm feature_engineering():
- Thêm các features thời gian: Date, Month, DayOfWeek, WeekOfYear, IsWeekend, IsMonthStart, IsMonthEnd.
- Thêm các features tuần hoàn: Day_sin, Day_cos, Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos.
- Thêm các features dựa trên giá: LogReturn, MA_5, MA_20, Volatility_20.
- Thêm các features dựa trên volume: Close_vs_MA5, MA5_vs_MA20, Volume_avg_5, Volume_spike.
- Thêm các features trong ngày: Intraday_range, Open_to_Close.
- Thêm các features chỉ báo kĩ thuật: RSI, MACD, BB_bbm.


Các features này giúp model hiểu được xu hướng và học được các đặc điểm quan trọng của thị trường.
#### 3.3. Data segregation 
- Hàm split_time_series_data(): Chia tập train/test theo tỷ lệ 80/20 dựa trên thứ tự thời gian, trong đó tập train bao gồm 80% dữ liệu đầu tiên theo trình tự thời gian, và tập test là 20% dữ liệu còn lại sau cùng. Dữ liệu không bị shuffle để đảm bảo không xảy ra hiện tượng data leakage.
- Trước khi được đưa vào model để train, các splited set sẽ được kiểm tra shape.
  
### 4. Model
#### 4.1. Training
- Mô hình được lựa chọn là LightGBM, với các thông số learning_rate=0.5, random_state=42, verbosity=-1. LightGBM được lựa chọn nhờ khả năng xử lý hiệu quả dữ liệu dạng bảng có nhiều features, tốc độ train nhanh, cũng như khả năng mô hình hóa tốt các quan hệ phi tuyến và tương tác giữa các biến, phù hợp với dữ liệu tài chính như giá cổ phiếu.
- Model sẽ được train và đánh giá bằng phương pháp cross-validation với số lượng splits là 3. Việc này giúp giảm nguy cơ overfitting trên tập train, đồng thời mang lại đánh giá ổn định hơn về hiệu suất model. Tập test chỉ được sử dụng sau cùng để đánh giá tổng quan mô hình, nhằm đảm bảo tính khách quan và tránh bias.
- Sau đó kết quả dự đoán sẽ được trực quan hóa bằng hàm plot_cv_predictions() và đưa ra nhận xét.
- Evaluation metric được sử dụng là Mean Absolute Error (MAE) do ít nhạy cảm với các outliers, đồng thời cung cấp cái nhìn trực quan về mức sai số trung bình theo đơn vị giá, tức mô hình trung bình dự đoán sai bao nhiêu đồng mỗi phiên giao dịch.
  
#### 4.2. Tunning 
- Sử dụng Optuna để tối ưu các hyperparameters: n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, colsample_bytree, random_state, verbosity.
- Kết quả vẫn được tối ưu dựa trên tập cross-validation.
- Loss function được sử dụng trong objective function là một hàm custom dựa trên RMSE, có dạng: loss = RMSE - 0.5 * correlation. Mặc dù việc tối ưu theo hàm này không nhất thiết làm giảm mạnh MAE, nhưng lại giúp mô hình học được pattern của dữ liệu tốt hơn so với tối ưu theo MAE.
  
#### 4.3. Retraining and features importance 
- Model sẽ được train lại một lần nữa với toàn bộ tập train nhằm tận dụng tối đa dữ liệu sẵn có và dự đoán với tập test để đánh giá hiệu suất tổng quát của mô hình.
- Features importance tìm ra 10 features có tác động, ảnh hưởng lớn nhất đến model,
- Trực quan hóa kết quả predicted và đưa ra nhận xét.
  
## IV. Kết quả đạt được
- FPT
  
MAE: 0.4388 | RMSE: 0.5415
![Image](https://github.com/user-attachments/assets/c6c21eaa-b870-4f2a-92ac-dc54e695c28a)
- VIC
  
MAE: 0.7170 | RMSE: 0.9234
![Image](https://github.com/user-attachments/assets/95797489-ec84-4198-b06a-e7280f28f902)
- PNJ
  
MAE: 0.8171 | RMSE: 1.1013
![Image](https://github.com/user-attachments/assets/41e6ec71-1dc0-4a80-9db0-d5646cb63b9c)
- MSN
  
MAE: 0.7973 | RMSE: 1.0688
![Image](https://github.com/user-attachments/assets/8dc6d9a8-f19a-45bc-b9fd-f190ce3f14e8)

Nhìn chung, sau khi model được tối ưu và đánh giá trên tập test, kết quả MAE có giảm, nhưng khả năng phản ứng với biến động lại kém hơn rất nhiều so với tập train, điều này thể hiện rõ qua các biểu đồ trực quan hóa phía trên. Model thể hiện hiệu suất kém trong các dataset có mức độ biến động cao. Khi phạm vi dao động của target tăng, độ chính xác của model có xu hướng giảm. Ví dụ, với mã MSN, target dao động trong khoảng từ -3 đến 3, trong khi giá trị dự đoán chỉ dao động quanh mức 0, cho thấy model không phản ứng tốt với những biến động lớn.

## V. Hạn chế và những điểm cần cải thiện
### 1. Kết Quả 
- Kết quả trực quan hóa và MAE cho thấy model của em có kết quả không được tốt. Để có thể ứng dụng trong thực tế, em cần tiếp tục cải thiện model và trau dồi thêm kiến thức, đặc biệt về các kỹ thuật xử lý dữ liệu và tối ưu hóa mô hình.
### 2. Dataset
- Input data cũng cần chi tiết và có tính liên tục hơn để đảm báo kết quả xây dựng model tốt hơn.
### 3. Model
- Khi đã tối ưu model LightGBM mà vẫn cho kết quả kém thì nên thử thêm các mô hình khác như LSTM/RNN/GRU. Trên thực tế, em cũng đã thử nghiệm với model LSTM 1 layer, tuy nhiên kết quả cũng không tốt hơn model LightGBM baseline khi chưa tunning. 
### 4. Khoảng Hyperparameter tối ưu
- Tăng khoảng tìm kiếm giá trị tối ưu của các hyperparmeter như max_depth, num_leaves, n_estimators, learning_rate. Đồng thời giảm regularization như min_child_samples, subsample, colsample_bytree.
### 5. Feature Engineering 
- Ngoài việc thêm các features ra cũng nên lọc bớt đi các feature yếu hoặc các features có tính trùng lặp.
