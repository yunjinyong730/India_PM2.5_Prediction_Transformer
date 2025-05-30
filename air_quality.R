# 미세먼지의 하루를 예측하다: 시간대별 PM2.5 딥러닝 모델링 
# 작성자 : 윤진용 2025.4.16

# 필요한 패키지 설치 및 불러오기
install.packages(c("tidyverse", "lubridate", "GGally", "scales", "patchwork", "plotly", "zoo"), dependencies = TRUE)

# 패키지 로드
library(tidyverse)   # 데이터 처리 및 시각화
library(lubridate)   # 날짜 처리
library(GGally)      # 상관관계 시각화
library(scales)      # 스케일 관련 유틸리티
library(patchwork)   # ggplot 결합
library(plotly)      # 인터랙티브 3D 시각화
library(zoo)         # 이동 평균 등 시계열 도구

# CSV 데이터 불러오기
df <- read_csv("C:/Users/PC/Desktop/빅분기/data/air-quality-india.csv")

# 문자열 형식의 날짜 데이터를 날짜-시간 객체로 변환
df <- df %>%
  mutate(
    Timestamp = ymd_hms(Timestamp),              # 문자열 → 날짜시간
    date = as.Date(Timestamp),                   # 날짜만 추출
    weekday = wday(Timestamp, label = TRUE, abbr = FALSE),  # 요일 이름
    month_label = month(Timestamp, label = TRUE),           # 월 이름
    year_month = format(Timestamp, "%Y-%m"),     # 연-월 포맷
    week = isoweek(Timestamp),                   # ISO 주차
    Year = year(Timestamp),                      # 연도 추출
    Month = month(Timestamp),                    # 월 숫자 추출
    Day = day(Timestamp),                        # 일자
    Hour = hour(Timestamp)                       # 시간
  )




# 1. 월별 및 요일별 PM2.5 평균 시각화
pm_by_month <- df %>%
  group_by(month_label) %>%
  summarise(avg_pm25 = mean(PM2.5, na.rm = TRUE))

plot_month <- ggplot(pm_by_month, aes(x = month_label, y = avg_pm25)) +
  geom_col(fill = "tomato") +
  labs(title = "월별 평균 PM2.5", x = "월", y = "평균 PM2.5") +
  theme_minimal()

pm_by_weekday <- df %>%
  group_by(weekday) %>%
  summarise(avg_pm25 = mean(PM2.5, na.rm = TRUE))

plot_weekday <- ggplot(pm_by_weekday, aes(x = weekday, y = avg_pm25)) +
  geom_col(fill = "steelblue") +
  labs(title = "요일별 평균 PM2.5", x = "요일", y = "평균 PM2.5") +
  theme_minimal()

# 두 그래프 나란히 출력
plot_month + plot_weekday



# 2. 연도별 PM2.5 분포 
ggplot(df, aes(x = as.factor(Year), y = PM2.5)) +
  geom_boxplot(fill = "orchid") +
  labs(title = "연도별 PM2.5 분포", x = "연도", y = "PM2.5") +
  theme_minimal()




# 3. 연도별 월 평균 PM2.5 변화 추이
df %>%
  group_by(Year, Month) %>%
  summarise(monthly_avg = mean(PM2.5, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = Month, y = monthly_avg, color = factor(Year))) +
  geom_line(size = 1.1) +
  labs(title = "연도별 월간 PM2.5 평균 추이", x = "월", y = "평균 PM2.5") +
  scale_x_continuous(breaks = 1:12) +
  theme_minimal()



# 4. 월-시간별 PM2.5 평균 히트맵
heatmap_data <- df %>%
  group_by(Month, Hour) %>%
  summarise(mean_pm25 = mean(PM2.5, na.rm = TRUE), .groups = "drop")

ggplot(heatmap_data, aes(x = Hour, y = Month, fill = mean_pm25)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c(option = "C") +
  labs(title = "월-시간별 평균 PM2.5 히트맵", x = "시간", y = "월") +
  theme_minimal()



# 5. 7일/30일 이동 평균 시계열
daily_pm <- df %>%
  group_by(date) %>%
  summarise(daily_avg = mean(PM2.5, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    MA7 = rollmean(daily_avg, 7, fill = NA, align = "right"),
    MA30 = rollmean(daily_avg, 30, fill = NA, align = "right")
  )

ggplot(daily_pm, aes(x = date)) +
  geom_line(aes(y = daily_avg), color = "gray60", alpha = 0.4) +
  geom_line(aes(y = MA7), color = "dodgerblue", size = 1) +
  geom_line(aes(y = MA30), color = "firebrick", size = 1) +
  labs(title = "PM2.5 일별 평균 (이동평균 포함)", x = "날짜", y = "PM2.5") +
  theme_minimal()



# 6. 상관관계 매트릭스 시각화
cor_vars <- df %>% select(Year, Month, Day, Hour, `PM2.5`)
ggcorr(cor_vars, label = TRUE, label_round = 2, palette = "RdBu", layout.exp = 1.2)



# 7. 3차원 PM2.5 시각화 
df_3d <- df %>%
  group_by(Year, Month, Hour) %>%
  summarise(mean_pm25 = mean(PM2.5, na.rm = TRUE), .groups = "drop")

plot_ly(data = df_3d,
        x = ~Month,
        y = ~Hour,
        z = ~mean_pm25,
        color = ~factor(Year),
        type = "scatter3d",
        mode = "markers",
        marker = list(size = 3)) %>%
  layout(
    title = "3D PM2.5 시각화",
    scene = list(
      xaxis = list(title = "월"),
      yaxis = list(title = "시간"),
      zaxis = list(title = "평균 PM2.5")
    )
  )

