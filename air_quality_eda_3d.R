
# 패키지 설치 및 로드
install.packages(c("tidyverse", "lubridate", "GGally", "scales", "patchwork", "plotly", "zoo"), dependencies = TRUE)
library(tidyverse)
library(lubridate)
library(GGally)
library(scales)
library(patchwork)
library(plotly)
library(zoo)

# 데이터 불러오기
df <- read.csv("C:\\Users\\PC\\Desktop\\빅분기\\data\\air-quality-india.csv")
df$Timestamp <- ymd_hms(df$Timestamp)

# 날짜 파생 변수 생성
df <- df %>%
  mutate(
    date = as.Date(Timestamp),
    weekday = wday(Timestamp, label = TRUE, abbr = FALSE),
    month_label = month(Timestamp, label = TRUE),
    year_month = format(Timestamp, "%Y-%m"),
    week = isoweek(Timestamp)
  )


# 1. 월/요일별 평균 PM2.5
p1 <- df %>%
  group_by(month_label) %>%
  summarise(avg_pm = mean(PM2.5, na.rm = TRUE)) %>%
  ggplot(aes(x = month_label, y = avg_pm)) +
  geom_bar(stat = "identity", fill = "salmon") +
  labs(title = "월별 평균 PM2.5", x = "Month", y = "Avg PM2.5") +
  theme_minimal()

p2 <- df %>%
  group_by(weekday) %>%
  summarise(avg_pm = mean(PM2.5, na.rm = TRUE)) %>%
  ggplot(aes(x = weekday, y = avg_pm)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "요일별 평균 PM2.5", x = "Weekday", y = "Avg PM2.5") +
  theme_minimal()

p1 + p2


# 2. 연도별 PM2.5 분포
ggplot(df, aes(x = factor(Year), y = PM2.5)) +
  geom_boxplot(fill = "plum") +
  labs(title = "연도별 PM2.5 분포", x = "Year", y = "PM2.5") +
  theme_minimal()


# 3. 연도별 월 평균 추이
df %>%
  group_by(Year, Month) %>%
  summarise(mean_pm = mean(PM2.5, na.rm = TRUE)) %>%
  ggplot(aes(x = Month, y = mean_pm, color = factor(Year))) +
  geom_line(size = 1.1) +
  labs(title = "연도별 월간 평균 PM2.5 추이", x = "Month", y = "Avg PM2.5") +
  scale_x_continuous(breaks = 1:12) +
  theme_minimal()


# 4. 월-시간별 평균 히트맵
df_heat <- df %>%
  group_by(Month, Hour) %>%
  summarise(mean_pm = mean(PM2.5, na.rm = TRUE))

ggplot(df_heat, aes(x = Hour, y = Month, fill = mean_pm)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c() +
  labs(title = "월-시간별 PM2.5 평균 히트맵", x = "Hour", y = "Month") +
  theme_minimal()



# 5. 이동 평균 시계열 (7일, 30일)
df_daily <- df %>%
  group_by(date) %>%
  summarise(pm_daily = mean(PM2.5, na.rm = TRUE)) %>%
  mutate(
    MA7 = zoo::rollmean(pm_daily, k = 7, fill = NA),
    MA30 = zoo::rollmean(pm_daily, k = 30, fill = NA)
  )

ggplot(df_daily, aes(x = date)) +
  geom_line(aes(y = pm_daily), alpha = 0.3, color = "gray") +
  geom_line(aes(y = MA7), color = "blue", size = 1) +
  geom_line(aes(y = MA30), color = "red", size = 1) +
  labs(title = "PM2.5 일 평균 (7일, 30일 이동평균)", y = "PM2.5", x = "Date") +
  theme_minimal()


# 6. 상관관계 분석
cor_data <- df %>% select(Year, Month, Day, Hour, PM2.5)
ggcorr(cor_data, label = TRUE, label_round = 2, palette = "RdBu", layout.exp = 1.2)


# 7. 3D 시각화 (plotly)
df_3d <- df %>%
  group_by(Year, Month, Hour) %>%
  summarise(mean_pm = mean(PM2.5, na.rm = TRUE))

plot_ly(df_3d, 
        x = ~Month, 
        y = ~Hour, 
        z = ~mean_pm, 
        color = ~factor(Year),
        type = "scatter3d", 
        mode = "markers",
        marker = list(size = 3)) %>%
  layout(title = "3D Visualization of PM2.5",
         scene = list(
           xaxis = list(title = "Month"),
           yaxis = list(title = "Hour"),
           zaxis = list(title = "Avg PM2.5")
         ))

