import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

# 1.데이터 로드
path = "./Netflix Dataset.csv"
netflix = pd.read_csv(path)

print(netflix.head(5))

print(netflix.info())

print(netflix.columns)
#  0   Show_Id       7789 non-null   object
#  1   Category      7789 non-null   object
#  2   Title         7789 non-null   object
#  3   Director      5401 non-null   object
#  4   Cast          7071 non-null   object
#  5   Country       7282 non-null   object
#  6   Release_Date  7779 non-null   object
#  7   Rating        7782 non-null   object 등급-점수가 엑셀파일에는 아님
#  8   Duration      7789 non-null   object
#  9   Type          7789 non-null   object
#  10  Description   7789 non-null   object

# 기묘한 이야기 검색
# m/s 비율
# 연도별 m/s 수치 시각화
# 월별 m/s 수치 시각화
# 나라별 타켓팅 연령 시각화
# 워드클라우드(핵심 단어 시각화)

print("-------------------------------------------")
# 1.상징색
netflix_palette = ["#E50909","#87B207","#831076","#B3B3B3"]
sns.set_palette(sns.color_palette(netflix_palette))
sns.palplot(netflix_palette)
# plt.show()

# 2.결측치 처리
netflix["Cast"] = netflix["Cast"].fillna("No Data")
netflix["Director"] = netflix["Director"].fillna("No Data")
netflix["Country"] = netflix["Country"].fillna("No Data")
netflix.dropna(axis=0, inplace=True) #결측치가 생긴 경우, x축을 다 지워버림
# print(netflix.isnull().sum()) #공간만 있는 데이터를 가져와서 합계를 도출해냄

# 3. 피처 엔지니어링
netflix["date_added"] = pd.to_datetime(netflix["Release_Date"].str.strip())
netflix["year_added"] = netflix["date_added"].dt.year
netflix["month_added"] = netflix["date_added"].dt.month

# 4.시청등급 mapping
# 넷플릭스 미국 시청 등급 및 핵심 키워드 매핑
age_group_map = {
    # --- TV 프로그램 등급 (TV Parental Guidelines) ---
    "TV-Y": "Kids",      # 모든 어린이 시청 가능 (주로 유아 대상)
    "TV-Y7": "Youth",    # 7세 이상의 어린이 대상 (가벼운 판타지 폭력 포함 가능)
    "TV-G": "General",   # 모든 연령층 시청 가능 (자극적인 내용 없음)
    "TV-PG": "Guidance",  # 부모의 지도가 필요함 (약간의 대사나 폭력성 가능)
    "TV-14": "Caution",   # 14세 미만에게 부적절할 수 있음 (강한 수위의 내용 포함)
    "TV-MA": "Mature",    # 성인 전용 (성적 내용, 폭력성, 언어 수위 매우 높음)

    # --- 영화 등급 (MPA) ---
    "G": "General",      # 모든 연령 시청 가능 (전체 관람가)
    "PG": "Guidance",    # 부모의 지도가 권장됨 (일부 내용이 어린이에게 부적절할 수 있음)
    "PG-13": "Teens",    # 13세 미만에게 부적절할 수 있는 내용 포함 (청소년 주의)
    "R": "Restricted",   # 17세 미만은 부모나 보호자 동반 필수 (제한적 관람)
    "NC-17": "Adults"    # 17세 이하 시청 불가 (완전한 성인 전용)
}

netflix["age_group"] = netflix["Rating"].map(age_group_map)

# 기묘한 이야기 검색 Stranger Things
# 검색할 검색어 변수
search_query = "Stranger Things"
search_results = netflix[netflix["Title"].str.contains(search_query,case=False)]
print(search_results [["Title","Country","Rating","Type"]])

# m/s 비율 시각화
plt.figure(figsize=(5,5))
ratio = netflix["Category"].value_counts()
plt.pie(ratio, colors= ["#102A43", "#D4AF37"])
plt.suptitle("movies & TV Show",fontsize = 16)
# plt.show()

import matplotlib.pyplot as plt

# 1. 데이터 준비 (netflix 데이터프레임이 있다고 가정)
ratio = netflix["Category"].value_counts()
labels = ratio.index

# 2. 미래지향적 스타일 설정 (다크 모드 및 네온 컬러)
plt.style.use('dark_background') # 배경을 어둡게 설정
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0e1117') # 전체 배경색 미세 조정
ax.set_facecolor('#0e1117')

# 미래적인 색상 조합: 네온 사이언 & 일렉트릭 퍼플
colors = ['#00e5ff', '#7000ff']
explode = [0.08, 0] # 첫 번째 요소(Movie)를 살짝 띄워 입체감 부여

# 3. 도넛형 파이 차트 그리기
wedges, texts, autotexts = ax.pie(
    ratio,
    labels=labels,
    # 수치 데이터(건수)와 퍼센트(%)를 동시에 표시하는 람다 함수
    autopct=lambda p: f'{p:.1f}%\n({int(p*sum(ratio)/100):,})',
    startangle=140,
    colors=colors,
    explode=explode,
    shadow=True,            # 그림자 효과 추가
    pctdistance=0.75,       # 수치 표시 위치
    textprops={'color': 'white', 'fontsize': 12, 'fontweight': 'bold'},
    # wedgeprops의 width를 조절하여 도넛 모양으로 변형
    wedgeprops={'width': 0.4, 'edgecolor': '#0e1117', 'linewidth': 3} 
)

# 4. 텍스트 및 타이틀 디자인
for text in texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')

# 차트 중앙에 텍스트 추가 (미래지향적인 UI 느낌)
ax.text(0, 0, 'NETFLIX\nSTATS', ha='center', va='center', 
        fontsize=18, fontweight='extra bold', color='#ffffff')

# 메인 타이틀 (네온 포인트 컬러 적용)
plt.suptitle("MOVIES & TV SHOWS", fontsize=22, fontweight='bold', color='#00e5ff', y=0.98)
plt.title("Futuristic Content Analysis", fontsize=12, color='#aaaaaa', pad=20)

# 범례(Legend) 추가
ax.legend(wedges, labels, title="Category", loc="upper right", bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.show()

#연도별 m/s 수치 시각화
#netflix["year_added"]

# 데이터 준비 (예시 데이터를 바탕으로 작성)
# 실제 netflix 데이터프레임이 있다고 가정하고 year_added의 빈도수를 계산합니다.
year_counts = netflix['year_added'].value_counts().sort_index().reset_index()
year_counts.columns = ['Year', 'Count']

# 2. 그래프 스타일 설정
sns.set_theme(style="whitegrid") # 배경 격자 설정
plt.figure(figsize=(15, 5))      # 요청하신 사이즈 설정

# 3. 막대그래프 생성
# 컬러 팔레트를 적용하여 시각적으로 더 다채롭게 만듭니다.
ax = sns.barplot(
    x='Year', 
    y='Count', 
    data=year_counts, 
    palette="viridis"
)

# 4. 세부 디자인 및 텍스트 추가
plt.title('Netflix Content Added per Year', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Year Added', fontsize=12)
plt.ylabel('Number of Titles', fontsize=12)

# 각 막대 상단에 정확한 수치 표시 (Annotate)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=10,
                   fontweight='bold',
                   color='black')

plt.tight_layout()
plt.show()

#월별 m/s 수치 시각화
netflix["month_added"]

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 월별 데이터 집계 (1월~12월 순서 보장)
# month_added가 숫자(1~12)라고 가정할 때의 정렬 방식입니다.
month_order = range(1, 13)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 데이터 카운트 및 재색인(reindex)으로 빈 달이 있어도 순서대로 정렬
month_counts = netflix['month_added'].value_counts().reindex(month_order).fillna(0).reset_index()
month_counts.columns = ['Month', 'Count']
month_counts['MonthName'] = month_names

# 2. 그래프 스타일 및 테마 설정
sns.set_theme(style="white") 
plt.figure(figsize=(15, 6))

# 3. 그라데이션 막대그래프 생성
# 'flare' 팔레트는 붉은색에서 보라색 계열로 부드럽게 변하는 그라데이션을 제공합니다.
ax = sns.barplot(
    x='MonthName', 
    y='Count', 
    data=month_counts, 
    palette="flare" 
)

# 4. 고급 디자인 디테일 추가
plt.title('Monthly Distribution of Netflix Content Added', fontsize=20, fontweight='bold', pad=25)
plt.xlabel('Month', fontsize=14, labelpad=15)
plt.ylabel('Number of Titles', fontsize=14, labelpad=15)

# 불필요한 테두리 제거 (깔끔한 Look)
sns.despine(left=True)

# 막대 상단에 데이터 수치(Label) 추가
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color='#444444')

plt.tight_layout()
plt.show()

# 3-5. 나라별 타겟팅 하는 연령 시각화(Heatmap)
# 1. 콘텐츠가 가장 많은 상위 10개국 추출 (No Data 제외)
top_countries = netflix[netflix['Country'] != 'No Data']['Country'].value_counts().head(10).index
df_top10 = netflix[netflix['Country'].isin(top_countries)]

# 2. 히트맵을 위한 교차표(Crosstab) 생성
country_age = pd.crosstab(df_top10['Country'], df_top10['age_group'])

# 3. 보기 좋게 열 순서 재정렬 (어린이 -> 성인 순)
# .fillna(0)와 .astype(int)를 추가하여 데이터가 없는 칸의 에러를 방지했습니다.
age_order = ['Kids', 'Older Kids', 'Teens', 'Adults']
country_age = country_age.reindex(columns=age_order).fillna(0).astype(int)

# 4. 시각화 설정
plt.figure(figsize=(12, 8))

# 5. 히트맵 그리기
sns.heatmap(country_age, 
            annot=True,          # 숫자 표시
            fmt='d',             # 정수형(decimal) 포맷
            cmap='Reds',         # 넷플릭스 테마의 레드 계열 색상
            linewidths=0.5,      # 칸 사이 경계선
            cbar_kws={'label': 'Content Count'}) # 컬러바 라벨

# 6. 마무리 디자인
plt.title("Target Age Groups by Top 10 Countries", fontsize=18, fontweight='bold', pad=25, color='#E50914')
plt.xlabel("Age Group", fontsize=12, fontweight='bold')
plt.ylabel("Country", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 워드클라우드
try:
    plt.figure(figsize=(15, 5))
    
    # 1. 텍스트 합치기 (결측치 제거를 위해 dropna() 추가 권장)
    text = " ".join(netflix["Description"].dropna())
    
    # 2. 마스크 이미지 로드
    mask = np.array(Image.open('./netflix_logo.jpg'))
    
    # 3. 워드클라우드 설정 및 생성
    # 인자들 사이의 쉼표(,)와 마지막 generate() 연결을 확인하세요.
    wordcloud = WordCloud(
        background_color="white", 
        width=1400, 
        height=1400, 
        max_words=170, 
        mask=mask
    ).generate(text)
    
    # 4. 화면 출력
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off") # 격자 숫자 제거
    plt.show()
    
except Exception as e:
    print(f"에러가 발생했습니다: {e}")


