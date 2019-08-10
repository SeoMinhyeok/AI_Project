#-*-coding:utf-8-*-
import pandas as pd

movies = 'movies.csv'
movies = pd.read_csv(movies)

"""
print(movies) # 다 출력함.

print(movies.head()) # 처음 5개를 출력함.

print(movies.tail()) # 마지막 5개를 출력함.

print(movies.sample(5)) # 무작위 5개를 출력함.

print(movies.columns) # 컬럼만 출력함.

print(movies.describe()) # 통계적인 수치를 출력함.

sheet_1 = movies[['Title','Year','Country']]
print(sheet_1) # 엑셀파일에서 부분적인 컬럼만 가져옴.

print(movies[15:20]) # 행 범위를 부분적으로 가져 오고 싶을 때.

print(movies.loc[0:5,['Year', 'Title']])
print(movies.iloc[3:5,0:2]) # 열, 행 두개를 섞어서 사용하는 법.

print(movies[movies.Year > 2015]) # 조건으로 검색하고 싶을 때.

title = movies.Title
print(title) # 특정 컬럼만 가져옴.

for col in movies.columns:
    msg = print('column : {:<30}\t Percent of NaN value:{:.2f}%'
                .format(col, 100* (movies[col].isnull().sum()/movies[col].shape[0])))
    # 없는 데이터가 얼마만큼이나 있는지 구하는 것.

print(movies.fillna(value=0)) # 결측치를 바꿔주고 싶을 때.

print(movies['Title'].value_counts()) # 데이터의 갯수를 세고 싶을 때.

excel_file = 'part_time.csv'
df = pd.read_csv(excel_file)
df['total_time'] = df['End_time'] - df['Start_time']
df['time_of_make'] = df['Make'] / df['total_time']
print(df) # 컬럼을 추가하고 계산하는 법.
df = df.sort_values(by=['time_of_make'], ascending=False)
print(df) # 그리고 내림차순 정리.
df.to_csv('df.csv') # 마지막으로 저장을 하는 역할
"""
