import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from (videos.csv and category_id.json) and answer the questions.
"""

def Q1():
    """
        1. How many rows are there in the videos.csv after removing duplications?
        - To access 'videos.csv', use the path '/data/videos.csv'.
    """
    # TODO: Paste your code here
    df = pd.read_csv("./data/videos.csv")
    df.drop_duplicates(inplace = True)

    return len(df)

def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO: Paste your code here
    return vdo_df[vdo_df["likes"] < vdo_df["dislikes"]]["video_id"].nunique()

def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    # TODO: Paste your code here
    return len(vdo_df[(vdo_df["trending_date"] == "18.22.01") & (vdo_df["comment_count"] > 10000)])

def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - videos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO:  Paste your code here
    return (vdo_df.groupby('trending_date')['comment_count'].mean()).idxmin()

def Q5(vdo_df):
    # 1. โหลดข้อมูลจาก category_id.json [cite: 51, 52]
    with open('./data/category_id.json', 'r') as f:
        categories_data = json.load(f)

    # 2. สร้าง Dictionary เพื่อ Map ระหว่าง category_id และ title [cite: 27, 30]
    # หมายเหตุ: ใน JSON 'id' เป็น String แต่ใน CSV มักจะเป็น Int จึงต้องแปลงให้ตรงกัน
    id_to_title = {
        item['id']: item['snippet']['title'] 
        for item in categories_data['items']
    }

    # 3. สร้างคอลัมน์ชื่อหมวดหมู่ใน vdo_df [cite: 49]
    vdo_df['category_title'] = vdo_df['category_id'].astype(str).map(id_to_title)

    # 4. กรองเฉพาะแถวที่เป็น "Sports" หรือ "Comedy" [cite: 49]
    target_df = vdo_df[vdo_df['category_title'].isin(['Sports', 'Comedy'])]

    # 5. คำนวณยอดวิวรวม (sum of views) แยกตามวันและหมวดหมู่ [cite: 49]
    daily_views = target_df.groupby(['trending_date', 'category_title'])['views'].sum().unstack(fill_value=0)

    # 6. ตรวจสอบว่ามีคอลัมน์ทั้งคู่หรือไม่ (ป้องกัน Error หากวันนั้นไม่มีข้อมูลหมวดใดหมวดหนึ่ง)
    if 'Sports' not in daily_views.columns: daily_views['Sports'] = 0
    if 'Comedy' not in daily_views.columns: daily_views['Comedy'] = 0

    # 7. นับจำนวนวันที่ยอดวิว Sports > Comedy [cite: 49]
    result = len(daily_views[daily_views['Sports'] > daily_views['Comedy']])

    return int(result)