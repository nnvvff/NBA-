import os
import time
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class NBA:
    def __init__(self):
        # 初始化 WebDriver 服务
        self.service = Service('D:/Dekstop/chromedriver-win64/chromedriver-win64/chromedriver.exe')  # 指定 ChromeDriver 的路径
        self.options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(service=self.service)
        self.url = 'https://china.nba.cn/statistics/playerstats '  # NBA 球员数据页面  #https://china.nba.cn/statistics/playerstats   https://www.nba.com/stats/leaders
        # 初始化数据列表#
        self.data_list = []

    def getData(self, page):
        # 打开目标网站
        self.driver.get(self.url)
        wait = WebDriverWait(self.driver, 10)  # 显式等待时间设置为10秒

        for i in range(page):
            if i != 0:
                # 等待下一页按钮加载并点击
                try:
                    # 定位下一页按钮
                    next_page_btn = wait.until(
                        EC.element_to_be_clickable((By.XPATH, '//li[contains(@class, "page-next")]'))
                    )
                    # 检查按钮是否可点击
                    if next_page_btn.is_displayed() and next_page_btn.is_enabled():
                        # 点击按钮
                        next_page_btn.click()
                        print(f"成功翻到第 {i + 1} 页")
                        time.sleep(2)  # 等待页面内容刷新
                    else:
                        print(f"下一页按钮不可点击: 第 {i + 1} 页")
                        break
                except Exception as e:
                    print(f"翻页失败: {e}")
                    break
            print(f'开始爬取第{i + 1}页的数据')

            try:
                # 等待表格内容加载完成
                table = wait.until(EC.presence_of_element_located(
                    (By.XPATH, "//*[@id='app']/div[2]/div/div[2]/section[1]/div[3]/div/div/div/table/tbody")))
                rows = table.find_elements(By.XPATH, "//tr")

                # 遍历每一行数据
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    # 确保该行至少有14列
                    if len(cols) >= 14:
                        # 提取每一列数据
                        data = {
                            "rank": cols[0].text,#排名
                            "pname": cols[1].text.replace('\n', '.'),#球员姓名
                            "team": cols[2].text,#球队
                            "gp": cols[3].text,#比赛场次
                            "time": cols[4].text,#出场时间
                            "score": cols[5].text,#得分
                            "tov": cols[6].text,#篮板
                            "as": cols[7].text,#助攻
                            "ftm": cols[8].text,#抢断
                            "fta": cols[9].text,#盖帽
                            "reb": cols[10].text,#投篮命中率
                            "3pm": cols[11].text,#三分命中数
                            "3p%": cols[12].text,#三分命中率
                            "reh": cols[13].text,#罚球命中率
                        }
                        self.data_list.append(data)  # 将每行数据存入列表
                    else:
                        print(f"跳过一行数据（列数不足14）: {row.text}")
                print(f"第{i + 1}页数据抓取成功")
            except Exception as e:
                print(f"抓取数据失败: {e}")

        self.save_data()  # 抓取完成后保存数据

    def save_data(self):
        output_file = "NBA球员数据.xlsx"
        # 检查文件是否已存在
        if os.path.exists(output_file):
            # 如果文件存在，读取现有数据
            df = pd.read_excel(output_file)
            # 将新数据转换为 DataFrame 并合并
            new_data = pd.DataFrame(self.data_list)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            # 如果文件不存在，直接创建
            df = pd.DataFrame(self.data_list)
        df.to_excel(output_file, index=False)  # 保存数据到 Excel 文件
        print(f"数据成功保存到 {output_file}")

    def quit(self):
        self.driver.quit()  # 关闭浏览器

# 主程序入口
if __name__ == '__main__':
    nba = NBA()
    nba.getData(5)  # 爬取前5页数据
    nba.quit()  # 结束时关闭浏览器