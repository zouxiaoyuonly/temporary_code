from textrank4zh import TextRank4Sentence, TextRank4Keyword
from bert_serving.client import BertClient
import requests
import json
import time
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
import pandas as pd
import numpy as np
from tqdm import tqdm

ONE_MINUTE = 60
# 创建对象的基类
Base = declarative_base()
# 预置分类
PRECLASSIFY = {
    "1024": ["17", "18", "19", "21"],
    "42区": ["1", "2", "4", "6", "11", "22", "23", "25"],
}
SERVERMYSQL = (
    "mysql+pymysql://newrank:Newrank123456@101.37.124.81:13307/message"  # 远端mysql服务地址
)


class UpdateId(object):
    def __init__(self):
        self.tr4w = TextRank4Keyword()
        self.tr4s = TextRank4Sentence()
        self.re_url = "https://www.weseepro.com/api/v1/message/webArticle/getData"
        self.sentence_delimiters = [
            "?",
            "!",
            ";",
            "？",
            "！",
            "。",
            "；",
            "……",
            "…",
            "...",
            "... ...",
            "\n",
        ]

    def update_uuid(self, url):
        data = {"url": url}
        try:
            # 这里防止采集失败
            for count in range(3):
                t = requests.post(self.re_url, data=data, timeout=(5, 5))
                if t.ok:
                    break
                time.sleep(1)
            if not t.ok:
                return ""
        except:
            return ""
        mm = json.loads(t.text)["data"]
        # 如果采集失败则删除
        if mm["status"] == "0":
            return ""
        text = {}
        for line in mm["text"]:
            if "1" in line:
                if line["1"] in text:
                    text[line["1"]] += 1
                else:
                    text[line["1"]] = 1
        content = ""
        for tt in text:
            if text[tt] < 3 and len(tt) > 10:
                content += tt
        self.tr4w.analyze(
            text=content,
            lower=True,
            window=5,
            vertex_source="all_filters",
            edge_source="all_filters",
        )
        keyphrases = self.tr4w.get_keyphrases(keywords_num=20, min_occur_num=3)
        keywords = self.tr4w.get_keywords(20, word_min_len=2)
        key_ph = {}
        for tt in keyphrases:
            key_ph[tt] = len(tt)
        keyphrases = []
        for ii in key_ph:
            for jj in key_ph:
                if key_ph[ii] < key_ph[jj] and ii in jj:
                    key_ph[ii] = 0
                    break
            if key_ph[ii] > 0:
                keyphrases.append(ii)
        title = mm["title"]
        result = title
        for tt in keywords:
            result += "," + tt.word
        for tt in keyphrases:
            result += "," + tt

        return result


# 定义User对象：
class User(Base):
    # 表的名字：
    __tablename__ = "message_spam_second_floor"

    # 表的结构:
    id = Column(Integer, primary_key=True)
    uuid = Column(String(32))
    message_uuid = Column(String(32))
    url = Column(String(500))
    comment_count = Column(String(2))
    title = Column(String(255))
    author = Column(String(255))
    suanfa_type = Column(Integer)
    suanfa_type_original = Column(Integer)
    appear_count = Column(String(50))
    status = Column(String(1))
    ori_description = Column(String(512))
    add_time = Column(DateTime)
    update_time = Column(DateTime)


class Recircle(object):
    def __init__(self):
        self.model = load_model("myclassify0707.h5")
        self.sqlquery = """ SELECT * FROM  message.message_spam_second_floor WHERE idx=1 AND suanfa_type =-1 AND `status` = 1 """

        self.bertclient = BertClient(ip="127.0.0.1")
        self.gettext = UpdateId()

    def recurrent(self):
        start_time = time.time()
        print(start_time)
        # 创建sql连接
        engine = create_engine(SERVERMYSQL)
        # 创建session类型
        DBSession = sessionmaker(bind=engine)
        # 创建session对象
        session = DBSession()
        # 查询待分类数据
        data = pd.read_sql(self.sqlquery, engine, index_col=None)
        for index in tqdm(range(len(data))):
            uuid, title, ori_description, url = (
                data["message_uuid"][index],
                data["title"][index],
                data["ori_description"][index],
                data["url"][index],
            )
            if ori_description:
                if ori_description == "1024":
                    label = "19"
                else:
                    label = "1"
            else:
                content = self.gettext.update_uuid(url)
                if content:
                    bertvec_wd = self.bertclient.encode([content])
                else:
                    bertvec_wd = self.bertclient.encode([title])
                #predict
                pre_result = self.model.predict(bertvec_wd)
                label = np.argmax(pre_result[0], axis=0) + 1
                label = str(label)
            for kk in range(3):
                try:
                    # 更新语句方式为，先查询数据出来，然后直接更改就可以
                    # 查出用户名为"MK"的，然后赋于新的值
                    user_info = (
                        session.query(User).filter(User.message_uuid == uuid).first()
                    )

                    if user_info:
                        user_info.suanfa_type = label
                        user_info.suanfa_type_original = label
                        session.commit()
                    # 这里成功了就退出
                    break
                except:
                    # 创建sql连接
                    engine = create_engine(SERVERMYSQL)
                    # 创建session类型
                    DBSession = sessionmaker(bind=engine)
                    # 创建session对象
                    session = DBSession()
        end_time = time.time()
        diff = end_time - start_time
        if diff < ONE_MINUTE:
            time.sleep(ONE_MINUTE - diff)


updatehandel = Recircle()

if __name__ == "__main__":
    while True:
        updatehandel.recurrent()
