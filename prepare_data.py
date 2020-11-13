import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import sqlite3

def prepare_data(path="/content/drive/My Drive/Module 3 shared folder/toxic_comment.db"): 
    con = sqlite3.connect(path)

    cur = con.cursor()
    cur.execute("CREATE TABLE comment (my_id INTEGER PRIMARY KEY AUTOINCREMENT, id, comment_text,toxic INTEGER, severe_toxic INTEGER, obscene INTEGER, threat INTEGER, insult INTEGER, identity_hate INTEGER);")

    with open('/content/drive/My Drive/Module 3 shared folder/train.csv.zip (Unzipped Files)/train.csv','r') as fin:
        dr = csv.DictReader(fin) # comma is default delimiter
        to_db = [(i['id'], i['comment_text'], i['toxic'],i['severe_toxic'], i['obscene'], i['threat'],i['insult'], i['identity_hate']) for i in dr]

    cur.executemany("INSERT INTO comment (id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate) VALUES (?,?,?,?,?,?,?,?);", to_db)
    con.commit()
    con.close()
