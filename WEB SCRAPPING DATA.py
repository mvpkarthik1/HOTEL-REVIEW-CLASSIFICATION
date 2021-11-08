import csv,requests
from bs4 import BeautifulSoup
import emoji


no_of_reviews = input("Enter no of reviews: ")
reviews_num = int(no_of_reviews)
URL = input("Enter URL: ")

header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}

#If reviews are less than input reviews this code qwill be usdefull

html = requests.get(URL,headers= header)
soup = BeautifulSoup(html.content,features="html.parser")
#reviews=[soup.find_all("q", {"class": "IRsGHoPm"})[i].span.string for i in range(5)]
avail = [item.get_text(strip=True) for item in soup.select("span._33O9dg0j")]
avail =avail[0].split(" reviews")[0].split(",")
b=""
for d in avail:
    b=b+d
    avail = int(b)

if avail < reviews_num:
    print("Choose URL which has more 5000 reviews ")
    print("Available review(s)", avail)
    exit

from bs4 import BeautifulSoup
q=0
import requests
with open('Reviews.csv','w') as f:
    write = csv.writer(f)
    write.writerow(['REVIEWS'])    
    try:
      URL.split()
      x = URL.split("-Reviews-", 1)
      for i in range(5,reviews_num+1,5):
        URL = x[0]+ f"-Reviews-or{i}-"+x[1]
        html = requests.get(URL,headers= header)
        soup = BeautifulSoup(html.content,features="html.parser")
        reviews=[soup.find_all("q", {"class": "IRsGHoPm"})[i].span.string for i in range(5)]
        
        for j in range(len(reviews)):

          reviews[j]=str(reviews[j])

          reviews[j] = emoji.demojize(reviews[j], delimiters=("", ""))

          write.writerow([reviews[j]])
          q+=1
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print (message)
              
# print("User input: ", reviews_num )
# print("Extracted reviews: ", q )
# print("Omitted reviews:",reviews_num-q)


import pandas as pd
reviews_df = pd.read_csv('Reviews.csv')
print(reviews_df.head())
print("Total Reviwes Extracted:", len(reviews_df))



