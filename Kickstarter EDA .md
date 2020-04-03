## EDA of Kickstarter Data

For this project we're going to explore a free dataset on Kaggle with Kickstarter project data from May 2009 to March 2018. We will perform some EDA on the data (Exploratory Data Analysis) to gather any insights. 

**Questions to Answer:**  
1a. Examine the `state` column to see unique values and counts.  
1b. Show a pie chart of the `state` project count for all projects.  
1c. Create a new "Completed" dataframe that removes any rows with state of 'live', 'undefined', or suspended.  
*note - from here out we'll be looking at the completed project data unless mentioned otherwise*

2a. What is the overall success rate for all completed kickstarter projects?  
2b. Which 5 projects were pledged the most money (usd_pledged_real)?  
2c. Which 5 projects had the most backers?  
2d. Which year had the most competition? (# of projects)  

3a. What is the success rate for all projects broken down by `main_category`?  
3b. Show a horizontal bar chart for project success rate by `main_category`, sorted by highest to lowest.  
3c. Within the Games `main_category`, what is the success rate for each `category` within it?  

4a. Calculate the 'pct_of_goal' for each completed project  
4b. What were the top 5 projects when looking at pct_of_goal for all time?  
4c. Plot a histogram distribution of all completed projects by pct_of_goal  
4d. Create 2 histogram subplots by pct_of_goal: 1) state=successful, and 2) all others (failed)  

5a. What is the average `usd_goal_real` for all *completed* kickstarter projects, broken down by `main_category`.  
5b. What is the median `usd_goal_real` for all *completed* kickstarter projects, broken down by `main_category`.  
5c. What is the average `usd_pledged_real` for all *completed* kickstarter projects, broken down by `main_category`.  
5d. What is the median `usd_pledged_real` for all *completed* kickstarter projects, broken down by `main_category`.  
5e. What insights does this information provide?  
5f. Based on this information, if someone wanted to choose the `main_category` with the highest combined success rate and pledged dollar amount, which one would you recommend?  

6a. Create a new column 'months' that shows how many months the project was active between launch and deadline.  
6b. Compare the avg months for successful projects vs non-successful.  Add visuals if you'd like.  
6c. Does the length of a project in months seem to have an impact?  

*Let's zoom in on Games: Video Games (main_category: category)*

7a. Calculate the expected value for the Games: Video Games category, with the expected value defined as (median of usd_pledged_real)* (success rate of completed projects).  
7b. Do this again but broken down by deadline year  
7c. Show this in a bar chart  
7d. What insights does this data provide you?  

*Let's zoom in on personal planners*

8a. Calculate the project count, success rate, and pct_of_goal for all projects with 'planner' in the name.  Check for spelling variations in upper/lowercase.  
8b. How about all projects with both 'planner' and 'Panda' in the name?  
8c. Congrats Panda Planner!  (That's my bro's company)


First, lets open the data set and have a look at some rows and columns:


```python
from csv import reader
import os
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%matplotlib inline


```


```python
ks_data = pd.read_csv('ks-data.csv')
```


```python
ks_data.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000002330</td>
      <td>The Songs of Adelaide &amp; Abullah</td>
      <td>Poetry</td>
      <td>Publishing</td>
      <td>GBP</td>
      <td>2015-10-09</td>
      <td>1000.0</td>
      <td>2015-08-11 12:12:28</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>GB</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1533.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000003930</td>
      <td>Greeting From Earth: ZGAC Arts Capsule For ET</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2017-11-01</td>
      <td>30000.0</td>
      <td>2017-09-02 04:43:57</td>
      <td>2421.0</td>
      <td>failed</td>
      <td>15</td>
      <td>US</td>
      <td>100.0</td>
      <td>2421.0</td>
      <td>30000.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000004038</td>
      <td>Where is Hank?</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2013-02-26</td>
      <td>45000.0</td>
      <td>2013-01-12 00:20:50</td>
      <td>220.0</td>
      <td>failed</td>
      <td>3</td>
      <td>US</td>
      <td>220.0</td>
      <td>220.0</td>
      <td>45000.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000007540</td>
      <td>ToshiCapital Rekordz Needs Help to Complete Album</td>
      <td>Music</td>
      <td>Music</td>
      <td>USD</td>
      <td>2012-04-16</td>
      <td>5000.0</td>
      <td>2012-03-17 03:24:11</td>
      <td>1.0</td>
      <td>failed</td>
      <td>1</td>
      <td>US</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5000.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000011046</td>
      <td>Community Film Project: The Art of Neighborhoo...</td>
      <td>Film &amp; Video</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2015-08-29</td>
      <td>19500.0</td>
      <td>2015-07-04 08:35:03</td>
      <td>1283.0</td>
      <td>canceled</td>
      <td>14</td>
      <td>US</td>
      <td>1283.0</td>
      <td>1283.0</td>
      <td>19500.00</td>
    </tr>
  </tbody>
</table>
</div>



### 1a. Examine the `state` column to see unique values and counts


```python
ks_data["state"].unique()
```




    array(['failed', 'canceled', 'successful', 'live', 'undefined',
           'suspended'], dtype=object)




```python
ks_data["state"].value_counts()
```




    failed        197719
    successful    133956
    canceled       38779
    undefined       3562
    live            2799
    suspended       1846
    Name: state, dtype: int64



### 1b. Show a pie chart of the `state` project count for all projects.


```python
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Failed', 'Successful', 'Canceled', 'Undefined', 'Live', 'Suspended'
sizes = [197719, 133956, 38779, 3562, 2799, 1846]
fig1, ax1 = plt.subplots(figsize=(10,10))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
```


![png](output_9_0.png)


### 1c. Create a new "Completed" dataframe that removes any rows with `state` of 'live', 'undefined', or suspended.

*note - from here out we'll be looking at the completed project data unless mentioned otherwise


```python
ks_f = ks_data.loc[ks_data["state"] == "failed"]
ks_s = ks_data.loc[ks_data["state"] == "successful"]
ks_c = ks_data.loc[ks_data["state"] =="canceled"]
ks_complete = pd.concat([ks_f, ks_s, ks_c])
```


```python
ks_complete.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000002330</td>
      <td>The Songs of Adelaide &amp; Abullah</td>
      <td>Poetry</td>
      <td>Publishing</td>
      <td>GBP</td>
      <td>2015-10-09</td>
      <td>1000.0</td>
      <td>2015-08-11 12:12:28</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>GB</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1533.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000003930</td>
      <td>Greeting From Earth: ZGAC Arts Capsule For ET</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2017-11-01</td>
      <td>30000.0</td>
      <td>2017-09-02 04:43:57</td>
      <td>2421.0</td>
      <td>failed</td>
      <td>15</td>
      <td>US</td>
      <td>100.0</td>
      <td>2421.0</td>
      <td>30000.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000004038</td>
      <td>Where is Hank?</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2013-02-26</td>
      <td>45000.0</td>
      <td>2013-01-12 00:20:50</td>
      <td>220.0</td>
      <td>failed</td>
      <td>3</td>
      <td>US</td>
      <td>220.0</td>
      <td>220.0</td>
      <td>45000.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000007540</td>
      <td>ToshiCapital Rekordz Needs Help to Complete Album</td>
      <td>Music</td>
      <td>Music</td>
      <td>USD</td>
      <td>2012-04-16</td>
      <td>5000.0</td>
      <td>2012-03-17 03:24:11</td>
      <td>1.0</td>
      <td>failed</td>
      <td>1</td>
      <td>US</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5000.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1000030581</td>
      <td>Chaser Strips. Our Strips make Shots their B*tch!</td>
      <td>Drinks</td>
      <td>Food</td>
      <td>USD</td>
      <td>2016-03-17</td>
      <td>25000.0</td>
      <td>2016-02-01 20:05:12</td>
      <td>453.0</td>
      <td>failed</td>
      <td>40</td>
      <td>US</td>
      <td>453.0</td>
      <td>453.0</td>
      <td>25000.00</td>
    </tr>
  </tbody>
</table>
</div>



### 2a. What is the overall success rate for all completed kickstarter projects?


```python
av_success = len(ks_s) / len(ks_complete)
rate = round(av_success, 2) * 100
```


```python
print("The overall success rate for completed Kickstarter projects is " + str(rate) +"%")
```

    The overall success rate for completed Kickstarter projects is 36.0%


### 2b. Which 5 projects were pledged the most money (usd_pledged_real)?


```python
ks_complete.sort_values(by="usd_pledged_real", ascending=False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>157270</th>
      <td>1799979574</td>
      <td>Pebble Time - Awesome Smartwatch, No Compromises</td>
      <td>Product Design</td>
      <td>Design</td>
      <td>USD</td>
      <td>2015-03-28</td>
      <td>500000.0</td>
      <td>2015-02-24 15:44:42</td>
      <td>20338986.27</td>
      <td>successful</td>
      <td>78471</td>
      <td>US</td>
      <td>20338986.27</td>
      <td>20338986.27</td>
      <td>500000.0</td>
    </tr>
    <tr>
      <th>250254</th>
      <td>342886736</td>
      <td>COOLEST COOLER: 21st Century Cooler that's Act...</td>
      <td>Product Design</td>
      <td>Design</td>
      <td>USD</td>
      <td>2014-08-30</td>
      <td>50000.0</td>
      <td>2014-07-08 10:14:37</td>
      <td>13285226.36</td>
      <td>successful</td>
      <td>62642</td>
      <td>US</td>
      <td>13285226.36</td>
      <td>13285226.36</td>
      <td>50000.0</td>
    </tr>
    <tr>
      <th>216629</th>
      <td>2103598555</td>
      <td>Pebble 2, Time 2 + All-New Pebble Core</td>
      <td>Product Design</td>
      <td>Design</td>
      <td>USD</td>
      <td>2016-06-30</td>
      <td>1000000.0</td>
      <td>2016-05-24 15:49:52</td>
      <td>12779843.49</td>
      <td>successful</td>
      <td>66673</td>
      <td>US</td>
      <td>12779843.49</td>
      <td>12779843.49</td>
      <td>1000000.0</td>
    </tr>
    <tr>
      <th>289915</th>
      <td>545070200</td>
      <td>Kingdom Death: Monster 1.5</td>
      <td>Tabletop Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2017-01-08</td>
      <td>100000.0</td>
      <td>2016-11-25 06:01:41</td>
      <td>12393139.69</td>
      <td>successful</td>
      <td>19264</td>
      <td>US</td>
      <td>5228482.00</td>
      <td>12393139.69</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>282416</th>
      <td>506924864</td>
      <td>Pebble: E-Paper Watch for iPhone and Android</td>
      <td>Product Design</td>
      <td>Design</td>
      <td>USD</td>
      <td>2012-05-19</td>
      <td>100000.0</td>
      <td>2012-04-11 06:59:04</td>
      <td>10266845.74</td>
      <td>successful</td>
      <td>68929</td>
      <td>US</td>
      <td>10266845.74</td>
      <td>10266845.74</td>
      <td>100000.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2c. Which 5 projects had the most backers?


```python
ks_complete.sort_values(by="backers", ascending=False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>187652</th>
      <td>1955357092</td>
      <td>Exploding Kittens</td>
      <td>Tabletop Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2015-02-20</td>
      <td>10000.0</td>
      <td>2015-01-20 19:00:19</td>
      <td>8782571.99</td>
      <td>successful</td>
      <td>219382</td>
      <td>US</td>
      <td>8782571.99</td>
      <td>8782571.99</td>
      <td>10000.0</td>
    </tr>
    <tr>
      <th>75900</th>
      <td>1386523707</td>
      <td>Fidget Cube: A Vinyl Desk Toy</td>
      <td>Product Design</td>
      <td>Design</td>
      <td>USD</td>
      <td>2016-10-20</td>
      <td>15000.0</td>
      <td>2016-08-30 22:02:09</td>
      <td>6465690.30</td>
      <td>successful</td>
      <td>154926</td>
      <td>US</td>
      <td>13770.00</td>
      <td>6465690.30</td>
      <td>15000.0</td>
    </tr>
    <tr>
      <th>292244</th>
      <td>557230947</td>
      <td>Bring Reading Rainbow Back for Every Child, Ev...</td>
      <td>Web</td>
      <td>Technology</td>
      <td>USD</td>
      <td>2014-07-02</td>
      <td>1000000.0</td>
      <td>2014-05-28 15:05:45</td>
      <td>5408916.95</td>
      <td>successful</td>
      <td>105857</td>
      <td>US</td>
      <td>5408916.95</td>
      <td>5408916.95</td>
      <td>1000000.0</td>
    </tr>
    <tr>
      <th>148585</th>
      <td>1755266685</td>
      <td>The Veronica Mars Movie Project</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2013-04-13</td>
      <td>2000000.0</td>
      <td>2013-03-13 15:42:22</td>
      <td>5702153.38</td>
      <td>successful</td>
      <td>91585</td>
      <td>US</td>
      <td>5702153.38</td>
      <td>5702153.38</td>
      <td>2000000.0</td>
    </tr>
    <tr>
      <th>182657</th>
      <td>1929840910</td>
      <td>Double Fine Adventure</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2012-03-14</td>
      <td>400000.0</td>
      <td>2012-02-09 02:52:52</td>
      <td>3336371.92</td>
      <td>successful</td>
      <td>87142</td>
      <td>US</td>
      <td>3336371.92</td>
      <td>3336371.92</td>
      <td>400000.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2d. Which year had the most competition? (# of projects)


```python
ks_complete["year"] = pd.to_datetime(ks_complete["deadline"]).dt.year
```


```python
ks_complete["year"]. value_counts()
```




    2015    74439
    2014    65377
    2016    57112
    2017    52393
    2013    44118
    2012    41507
    2011    25055
    2010     9090
    2009      902
    2018      461
    Name: year, dtype: int64



### 3a. What is the success rate for all projects broken down by `main_category`?


```python
pct_success = (ks_complete['main_category'][ks_complete['state'] == 'successful']
.value_counts() / ks_complete['main_category'].value_counts()
* 100).sort_values(ascending=False)
```


```python
print(pct_success)
```

    Dance           62.580300
    Theater         60.221198
    Comics          54.496269
    Music           49.126974
    Art             41.309263
    Film & Video    37.929097
    Games           36.051032
    Design          35.743326
    Publishing      31.500499
    Photography     30.960187
    Food            25.077272
    Fashion         24.940914
    Crafts          24.419813
    Journalism      21.660959
    Technology      20.254998
    Name: main_category, dtype: float64


`Dance` had the highest success rate, with a surprising 62.5%

### 3b. Show a horizontal bar chart for project success rate by main_category, sorted by highest to lowest.


```python
pct_success = pct_success.sort_values(ascending=True)
pct_success.plot.barh(title = "Percent Completion by Main Category", figsize=(20, 20))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b116190>




![png](output_28_1.png)


### 3c. Within the Games main_category, what is the success rate for each category within it?


```python
games = ks_complete[ks_complete['main_category']=='Games']
games.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>1000056157</td>
      <td>G-Spot Place for Gamers to connect with eachot...</td>
      <td>Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2016-03-25</td>
      <td>200000.0</td>
      <td>2016-02-09 23:01:12</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>US</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200000.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1000170964</td>
      <td>Penny Bingo Playing Card Game fun for the whol...</td>
      <td>Tabletop Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2017-03-27</td>
      <td>1500.0</td>
      <td>2017-03-02 04:01:43</td>
      <td>856.0</td>
      <td>failed</td>
      <td>25</td>
      <td>US</td>
      <td>324.0</td>
      <td>856.0</td>
      <td>1500.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1000328150</td>
      <td>Legacy of Svarog | a Unique 3D Action RPG and ...</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2015-10-30</td>
      <td>50000.0</td>
      <td>2015-08-31 06:33:31</td>
      <td>1410.0</td>
      <td>failed</td>
      <td>38</td>
      <td>US</td>
      <td>1410.0</td>
      <td>1410.0</td>
      <td>50000.0</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
games_pct = pd.crosstab(games['category'], games['state'], normalize='index')
games_pct
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>state</th>
      <th>canceled</th>
      <th>failed</th>
      <th>successful</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Games</th>
      <td>0.169663</td>
      <td>0.584674</td>
      <td>0.245664</td>
    </tr>
    <tr>
      <th>Gaming Hardware</th>
      <td>0.246114</td>
      <td>0.497409</td>
      <td>0.256477</td>
    </tr>
    <tr>
      <th>Live Games</th>
      <td>0.137352</td>
      <td>0.684783</td>
      <td>0.177866</td>
    </tr>
    <tr>
      <th>Mobile Games</th>
      <td>0.176370</td>
      <td>0.736301</td>
      <td>0.087329</td>
    </tr>
    <tr>
      <th>Playing Cards</th>
      <td>0.178995</td>
      <td>0.425828</td>
      <td>0.395178</td>
    </tr>
    <tr>
      <th>Puzzles</th>
      <td>0.115044</td>
      <td>0.495575</td>
      <td>0.389381</td>
    </tr>
    <tr>
      <th>Tabletop Games</th>
      <td>0.163414</td>
      <td>0.276250</td>
      <td>0.560336</td>
    </tr>
    <tr>
      <th>Video Games</th>
      <td>0.202121</td>
      <td>0.593790</td>
      <td>0.204089</td>
    </tr>
  </tbody>
</table>
</div>




```python
games_pct = games_pct['successful'].sort_values(ascending=False)
games_pct
```




    category
    Tabletop Games     0.560336
    Playing Cards      0.395178
    Puzzles            0.389381
    Gaming Hardware    0.256477
    Games              0.245664
    Video Games        0.204089
    Live Games         0.177866
    Mobile Games       0.087329
    Name: successful, dtype: float64



## 4a. Calculate the 'pct_of_goal' for each completed project


```python
ks_complete.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000002330</td>
      <td>The Songs of Adelaide &amp; Abullah</td>
      <td>Poetry</td>
      <td>Publishing</td>
      <td>GBP</td>
      <td>2015-10-09</td>
      <td>1000.0</td>
      <td>2015-08-11 12:12:28</td>
      <td>0.0</td>
      <td>failed</td>
      <td>0</td>
      <td>GB</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1533.95</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000003930</td>
      <td>Greeting From Earth: ZGAC Arts Capsule For ET</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2017-11-01</td>
      <td>30000.0</td>
      <td>2017-09-02 04:43:57</td>
      <td>2421.0</td>
      <td>failed</td>
      <td>15</td>
      <td>US</td>
      <td>100.0</td>
      <td>2421.0</td>
      <td>30000.00</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000004038</td>
      <td>Where is Hank?</td>
      <td>Narrative Film</td>
      <td>Film &amp; Video</td>
      <td>USD</td>
      <td>2013-02-26</td>
      <td>45000.0</td>
      <td>2013-01-12 00:20:50</td>
      <td>220.0</td>
      <td>failed</td>
      <td>3</td>
      <td>US</td>
      <td>220.0</td>
      <td>220.0</td>
      <td>45000.00</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000007540</td>
      <td>ToshiCapital Rekordz Needs Help to Complete Album</td>
      <td>Music</td>
      <td>Music</td>
      <td>USD</td>
      <td>2012-04-16</td>
      <td>5000.0</td>
      <td>2012-03-17 03:24:11</td>
      <td>1.0</td>
      <td>failed</td>
      <td>1</td>
      <td>US</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5000.00</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1000030581</td>
      <td>Chaser Strips. Our Strips make Shots their B*tch!</td>
      <td>Drinks</td>
      <td>Food</td>
      <td>USD</td>
      <td>2016-03-17</td>
      <td>25000.0</td>
      <td>2016-02-01 20:05:12</td>
      <td>453.0</td>
      <td>failed</td>
      <td>40</td>
      <td>US</td>
      <td>453.0</td>
      <td>453.0</td>
      <td>25000.00</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
ks_complete['pct_of_goal'] = ((ks_complete['usd_pledged_real']) / (ks_complete['usd_goal_real'])*100).round(2)
ks_complete['pct_of_goal'].describe()
```




    count    3.704540e+05
    mean     3.272709e+02
    std      2.697464e+04
    min      0.000000e+00
    25%      4.800000e-01
    50%      1.367000e+01
    75%      1.066700e+02
    max      1.042779e+07
    Name: pct_of_goal, dtype: float64



## 4b. What were the top 5 projects when looking at pct_of_goal for all time?


```python
ks_complete.sort_values(by='pct_of_goal', inplace=True, ascending=False)
ks_complete.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>year</th>
      <th>pct_of_goal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>369176</th>
      <td>9509582</td>
      <td>VULFPECK /// The Beautiful Game</td>
      <td>Music</td>
      <td>Music</td>
      <td>USD</td>
      <td>2016-10-17</td>
      <td>1.0</td>
      <td>2016-08-18 09:04:03</td>
      <td>104277.89</td>
      <td>successful</td>
      <td>3917</td>
      <td>US</td>
      <td>23874.13</td>
      <td>104277.89</td>
      <td>1.0</td>
      <td>2016</td>
      <td>10427789.0</td>
    </tr>
    <tr>
      <th>186096</th>
      <td>1947298033</td>
      <td>Re-covering with Friends</td>
      <td>Rock</td>
      <td>Music</td>
      <td>USD</td>
      <td>2016-12-13</td>
      <td>1.0</td>
      <td>2016-10-14 19:04:27</td>
      <td>68764.10</td>
      <td>successful</td>
      <td>955</td>
      <td>US</td>
      <td>9306.00</td>
      <td>68764.10</td>
      <td>1.0</td>
      <td>2016</td>
      <td>6876410.0</td>
    </tr>
    <tr>
      <th>360721</th>
      <td>907870443</td>
      <td>VULFPECK /// Thrill of the Arts</td>
      <td>Music</td>
      <td>Music</td>
      <td>USD</td>
      <td>2015-10-09</td>
      <td>1.0</td>
      <td>2015-08-10 19:31:56</td>
      <td>55266.57</td>
      <td>successful</td>
      <td>1673</td>
      <td>US</td>
      <td>55266.57</td>
      <td>55266.57</td>
      <td>1.0</td>
      <td>2015</td>
      <td>5526657.0</td>
    </tr>
    <tr>
      <th>76290</th>
      <td>1388400809</td>
      <td>Energy Hook</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2013-06-10</td>
      <td>1.0</td>
      <td>2013-05-10 01:22:38</td>
      <td>41535.01</td>
      <td>successful</td>
      <td>1622</td>
      <td>US</td>
      <td>41535.01</td>
      <td>41535.01</td>
      <td>1.0</td>
      <td>2013</td>
      <td>4153501.0</td>
    </tr>
    <tr>
      <th>81368</th>
      <td>1413857335</td>
      <td>Band of Brothers 2nd Chance</td>
      <td>Tabletop Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2016-08-02</td>
      <td>1.0</td>
      <td>2016-07-12 00:29:12</td>
      <td>32843.00</td>
      <td>successful</td>
      <td>268</td>
      <td>US</td>
      <td>26095.00</td>
      <td>32843.00</td>
      <td>1.0</td>
      <td>2016</td>
      <td>3284300.0</td>
    </tr>
  </tbody>
</table>
</div>



## 4c. Plot a histogram distribution of all completed projects by pct_of_goal


Since there were so many kickstarts that failed with \$0 pledged, let's only graph those with at least 10\% pledged


```python
def over_300(val):
    if val > 300:
        return 301
    else:
        return val
```


```python
ks_complete['over_300'] = ks_complete['pct_of_goal'].map(over_300)
ks_complete['over_300'].astype('float')
ks_complete['over_300'].value_counts()
```




    0.00      54648
    301.00    15624
    100.00     4174
    0.01       3894
    0.02       3187
              ...  
    194.30        1
    279.28        1
    78.94         1
    171.26        1
    70.04         1
    Name: over_300, Length: 22412, dtype: int64




```python
fig, ax = plt.subplots()
ax.hist(ks_complete['over_300'], bins=30, range=(0,310))
plt.show()
```


![png](output_42_0.png)


## 4d. Create 2 histogram subplots by pct_of_goal: 1) state=successful, and 2) all others (failed)


```python
pct_pt = ks_complete.pivot_table(values='pct_of_goal', index='state')
pct_pt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pct_of_goal</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>canceled</th>
      <td>123.990150</td>
    </tr>
    <tr>
      <th>failed</th>
      <td>9.061944</td>
    </tr>
    <tr>
      <th>successful</th>
      <td>855.794954</td>
    </tr>
  </tbody>
</table>
</div>




```python
pct_gb = ks_complete.groupby('state')['pct_of_goal']
pct_gb_counts = pct_gb.value_counts()
pct_gb_counts
```




    state       pct_of_goal
    canceled    0.00           12399
                0.01             394
                0.02             296
                0.20             259
                0.10             255
                               ...  
    successful  3284300.00         1
                4153501.00         1
                5526657.00         1
                6876410.00         1
                10427789.00        1
    Name: pct_of_goal, Length: 40628, dtype: int64




```python
ks_success = ks_complete[ks_complete['state'] == 'successful']
```


```python
ks_not_success = ks_complete[ks_complete['state'] != 'successful'] 
ks_not_success['over_300'].value_counts()
```




    0.00      54648
    0.01       3894
    0.02       3187
    0.10       2208
    1.00       2085
              ...  
    96.58         1
    221.00        1
    67.12         1
    53.93         1
    192.00        1
    Name: over_300, Length: 8630, dtype: int64




```python
ks_success['over_300'].value_counts()
```




    301.00    15489
    100.00     4123
    101.00      564
    102.00      511
    105.00      501
              ...  
    246.42        1
    219.19        1
    154.81        1
    202.28        1
    249.24        1
    Name: over_300, Length: 14222, dtype: int64




```python
# matplotlib hist:
fig, ax = plt.subplots(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.hist(ks_success['over_300'], bins=50)
ax1.set_ylim(0,25000)

ax2.hist(ks_not_success['over_300'], bins=50)
ax2.set_ylim(0, 25000)
#ax2.tick_params(bottom='off')

plt.show()

#why x,y labels and ticks incorrect?
```


![png](output_49_0.png)



```python
# seaborn kde plot:
sns.kdeplot(ks_success['over_300'], shade=True)
plt.show()
```


![png](output_50_0.png)



```python
sns.kdeplot(ks_not_success['over_300'], shade=True)
plt.show()
```


![png](output_51_0.png)



```python
# seaborn FacetGrid:

g = sns.FacetGrid(ks_complete, col='state')
g.map(plt.hist, 'pct_of_goal' )
```




    <seaborn.axisgrid.FacetGrid at 0x1a229bf750>




![png](output_52_1.png)


## 5a/b. What is the average/median usd_goal_real for all completed kickstarter projects, broken down by main_category


```python
grouped = ks_complete.groupby('main_category')['usd_goal_real']
grouped.agg([np.mean, np.median, np.min, np.max])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>amin</th>
      <th>amax</th>
    </tr>
    <tr>
      <th>main_category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Art</th>
      <td>39467.623304</td>
      <td>3000.00</td>
      <td>0.01</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Comics</th>
      <td>19675.773316</td>
      <td>3500.00</td>
      <td>0.72</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Crafts</th>
      <td>10423.092597</td>
      <td>2330.35</td>
      <td>1.00</td>
      <td>1.000000e+07</td>
    </tr>
    <tr>
      <th>Dance</th>
      <td>9408.592117</td>
      <td>3310.00</td>
      <td>5.00</td>
      <td>2.000000e+06</td>
    </tr>
    <tr>
      <th>Design</th>
      <td>42199.323873</td>
      <td>10000.00</td>
      <td>0.75</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Fashion</th>
      <td>22530.494784</td>
      <td>5983.55</td>
      <td>0.77</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Film &amp; Video</th>
      <td>82375.686055</td>
      <td>7000.00</td>
      <td>0.15</td>
      <td>1.513959e+08</td>
    </tr>
    <tr>
      <th>Food</th>
      <td>48661.356201</td>
      <td>10000.00</td>
      <td>0.88</td>
      <td>1.663614e+08</td>
    </tr>
    <tr>
      <th>Games</th>
      <td>45148.243871</td>
      <td>8000.00</td>
      <td>0.75</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Journalism</th>
      <td>65528.614392</td>
      <td>5000.00</td>
      <td>1.00</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Music</th>
      <td>15719.283867</td>
      <td>4000.00</td>
      <td>0.74</td>
      <td>5.000000e+07</td>
    </tr>
    <tr>
      <th>Photography</th>
      <td>12266.908892</td>
      <td>4000.00</td>
      <td>1.00</td>
      <td>1.000000e+07</td>
    </tr>
    <tr>
      <th>Publishing</th>
      <td>22590.745149</td>
      <td>5000.00</td>
      <td>0.01</td>
      <td>1.000000e+08</td>
    </tr>
    <tr>
      <th>Technology</th>
      <td>102154.155936</td>
      <td>20000.00</td>
      <td>0.74</td>
      <td>1.101698e+08</td>
    </tr>
    <tr>
      <th>Theater</th>
      <td>27147.451041</td>
      <td>3300.00</td>
      <td>1.00</td>
      <td>4.000000e+07</td>
    </tr>
  </tbody>
</table>
</div>



## 5c/d. What is the average/median usd_pledged_real for all completed kickstarter projects, broken down by main_category


```python
grouped_complete = ks_success.groupby('main_category')['usd_goal_real']
grouped_complete_values = grouped_complete.agg([np.mean, np.median, np.min, np.max])
print(grouped_complete_values)
```

                           mean    median  amin        amax
    main_category                                          
    Art             4410.086374   2000.00  0.01   600000.00
    Comics          5397.352176   2619.71  0.72   250000.00
    Crafts          3013.430794   1000.00  1.00    75783.09
    Dance           4601.094769   3000.00  5.00   125000.00
    Design         15408.835300   7500.00  0.75  1000000.00
    Fashion         9080.955291   5000.00  1.00   300000.00
    Film & Video   11145.188408   5000.00  0.77  2000000.00
    Food           11633.962945   7343.91  0.88   350000.00
    Games          14857.224807   5000.00  0.75  2015608.88
    Journalism      8148.647846   3000.00  1.00   177794.64
    Music           5736.980040   3210.12  0.74   250000.00
    Photography     6490.599673   3000.00  1.00   400000.00
    Publishing      5897.848654   3000.00  0.55   250000.00
    Technology     26286.354186  10000.00  0.75  1000000.00
    Theater         5198.043532   2650.00  1.00   333841.51


##  5e. What insights does this information provide?

`Technology` has the highest mean and median goal in both `completed` and `successful` category. `Art` has the lowest min starting `usd_goal_real` in both categories.

## 5f. Based on this information, if someone wanted to choose the main_category with the highest combined success rate and pledged dollar amount, which one would you recommend?


```python
overall = pd.concat([grouped_complete_values, pct_success], axis=1)
overall = overall.rename(columns={'main_category':'percent_complete'})
overall.sort_values(by='percent_complete', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>amin</th>
      <th>amax</th>
      <th>percent_complete</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dance</th>
      <td>4601.094769</td>
      <td>3000.00</td>
      <td>5.00</td>
      <td>125000.00</td>
      <td>62.580300</td>
    </tr>
    <tr>
      <th>Theater</th>
      <td>5198.043532</td>
      <td>2650.00</td>
      <td>1.00</td>
      <td>333841.51</td>
      <td>60.221198</td>
    </tr>
    <tr>
      <th>Comics</th>
      <td>5397.352176</td>
      <td>2619.71</td>
      <td>0.72</td>
      <td>250000.00</td>
      <td>54.496269</td>
    </tr>
    <tr>
      <th>Music</th>
      <td>5736.980040</td>
      <td>3210.12</td>
      <td>0.74</td>
      <td>250000.00</td>
      <td>49.126974</td>
    </tr>
    <tr>
      <th>Art</th>
      <td>4410.086374</td>
      <td>2000.00</td>
      <td>0.01</td>
      <td>600000.00</td>
      <td>41.309263</td>
    </tr>
    <tr>
      <th>Film &amp; Video</th>
      <td>11145.188408</td>
      <td>5000.00</td>
      <td>0.77</td>
      <td>2000000.00</td>
      <td>37.929097</td>
    </tr>
    <tr>
      <th>Games</th>
      <td>14857.224807</td>
      <td>5000.00</td>
      <td>0.75</td>
      <td>2015608.88</td>
      <td>36.051032</td>
    </tr>
    <tr>
      <th>Design</th>
      <td>15408.835300</td>
      <td>7500.00</td>
      <td>0.75</td>
      <td>1000000.00</td>
      <td>35.743326</td>
    </tr>
    <tr>
      <th>Publishing</th>
      <td>5897.848654</td>
      <td>3000.00</td>
      <td>0.55</td>
      <td>250000.00</td>
      <td>31.500499</td>
    </tr>
    <tr>
      <th>Photography</th>
      <td>6490.599673</td>
      <td>3000.00</td>
      <td>1.00</td>
      <td>400000.00</td>
      <td>30.960187</td>
    </tr>
    <tr>
      <th>Food</th>
      <td>11633.962945</td>
      <td>7343.91</td>
      <td>0.88</td>
      <td>350000.00</td>
      <td>25.077272</td>
    </tr>
    <tr>
      <th>Fashion</th>
      <td>9080.955291</td>
      <td>5000.00</td>
      <td>1.00</td>
      <td>300000.00</td>
      <td>24.940914</td>
    </tr>
    <tr>
      <th>Crafts</th>
      <td>3013.430794</td>
      <td>1000.00</td>
      <td>1.00</td>
      <td>75783.09</td>
      <td>24.419813</td>
    </tr>
    <tr>
      <th>Journalism</th>
      <td>8148.647846</td>
      <td>3000.00</td>
      <td>1.00</td>
      <td>177794.64</td>
      <td>21.660959</td>
    </tr>
    <tr>
      <th>Technology</th>
      <td>26286.354186</td>
      <td>10000.00</td>
      <td>0.75</td>
      <td>1000000.00</td>
      <td>20.254998</td>
    </tr>
  </tbody>
</table>
</div>



`Dance`, `Theater` and `Comics` all have a higher than 50% completion, with `Comics` having the highest mean value at \$5\,397. The top 3 highest mean inculde: `Technology`, `Design` and `Games`, with the highest competetion rate going to `games` with 36%.

A "safer" option would be `Theater` or `Comics`, while a "riskier" option is `Technology` with the highest payoff but lowest completion rate.

## 6a. Create a new column 'months' that shows how many months the project was active between launch and deadline.


```python
ks_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 378661 entries, 0 to 378660
    Data columns (total 15 columns):
     #   Column            Non-Null Count   Dtype  
    ---  ------            --------------   -----  
     0   ID                378661 non-null  int64  
     1   name              378657 non-null  object 
     2   category          378661 non-null  object 
     3   main_category     378661 non-null  object 
     4   currency          378661 non-null  object 
     5   deadline          378661 non-null  object 
     6   goal              378661 non-null  float64
     7   launched          378661 non-null  object 
     8   pledged           378661 non-null  float64
     9   state             378661 non-null  object 
     10  backers           378661 non-null  int64  
     11  country           378661 non-null  object 
     12  usd pledged       374864 non-null  float64
     13  usd_pledged_real  378661 non-null  float64
     14  usd_goal_real     378661 non-null  float64
    dtypes: float64(5), int64(2), object(8)
    memory usage: 43.3+ MB



```python
# first we'll need to convert 'launched' and 'deadline' to type datetime

ks_data['launched'] = pd.to_datetime(ks_data['launched'])
ks_data['deadline'] = pd.to_datetime(ks_data['deadline'])

#now lets generarte the new column
ks_data['months'] = (ks_data['deadline'] - ks_data['launched']) / np.timedelta64(1, 'M')
ks_data['months']
```




    0         1.921726
    1         1.964814
    2         1.477994
    3         0.980988
    4         1.828122
                ...   
    378656    0.982099
    378657    0.882171
    378658    1.484391
    378659    0.993543
    378660    0.907439
    Name: months, Length: 378661, dtype: float64



## 6b. Compare the avg months for successful projects vs non-successful. Add visuals if you'd like.


```python
grouped_months = ks_data.groupby('state')['months']
grouped_months_mean = grouped_months.mean()
grouped_months_values = grouped_months.agg([np.mean, np.median, np.min, np.max])
print(grouped_months_values)
```

                    mean    median      amin        amax
    state                                               
    canceled    1.238679  0.980431  0.003143  488.452193
    failed      1.137120  0.976997  0.001166    3.021422
    live        1.289052  1.083030  0.118394    1.972557
    successful  1.037793  0.968118  0.000166    3.012157
    suspended   1.442903  0.979237  0.028089  549.956536
    undefined   1.061652  0.969542  0.036421    2.942919



```python
grouped_months_mean.plot(kind='bar')
plt.title('Average Active Time in Months')
plt.ylabel('Months')
```




    Text(0, 0.5, 'Months')




![png](output_66_1.png)


## 6c. Does the length of a project in months seem to have an impact?

According to our graphic above, `successful` projects appear to have the shortest timeline of the all, with `undefined` being nearly tied. It is wirth noting that the overall difference between them is only 0.4 months, or about 12 days. The medians are even closer, within 0.12.

## 7a. Calculate the expected value for the Games: Video Games category, with the expected value defined as (median of usd_pledged_real)* (success rate of completed projects).


```python
#first, a pivot table of video games by state and median
ks_data["year"] = pd.to_datetime(ks_data["deadline"]).dt.year
video_games = ks_data[ks_data['category'] == 'Video Games']
video_games_pt = pd.pivot_table(video_games, index = 'state', values = 'usd_pledged_real', aggfunc=(np.median, 'count'))
video_games_pt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>median</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>canceled</th>
      <td>2363</td>
      <td>151.000</td>
    </tr>
    <tr>
      <th>failed</th>
      <td>6942</td>
      <td>184.000</td>
    </tr>
    <tr>
      <th>live</th>
      <td>86</td>
      <td>153.435</td>
    </tr>
    <tr>
      <th>successful</th>
      <td>2386</td>
      <td>11227.920</td>
    </tr>
    <tr>
      <th>suspended</th>
      <td>53</td>
      <td>10.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#now we add a success rate column
video_games_pt['success_rate'] = ((video_games_pt['count'] / video_games_pt['count'].sum()))
video_games_pt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>median</th>
      <th>success_rate</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>canceled</th>
      <td>2363</td>
      <td>151.000</td>
      <td>0.199746</td>
    </tr>
    <tr>
      <th>failed</th>
      <td>6942</td>
      <td>184.000</td>
      <td>0.586813</td>
    </tr>
    <tr>
      <th>live</th>
      <td>86</td>
      <td>153.435</td>
      <td>0.007270</td>
    </tr>
    <tr>
      <th>successful</th>
      <td>2386</td>
      <td>11227.920</td>
      <td>0.201691</td>
    </tr>
    <tr>
      <th>suspended</th>
      <td>53</td>
      <td>10.000</td>
      <td>0.004480</td>
    </tr>
  </tbody>
</table>
</div>




```python
#let's generate the expected value
video_games_pt['expected_value'] = video_games_pt['median'] * video_games_pt['success_rate']
video_games_pt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>median</th>
      <th>success_rate</th>
      <th>expected_value</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>canceled</th>
      <td>2363</td>
      <td>151.000</td>
      <td>0.199746</td>
      <td>30.161708</td>
    </tr>
    <tr>
      <th>failed</th>
      <td>6942</td>
      <td>184.000</td>
      <td>0.586813</td>
      <td>107.973626</td>
    </tr>
    <tr>
      <th>live</th>
      <td>86</td>
      <td>153.435</td>
      <td>0.007270</td>
      <td>1.115419</td>
    </tr>
    <tr>
      <th>successful</th>
      <td>2386</td>
      <td>11227.920</td>
      <td>0.201691</td>
      <td>2264.566113</td>
    </tr>
    <tr>
      <th>suspended</th>
      <td>53</td>
      <td>10.000</td>
      <td>0.004480</td>
      <td>0.044801</td>
    </tr>
  </tbody>
</table>
</div>



We can see the expected value for a successful video game is $2,264

## 7b. Do this again but broken down by deadline year


```python

#lets add a successful column to aid in the table

temp = pd.get_dummies(video_games['state'])
temp = temp['successful']
video_games = pd.concat([video_games, temp], axis=1)
video_games.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>months</th>
      <th>year</th>
      <th>successful</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79</th>
      <td>1000328150</td>
      <td>Legacy of Svarog | a Unique 3D Action RPG and ...</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2015-10-30</td>
      <td>50000.0</td>
      <td>2015-08-31 06:33:31</td>
      <td>1410.00</td>
      <td>failed</td>
      <td>38</td>
      <td>US</td>
      <td>1410.00</td>
      <td>1410.00</td>
      <td>50000.00</td>
      <td>1.962315</td>
      <td>2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>1000524949</td>
      <td>Operation: Make Stuff</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2012-10-18</td>
      <td>200.0</td>
      <td>2012-09-18 04:46:06</td>
      <td>306.72</td>
      <td>successful</td>
      <td>36</td>
      <td>US</td>
      <td>306.72</td>
      <td>306.72</td>
      <td>200.00</td>
      <td>0.979119</td>
      <td>2012</td>
      <td>1</td>
    </tr>
    <tr>
      <th>159</th>
      <td>1000648918</td>
      <td>ADVENT SAGA, DIGITAL TACTICAL CARD GAME</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2014-07-31</td>
      <td>70000.0</td>
      <td>2014-07-01 16:01:41</td>
      <td>15542.11</td>
      <td>failed</td>
      <td>260</td>
      <td>US</td>
      <td>15542.11</td>
      <td>15542.11</td>
      <td>70000.00</td>
      <td>0.963705</td>
      <td>2014</td>
      <td>0</td>
    </tr>
    <tr>
      <th>192</th>
      <td>1000786724</td>
      <td>Harold VS The Horde</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>USD</td>
      <td>2015-01-18</td>
      <td>14000.0</td>
      <td>2014-12-19 07:49:11</td>
      <td>113.00</td>
      <td>failed</td>
      <td>18</td>
      <td>US</td>
      <td>113.00</td>
      <td>113.00</td>
      <td>14000.00</td>
      <td>0.974942</td>
      <td>2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1000811882</td>
      <td>The Sword of Asumi Visual Novel</td>
      <td>Video Games</td>
      <td>Games</td>
      <td>GBP</td>
      <td>2014-11-10</td>
      <td>2500.0</td>
      <td>2014-10-19 20:35:23</td>
      <td>3097.00</td>
      <td>successful</td>
      <td>107</td>
      <td>GB</td>
      <td>4983.78</td>
      <td>4848.23</td>
      <td>3913.65</td>
      <td>0.694621</td>
      <td>2014</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#now lets create a pivot table

video_games_pt = pd.pivot_table(video_games, index='year', values=['name', 'usd_pledged_real', 'successful'], 
                                aggfunc={'name':'count', 'usd_pledged_real': 'median', 'successful':'sum'})
video_games_pt['success_rate'] = video_games_pt['successful'] / video_games_pt['name']
video_games_pt['excepted_value'] = video_games_pt['usd_pledged_real']*video_games_pt['success_rate']
video_games_pt = video_games_pt.reset_index()
print(video_games_pt)
video_games_pt.info()
```

       year  name  successful  usd_pledged_real  success_rate  excepted_value
    0  2009    24         9.0            230.00      0.375000       86.250000
    1  2010   151        40.0            305.00      0.264901       80.794702
    2  2011   331        85.0            310.00      0.256798       79.607251
    3  2012  1412       293.0            798.04      0.207507      165.598952
    4  2013  1846       436.0           1158.21      0.236186      273.553391
    5  2014  2107       421.0            501.00      0.199810      100.104888
    6  2015  2221       373.0            183.54      0.167942       30.824142
    7  2016  1945       382.0            247.12      0.196401       48.534622
    8  2017  1695       347.0            275.00      0.204720       56.297935
    9  2018    97         0.0            121.37      0.000000        0.000000
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   year              10 non-null     int64  
     1   name              10 non-null     int64  
     2   successful        10 non-null     float64
     3   usd_pledged_real  10 non-null     float64
     4   success_rate      10 non-null     float64
     5   excepted_value    10 non-null     float64
    dtypes: float64(4), int64(2)
    memory usage: 608.0 bytes


## 7c. Show this in a bar chart


```python
#here i discovered my earlier misspelling of expected...
sns.barplot(x='year', y='excepted_value', data=video_games_pt,
            label="Expected Value", color="b")
plt.title('Expected Value of Video Games')
```




    Text(0.5, 1.0, 'Expected Value of Video Games')




![png](output_77_1.png)


## 7d. What insights does this data provide you?

2009 to 2011 were fairly stable and then the value climbed in 2012 and peaked in 2013. The expected value dropped precipitously the next two years and since leveled off. As of 2017 it wa approaching, but not yet at, 2009-2011 levels.

## 8a. Calculate the project count, success rate, and pct_of_goal for all projects with 'planner' in the name.
Check for spelling variations in upper/lowercase.


```python
plan = ks_data['name'].str.lower().str.contains('planner', na=False)
```


```python
planner = ks_data[plan]
planner.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 367 entries, 3119 to 378485
    Data columns (total 17 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   ID                367 non-null    int64         
     1   name              367 non-null    object        
     2   category          367 non-null    object        
     3   main_category     367 non-null    object        
     4   currency          367 non-null    object        
     5   deadline          367 non-null    datetime64[ns]
     6   goal              367 non-null    float64       
     7   launched          367 non-null    datetime64[ns]
     8   pledged           367 non-null    float64       
     9   state             367 non-null    object        
     10  backers           367 non-null    int64         
     11  country           367 non-null    object        
     12  usd pledged       364 non-null    float64       
     13  usd_pledged_real  367 non-null    float64       
     14  usd_goal_real     367 non-null    float64       
     15  months            367 non-null    float64       
     16  year              367 non-null    int64         
    dtypes: datetime64[ns](2), float64(6), int64(3), object(6)
    memory usage: 51.6+ KB


There are 367 projects in `planner`


```python
success_rate = len(planner[planner['state']=='successful']) / len(planner[planner['state']!='successful'])
success_rate
```




    0.5355648535564853



The mean success rate for `planner` is 53.5%


```python
planner_mean = planner.copy()
planner_mean['pct_of_goal'] = ((planner_mean['usd_pledged_real']) / (planner_mean['usd_goal_real'])*100).round(2)
mean_pct_goal = planner_mean['pct_of_goal'].mean()
mean_pct_goal
```




    194.64847411444134



The mean `pct_of_goal` for `planner` is 194.6

## 8b. How about all projects with both 'planner' and 'Panda' in the name?


```python
planner_pandas = planner['name'].str.lower().str.contains('panda')
planner_pandas = planner[planner_pandas]
planner_pandas.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>name</th>
      <th>category</th>
      <th>main_category</th>
      <th>currency</th>
      <th>deadline</th>
      <th>goal</th>
      <th>launched</th>
      <th>pledged</th>
      <th>state</th>
      <th>backers</th>
      <th>country</th>
      <th>usd pledged</th>
      <th>usd_pledged_real</th>
      <th>usd_goal_real</th>
      <th>months</th>
      <th>year</th>
      <th>pct_of_goal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>374940</th>
      <td>980774782</td>
      <td>Panda Planner Pro: Happiness + Productivity = ...</td>
      <td>Product Design</td>
      <td>Design</td>
      <td>USD</td>
      <td>2016-04-02</td>
      <td>10000.0</td>
      <td>2016-02-29 17:00:53</td>
      <td>26944.0</td>
      <td>successful</td>
      <td>829</td>
      <td>US</td>
      <td>26944.0</td>
      <td>26944.0</td>
      <td>10000.0</td>
      <td>1.060919</td>
      <td>2016</td>
      <td>269.44</td>
    </tr>
  </tbody>
</table>
</div>



## Congrats!
