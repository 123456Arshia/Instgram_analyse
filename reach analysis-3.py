


# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Load data from a CSV file
data = pd.read_csv("/Instagram data.csv", encoding='latin1')

# Data Visualization

# Distribution of Impressions From Home using seaborn histplot
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'])
plt.show()

# Distribution of Impressions From Hashtags using seaborn histplot
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'])
plt.show()

# Pie chart to show the distribution of impressions from various sources
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels,
             title='Impressions on Instagram Posts From Various Sources', hole=0.3)
fig.show()

# Word Cloud for Caption
text_caption = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud_caption = WordCloud(stopwords=stopwords, background_color="white").generate(text_caption)
plt.style.use('classic')
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud_caption, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Caption")
plt.show()

# Word Cloud for Hashtags
text_hashtags = " ".join(i for i in data.Hashtags)
wordcloud_hashtags = WordCloud(stopwords=stopwords, background_color="white").generate(text_hashtags)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud_hashtags, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Hashtags")
plt.show()

# Scatter plots to explore relationships between Impressions and other variables
figure_likes = px.scatter(data_frame=data, x="Impressions", y="Likes", size="Likes", trendline="ols",
                          title="Relationship Between Likes and Impressions")
figure_likes.show()

figure_comments = px.scatter(data_frame=data, x="Impressions", y="Comments", size="Comments", trendline="ols",
                             title="Relationship Between Comments and Total Impressions")
figure_comments.show()

figure_shares = px.scatter(data_frame=data, x="Impressions", y="Shares", size="Shares", trendline="ols",
                           title="Relationship Between Shares and Total Impressions")
figure_shares.show()

# Correlation matrix between Impressions and other numeric columns
correlation_matrix = data.corr()
print("Correlation between Impressions and other numeric columns:")
print(correlation_matrix["Impressions"].sort_values(ascending=False))

# Conversion rate calculation
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print("Conversion rate (Follows / Profile Visits):", conversion_rate)

# Machine Learning: Passive Aggressive Regressor model

# Prepare the data and split into training and testing sets
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = PassiveAggressiveRegressor()
model.fit(x_train, y_train)

# Evaluate the model on the testing data
r_squared_score = model.score(x_test, y_test)
print("R-squared score of the model:", r_squared_score)

# Make predictions using the trained model
example_features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
predicted_impressions = model.predict(example_features)
print("Predicted Impressions:", predicted_impressions)





