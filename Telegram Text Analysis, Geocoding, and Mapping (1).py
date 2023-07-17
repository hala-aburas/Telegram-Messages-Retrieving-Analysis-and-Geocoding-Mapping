#!/usr/bin/env python
# coding: utf-8

# # Retrieving Telegram Messages

# In[ ]:


from telethon.sync import TelegramClient
import pandas as pd
from datetime import datetime, timedelta
import pytz

api_id = your api-id
api_hash = 'Your api-hash'
name = 'Your name'
chat = 'Telegram Chat, from where the messages will be retrieved'

client = TelegramClient(name, api_id, api_hash)
client.start()

df = pd.DataFrame()
chat_timezone = pytz.timezone('desired time zone')
now = datetime.now(chat_timezone)
two_hours_ago = now - timedelta(hours=2)

messages = client.get_messages(chat, offset_date=two_hours_ago, reverse=True)
for message in messages:
    print(message)
    data = {"text": message.text, "Date": message.date}
    temp_df = pd.DataFrame(data, index=[1])
    df = df.append(temp_df)


# # Telegram Messages processing using NLTK

# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def preprocess_text(text):
    # Remove non-letter characters and digits (except question marks)
    text = re.sub(r'(?<![ء-ي\s])[^ء-ي\s?]', '', text)
    
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('arabic'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join filtered words into a string
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text


# # Telegram Messages processing using regular expression

# In[ ]:


import nltk
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import re
import matplotlib.pyplot as plt

# Define regular expression
regex = r'\b(\S+)\s+(ازمه|ازمة|مسكره|سالكه|سالكة|مغلقه|مغلقة|مغلق|مفتوح|مسكر|سالك)\b'

checkpoints = []
statuses = []
times = []

# Loop over rows of table and extract checkpoint names and statuses
for index, row in df.iterrows():
    text = row['preprocessed_text']
    
    # Skip sentences with a question mark
    if '?' in text:
        continue
    
    match = re.search(regex, text)
    if match:
        checkpoint_name = match.group(1)
        checkpoint_status = match.group(2)
        checkpoints.append(checkpoint_name)
        statuses.append(checkpoint_status)
        times.append(row['time_col'])

data = {'checkpoint': checkpoints, 'status': statuses, 'time_col': times}
df = pd.DataFrame(data)
df


# # Geocoding the results (Checkpoint names)

# In[ ]:


pip install opencage


# In[ ]:


import pandas as pd
from opencage.geocoder import OpenCageGeocode

# Define your OpenCage Geocoder API key
api_key = 'YOUR_API_KEY'

# Define the bounding box coordinates [southwest_lng, southwest_lat, northeast_lng, northeast_lat]
# Replace with the desired bounding box coordinates for your specific geographic area
lower_lng, lower_lat, upper_lng, upper_lat = southwest_lng, southwest_lat, northeast_lng, northeast_lat

# Create an instance of the geocoder
geocoder = OpenCageGeocode(api_key)

# Geocode the checkpoint names and retrieve the coordinates
geocoded_data = []
for checkpoint_name in df['checkpoint']:
    result = geocoder.geocode(checkpoint_name)
    if result and 'geometry' in result[0]:
        latitude = result[0]['geometry']['lat']
        longitude = result[0]['geometry']['lng']
        
        # Filter the results based on the desired geographic area
        if lower_lng <= longitude <= upper_lng and lower_lat <= latitude <= upper_lat:
            geocoded_data.append((checkpoint_name, latitude, longitude))
        else:
            geocoded_data.append((checkpoint_name, None, None))
    else:
        geocoded_data.append((checkpoint_name, None, None))

# Create a new DataFrame with the geocoded data
geocoded_df = pd.DataFrame(geocoded_data, columns=['checkpoint', 'latitude', 'longitude'])

# Merge the geocoded DataFrame with the original DataFrame
merged_df = pd.merge(df, geocoded_df, on='checkpoint')

# Drop rows with NaN values in latitude and longitude columns
merged_df.dropna(subset=['latitude', 'longitude'], inplace=True)

# Print the merged DataFrame with geocoded data
print(merged_df)


# # Mapping the Geocoded results (Checkpoint names)

# In[ ]:


pip install arcgis


# In[ ]:


import pandas as pd
from arcgis.gis import GIS
from arcgis.features import FeatureLayer, FeatureSet

# Connect to your ArcGIS Online account
gis = GIS("https://www.arcgis.com", "YOUR_USERNAME", "YOUR_PASSWORD")

# Create a new web map
webmap = gis.map()

# Create a FeatureSet from the merged DataFrame
features = merged_df.apply(lambda row: {
    'geometry': {'x': row['longitude'], 'y': row['latitude']},
    'attributes': {'checkpoint': row['checkpoint'], 'status': row['status'], 'time_col': row['time_col']}
}, axis=1).tolist()
feature_set = FeatureSet(features)

# Add the FeatureSet as a layer to the web map
webmap.add_layer(feature_set)

# Display the web map
webmap

