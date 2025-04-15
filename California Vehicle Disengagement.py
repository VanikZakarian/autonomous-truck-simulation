#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv('2022-autonomous-vehicle-disengagement-reports-csv.csv', encoding='ISO-8859-1')
print(df.columns)


# In[3]:


#Summary of disengagements by Manufacturer
manufacturer_summary = df.groupby('Manufacturer').size()
print("Disengagements by Manufacturer:\n", manufacturer_summary, "\n")


# In[4]:


#Percentage of vehicles capable of operating without a driver
capable_vehicles = df[df['VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER\n(Yes or No)'].str.upper() == 'YES'].shape[0]
total_vehicles = df.shape[0]
percentage_capable = (capable_vehicles / total_vehicles) * 100

print(f"Percentage of Vehicles Capable of Operating Without a Driver: {percentage_capable:.2f}%\n")


# In[16]:


# Renamed column
df = df.rename(columns={
    'DISENGAGEMENT INITIATED BY\n(AV System, Test Driver, Remote Operator, or Passenger)': 'DisengagementBy'
})



#Percentage of disengagements initiated by each entity
disengagement_by_entity = df['DisengagementBy'].value_counts(normalize=True) * 100
print("Disengagements Initiated By (in %):\n", disengagement_by_entity, "\n")


# In[6]:


import re

def extract_info(description):
    event_pattern = r"(.*?)[.]\s*Root cause:"
    root_cause_pattern = r"Root cause:\s*(.*?)[.]\s*Conditions:"
    conditions_pattern = r"Conditions:\s*(.*?)[.]"

    event = re.search(event_pattern, description)
    root_cause = re.search(root_cause_pattern, description)
    conditions = re.search(conditions_pattern, description)

    return event.group(1) if event else None, root_cause.group(1) if root_cause else None, conditions.group(1) if conditions else None


df['Event'], df['Root Cause'], df['Conditions'] = zip(*df['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'].apply(extract_info))

print(df['Root Cause'].value_counts())


# In[7]:


#Traffic situtation subset
traffic_situation_descriptions = df[df['Root Cause'] == 'traffic situation']['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT']
print(traffic_situation_descriptions.sample(10))


# In[8]:


traffic_situation_events = df[df['Root Cause'] == 'traffic situation']['Event']
print(traffic_situation_events.value_counts().head(10))

traffic_situation_conditions = df[df['Root Cause'] == 'traffic situation']['Conditions']
print(traffic_situation_conditions.value_counts().head(10))


# In[25]:


from fuzzywuzzy import process


common_causes = [
    "object detection issue",
    "traffic situation",
    "Navigation/Localisation issue",
    "lane detection issue",
    "bad lane detection in exit/merge lane",
    "limited control actuation, detection issue",
    "object or lane detection issue"
]

# Function to classify description into one of the common causes
def classify_description(desc):
    match, score = process.extractOne(desc, common_causes)
    if score > 80:  
        return match
    return "Other"

df['Cause Category'] = df['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'].apply(classify_description)  # Replace 'DescriptionColumnName' with actual column name


print(df['Cause Category'].value_counts())


# In[28]:


#Percentages of each category
percentages = (counts / counts.sum()) * 100

print(percentages)


# In[29]:


from fuzzywuzzy import process


common_causes = [
    "Unnecessary lane change",
    "test vehicle could have got too close to another vehicle",
    "SW couldnt perform the manuever safely",
    "The test vehicle attepted to lane change to shoulder/express lane",
    "During a lane change a faster car approached in the target lane",
    "limited control actuation, detection issue",
    "object or lane detection issue"
]

# Function to classify description into one of the common causes
def classify_description(desc):
    match, score = process.extractOne(desc, common_causes)
    if score > 80:  # You can adjust this threshold based on your needs
        return match
    return "Other"


df['Cause Category'] = df['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'].apply(classify_description)  # Replace 'DescriptionColumnName' with actual column name

# Print counts of each category
print(df['Cause Category'].value_counts())

