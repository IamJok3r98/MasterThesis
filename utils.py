import pandas as pd
import base64
import numpy as np
import json

def transform_date_DMY(date, formats):
	"""
	Transforms a date into the given format(s)
	"""
	for format in formats:
		try:
			return pd.to_datetime(date, format = format)
		except:
			print('Nothing worked')

def get_long_lat_from_id(idd):
	"""
	This function returns the latitude and longitude from a base64 encoded LocationId.
	"""
	base64_bytes = idd.encode('ascii')
	message_bytes = base64.b64decode(base64_bytes)
	message = message_bytes.decode('ascii').split(',')
	latitude  = message[6]
	longitude = message[5]
	return float(latitude),float(longitude)
    
def count_impact_on_day(xi, start, end): #Function used in the vehicle availability/customer demand calculation
    """
    Given an array of arrays of start and finish time, returns the weight of each array demand/availability on the overall day demand/availability.
    """
    dif = (end-start).total_seconds()/60
    samples = np.linspace(0,dif,200)
    return ((xi[0].total_seconds()/60 < samples) & (samples <= xi[-1].total_seconds()/60))/(xi[-1].total_seconds()/60-xi[0].total_seconds()/60)

def count(xi):
    samples = np.linspace(0, 660 , 200)
    return ((xi[0].total_seconds()/60 < samples) & (samples <= xi[-1].total_seconds()/60))/(xi[-1].total_seconds()/60-xi[0].total_seconds()/60)

def perc_cal(x,y):
        return ' '.join([str(float(a)/x) for a in list(y.split(' '))])
    
    
def return_tasks(filepath):
    """
    This function has the purpose of returning all information about the orders made by the customers
    """
    with open(filepath, "r") as infile:
        data = json.load(infile)
        tasks = pd.DataFrame(columns=['id', 'type', 'address_lon', 'address_lat', 'address_id', 'unit1', 'unit2', 'capabilities', 'time_from', 'time_till', 'addressHandlingDuration', 'tag'])
    for j in data['tasks']:
        to_add = pd.DataFrame([[j['id'], j['type'], j['address']['longitude'], j['address']['latitude'], j['address']['mapLocationId'], j['amounts'][0]['value'], j['amounts'][1]['value'], j['capabilities'], j['timeWindow']['from'], j['timeWindow']['till'], j['addressHandlingDuration'], j['tags'][0]]], columns=['id', 'type', 'address_lon', 'address_lat', 'address_id', 'unit1', 'unit2', 'capabilities', 'time_from', 'time_till', 'addressHandlingDuration', 'tag'])
        tasks = pd.concat([tasks, to_add], ignore_index=True)
    tasks = tasks.astype({'id':'str', 'type':'str', 'address_lon':'float', 'address_lat':'float', 'address_id':'str', 'unit1':'float', 'unit2':'float', 'capabilities':'str', 'time_from':'datetime64', 'time_till':'datetime64', 'addressHandlingDuration':'str', 'tag':'str'})
    tasks['addressHandlingDuration']=pd.to_timedelta(tasks['addressHandlingDuration'])
    return tasks

#Functions to read the depots, tasks, routes
def return_routes(filepath):
    """
    This function has the purpose of returning all information about the vehicles available. 
    """
    with open(filepath, "r") as infile:
        data = json.load(infile)
    return data['routes'][0]['start']['earliestTime'], data['routes'][0]['finish']['latestTime']
