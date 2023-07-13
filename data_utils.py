
##Import 
import pandas as pd
import json
import numpy as np
from utils import get_long_lat_from_id, count_impact_on_day, return_tasks, return_routes , count, perc_cal
from geopy import distance
import credentials as creds
import matrix_calculation as matrix
from tqdm import tqdm
import os
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mlp
from scipy import stats
import scipy
import uuid
import math
import re
import statsmodels.api as sm
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

# Set the font to the standard LaTeX font

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# Increase the font size
plt.rcParams['font.size'] = 16
## Useful function
def save_to_excel(filename, dfs):
    """
    Given a dict of dataframes, for example:
    dfs = {'gadgets': df_gadgets, 'widgets': df_widgets}
    """

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheetname, df in dfs.items():  # loop through `dict` of dataframes
        df.to_excel(writer, sheet_name=sheetname)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width
    writer.save() # type: ignore
    return

## Extract vehicles, orders, depots from OHD format
class OHD_to_Dataframes: #class created for all import purposes of OHD format file to dataframe.
    """
    From a JSON file, this class has several function to return dataframes containing vehicles, tasks and depots.
    """
    def __init__(self,filepath):
        self.filepath=filepath
    
    def return_all_depots(self): #returns a dataframe with all depots with their latitude and longitude
        """
        Return a dataframe with all depots in an instance
        """
        with open(self.filepath,"r") as infile:
            data=json.load(infile)
            df_nest = pd.json_normalize(data, record_path=['depots'])# this is done because not every ORTEC client has the same file structure. In that way, we import in a more robust way.
            depots = df_nest
            depots['latitude'] = np.nan #initialize column 
            depots['longitude'] = np.nan #initialize column
            for i in range(len(depots)): #loop through all depots in file
                latitude, longitude = get_long_lat_from_id(depots['mapLocationId'][i]) #return the latitude and longitude
                depots.at[i,'latitude']= latitude
                depots.at[i,'longitude']= longitude
        return depots
    
    def return_all_orders(self): #return a dataframe with all orders along with their latitude and longitude
        """
        Return a dataframe with all orders along with their latitude and longitude 
        """
        with open(self.filepath,"r") as infile:
            data =json.load(infile)
            df_nest= pd.json_normalize(data,record_path=['tasks'])
            orders = df_nest
            orders['latitude'] = np.nan
            orders['longitude'] = np.nan
            if 'address.latitude' in df_nest.columns:
                orders['latitude']=df_nest['address.latitude']
            else:
                for i in range(len(orders)):
                    latitude,longitude = get_long_lat_from_id(orders['address.mapLocationId'][i])
                    orders.at[i,'latitude']=latitude
                    orders.at[i,'longitude']=longitude	
            if 'address.longitude' in df_nest.columns:
                orders['longitude']=df_nest['address.longitude']
            else:
                pass
            if 'priority' in df_nest.columns:
                orders['priority']=df_nest['priority']
            else:
                pass
        return orders
    
    def return_all_routes(self):
        """
        Return all routes/vehicles available for that instance
        """
        with open(self.filepath,"r") as infile:
            data =json.load(infile)
            df_nest= pd.json_normalize(data,record_path=['routes'])
        return df_nest

## Extract features from a dataframe from OHD_to_Dataframes class
class GetFeatures:
    """
    The purpose of this class is to get several features calculated given an instance object. Orders, vehicles and depot are dataframe objects with one row for each order/vehicle/depot in the given instance.
    """
    def __init__(self,orders,vehicles,depots,continent='Europe'):
        self.orders = orders
        self.vehicles = vehicles
        self.depots = depots
        self.continent = continent
        self.dist_matrix = np.empty((len(self.orders), len(self.orders)))
        #self.centroid = np.empty(2)
        # self.depot = np.empty(2)
        self.time_matrix = np.empty((len(self.orders), len(self.orders)))

    def __str__(self):
        return f"We have an instance containing {len(self.orders)} orders, this will lead to a distance matrix of size {self.dist_matrix.shape}. The centroid of these points is {self.centroid}."

    def get_time_matrix_ortec(self):
        """ This is a function calculating the time matrix using Ortec's API, it takes no additional argument than the class needs.
            It returns the time matrix but also set the value of time_matrix for this class to be the distance matrix.
            In that way we can easily access it later on via *instance of GetFeatures*.time_matrix.
        """
        origin_dest_array = []
        for i in range(len(self.orders)):
                origin_dest_array.append(str(self.orders['latitude'][i])+','+str(self.orders['longitude'][i]))
        if self.continent == 'Europe':
            url = 'https://map-europe-test.orteccloudservices.com' #for EU instances
        elif self.continent == 'NA' : 
            url = 'https://map-namerica-test.orteccloudservices.com' #for NA instances
        else :
            print("Continent given is not supported yet. Please try \'Europe\' or \'NA\' for the continent argument")
            return 
        credentials = creds.Credentials(continent=self.continent,verify=False) #Add a NA creds possibility
        route_client = matrix.MatrixCalculation(url, credentials)
        route_response = route_client.time_distance(origin_dest_array, origin_dest_array)
        rows = list()
        for row_json in route_response['origins']:
               rows.append([x['time'] for x in row_json['destinations']])
        self.time_matrix = np.array(rows)
        return self.time_matrix
    
    def get_dist_matrix_ortec(self):
        """ This is a function calculating the distance matrix using Ortec's API, it takes no additional argument than the class needs.
            It returns the distance matrix but also set the value of dist_matrix for this class to be the distance matrix.
            In that way we can easily access it later on via *instance of GetFeatures*.dist_matrix.
        """
        origin_dest_array = []
        for i in range(len(self.orders)):
                origin_dest_array.append(str(self.orders['latitude'][i])+','+str(self.orders['longitude'][i]))
        if self.continent == 'Europe':
            url = 'https://map-europe-test.orteccloudservices.com' #for EU instances
        elif self.continent == 'NA' : 
            url = 'https://map-namerica-test.orteccloudservices.com' #for NA instances
        else :
            print("Continent given is not supported yet. Please try \'Europe\' or \'NA\' for the continent argument")
            return
        credentials = creds.Credentials(continent=self.continent,verify=False)
        route_client = matrix.MatrixCalculation(url, credentials)
        route_response = route_client.time_distance(origin_dest_array, origin_dest_array)
        rows = list()
        for row_json in route_response['origins']:
               rows.append([x['distance'] for x in row_json['destinations']])
        self.dist_matrix = np.array(rows)
        return self.dist_matrix
    
    def get_dist_matrix_geopy(self):
        """ 
            Please, be aware that this function is not the fastest way to do it and you might want to improve it or use Ortec calculation.

            This is a function calculating the distance matrix using geopy distances, bird fly distances around the earth.
            It takes no additional argument than the class needs.
            It returns the distance matrix but also set the value of dist_matrix for this class to be the distance matrix.
            In that way we can easily access it later on via *instance of GetFeatures*.dist_matrix.
        """
        pos = list(zip(self.orders.latitude, self.orders.longitude))
        dist = np.empty((len(pos), len(pos)))
        for i in tqdm(range(len(pos))):
            for j in range(i + 1):
                if i == j:
                    dist.itemset((i, j), 0)
                else:
                    dist.itemset((i, j), distance.distance(pos[i], pos[j]).m)
                    dist.itemset((j, i), dist[i, j])
        self.dist_matrix = dist
        return self.dist_matrix
    
    def get_dist_matrix_mean_std(self):
        """ This is a function that returns the mean of the distance matrix as well as the std.
        Pay attention, the distance matrix should have been calculated beforehand.
        """
        return self.dist_matrix.mean(), self.dist_matrix.std()

    def get_centroid(self):
        """ It is a function that gets the coordinates of the centroid of all the clients' addresses.
        It takes no additional argument and return a tuple with the coordinates and
        changes the value of self.centroid to the calculated centroid value
        """
        length = len(self.orders)
        self.centroid = sum(self.orders.latitude[i] / length for i in range(length)), sum(self.orders.longitude[i] / length for i in range(length))
        return self.centroid

    def get_dist_depot_centroid(self): 
        """ This function requires a depot dataframe as input.
        It calculates the mean distance between the centroid and all depot. No difference is made between main and sub-depot.
        Distance calculated by bird flight because the centroid might not be an accessible address.
        """
        if hasattr(self,"centroid"):
            pass
        else:
            self.get_centroid()
        depot_pos = list(zip(self.depots.latitude, self.depots.longitude))
        dist = []
        for i in range(len(self.depots)):
            dist.append(distance.distance(depot_pos[i], self.centroid).km)
        return np.mean(dist)

    def find_number_clusters(self):
        """ Function that finds the number of clusters in a given instance.
        The algorithm used for clustering is DBSCAN, the optimal value for distance is found by nearest neighbor clustering analysis and kneelocation.
        Min_samples is set to 4 because it gives quite scattered number of clusters which might be better
        to separate based on number of clusters. Also from literature min_samples= 2*ndim which is 2 in our case. (https://towardsdatascience.com/detecting-knee-elbow-points-in-a-graph-d13fc517a63c, https://raghavan.usc.edu/papers/kneedle-simplex11.pdf)
        """
        neigh = NearestNeighbors(n_neighbors=4, metric='precomputed') #compute nearest neighbors using the distance matrix 
        nbrs = neigh.fit(self.dist_matrix)
        distances, indices = nbrs.kneighbors(self.dist_matrix) #get the distances and indices of each nearest neighbor
        ar = np.array(range(0, len(self.dist_matrix))) #reshape for use in KneeLocator function
        ar2 = np.reshape(np.array(pd.DataFrame(np.sort(distances, axis=0))[[1]]), len(self.dist_matrix)) #reshape for use in KneeLocator function
        kl = KneeLocator(x=ar, y=ar2, curve='convex') #automatically find the elbow in the graph
        opt_eps = kl.elbow_y
        clustering = DBSCAN(eps=opt_eps, min_samples=4, metric='precomputed').fit(self.dist_matrix) # type: ignore
        return len(np.unique(clustering.labels_))

    def retrieve_cluster_labels(self):
        """ Function that finds returns the cluster labels for visualization purposes.
        """
        neigh = NearestNeighbors(n_neighbors=2, metric='precomputed')
        nbrs = neigh.fit(self.dist_matrix)
        distances, indices = nbrs.kneighbors(self.dist_matrix)
        ar = np.array(range(0, len(self.dist_matrix)))
        ar2 = np.reshape(np.array(pd.DataFrame(np.sort(distances, axis=0))[[1]]), len(self.dist_matrix))
        kl = KneeLocator(x=ar, y=ar2, curve='convex')
        opt_eps = kl.elbow_y
        clustering = DBSCAN(eps=opt_eps, min_samples=4, metric='precomputed').fit(self.dist_matrix) # type: ignore
        return clustering.labels_
    
    def get_time_window_mean_sd(self):
        """Function that returns the mean and std of windows of customer demand.
        """
        tw_cols = [col for col in self.orders.columns if 'timeWindow' in col] #finds all columns containing the time window tag (either from or till) (syntax differs between ORTEC clients)
        only_tw_range = self.orders[tw_cols]
        from_col = [col for col in only_tw_range.columns if 'rom' in col]
        tw_from = only_tw_range[from_col]
        till_col = [col for col in only_tw_range.columns if 'ill' in col]
        tw_till = only_tw_range[till_col]
        dif_minutes = np.empty_like(tw_from)
        for i in range(len(tw_from)):
            time_from = datetime.strptime(tw_from.iloc[i,0], '%Y-%m-%dT%H:%M:%S')
            time_till = datetime.strptime(tw_till.iloc[i,0], '%Y-%m-%dT%H:%M:%S')
            dif = time_till - time_from
            dif_minutes[i] = dif.total_seconds()/60
        return np.mean(dif_minutes), np.std(dif_minutes)
    
    def get_number_customers(self):
        """Function that returns the number of customers in the instance.
        """
        return len(self.orders)
    
    def get_number_vehicles(self):
        """Function that requires a vehicle dataframe as input argument.
        It calculates the number of vehicles available for this instance.(NB: They might not all be used after optimization)
        """
        return len(self.vehicles)
    
    def get_working_window_mean_sd(self):
        """Function that requires a vehicle dataframe as input argument.
        It calculates the mean and std of the working time window of all vehicles in the instance.
        Returns a mean and a sd.
        """
        ww_cols = [col for col in self.vehicles.columns if 'maxDurationInMinutes' in col or 'maximumDuration' in col or 'finish.latestTime' in col or 'start.earliestTime' in col] #This is to make it robust to the several syntaxes that exist for the different ORTEC customer, if another syntax is created, pay attention to change this!
        working_window = self.vehicles[ww_cols]
        if 'maxDurationInMinutes' in ww_cols: #if maxDuration is specified, we simply take that value
            return working_window.mean().get('maxDurationInMinutes'), round(np.std(working_window).get('maxDurationInMinutes'),1)
        elif 'finish.latestTime' in ww_cols and 'start.earliestTime' in ww_cols: #otherwise, we compute that value by being the difference between latest finish time and earlier start time.
            dif_minutes = np.empty(len(working_window))
            for i in range(len(working_window)):
                time_from = datetime.strptime(working_window.at[i,'start.earliestTime'], '%Y-%m-%dT%H:%M:%S')
                time_till = datetime.strptime(working_window.at[i,'finish.latestTime'], '%Y-%m-%dT%H:%M:%S')
                dif = time_till - time_from
                dif_minutes[i] = dif.total_seconds()/60 #express it in minutes
            return np.mean(dif_minutes) , np.std(dif_minutes)
        elif any([True if 'maximumDuration' in col else False for col in ww_cols]): #for other ORTEC client, maxDurationInMinutes does not exist and we thus take maximumDuration (the order of the if conditions is set such that for simulated instances, it is always the difference between latest-earliest that's taken into account)
            ww = np.empty_like(working_window)
            for i in range(len(working_window)):
                timedelta = pd.to_timedelta(working_window.iloc[i,0])
                totsec = timedelta.total_seconds()
                ww[i] = totsec/60
            return np.mean(ww), np.std(ww)
        else :
            print("No supported time window format found")
            return np.nan, np.nan
        
    def extract_customer_demand_location(self):
        """
        Extract the peak customer demand location after calculating the weight of each customer demand on the overall demand for each timestamp.
        Returns a value between 0 and 1. 0 close to the start of the day, 1 close to the end of the day. A day being defined by earliest start of a vehicle to latest finish of another.
        """
        interest = []
        start = pd.to_datetime(self.vehicles['start.earliestTime'].min())
        end = pd.to_datetime(self.vehicles['finish.latestTime'].max())
        for index, row in self.orders.iterrows():
            interest.append([pd.to_datetime(row['timeWindow.from'])-start, pd.to_datetime(row['timeWindow.till'])-start])
        is_in_range = np.apply_along_axis(count_impact_on_day,arr=interest, axis=1, start=start, end=end)
        density = np.nansum(is_in_range,axis=0)
        if ((end-start).total_seconds()/60)> 660 :
            density_morning = density[:100]
            location_value_morning = np.mean(np.where(density_morning == density_morning.max())[0])
            density_evening = density[100:]
            location_value_evening = np.mean(np.where(density_evening == density_evening.max())[0])
            location_value = np.round((location_value_evening+location_value_morning)/2)
        else:
            density = density
            location_value = np.round(np.mean(np.where(density == density.max())[0]))
        return location_value/200
    
    def extract_customer_demand_density(self):
        """
        Extract the customer demand array/density after calculating the weight of each customer demand on the overall demand for each timestamp.
        """
        interest = []
        start = pd.to_datetime(self.vehicles['start.earliestTime'].min())
        end = pd.to_datetime(self.vehicles['finish.latestTime'].max())
        for index, row in self.orders.iterrows():
            interest.append([pd.to_datetime(row['timeWindow.from'])-start, pd.to_datetime(row['timeWindow.till'])-start]) #-pd.to_datetime("2023-04-05T04:00:00"),
        is_in_range = np.apply_along_axis(count_impact_on_day,arr=interest, axis=1, start=start, end=end)
        density = np.sum(is_in_range,axis=0,)
        return density
    
    def extract_customer_demand_location_from_simulation(self):
        """
        Extract the peak customer demand location after calculating the weight of each customer demand on the overall demand for each timestamp for a simulated file. Start of day and end of day are thus fixed to given values!
        Returns a value between 0 and 1. 0 close to the start of the day, 1 close to the end of the day. A day being defined by earliest start of a vehicle to latest finish of another.
        """
        interest = []
        start = pd.to_datetime("2023-04-05T04:00:00")
        end = pd.to_datetime("2023-04-05T15:00:00")
        for index, row in self.orders.iterrows():
            interest.append([pd.to_datetime(row['timeWindow.from'])-start, pd.to_datetime(row['timeWindow.till'])-start]) #-pd.to_datetime("2023-04-05T04:00:00"),
        is_in_range = np.apply_along_axis(count_impact_on_day,arr=interest, axis=1, start=start, end=end)
        density = np.sum(is_in_range,axis=0)
        location_value = np.round(np.mean(np.where(density == density.max())[0]))
        return location_value/200
    
    def extract_vehicle_availability_density(self):
        """
        Extract the vehicle availability array/density after calculating the weight of each vehicle availability on the overall availability for each timestamp.
        """
        interest = []
        start = pd.to_datetime(self.vehicles['start.earliestTime'].min())
        end = pd.to_datetime(self.vehicles['finish.latestTime'].max())
        for index, row in self.vehicles.iterrows():
            interest.append([pd.to_datetime(row['start.earliestTime'])-start, pd.to_datetime(row['finish.latestTime'])-start])
        is_in_range = np.apply_along_axis(count_impact_on_day,arr=interest, axis=1, start=start, end=end)
        density = np.sum(is_in_range,axis=0)
        return density
    
    def extract_vehicle_availability_location(self):
        """
        Extract the vehicle availability array/density after calculating the weight of each vehicle availability on the overall availability for each timestamp.
        Returns a value between 0 and 1. 0 close to the start of the day, 1 close to the end of the day. A day being defined by earliest start of a the earliest vehicle to latest finish of the latest one.
        """
        interest = []
        start = pd.to_datetime(self.vehicles['start.earliestTime'].min())
        end = pd.to_datetime(self.vehicles['finish.latestTime'].max())
        for index, row in self.vehicles.iterrows():
            interest.append([pd.to_datetime(row['start.earliestTime'])-start, pd.to_datetime(row['finish.latestTime'])-start])
        is_in_range = np.apply_along_axis(count_impact_on_day,arr=interest, axis=1, start=start, end=end)
        density = np.sum(is_in_range,axis=0)
        if ((end-start).total_seconds()/60)> 660 : #cut in half if the day is longer than 11 hours, that means that we usually have two peaks.
            density_morning = density[:100]
            location_value_morning = np.mean(np.where(density_morning == density_morning.max())[0])
            density_evening = density[100:]
            location_value_evening = np.mean(np.where(density_evening == density_evening.max())[0])
            location_value = np.round((location_value_evening+location_value_morning)/2)
        else:
            density = density
            location_value = np.round(np.mean(np.where(density == density.max())[0]))
        return location_value/200
    
    def extract_vehicle_availability_location_from_simulation(self):

        """
        Extract the vehicle availability array/density after calculating the weight of each vehicle availability on the overall availability for each timestamp. Start of day and end of day are thus fixed to given values!
        Returns a value between 0 and 1. 0 close to the start of the day, 1 close to the end of the day. A day being defined by earliest start of a vehicle to latest finish of another.
        """
        interest = []
        start = pd.to_datetime("2023-04-05T04:00:00")
        end = pd.to_datetime("2023-04-05T15:00:00")
        for index, row in self.vehicles.iterrows():
            interest.append([pd.to_datetime(row['start.earliestTime'])-start, pd.to_datetime(row['finish.latestTime'])-start]) #-pd.to_datetime("2023-04-05T04:00:00"),
        is_in_range = np.apply_along_axis(count_impact_on_day,arr=interest, axis=1, start=start, end=end)
        density = np.sum(is_in_range,axis=0)
        location_value = np.round(np.mean(np.where(density == density.max())[0]))
        return location_value/200
    
## Extract features from logs
class OutputExtractionFromLog:
    """
    This class serves the purpose of extracting interesting information from logs.
    """
    def __init__(self,filepath) -> None:
        if filepath.endswith('.json'):
            self.filepath = filepath
        else:
            raise SyntaxError("The format should be a JSON file")

    def extract_time_cost(self):
        """
        Extract time and cost for each GlobalSolutionImprovements reported in the logs of the OHD algorithm. Takes as input the filepath and returns an array containing the costs and an array containing the timestamp of each better solution.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        cost_array = []
        time_array = []
        for improvement_index in data['globalSolutionImprovements']:
            cost = improvement_index['kpis']['costs']
            time = improvement_index['timestamp']
            cost_array.append(cost)
            time_array.append(time)
        return cost_array,time_array

    def extract_time(self):
        """
        Extract the time for each GlobalSolutionImprovements reported in the logs of the OHD algorithm. Takes as input the filepath and returns an array containing the timestamp of each better solution.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        time_array = []
        for improvement_index in data['globalSolutionImprovements']:
            time = improvement_index['timestamp']
            time_array.append(time)
        return time_array

    def extract_final_time(self):
        """
        Extract the final time from GlobalSolutionImprovements reported in the logs of the OHD algorithm. Takes as input the filepath and returns a value containing the timestamp of the final better solution.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        final_time = data['globalSolutionImprovements'][-1]['timestamp']
        return final_time

    def extract_cost(self):
        """
        Extract the cost for each GlobalSolutionImprovements reported in the logs of the OHD algorithm. Takes as input the filepath and returns an array containing the cost of each better solution.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        cost_array = []
        for improvement_index in data['globalSolutionImprovements']:
            cost = improvement_index['kpis']['costs']
            cost_array.append(cost)
        return cost_array

    def extract_final_cost(self):
        """
        Extract the final cost from GlobalSolutionImprovements reported in the logs of the OHD algorithm. Takes as input the filepath and returns a value containing the cost of the final better solution.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        final_cost = data['globalSolutionImprovements'][-1]['kpis']['costs']
        return final_cost

    def extract_number_of_planned_tasks(self):
        """
        Extract the number of planned tasks for each GlobalSolutionImprovements reported in the logs of the OHD algorithm. Takes as input the filepath and returns an array containing the number of planned tasks of each better solution.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        planned_tasks_array = []
        for improvement_index in data['globalSolutionImprovements']:
            planned_tasks = improvement_index['kpis']['numberOfPlannedTasks']
            planned_tasks_array.append(planned_tasks)
        return planned_tasks_array

## Extract from response file (less information than from the logs)

class OutputExtractionFromResponse:
    """
    This class serves the purpose of extracting interesting information from the response file.
    """
    def __init__(self,filepath) -> None:
        if filepath.endswith('.json'):
            self.filepath = filepath
        else:
            raise SyntaxError("The format should be a JSON file")
        
    def extract_number_of_planned_tasks(self):
        """
        Extract the number of planned tasks for the best solution reported in kpis in the response of the OHD algorithm. Takes as input the filepath and returns a value containing the number of planned tasks for the best solution found.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        nb_planned_tasks = data['kpis']['numberOfPlannedTasks']
        return nb_planned_tasks

    def extract_final_cost(self):
        """
        Extract the final cost for the best solution reported in kpis in the response of the OHD algorithm. Takes as input the filepath and returns a value containing the number of planned tasks for the best solution found.
        """
        with open(self.filepath,"r") as infile:
            data = json.load(infile)
        final_cost = data['kpis']['costs']
        return final_cost

def extract_output_features_from_folder(response_folder,log_folder):
    """
    Takes a response folder and a log folder as input and outputs a dataframe with columns being extracted output features.
    Args:
        response_folder (_str_): _Responsepath_
        log_folder (_str_): _Folderpath_
    """
    final_dict = {'Dataset':[],'Final Cost':[],'Final Time':[],'Number of planned tasks':[],'Cost Array':[], 'Time Array':[], 'Number of planned tasks Array':[]}
    print(f"Extracting from {len([entry for entry in os.listdir(response_folder) if os.path.isfile(os.path.join(response_folder, entry))])} instances")
    for filename in tqdm(os.scandir(response_folder)):
        output = OutputExtractionFromResponse(filename.path)
        cost = output.extract_final_cost()
        nplannedtasks = output.extract_number_of_planned_tasks()
        name = filename.path.split("\\")[-1]
        name2 = name.split('.json')[0]
        name3 = name2.split('_withConfiguration_OHD_Default')[0]
        for files in os.listdir(log_folder):
            if name3 in files:
                output_path = log_folder+"\\"+files
                log_extraction = OutputExtractionFromLog(output_path)
                final_time = log_extraction.extract_final_time()
                nb_planned_tasks_array = log_extraction.extract_number_of_planned_tasks()
                cost_array,time_array = log_extraction.extract_time_cost()
                final_cost_array = ' '.join(str(e) for e in cost_array)
                final_time_array = ' '.join(str(e) for e in time_array)
                final_nb_planned_tasks_array = ' '.join(str(e) for e in nb_planned_tasks_array)
                final_dict['Dataset'].append(name3)
                final_dict['Cost Array'].append(final_cost_array)
                final_dict['Time Array'].append(final_time_array)
                final_dict['Final Cost'].append(cost)
                final_dict['Final Time'].append(final_time)
                final_dict['Number of planned tasks'].append(nplannedtasks)
                final_dict['Number of planned tasks Array'].append(final_nb_planned_tasks_array)
    final = pd.DataFrame.from_dict(final_dict)
    return final
                
        
        
def extract_input_features_from_folder(folderpath):
    """
    Takes a folderpath where the input instances are (NO CONFIGURATION ON THE INPUT/ RAW INPUT) as input and outputs a dataframe with columns being features and rows being instances
    """
    final_dict = {'Dataset':[],'Mean Dist Matrix':[],'Std Dist Matrix':[],'Mean Distance Depot Centroid':[],'Number of clusters':[],
                              'Customer Time Window Mean':[],'Customer Time Window Std':[],'Number of customers':[],'Number of vehicles':[],
                              'Vehicles Time Window Mean':[],'Vehicles Time Window Std':[],'Ortecs Customer':[],'Customer Demand Location':[],'Vehicle Availability':[]}
    print(f"Extracting from {len([entry for entry in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, entry))])} instances")
    for filename in tqdm(os.scandir(folderpath)):
        if filename.is_file():
            if filename.path.endswith(".json"):
                y = OHD_to_Dataframes(filename.path)
                orders = y.return_all_orders()
                depots = y.return_all_depots()
                vehicles = y.return_all_routes()
                name = filename.path.split('\\')[-1].split(".json")[0]
                if ('GIJ' or 'GKB') in name:
                    continent = 'NA'
                else:
                    continent = 'Europe'
                feat = GetFeatures(orders,vehicles,depots,continent)
                feat.get_dist_matrix_ortec()
                dist_mean, dist_std = feat.get_dist_matrix_mean_std()
                dist_centroid = feat.get_dist_depot_centroid()
                num_clusters = feat.find_number_clusters()
                tw_mean, tw_std = feat.get_time_window_mean_sd()
                number_cust = feat.get_number_customers()
                number_vehicles = feat.get_number_vehicles()
                ww_mean,ww_std = feat.get_working_window_mean_sd()
                vehicle_availability = feat.extract_vehicle_availability_location()
                demand_location = feat.extract_customer_demand_location()
                if "DONE_" in name:
                    dataset = name.split('DONE_')[-1]
                elif "responses\\" in name:
                    dataset = name.split("responses\\")[-1]
                else:
                    dataset = name
                if 'Freshful' in name:
                    ortec_customer = 'Client 3'
                elif 'task_num' in name:
                    ortec_customer = 'Simulated'
                elif 'request_converted_converted' in name:
                    ortec_customer = 'Client 1'
                elif '_request' in name:
                    ortec_customer = 'Client 2'
                elif '_converted' in name:
                    ortec_customer = 'Client 4'
                elif ('GIJ' or 'GKB') in name:
                    ortec_customer = 'Client 5'
                else:
                    ortec_customer = "Unknown"
                final_dict['Dataset'].append(dataset)
                final_dict['Mean Dist Matrix'].append(dist_mean)
                final_dict['Std Dist Matrix'].append(dist_std)
                final_dict['Mean Distance Depot Centroid'].append(dist_centroid)
                final_dict['Number of clusters'].append(num_clusters)
                final_dict['Customer Time Window Mean'].append(tw_mean)
                final_dict['Customer Time Window Std'].append(tw_std)
                final_dict['Number of customers'].append(number_cust)
                final_dict['Number of vehicles'].append(number_vehicles)
                final_dict['Vehicles Time Window Mean'].append(ww_mean)
                final_dict['Vehicles Time Window Std'].append(ww_std)
                final_dict['Ortecs Customer'].append(ortec_customer)
                final_dict['Customer Demand Location'].append(demand_location)
                final_dict['Vehicle Availability'].append(vehicle_availability)
    final = pd.DataFrame.from_dict(final_dict)
    return final


def plot_anonymized_map(depots,orders,point_color=None):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # Increase the font size
    plt.rcParams['font.size'] = 16
    depot_x = []
    depot_y = []
    for i in depots.iterrows():
        depot_y.append(i[1]['latitude'])
        depot_x.append(i[1]['longitude'])
    interest_x = []
    interest_y = []
    for i in orders.iterrows():
        interest_y.append(i[1]['latitude'])
        interest_x.append(i[1]['longitude'])
    #Plot figure
    fig, ax = plt.subplots(figsize=(15,10),dpi=300)
    ax.scatter(interest_x,interest_y,s=50,c=point_color,edgecolor='k',alpha=0.8,label='Tasks')
    ax.scatter(depot_x,depot_y,s=120,facecolor='c',edgecolor='k',label='Depots')
    ax.set_axis_on()
    ax.tick_params(axis='both',which='both',direction='inout',bottom=True,left=True,labelbottom = False, labelleft= False)
    plt.legend(loc='upper left')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    plt.show()

def timewindow_summary(orders):
        print("Unique timewindows in minutes: "+ str(np.sort((pd.to_datetime(orders['timeWindow.till'])-pd.to_datetime(orders['timeWindow.from'])).unique().astype('timedelta64[m]'))))
        print("Unique window start times: "+ str(np.sort(pd.to_datetime(orders['timeWindow.from']).dt.time.astype('str').unique())))
        print("Unique window end times: "+ str(np.sort(pd.to_datetime(orders['timeWindow.till']).dt.time.astype('str').unique())))
        pivot=orders[['timeWindow.from', 'timeWindow.till', 'id']].rename(columns={'timeWindow.from':'From', 'timeWindow.till':'Till'})
        pivot['From']=pd.to_datetime(pivot['From']).dt.time.astype('str')
        pivot['Till']=pd.to_datetime(pivot['Till']).dt.time.astype('str')
        fig,ax = plt.subplots(figsize=(15,10),dpi=300)
        sns.heatmap(pd.pivot_table(pivot, values='id', index='From', columns='Till',
                       aggfunc='count'), cmap="Blues", annot=True, fmt="")
        ax.set_axis_on()
        plt.show()
        
def vehicle_costs_plot(routes, type_of_cost):
    if type_of_cost == 'perRoute':
        column = 'costs.perRoute'
        title = 'Cost per Route'
    elif type_of_cost == 'perKilometer':
        column = 'costs.perKilometer'
        title = 'Cost per Kilometer'
    elif type_of_cost == 'perTask':
        column = 'costs.perTask'
        title = 'Cost per Task'
    elif type_of_cost == 'perHour':
        column = 'costs.perHour'
        title = 'Cost per Hour'
    else:
        return
    counts,bins= np.histogram(routes[column])
    fig,ax = plt.subplots(figsize=(15,10),dpi = 300)
    ax.stairs(counts,bins,fill=True)
    ax.set_xlabel(title)
    ax.set_ylabel('Count')
    plt.show()

def plot_complete_anonymized_map(depots,orders,solution_filepath,legend_location='upper left'):
    with open(solution_filepath, 'r') as infile:
        data = json.load(infile)
    depot_x = []
    depot_y = []
    for i in depots.iterrows():
        depot_y.append(i[1]['latitude'])
        depot_x.append(i[1]['longitude'])
    interest_x = []
    interest_y = []
    for i in orders.iterrows():
        interest_y.append(i[1]['latitude'])
        interest_x.append(i[1]['longitude'])
    route_number = 1
    array_of_routes_array_latitude = []
    array_of_routes_array_longitude = []
    for i in data['routes']:
        for j in i['trips']:
            if j['activities'] != []:
                route_array_latitude = []
                route_array_longitude = []
                for u in j['activities']:
                    if 'depotId' in u.keys():
                        route_array_latitude.append(depots[depots['id']==u['depotId']]['latitude'].values[0])
                        route_array_longitude.append(depots[depots['id']==u['depotId']]['longitude'].values[0])
                    elif 'taskId' in u.keys():
                        route_array_latitude.append(orders[orders['id']==u['taskId']]['latitude'].values[0])
                        route_array_longitude.append(orders[orders['id']==u['taskId']]['longitude'].values[0])
                route_number = route_number+1
                array_of_routes_array_latitude.append(route_array_latitude)
                array_of_routes_array_longitude.append(route_array_longitude)
    map = mlp.colormaps['tab20']
    number_of_routes = len(array_of_routes_array_latitude)
    cmap = map(np.linspace(0,1,number_of_routes))
    fig, ax = plt.subplots(figsize=(15,10),dpi=300)
    for i in range(0,len(array_of_routes_array_latitude)):
        ax.plot(array_of_routes_array_longitude[i],array_of_routes_array_latitude[i],'-',color=cmap[i],zorder=1,label=f'Route {i}')
        ax.scatter(array_of_routes_array_longitude[i],array_of_routes_array_latitude[i],s=50,alpha=0.5,zorder=3,color=cmap[i],edgecolors='k')
    ax.scatter(interest_x,interest_y,s=50,facecolor='C0',edgecolor='k',alpha=0.3,label='Tasks',zorder=1)
    #ax.plot(array_of_routes_array_longitude[0],array_of_routes_array_latitude[0],'-',color=cmap[0],zorder=1,label='Routes')
    ax.scatter(depot_x,depot_y,s=120,facecolor='c',edgecolor='k',label='Depots',zorder=3)
    ax.set_axis_on()
    ax.tick_params(axis='both',which='both',direction='inout',bottom=True,left=True,labelbottom = False, labelleft= False)
    plt.legend(loc=legend_location)
    plt.grid(visible=False)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    plt.show()  

    
def show_output_algorithm_in_table(output_filepath):
    with open(output_filepath, 'r') as infile:
        data = json.load(infile)
    dataframe = pd.DataFrame()
    u = 0
    for i in data['kpis'].keys():
        dataframe.loc[u,'KPI'] = i
        dataframe.loc[u,'Value'] = data['kpis'].get(i)
        u = u+1
    return dataframe

def plot_knee_from_array(x_array,y_array):
    fig, ax = plt.subplots(figsize=(15,10),dpi=300,facecolor=(1,1,1))
    fig.tight_layout()
    kl = KneeLocator(x=x_array,y=y_array,curve='convex')
    plt.plot(x_array,y_array, color='b', label = 'Data')
    plt.axvline(x =kl.knee , color = 'b', linestyle='--', label = 'Knee')
    plt.xlabel("Data points (sorted by distance)", fontsize=16)
    plt.ylabel("k-distance value", fontsize=16)
    plt.legend(loc="upper left")

def plot_clustered_example(filepath,legend_location='lower left'):
    y = OHD_to_Dataframes(filepath)
    orders = y.return_all_orders()
    depots = y.return_all_depots()
    vehicles = y.return_all_routes()
    feat = GetFeatures(orders,vehicles,depots)
    feat.get_dist_matrix_ortec()
    clust_labels = feat.retrieve_cluster_labels()
    num_clust = feat.find_number_clusters()
    interest_x = []
    interest_y = []
    for i in orders.iterrows():
        interest_y.append(i[1]['latitude'])
        interest_x.append(i[1]['longitude'])
    depot_x = []
    depot_y = []
    for i in depots.iterrows():
        depot_y.append(i[1]['latitude'])
        depot_x.append(i[1]['longitude'])
    map = mlp.colormaps['tab20']
    cmap = map(np.linspace(0,1,num_clust))
    plottingdf = pd.DataFrame(data={'X':interest_x,'Y':interest_y,'Clust':clust_labels})
    fig, ax = plt.subplots(figsize=(15,10),dpi=300)
    for i in np.unique(clust_labels):
        if i == -1:
            ax.scatter(plottingdf[plottingdf['Clust']==i]['X'].values,plottingdf[plottingdf['Clust']==i]['Y'].values,s=20,facecolor = '0.8',label=f'Unclustered',alpha=0.8,edgecolor='k')
        else : 
            ax.scatter(plottingdf[plottingdf['Clust']==i]['X'].values,plottingdf[plottingdf['Clust']==i]['Y'].values,s=50,facecolor = cmap[clust_labels[clust_labels==i]],label=f'Cluster {i}',alpha=0.8,edgecolor='k')
    ax.scatter(depot_x,depot_y,s=100,facecolor='r',edgecolor='k',label='Depots')
    ax.set_axis_on()
    ax.tick_params(axis='both',which='both',direction='inout',bottom=True,left=True,labelbottom = False, labelleft= False)
    plt.legend(loc=legend_location)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
 
def plot_density(filepath,mode='customer demand',mode2='simulation',show_line = True):
    dataframe = OHD_to_Dataframes(filepath)
    feat = GetFeatures(dataframe.return_all_orders(),dataframe.return_all_routes(),dataframe.return_all_depots())
    if (mode == 'customer demand' and mode2 !='simulation'):
        dens = feat.extract_customer_demand_density()
        loc = feat.extract_customer_demand_location()
        label_plot = 'Customer Demand'
        label_line = 'Demand Location'
        y_lab = 'Customer Demand (arbitrary unit)'
    elif (mode == 'customer demand' and mode2 == 'simulation'):
        dens = feat.extract_customer_demand_density()
        loc = feat.extract_customer_demand_location_from_simulation()
        label_plot = 'Customer Demand'
        label_line = 'Demand Location'
        y_lab = 'Customer Demand (arbitrary unit)'
    elif (mode == 'vehicle availability' and mode2 == 'simulation'):
        dens = feat.extract_vehicle_availability_density()
        loc = feat.extract_vehicle_availability_location_from_simulation()
        label_plot = 'Vehicle Availability'
        label_line = 'Availability Location'
        y_lab = 'Vehicle Availability (arbitrary unit)'
    else:
        dens = feat.extract_vehicle_availability_density()
        loc = feat.extract_vehicle_availability_location()
        label_plot = 'Vehicle Availability'
        label_line = 'Availability Location'
        y_lab = 'Vehicle Availability (arbitrary unit)'
        
    fig,ax = plt.subplots(figsize=(15,10),dpi=300)
    ax.plot(np.linspace(0,1,200),dens,label = label_plot)
    if show_line:
        ax.axvline(x=loc,color = 'k',linestyle='--',label=label_line)
    ax.set_xlabel('Time since earliest start of a vehicle time window', fontsize= 16)
    ax.set_ylabel(y_lab, fontsize= 16)
    plt.legend(loc='upper left')


def plot_histogram(file,x,hue):
    fig,ax = plt.subplots(figsize = (15,10),dpi=300)
    sns.histplot(file,hue=hue,x =x, palette='tab10',kde=True,bins=20)
    
def plot_histogram_start_at_0(file,x,hue,start=0,end=None):
    fig,ax = plt.subplots(figsize = (15,10),dpi=300)
    sns.histplot(file,hue=hue,x =x, palette='tab10',kde=True,bins=20)
    ax.set_xlim(left = start, right= end)
    
def plot_skewness(mean,mean2,variance,variance2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5), facecolor=(1, 1, 1))
    x = np.linspace(0,1 ,100)
    axes[0].plot(x, stats.beta.pdf(x, ((1-mean)/variance-1/mean)*mean**2, ((1-mean)/variance-1/mean)*mean**2*(1/mean-1)))
    axes[0].title.set_text(f'Mean {mean}, variance {variance}')
    axes[1].plot(x, stats.beta.pdf(x, ((1-mean2)/variance2-1/mean2)*mean2**2, ((1-mean2)/variance2-1/mean2)*mean2**2*(1/mean2-1)))
    axes[1].title.set_text(f'Mean {mean2}, variance {variance2}')
    
    
def from_original_to_smoothed(filepath_original,filepath_smoothed):
    for filename in os.listdir(filepath_original):
        f = os.path.join(filepath_original, filename)
        with open(f, "r") as infile:
            data = json.load(infile)

        new_tasks=return_tasks(f)
        routes_start=return_routes(f)[0]
        routes_finish=return_routes(f)[1]

        interest=[]
        for i, row in new_tasks.iterrows():
            interest.append([pd.to_datetime(row['time_from'])-pd.to_datetime(routes_start), pd.to_datetime(row['time_till'])-pd.to_datetime(routes_start)])

        interest=np.array(interest)

        manipulated=new_tasks
        affordable=min(min(interest[:,0])-pd.to_timedelta(15, unit='m'), pd.to_datetime(routes_finish)-pd.to_datetime(routes_start)-max(interest[:,1])-pd.to_timedelta(15, unit='m'))
        to_add=np.linspace(-affordable.total_seconds()/60, affordable.total_seconds()/60, len(new_tasks))
        np.random.shuffle(to_add)
        manipulated['time_from_new']=manipulated['time_from']-pd.to_timedelta(to_add, unit='m')
        manipulated['time_till_new']=manipulated['time_till']-pd.to_timedelta(to_add, unit='m')

        new={'routes':None, 'tasks': None, 'depots':None}
        new['routes']=data['routes']
        new['tasks']=data['tasks']
        new['depots']=data['depots']

        for i, row in new_tasks.iterrows():
            new['tasks'][i]['timeWindow']['from']=manipulated['time_from_new'][i].strftime('%Y-%m-%dT%H:%M:%S')
            new['tasks'][i]['timeWindow']['till']=manipulated['time_till_new'][i].strftime('%Y-%m-%dT%H:%M:%S')
            
        with open(filepath_smoothed+"\\"+filename, "w") as outfile:
            json.dump(new, outfile)
            
def plot_original_smoothed(filepath_original,filepath_smoothed):
    fig, ax = plt.subplots(figsize = (16, 8), dpi=300, facecolor=(1, 1, 1))
    interest=[]
    
    with open(filepath_smoothed, "r") as infile:
        data = json.load(infile)
        for i in data['tasks']:
            interest.append([pd.to_datetime(i['timeWindow']['from'])-pd.to_datetime(data['routes'][0]['start']['earliestTime']), pd.to_datetime(i['timeWindow']['till'])-pd.to_datetime(data['routes'][0]['start']['earliestTime'])])
            
    is_in_range = np.apply_along_axis(count, arr=interest, axis=1)
    density = np.sum(is_in_range, axis=0)
    plt.plot(np.linspace(0, 660, 200), density, label='Smoothed')
    interest=[]
    
    with open(filepath_original, "r") as infile:
        data = json.load(infile)
        for i in data['tasks']:
            interest.append([pd.to_datetime(i['timeWindow']['from'])-pd.to_datetime(data['routes'][0]['start']['earliestTime']), pd.to_datetime(i['timeWindow']['till'])-pd.to_datetime(data['routes'][0]['start']['earliestTime'])])
            
    is_in_range = np.apply_along_axis(count, arr=interest, axis=1)
    density = np.sum(is_in_range, axis=0)
    plt.plot(np.linspace(0, 660, 200), density, label='Original')
    plt.legend(loc="upper left")
    plt.ylabel('Demand', fontsize=15)
    plt.xlabel('Minutes after the delivery period start time', fontsize=15)


def plot_cost_ori_smoothed(dataframe_original,dataframe_smoothed):
    for i in [dataframe_original, dataframe_smoothed]:
        i['final_cost']=i['Cost Array'].apply(lambda x: [float(a) for a in list(x.split(' '))][-1])
    dataframe_smoothed=dataframe_smoothed.sort_values(by='Dataset')
    dataframe_original=dataframe_original.sort_values(by='Dataset')
    fig, ax = plt.subplots(figsize = (16, 8), dpi=300, facecolor=(1, 1, 1))
    plt.scatter(dataframe_original['final_cost'], dataframe_smoothed['final_cost'])
    plt.ylabel('Smoothed Instance Cost', fontsize=15)
    plt.xlabel('Original Instance Cost', fontsize=15)
    plt.show()
    return dataframe_original,dataframe_smoothed

 
def sample_ind(task_num, time_mean_task, time_std_task, demand_loc_task, clusters, distance, dist_var, route_num, time_mean_route, time_std_route, demand_loc_route,pooled_tasks,pooled_routes):
    
    #Task time windows (task_num, time_mean_task, time_std_task, demand_loc_task)
    time_len=np.random.uniform(low=time_mean_task-np.sqrt(3)*time_std_task, high=time_mean_task+np.sqrt(3)*time_std_task, size=task_num*150)
    mean, variance =demand_loc_task, 0.02
    mid=np.random.beta(a=((1-mean)/variance-1/mean)*mean**2, b=((1-mean)/variance-1/mean)*mean**2*(1/mean-1), size=task_num*150)*660
    beginning=mid-time_len/2
    end=mid+time_len/2
    mask=(end<660) & (beginning>0)
    beginning=beginning[mask]
    end=end[mask]
    indices=np.random.choice(len(beginning), size=task_num, replace=False)
    task_windows=np.stack((beginning[indices], end[indices]), axis=-1)
    length=task_windows[:, 1]-task_windows[:, 0]
    mean=np.mean(task_windows[:, 1]-task_windows[:, 0])
    std=np.std(task_windows[:, 1]-task_windows[:, 0])
    addition=length*(time_std_task/std-1)+time_mean_task-mean*time_std_task/std
    left_possible_only=np.logical_and(task_windows[:,0]>addition, task_windows[:,1]>660-addition)
    right_possible_only=np.logical_and(task_windows[:,0]<addition, task_windows[:,1]<660-addition)
    both_possible=np.logical_and(task_windows[:,0]>addition, task_windows[:,1]<660-addition)
    task_windows[:, 0]=task_windows[:, 0]-addition*left_possible_only-addition*both_possible/2
    task_windows[:, 1]=task_windows[:, 1]+addition*right_possible_only+addition*both_possible/2

    #Task locations (clusters, distance, dist_var)
    pool=pooled_tasks
    cluster_centroids=pool.groupby('cluster_label').mean()[['address_lat', 'address_lon']].reset_index()
    seed=np.random.choice(list(pool[pool['cluster_label']==-1].index), replace=False)
    seed_coordinates=pool.iloc[seed]
    seed_coordinates=np.array(seed_coordinates[3:1:-1].values)
    seed_coordinates=seed_coordinates.reshape(1,2)
    seed_coordinates=seed_coordinates.astype(float)
    centroids=np.array(cluster_centroids[['address_lat', 'address_lon']]).astype(float)
    dist = scipy.spatial.distance.cdist(seed_coordinates, centroids)
    cluster_centroids['distance_from_seed']=dist[0]
    cluster_centroids=cluster_centroids.sort_values('distance_from_seed')[['cluster_label', 'distance_from_seed']].reset_index(drop=True)
    cluster_centroids=cluster_centroids[cluster_centroids['cluster_label']!=-1].reset_index(drop=True)

    mean, variance, n = distance, dist_var, 385
    T=n*variance/mean-n+mean
    rv=scipy.stats.betabinom(n, a=(n*mean-mean**2-variance)/T, b=(n-mean)*(n-(mean**2+variance)/mean)/T).pmf(k=range(n+1))
    array=np.random.choice(range(n+1), size=clusters, replace=False, p=rv)
    if len(pooled_tasks[pooled_tasks['cluster_label'].isin(array)])<task_num:
        sampled_tasks_index=np.concatenate((pooled_tasks[pooled_tasks['cluster_label'].isin(array)].index, np.random.choice(pooled_tasks[pooled_tasks['cluster_label']==-1].index, size=task_num-len(pooled_tasks[pooled_tasks['cluster_label'].isin(array)]), replace=False)))
    else:
        sampled_tasks_index=np.concatenate((np.random.choice(pooled_tasks[pooled_tasks['cluster_label'].isin(array)].index, size=math.floor(task_num*(1-0.063790)), replace=False), np.random.choice(pooled_tasks[pooled_tasks['cluster_label']==-1].index, size=math.ceil(task_num*0.063790), replace=False)))

    #Routes (route_num)
    sampled_routes=pooled_routes.groupby(by=['priority', 'start_depot']).sample(frac=route_num/len(pooled_routes))

    route_num=len(sampled_routes)

    #Route time windows (time_mean_route, time_std_route, demand_loc_route)
    time_len=np.random.uniform(low=time_mean_route-np.sqrt(3)*time_std_route, high=time_mean_route+np.sqrt(3)*time_std_route, size=route_num*100)
    mean, variance =demand_loc_route, 0.02
    mid=np.random.beta(a=((1-mean)/variance-1/mean)*mean**2, b=((1-mean)/variance-1/mean)*mean**2*(1/mean-1), size=route_num*100)*660
    beginning=mid-time_len/2
    end=mid+time_len/2
    mask=(end<660) & (beginning>0)
    beginning=beginning[mask]
    end=end[mask]
    indices=np.random.choice(len(beginning), size=route_num, replace=False)
    route_windows=np.stack((beginning[indices], end[indices]), axis=-1)
    length=route_windows[:, 1]-route_windows[:, 0]
    mean=np.mean(route_windows[:, 1]-route_windows[:, 0])
    std=np.std(route_windows[:, 1]-route_windows[:, 0])
    addition=length*(time_std_route/std-1)+time_mean_route-mean*time_std_route/std
    left_possible_only=np.logical_and(route_windows[:,0]>addition, route_windows[:,1]>660-addition)
    right_possible_only=np.logical_and(route_windows[:,0]<addition, route_windows[:,1]<660-addition)
    both_possible=np.logical_and(route_windows[:,0]>addition, route_windows[:,1]<660-addition)
    route_windows[:, 0]=route_windows[:, 0]-addition*left_possible_only-addition*both_possible/2
    route_windows[:, 1]=route_windows[:, 1]+addition*right_possible_only+addition*both_possible/2

    if len(sampled_tasks_index)==0:
        return None
    return sampled_tasks_index, sampled_routes, task_windows, route_windows

def sample_instance(task_num, time_mean_task, time_std_task, demand_loc_task, clusters, distance, dist_var, route_num, time_mean_route, time_std_route, demand_loc_route, folder,pooled_tasks,pooled_depots,pooled_routes):
    depots_list=[
        {
            "id": "fe0d96ff-dd63-4972-ce37-08d737651eab",
            "latitude": 51.92671,
            "longitude": 4.43625,
            "mapLocationId": "MywxMTgsNjI5OTk5NDIsNjI5OTk5NDMsOTYsNC40MzYyNSw1MS45MjY3MSwyLDA=",
            "tags": [
                "1398",
                "MainDepot"
            ]
        },
        {
            "id": "12ce7189-3db2-42cd-ce39-08d737651eab",
            "latitude": 51.77544,
            "longitude": 4.639151,
            "mapLocationId": "MywxMTgsNzI3Mjc4ODgyLDU4NDUwMTkxNSwzNyw0LjYzOTE1MSw1MS43NzU0NCwyLDA=",
            "capacities": [
                {
                    "unit": "unit1",
                    "value": 7980
                },
                {
                    "unit": "unit2",
                    "value": 50000
                }
            ],
            "tags": [
                "13984",
                "SubDepot"
            ]
        },
        {
            "id": "76630d6f-4d64-4ebc-ce3b-08d737651eab",
            "latitude": 52.086249,
            "longitude": 4.859188,
            "mapLocationId": "MywxMDcsNjQwMjM0ODAsNjQwMjM0ODIsMzgsNC44NTkxODgsNTIuMDg2MjQ5LDIsMA==",
            "capacities": [
                {
                    "unit": "unit1",
                    "value": 7980
                },
                {
                    "unit": "unit2",
                    "value": 50000
                }
            ],
            "tags": [
                "13986",
                "SubDepot"
            ]
        },
        {
            "id": "e4ff87e1-edde-496d-8ab1-08d90f061eb5",
            "latitude": 51.4718,
            "longitude": 3.5929,
            "mapLocationId": "MywxMTgsNTQyNzA1ODE5LDk1NDI0MzI3Myw5NywzLjU5MjksNTEuNDcxOCwyLDA=",
            "capacities": [
                {
                    "unit": "unit1",
                    "value": 3990
                },
                {
                    "unit": "unit2",
                    "value": 25000
                }
            ],
            "tags": [
                "13987",
                "SubDepot"
            ]
        }
    ]
    dictionary ={"routes":[], "tasks":[], "depots":[]}
    #Indices and rows simulation
    samples=sample_ind(task_num, time_mean_task, time_std_task, demand_loc_task, clusters, distance, dist_var, route_num, time_mean_route, time_std_route, demand_loc_route,pooled_tasks,pooled_routes)
    sampled_tasks_index=samples[0]
    sampled_routes=samples[1]
    task_windows=samples[2]
    route_windows=samples[3]

    #Append depots    
    for depot in depots_list:
        dictionary['depots'].append(depot)

    #Append tasks
    task_count=0
    for index, row in pooled_tasks.iloc[sampled_tasks_index].iterrows():
        dictionary['tasks'].append({
                "id": uuid.uuid1().hex ,
                "type": row['type'],
                "allowedDepots": [],
                "address": {
                    "latitude": row['address_lat'],
                    "longitude": row['address_lon'],
                    "mapLocationId": row['address_id']
                },
                "amounts": [
                    {
                        "unit": "unit1",
                        "value": row['unit1']
                    },
                    {
                        "unit": "unit2",
                        "value": row['unit2']
                    }
                ],
                "timeWindow": {
                    "from": (pd.to_datetime("2023-04-05T04:00:00")+pd.to_timedelta(task_windows[task_count][0], unit='m')).strftime('%Y-%m-%dT%H:%M:%S'),
                    "till": (pd.to_datetime("2023-04-05T04:00:00")+pd.to_timedelta(task_windows[task_count][1], unit='m')).strftime('%Y-%m-%dT%H:%M:%S')
                },
                "addressHandlingDuration": row['addressHandlingDuration'].isoformat()[0]+row['addressHandlingDuration'].isoformat()[3:],
            })
        task_count=task_count+1


    #Append routes
    route_count=0
    for index, row in sampled_routes.iterrows():
        if route_windows[route_count][1]-route_windows[route_count][0] > 480:
            x = (pd.to_datetime((pd.to_datetime("2023-04-05T04:00:00")+pd.to_timedelta(route_windows[route_count][1], unit='m')).strftime('%Y-%m-%dT%H:%M:%S'))-pd.to_datetime((pd.to_datetime("2023-04-05T04:00:00")+pd.to_timedelta(route_windows[route_count][0], unit='m')).strftime('%Y-%m-%dT%H:%M:%S'))).isoformat()
            sp=re.split('T|H|M|S, ', x)
            maxdur = 'PT'+sp[1]+'H'+sp[2]+'M'+sp[3]
        else :
            maxdur = 'PT8H'
        dictionary['routes'].append({
                "id": uuid.uuid1().hex,
                "start": {
                    "depotId": str(row['start_depot']),
                    "earliestTime": (pd.to_datetime("2023-04-05T04:00:00")+pd.to_timedelta(route_windows[route_count][0], unit='m')).strftime('%Y-%m-%dT%H:%M:%S'),
                },
                "finish": {
                    "depotId": str(row['finish_depot']),
                    "latestTime": (pd.to_datetime("2023-04-05T04:00:00")+pd.to_timedelta(route_windows[route_count][1], unit='m')).strftime('%Y-%m-%dT%H:%M:%S')
                },
                "preparationDuration": row['preparationDuration'].isoformat()[0]+row['preparationDuration'].isoformat()[3]+row['preparationDuration'].isoformat()[6:-2],
                "turnaroundDuration": row['turnaroundDuration'].isoformat()[0]+row['turnaroundDuration'].isoformat()[3]+row['turnaroundDuration'].isoformat()[6:-2],
                "timeDistanceContext": 'car_fastest',
                "capacities": [
                    {
                        "unit": "unit1",
                        "value": row['capacity1']
                    },
                    {
                        "unit": "unit2",
                        "value": row['capacity2']
                    }
                ],
                "rules": {
                    "maximumDuration": maxdur,
                    "maximumNumberOfTrips": 1,
                    "breakRule": {
                        "breakWindows": [
                            {
                                "minimumWorkingTimeBeforeBreak": "PT1H15M",
                                "maximumWorkingTimeBeforeBreak": "PT3H30M",
                                "breakDuration": "PT15M"
                            },
                            {
                                "minimumWorkingTimeBeforeBreak": "PT3H45M",
                                "maximumWorkingTimeBeforeBreak": "PT5H10M",
                                "breakDuration": "PT15M"
                            }
                        ]
                    }
                },
                "costs": {
                    "perRoute": row['perRoutecost'],
                    #{0, 2000, 4000}
                    "perKilometer": row['perKilometercost'],
                    #{0.1, 0.2, 8}
                    "perTask": row['perTaskcost'],
                    #{10, 20, 40}
                    "perHour": row['perHourcost'],
                    #{60, 200, 400}
                    "perUsedTrip": row['perUsedTripcost'],
                    #0
                    "perHourOvertime": row['perHourOvertimecost']
                    #0
                },
                "travelTimeFactor": row['travelTimeFactor'],
                #{1, 1.1764706}
                "addressHandlingTimeFactor": row['addressHandlingTimeFactor'],
                #{1, 1.5}
                "tags": [
                    row['priority'],
                    #{'Learning', 'Normal', 'Overflow'}
                    row['depot']
                    #{'MainDepot', 'SubDepot'}
                ]
            })
        route_count=route_count+1
        
        #Write the simulation to json file
    with open( folder + "\\" + "task_num-"+str(task_num) + "_task_tw_mean-" + str(time_mean_task)+ "_task_tw_std-" + str(time_std_task)+ "_task_tw_dmd-"+str(demand_loc_task)+ "_num_clusters-"+str(clusters)+ "_dm_mean-"+str(distance)+ "_dm_std-"+str(dist_var)+ "_vhc_num-"+str(route_num)+ "_vhc_tw_mean-" + str(time_mean_route)+ "_vhc_tw_std-"+str(time_std_route)+ "_vhc_tw_dmd-"+str(demand_loc_route)+"_TIME-"+str(datetime.now().strftime("%H_%M_%S_%f"))+".json", "w") as outfile:
        json.dump(dictionary, outfile)


def plot_corr_map(full_mat):
    corr = np.abs(full_mat.corr())
    fig,ax = plt.subplots(figsize=(15,10),dpi=300)
    ax = sns.heatmap(corr,vmin=0, vmax=1, center=0.5,
        cmap="binary",
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
def scatter_plot_simulation_check(file,x_axis,y_axis):
    fig,ax = plt.subplots(figsize=(15,10),dpi=300)
    ax.scatter(file[x_axis],file[y_axis])
    ax.set_xlabel(f"{x_axis}")
    ax.set_ylabel(f"{y_axis}")
    plt.show()       
    
    
def plot12_hist_classification(file):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), facecolor=(1, 1, 1))
    fig.suptitle('Histogram of feature values for the 1020 instances simulated', fontsize=20)
    fig.tight_layout()
    for index, colname in enumerate(['Mean_Dist_Matrix', 'Std_Dist_Matrix',
        'Mean_Distance_Depot_Centroid', 'Number_of_clusters',
        'Customer_Time_Window_Mean', 'Customer_Time_Window_Std',
        'Number_of_customers', 'Number_of_vehicles',
        'Vehicles_Time_Window_Mean', 'Vehicles_Time_Window_Std', 'Customer_Demand_Location',
        'Vehicle_Availability_Location']):
        sns.histplot(file, hue='Complete',palette='tab10', x=colname, kde=True, ax=axs[index//4, index%4])


def plot12_hist_regression(file,y_name):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
    fig.tight_layout()
    for index, colname in enumerate(['Mean_Dist_Matrix', 'Std_Dist_Matrix',
       'Mean_Distance_Depot_Centroid', 'Number_of_clusters',
       'Customer_Time_Window_Mean', 'Customer_Time_Window_Std',
       'Number_of_customers', 'Number_of_vehicles',
       'Vehicles_Time_Window_Mean', 'Vehicles_Time_Window_Std', 'Customer_Demand_Location',
       'Vehicle_Availability_Location']):
        sns.regplot(data=file, x=colname, y=y_name, ax=axs[index//4, index%4],line_kws = {'color':'g'})

def coloured_scatter_plot(file,x_axis,y_axis,hue):
    fig,ax = plt.subplots(figsize=(15,10),dpi=300)
    points = ax.scatter(file[x_axis],file[y_axis],c = file[hue],cmap='winter')
    ax.set_xlabel(f"{x_axis}")
    ax.set_ylabel(f"{y_axis}")
    fig.colorbar(points)
    plt.show()
    
def forward_selection(X, y,
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.05, 
                       verbose=True):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded,dtype="float64")
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_selection(X, y,
                           initial_list=[], 
                           threshold_in=0.05, 
                           threshold_out = 0.05, 
                           verbose=True):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop  {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def give_measure_of_fit(x_train,y_train,y_pred_train,x_test,y_test,y_pred):
    r2_train = r2_score(y_train, y_pred_train)
    n_train = x_train.shape[0]  # number of observations
    k_train = x_train.shape[1]  # number of predictors
    adj_r2_train = 1 - ((1 - r2_train) * (n_train - 1) / (n_train - k_train - 1))
    print(f'Training Adjusted R-squared: {adj_r2_train}')
    print("####################################################")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    evr = explained_variance_score(y_test,y_pred)
    n = x_test.shape[0]  # number of observations
    k = x_test.shape[1]  # number of predictors
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    print(f'R-squared: {r2}')
    print(f'Adjusted R-squared: {adj_r2}')
    print(f'MAE : {mae}')
    print(f'MAE% : {mae/y_pred.mean()}')
    mape = mean_absolute_percentage_error(y_test,y_pred)
    print(f'MAPE : {mape}')
    print(f'RMSE: {rmse}')
    print(f'Explained variance ratio: {evr}')
    print("####################################################\n")
    
def perc_time_saved(perc, time_array, cost_array, task_array):
    times = [float(a) for a in list(time_array.split(' '))]
    costs = [float(a) for a in list(cost_array.split(' '))]
    tasks = [float(a) for a in list(task_array.split(' '))]
    final_time = times[-1]
    final_cost = costs[-1]
    final_task = tasks[-1]
    task_full = tasks.index(final_task)
    plateau = final_cost*(1+perc)
    cost_diff = np.absolute(np.array(costs)-plateau)
    cost_diff = cost_diff[task_full:]
    plateau_index = list(cost_diff).index(min(cost_diff))+task_full
    plateau_cost = costs[plateau_index]
    plateau_time = times[plateau_index]
    time_saved_perc = (final_time-plateau_time)/final_time
    return time_saved_perc, plateau_time, plateau_cost, plateau_index

def time_saving_array(time_array, cost_array, task_array):
    savings = []
    for perc in np.linspace(0.01, 1, 100, endpoint=True):
        savings.append(
            str(perc_time_saved(perc, time_array, cost_array, task_array)[0]))
    return ' '.join(savings)

def plot_knee(savings):
    fig, axs = plt.subplots(figsize=(15, 10),dpi=300,facecolor=(1, 1, 1))
    fig.tight_layout()
    plt.plot(np.linspace(0.01, 1, 100),savings, color='b', label = 'Data')
    plt.axvline(x = 0.22, color = 'b', linestyle='--', label = 'Knee')
    plt.xlabel("Cost increase in %")
    plt.ylabel("Time saved in %")
    plt.legend(loc="upper left")
    
def plot_cost(file,xaxis, pct, colname, sample_frac, sample):
    if sample=='complete':
        new_file=file[file['Complete']==True]
    elif sample=='incomplete':
        new_file=file[file['Complete']==False]
    else:
        new_file=file
        
    file_sample = new_file.groupby(colname).apply(
        lambda x: x.sample(frac=sample_frac))

    cmap=cm.rainbow(np.array(file[colname].unique())/np.mean(file[colname].unique()))
    res = dict(zip(file[colname].unique(), cmap))

    fig, ax = plt.subplots(figsize = (16, 8), dpi=1200, facecolor=(1, 1, 1))
    fig.tight_layout()

    xlim=0
    ylim=0

    for i in file_sample.index:

        if pct==True:
            cost = list(file_sample['final_cost_pct'][i].split(' '))
            cost = [float(a) for a in cost] 
        elif pct==False:  
            cost = list(file_sample['Cost_Array'][i].split(' '))
            cost = [float(a) for a in cost]

        if xaxis=='time':
            time = list(file_sample['Time_Array'][i].split(' '))
            time = [float(a) for a in time]
        elif xaxis=='iterations':
            time = range(0, len(cost))
        

        color=res[file_sample[colname][i]]

        xlim=max(xlim, max(time))
        ylim=max(ylim, max(cost))
        if file_sample['Complete'][i]:
            linestyle = 'solid'
        else:
            linestyle = 'dashed'
        ax.plot(time, cost, color = color, linestyle = linestyle, alpha=0.6, linewidth=0.7)

        
        if xaxis=='time' and pct==True:
            plt.plot(file_sample['complete_time'], file_sample['complete_cost_pct'], marker='.', ls='none', ms=8, color='black')
        elif xaxis=='time' and pct==False:
            plt.plot(file_sample['complete_time'], file_sample['complete_cost'], marker='.', ls='none', ms=8, color='black')
            plt.plot(file_sample['time_22'], file_sample['cost_22'], marker='.', ls='none', ms=8, color='blue')
        elif xaxis=='iterations' and pct==True:
            plt.plot(file_sample['complete_index'], file_sample['complete_cost_pct'], marker='.', ls='none', ms=8, color='black')
        elif xaxis=='iterations' and pct==False:
            plt.plot(file_sample['complete_index'], file_sample['complete_cost'], marker='.', ls='none', ms=8, color='black')
            plt.plot(file_sample['iteration_22'], file_sample['cost_22'], marker='.', ls='none', ms=8, color='blue')
            

    plt.colorbar(cm.ScalarMappable(norm=mlp.colors.Normalize(vmin=min(file[colname].unique()), vmax=max(file[colname].unique())), cmap='rainbow'))
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])

    if xaxis=='time':
        ax.set_xlabel('Time (s)')
    elif xaxis=='iterations':
        ax.set_xlabel('Iteration number')

    if pct==True:
        ax.set_ylabel('Percent of final cost')
    elif pct==False:  
        ax.set_ylabel('Cost')

    plt.show()

def return_first_pos(x, y, z):
    if z==False:
        pos=-1
    else:
        tasks=[float(a) for a in list(x.split(' '))]
        pos=tasks.index(tasks[-1])
    interest=[float(a) for a in list(y.split(' '))]
    return pos, interest[pos]

def perc_cal(x,y):
    return ' '.join([str(float(a)/x) for a in list(y.split(' '))])


if __name__ == "__main__":
    output= OutputExtractionFromResponse(".\\data\\intermediate\\responses optimization\\task_num-100_task_tw_mean-60.0_task_tw_std-0.0_task_tw_dmd-0.05_num_clusters-40_dm_mean-100_dm_std-1000_vhc_num-24_vhc_tw_mean-220.0_vhc_tw_std-0.0_vhc_tw_dmd-0.95_TIME-15_09_10_768987_withConfiguration_OHD_Default.json")
    print(output.extract_final_cost())