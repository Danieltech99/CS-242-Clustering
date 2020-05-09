import matplotlib.pyplot as plt
import csv

def save_test_results(results):
    with open('results.csv', 'w', newline='') as csvfile:
        r_file = csv.writer(csvfile, delimiter=',')
        r_file.writerow(["Data Set (ARI)", "Type"] + list(list(results.values())[0].keys()))
        for data_pair,pair_results in results.items():
            r_file.writerow(list(data_pair) + [i["end"].value for i in list(pair_results.values())])

def plot_rounds(results):
    l = len(results.keys())
    if l > 1:
        fig, axs = plt.subplots(len(results.keys()))
        i = 0
        for data_pair,pair_results in results.items():
            file_name = ' '.join(data_pair)
            axs[i].set_title(file_name)
            for name,data in list(pair_results.items()):
                x = list(range(1, len(data["rounds"])+1))
                y = data["rounds"]
                axs[i].plot(x,y,label=name)
            i+=1
            plt.legend(loc=0,bbox_to_anchor=(1,0.5))
            # plt.savefig('{}.png'.format(file_name), bbox_inches='tight')
        plt.savefig('rounds-algs.png', bbox_inches='tight')
    else:
        for data_pair,pair_results in results.items():
            file_name = ' '.join(data_pair)
            plt.title(file_name)
            for name,data in list(pair_results.items()):
                x = list(range(1, len(data["rounds"])+1))
                y = data["rounds"]
                plt.plot(x,y,label=name)
            plt.legend(loc=0,bbox_to_anchor=(1,0.5))
            # plt.savefig('{}.png'.format(file_name), bbox_inches='tight')
        plt.savefig('rounds-algs.png', bbox_inches='tight')



class MajorPlotter:
    def __init__(self, rounds, devices):
        self.height = int(devices) + 2 # plus server centers and server pred
        self.width = int(rounds)
        # plt.figure(num=None, figsize=(1, 1), dpi=300)
        _, self.axs = plt.subplots(self.height, self.width, sharex=True, sharey=True,figsize=(10,10))
    
    def plot(self, p, data, **kwargs):
        p.scatter(data[:, 0], data[:, 1], s=1, **kwargs)

    def plot_d(self, round, device, data, centers):
        self.plot(self.axs[device, round], data)
        self.plot(self.axs[device, round], centers, marker=">")
    
    def plot_s(self, round, data, centers):
        self.plot(self.axs[-2, round], data)
        self.plot(self.axs[-2, round], centers, marker=">")
    
    def plot_a(self, round, data, pred_labels):
        self.plot(self.axs[-1, round], data, c=pred_labels)

    def save(self, name):
        plt.savefig('Plots/' + str(name) + '.png', dpi=200)




# importing libraries 
import time 
import math 
  
# decorator to calculate duration 
# taken by any function. 
def calculate_time(func): 
      
    # added arguments inside the inner1, 
    # if function takes any arguments, 
    # can be added like this. 
    def inner1(*args, **kwargs): 
  
        # storing time before function execution 
        begin = time.time() 
          
        func(*args, **kwargs) 
  
        # storing time after function execution 
        delta = time.time() - begin
        print('That took {} seconds / {} minutes'.format(delta, delta/60)) 
    return inner1 