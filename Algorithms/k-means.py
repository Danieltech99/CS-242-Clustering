PARAMS = {}

class K_Means_Device:
# In chronological order of calling
    def __init__(self, indicies_for_data_subset):
        """Initialize k-means member variables"""
        
    def run_on_device(self):
        """Trains the k-means"""
        # Sets data indices for training data input
        
    def get_report_for_server(self):
        updates_for_server = []
        return updates_for_server

    def update_device(self, reports_from_server):
        pass


class Hierarchal_Server:
    # In chronological order of calling
    def __init__(self):
        pass
    def update_server(self, reports_from_devices):
        pass
     def run_on_server(self):
        # r
        pass
    def get_reports_for_devices(self):
        updates_for_devices = []
        return updates_for_devices