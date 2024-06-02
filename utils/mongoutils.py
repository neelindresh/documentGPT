from pymongo import MongoClient



class MongoConnect:
    def __init__(self,uri,db,collection) -> None:
        self.client = MongoClient(uri)
        self.db=self.client[db]
        self.collection=self.db[collection]
        
    def get_data_by_id(self,idx):
        return self.collection.find_one({'id':idx})
    
    def get_data(self,filters):
        return [i for i in self.collection.find(filters)]
    
    def update_data_by_id(self,idx,record):
        try:
            self.collection.update_one({"id":idx},{'$set': record})
            return True
        except:
            return False
    
    def add_meta_data(self, idx, meta_data,storage_name) :
        record=self.collection.find_one({'id' : idx})
        push2db={}
        if "data_sources" not in record:
            push2db["data_sources"]={}
        else:
            push2db["data_sources"]=record["data_sources"]
            
        if "vectorDB" not in push2db["data_sources"]:
            push2db["data_sources"]["vectorDB"]=[]
        else:
            push2db["data_sources"]["vectorDB"]=record["data_sources"]["vectorDB"]
            
        if "meta_data" not in push2db["data_sources"]:
            push2db["data_sources"]["meta_data"]=[]
        else:
            push2db["data_sources"]["meta_data"]=record["data_sources"]["meta_data"]
            

        push2db["data_sources"]["vectorDB"].append({"storage_name" : storage_name, "collection_name" : meta_data[0]["collection_name"]})
        push2db["data_sources"]["meta_data"].append(meta_data[0])
        try :
            update_result = self.collection.update_one(
                {'id' : idx}, 
                {
                    "$set" :push2db
                }
            )

            print("modified count --> ", update_result.modified_count)

            return True 
        except : 
            return False 
            
            
        
    def get_meta_data(self, idx) : 
        return self.collection.find_one({'id' : idx})['data_sources']['meta_data']
    
    def get_meta_data_without_id(self) : 
        output = []
        for usecase in self.collection.find() : 
            try : 
                output += usecase['data_sources']['meta_data']
            except : 
                pass 
        return output

class MongoIngestionStatus(MongoConnect):
    
        
    def set_status(self,status,idx,info):
        if status=="QUEUED":
            self.collection.update_one(
                {"id":idx},
                {"$addToSet":{"ingestion_status":info}}
            )
        elif status=="PROCESSING":
            self.collection.update_one({"id":idx},{'$set':{
                    "ingestion_status.$[updateFriend].status" : status,
                    "ingestion_status.$[updateFriend].start_time" : info['start_time'],
                    
                    }},
                    array_filters=[
                    {"updateFriend.doc_name" : info["doc_name"]},
                    ]
                )
        elif status=="COMPLETED" or status=="FAILED":
            self.collection.update_one({"id":idx},{'$set':{
                    "ingestion_status.$[updateFriend].status" : status,
                    "ingestion_status.$[updateFriend].end_time" : info['end_time'],
                    }},
                    array_filters=[
                    {"updateFriend.doc_name" : info["doc_name"]},
                    ]
                )
        