import embeddings
import torch
import json
import psycopg2
import io
from PIL import Image
import logging as log
from base64 import b64decode
import sys
import os
import torchvision

log.basicConfig(filename="logs_comparator.log",filemode="w+",level=log.INFO,format="Level:%(levelname)s Message: \t\t %(message)s")
 

def main():
    print("hi")
    '''
    Arguments Expected:
    1: Threshold
    
    '''
    
    arguments=sys.argv

    '''
    Connect to database
    '''
    
    try:
        connection = psycopg2.connect(
            dbname="lost_and_found",
            user="testuser",
            password="testuser",
            host="localhost"
        )
        log.info("Connected to database successfully")
    except psycopg2.Error as e:
        log.error("Unable to connect to the database:", e)
        return


    connection.set_session(autocommit=True)
    curr = connection.cursor()

    
    #loading in model wieghts
    if(os.path.exists("./weights.pt")):
        log.info("Existing weights found")
        model.load_state_dict(torch.load("./weights.pt"))
    else:
        log.error("Weights not found")

    
    data_uri = input()
    header, encoded = data_uri.split("base64,", 1)
    data = b64decode(encoded)
    img = Image.open(io.BytesIO(data))
    

    #Resize model
    transform = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize((224,224)),
    ])

    img = transform(img)
    img = img.type(torch.float32)
    img = torch.unsqueeze(img,dim=0)
    
    model = embeddings.embeddings()
    image_emb = model(img)

    matching_id = []
    threshold =  float(arguments[1])
    
    curr.execute("SELECT id, embedding FROM item WHERE embedding IS NOT NULL")
    found_data = curr.fetchall()

    for data in found_data:
        found_emb = data[1]
        found_emb = json.loads(found_emb)
        found_emb = torch.tensor(found_emb)
        found_emb = torch.unsqueeze(found_emb,dim=0)
        
        result=model.comparator(found_emb,image_emb).item()
        
        if(result>=threshold):
            matching_id.append(str(data[0]))
        

    matching_ids = " ".join(matching_id)
    print(matching_ids)

    return

if _name=="main_":
    main()