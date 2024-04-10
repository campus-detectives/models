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
from torchvision import transforms

log.basicConfig(filename="logs_upload.log",filemode="w+",level=log.INFO,format="Level:%(levelname)s Message: \t\t %(message)s")
 

def main():

    '''
    Arguments Expected:
    1: UniqueID for row
    
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
    model = torchvision.models.resnet18(weights = "DEFAULT")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256 , 256)) ,
        transforms.ToTensor() ,
        transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
    ])
    
    activation = {}
    def get_activation(name):
        def hook(model , input , output):
            activation[name] = output.detach()
        return hook
    
    model.avgpool.register_forward_hook(get_activation("avgpool"))


    found_id = arguments[1]
    curr.execute(f"SELECT image FROM item WHERE id={found_id}")
    data_uri = curr.fetchall()[0][0]
    header, encoded = data_uri.split("base64,", 1)
    data = b64decode(encoded)
    img = Image.open(io.BytesIO(data))
    

    img = transform(img)
    img = img.type(torch.float32)
    img = torch.unsqueeze(img,dim=0)
    
    
    image_emb = model(img)
    image_emb = activation["avgpool"].numpy().squeeze()[None , ...]

    log.info(f"Embeddings successfully generated.  Shape {image_emb.shape}")
    
    image_emb = image_emb.tolist()[0]
    image_emb = json.dumps(image_emb)

    curr.execute(f"UPDATE item SET embedding='{image_emb}' WHERE id={found_id}")

    return

if __name__=="__main__":
    main()