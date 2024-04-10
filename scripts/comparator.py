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
import numpy as np
from torchvision import transforms
import scipy.spatial.distance as cdist

log.basicConfig(filename="logs_comparator.log",filemode="w+",level=log.INFO,format="Level:%(levelname)s Message: \t\t %(message)s")


def test_model(input1,input2):

    diff = input1 - input2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    return dist.item()
    

def main():
    '''
    Arguments Expected:
    1: No. of results
    
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

    all_names = []
    all_vecs = None
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

    
    data_uri = input()
    header, encoded = data_uri.split("base64,", 1)
    data = b64decode(encoded)
    img = Image.open(io.BytesIO(data))


    img = transform(img)
    img = img.type(torch.float32)
    img = torch.unsqueeze(img,dim=0)

    with torch.inference_mode():
        image_emb = model(img)
        image_emb  = activation["avgpool"].numpy().squeeze()[None , ...]

    matching_id = []
    n =  int(arguments[1])

    curr.execute("SELECT COUNT(*) from item")

    m = curr.fetchone()[0]
    #print(curr.fetchone())
    if(n>=m):
        n=m
    
    
    curr.execute("SELECT id, embedding FROM item WHERE embedding IS NOT NULL")
    found_data = curr.fetchall()

    final_output=""
    vec=None
    for data in found_data:
        found_emb = data[1]
        matching_id.append(data[0])
        found_emb = json.loads(found_emb)
        found_emb = torch.tensor(found_emb)
        found_emb = torch.unsqueeze(found_emb,dim=0)
        result=test_model(found_emb,image_emb)
        log.info(f"Distance between {data[0]} and input image is {result}")
        '''
        if(result<(35-threshold/5)):
            final_output+=str(data[0])+" "'''
        if(vec is None):
            vec = found_emb
        else:
            vec=np.vstack((vec,found_emb))
    
    #print(vec.shape)
    result=cdist.cdist(vec,image_emb).squeeze().argsort()[0:n]
    #print(result)

    #print(result)
    for i in result:
        final_output+=str(matching_id[i])+" "

    print(final_output)
    
    return

if __name__=="__main__":
    main()