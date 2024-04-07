import embeddings
import torch
import json
import psycopg2
import io
from PIL import Image
import logging as log
from base64 import b64decode

log.basicConfig(filename="logs.log",filemode="w+",level=log.INFO,format="Level:%(levelname)s Message: \t\t %(message)s")
 

def main():

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
            dbname="postgres",
            user="postmoose",
            password="postmoose",
            host="localhost"
        )
        log.info("Connected to database successfully")
    except psycopg2.Error as e:
        log.error("Unable to connect to the database:", e)


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
    image = Image.open(io.BytesIO(data))
    

    #Resize model
    transform = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(224,224),
    ])

    img = transform(img)
    img = img.type(torch.float32)
    img = torch.unsqueeze(img,dim=0)
    
    model = embeddings.embeddings()
    image_emb = model(img)

    matching_id = []
    threshold =  arguments[1]
    
    curr.execute("SELECT * FROM found")
    found_data = curr.fetchall()

    for data in found_data:
        found_emb = data[1]
        found_emb = json.loads(found_emb)
        found_emb = torch.tensor(found_emb)
        found_emb = torch.unsqueeze(found_emb,dim=0)
        
        result=model.comparator(found_emb,image_emb).item()
        
        if(result>=threshold):
            matching_id.appned([data[0],result])
        

    print(matching_id)

if __name__==__main__:
    main()