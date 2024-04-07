import embeddings
import torch
import json
import psycopg2
import io
from PIL import Image

def main():

    '''
    Arguments Expected:
    1: Image
    2: Threshold
    
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
        print("Connected to database successfully")
    except psycopg2.Error as e:
        print("Unable to connect to the database:", e)


    connection.set_session(autocommit=True)
    curr = connection.cursor()

    
    #loading in model wieghts
    if(os.path.exists("./weights.pt")):
        print("Existing weights found")
        model.load_state_dict(torch.load("./weights.pt"))
    else:
        print("Weights not found")

    
    img = io.StringIO(arguments[1])
    img = Image.open(img)

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
    threshold =  arguments[2]
    
    curr.execute("SELECT * FROM found")
    found_data = curr.fetchall()

    for data in found_data:
        found_emb = data[1]
        found_emb = json.loads(found_emb)
        found_emb = torch.tensor(found_emb)

        result=model.comparator(found_emb,image_emb)
        
        if(result>=threshold):
            matching_id.appned(data[0])
        
    

if __name__==__main__:
    main()