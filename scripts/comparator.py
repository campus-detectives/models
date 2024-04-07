import embeddings
import torch
import json
import psycopg2
import io
from PIL import Image

def main():

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

    
    img = Image.open(io.StringIO(png))

    #Resize model
    transform = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(224),
    ])

    img = transform(img)
    
    model = embeddings.embeddings()
    image_emb = model(img)

    curr.execute("SELECT * FROM found")

    found_data=curr.fetchall()
    

if __name__==__main__:
    main()