# +
from SSD import *
import argparse
def main(args):
    data_dir = args.dir #"/dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
    dataset = CustomDataset(data_dir, args.data)
    #dataset.labels = {i:dataset.labels[i] for i in range(120)}
    #dataset.images = {i:dataset.images[i] for i in range(120)}
    model = Model(num_classes=dataset.num_classes, device = device, model_name = args.model, batch_size=args.batch, parallel=False) # if there is no ckpt to load, pass model_name=None 
    model.fit(dataset, max_epochs=args.epoch)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("--dir", default="/dataset", type=str, help="path to the dataset folder")
    parser.add_argument("--data", default="ssd_data.pt", type=str, help="dataset name to create OR load")
    parser.add_argument("--model", default="ssd_model_130.pt", type=str, help="model name to begin with")
    parser.add_argument("--epoch", default=10, type=int, help="number of epochs to train")
    parser.add_argument("--batch", default=128, type=int, help="the batch size")

    args = parser.parse_args()
    main(args)
    
