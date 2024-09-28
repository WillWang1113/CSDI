import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument("--device", default="cuda:0", help="Device for Attack")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--seq_len", type=int, default=96)
parser.add_argument("--pred_len", type=int, default=96)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/user/data/THU-timeseries/electricity/electricity.csv",
)
parser.add_argument("--target", type=str, default="OT")
parser.add_argument("--features", type=str, default="S")
parser.add_argument("--scale", type=bool, default=True)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)



config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + f"_{args.seed}_{args.pred_len}" + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler, target_dim = get_dataloader(
    datatype=args.datatype,
    device=args.device,
    seq_len=args.seq_len,
    pred_len=args.pred_len,
    data_dir=args.data_dir,
    target=args.target,
    scale=args.scale,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    pred_len=args.pred_len
)
