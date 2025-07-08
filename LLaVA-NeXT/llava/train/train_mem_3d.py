from llava.train.train_3d import train
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

if __name__ == "__main__":
    train()