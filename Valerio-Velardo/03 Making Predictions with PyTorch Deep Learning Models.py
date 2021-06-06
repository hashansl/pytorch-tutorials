#3
import torch
from Implementing_and_Training_a_Neural_Network_with_PyTorch import FeedForwardNet, download_mnist_datasets


class_mapping = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine"
]

def predict(model, input, target,class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor objects---Tensor(1#data,10#outputs) -> [[0.1, 0.01, ..., 0.6]]

        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

        return predicted,expected

if __name__ == "__main__":
    #load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    #load MNIST validation dataset
    _,validation_data = download_mnist_datasets()

    #get a sample from the validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    #make an inference
    predicted, expected = predict(feed_forward_net, input, target,class_mapping)
    print(f"Predicted: '{predicted}','expected: '{expected}'")