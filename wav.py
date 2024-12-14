import torch
import torch.nn as nn

from DataProcessing.net_feature_extractor import net_feature_loader
from runTorch import ResnetDropoutFull

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(filepath, model=ResnetDropoutFull()):
    """
    加载预训练的模型。

    参数:
    - filepath: 预训练模型的文件路径。
    - model: 要加载权重的模型实例，默认为ResnetDropoutFull的实例。

    返回:
    - 加载了预训练权重的模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"
    print(f"Loading model from: {filepath}")
    model.load_state_dict(torch.load(filepath, map_location=map_location))

    return model


def predict_features(model, features):
    # Predict
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    example_input = torch.randn(1, 3, 64, 2).to(device)

    # 使用 example_input 将模型转换为 Torch Script 格式
    traced_model = torch.jit.trace(model, example_input)

    # 保存 Torch Script 模型
    # torch.jit.save(traced_model, 'model_BinaryPresent.pt')
    with torch.no_grad():
        if device is None:
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_loss = 0.0

        all_y = []
        all_y_pred = []
        counter = 1
        i = 1
        print(features.shape)

        for inputs in features:

            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            x = x.unsqueeze(0)
            y = torch.argmax(inputs[1], dim=1, keepdim=True).float()

            print(x.shape)
            if len(x) == 1:
                x = x[0]
            print(x.shape)
            y_pred = model(x)
            print(f"Predicted Class for data point {i + 1}: {y_pred}")

    return y_pred


def run(modelname: str, data_directory: str):
    # Load the trained model (adjust the path to your model)
    model = load_model(modelname)

    # Load features from the directory
    (spectrograms_train, murmurs_train, outcomes_train, s_pitchs_train, d_pitchs_train, locations) = net_feature_loader(
        data_directory)
    print(murmurs_train)

    knowledge_train = torch.zeros((murmurs_train.shape[0], 2))
    for i in range(len(murmurs_train)):
        if (
                torch.argmax(murmurs_train[i]) == 1
                or torch.argmax(murmurs_train[i]) == 2
        ):
            knowledge_train[i, 1] = 1
        else:
            knowledge_train[i, 0] = 1
    y_train = knowledge_train.to(device)
    y_train = torch.tensor(y_train).float()
    y_train = y_train.unsqueeze(0).repeat(1, 3, 1, 1).float()
    for i in range(len(y_train)):
        # Convert spectrograms to tensors

        # Assuming you need to predict using combined features
        predicted_class = predict_features(model, y_train)  # Adjust function as needed

    return (float(predicted_class))
