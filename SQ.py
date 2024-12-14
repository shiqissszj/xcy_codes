import torch

from DataProcessing.net_feature_extractor import net_feature_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(filepath):
    """
    Loads a pre-trained TorchScript model from a .pt file.

    Parameters:
    - filepath: Path to the pre-trained model file.

    Returns:
    - The loaded TorchScript model ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directly load the TorchScript model
    model = torch.jit.load(filepath, map_location=device)

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

        for inputs in features:

            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            x = x.unsqueeze(0)
            y = torch.argmax(inputs[1], dim=1, keepdim=True).float()

            if len(x) == 1:
                x = x[0]

            y_pred = model(x)

            # print(f"所有分类任务的概率 {i}: {y_pred}")

            probabilities = torch.softmax(y_pred, dim=0)

            # Find the predicted class with the highest probability
            _, predicted_label = torch.max(probabilities, dim=0)

            # Output the predicted class
            # print(f"最高概率的分类: {predicted_label.item() + 1}")
            i = i + 1
    return predicted_label.item() + 1


def run(choice: str, data_directory: str):
    # Load the trained model (adjust the path to your model)
    model_choices = {
        "1": "model_BinaryPresent.pt",
        "2": "model_BinaryUnknown.pt",
        "3": "model_Grading.pt",
        "4": "model_Murmur.pt",
        "5": "model_Outcome.pt",
        "6": "model_Pitchs.pt"
    }
    # 重构run方法，无需打印菜单
    # Display the choices to the user
    # print("请选择您要加载的模型文件：")
    # print("1: model_BinaryPresent.pt")
    # print("2: model_BinaryUnknown.pt")
    # print("3: model_Grading.pt")
    # print("4: model_Murmur.pt")
    # print("5: model_Outcome.pt")
    # print("6: model_Pitchs.pt")

    # Get the user's choice
    # choice = input("请输入选择的测验编号: ")

    # Validate the choice and get the corresponding model name
    if choice in model_choices:
        modelname = model_choices[choice]
        model = load_model(modelname)
        # print(f"已加载模型: {modelname}")
    else:
        return -1

    # Specify the directory containing the data
    # data_directory = 'test'

    # Load features from the directory
    (spectrograms_train, murmurs_train, outcomes_train, s_pitchs_train, d_pitchs_train, locations) = net_feature_loader(
        data_directory)

    if (modelname != ""):
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
    return predicted_class
