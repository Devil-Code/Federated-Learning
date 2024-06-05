# Federated Learning Project

## Overview

This project implements federated learning for melanoma classification using PyTorch and Flower (FL). Federated learning is a machine learning setting where many clients (in this case, devices owned by end-users) collaboratively train a model under the orchestration of a central server (in this case, implemented using Flower).

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Contributors](#contributors)
- [License](#license)
- [Contact Information](#contact-information)

## Project Structure

The project consists of the following Python files:

1. **server.py**: This file contains the code for the central server that coordinates the federated learning process.
2. **client.py**: This file contains the code for the client devices participating in the federated learning process.
3. **utils.py**: This file contains utility functions and classes used across the project, including data loading, model creation, training, and evaluation.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Flower (FL)
- EfficientNet-PyTorch

You can install these dependencies using pip:

```bash
pip install torch torchvision numpy flwr efficientnet-pytorch
```

## Running the Project

To start the federated learning process, follow these steps:

1. **Start the Server**: Run the `server.py` script to start the central server. You can specify the server address and configuration parameters such as the number of rounds for federated training.

2. **Start the Clients**: Run the `client.py` script on each client device to participate in the federated learning process. You can specify the server address and other parameters such as the model architecture and dataset path.

3. **Monitor Training**: Monitor the training process on the server as it progresses through multiple rounds of federated learning. You can track metrics such as loss, accuracy, and AUC score.

## Configuration

You can configure various aspects of the federated learning process by modifying the arguments passed to the server and client scripts. These arguments include model architecture, dataset paths, number of rounds, batch size, and more.

## Contributors

- [Pritesh Gandhi]
- [Tarika Jain]

## License

This project is licensed under the [GNU GPL v3.0 License](LICENSE).

## Contact Information

For any inquiries or issues, please contact:
- **Pritesh Gandhi**
- **Email**: pgandhi1412@gmail.com
- **GitHub**: [GitHubProfile](https://github.com/Devil-Code)
